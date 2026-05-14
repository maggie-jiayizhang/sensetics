import re, json
from pathlib import Path, PureWindowsPath, PurePosixPath
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
from scripts.data_qc import get_approx_sampling_rate, get_dropout_idx


# ---------- filename parsing ----------
FNAME_RE = re.compile(r"^(x_(?P<x>-?\d+)_(?P<date>\d{8})_(?P<time>\d{4}))\.csv$")

# def parse_filename(csv_path: Path):
#     m = FNAME_RE.match(csv_path.name)
#     if not m:
#         return None
#     d = m.groupdict()
#     session_id = m.group(1)  # x_<x>_<date>_<time>
#     return {
#         "session_id": session_id,
#         "x_um": int(d["x"]),
#         "date_yyyymmdd": d["date"],
#         "time_hhmm": d["time"],
#     }

# def parse_filename_alt(csv_path: Path):
#     ts = datetime.strptime(csv_path.name, "debug_%Y%m%d_%H%M.csv")
#     return {
#         "session_id": csv_path.stem,  # e.g. debug_20260128_1410
#         "x_um": None,
#         "date_yyyymmdd": ts.strftime("%Y%m%d"),
#         "time_hhmm": ts.strftime("%H%M"),
#     }

def safe_read_meta(meta_path: Path):
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {"_meta_read_error": True}
    

def parse_session_id(session_id: str) -> dict | None:
    # pattern 1: x_<x>_<date>_<time>
    m = re.match(r"^x_(?P<x>-?\d+)_(?P<date>\d{8})_(?P<time>\d{4})$", session_id)
    if m:
        d = m.groupdict()
        return {
            "session_id": session_id,
            "x_um": int(d["x"]),
            "y_um": None,
            "date_yyyymmdd": d["date"],
            "time_hhmm": d["time"],
        }
    # pattern 2: y_<y>_<date>_<time>
    m = re.match(r"^y_(?P<y>-?\d+)_(?P<date>\d{8})_(?P<time>\d{4})$", session_id)
    if m:
        d = m.groupdict()
        return {
            "session_id": session_id,
            "x_um": None,
            "y_um": int(d["y"]),
            "date_yyyymmdd": d["date"],
            "time_hhmm": d["time"],
        }
    # pattern 3: <anything>_<date>_<time>
    m = re.match(r"^.+_(?P<date>\d{8})_(?P<time>\d{4})$", session_id)
    if m:
        d = m.groupdict()
        return {
            "session_id": session_id,
            "x_um": None,
            "y_um": None,
            "date_yyyymmdd": d["date"],
            "time_hhmm": d["time"],
        }
    return None

def build_sessions_index(data_dir: str | Path) -> pl.DataFrame:
    data_dir = Path(data_dir)
    rows = []

    # each session has a metadata json
    for meta_path in sorted(data_dir.glob("*_metadata.json")):
        meta = safe_read_meta(meta_path) # load metadata json
        meta_dict = meta if isinstance(meta, dict) else {}

        session_id = meta_dict.get("session_id") or meta_path.stem.replace("_metadata", "")

        # parse the filename
        parsed = parse_session_id(session_id)
        # location info if given
        x_um = parsed.get("x_um") if parsed else None
        y_um = parsed.get("y_um") if parsed else None
        # time info if given
        date_yyyymmdd = parsed.get("date_yyyymmdd") if parsed else None
        time_hhmm = parsed.get("time_hhmm") if parsed else None

        total_captured_sweeps = meta_dict.get("total_captured_sweeps")
        saved_sweeps = meta_dict.get("saved_sweeps")

        # config
        config = meta_dict.get("configuration") or {}
        sweeps_per_block = config.get("buffer_sweeps_per_block")
        samples_per_block = config.get("buffer_total_samples")

        # data paths
        timing_csv = meta_dict.get("block_timing_csv")
        timing_csv = meta_path.parent / PureWindowsPath(timing_csv).name if timing_csv else None
        fs_hz_channel = None
        try:
            assert(timing_csv is None or timing_csv.exists()), f"Expected timing CSV not found: {timing_csv}"
        except AssertionError as e:
            filtering = meta_dict.get("filtering")
            if filtering:
                fs_hz_channel = filtering.get("per_channel_sample_rates_hz")
            timing_csv = None
                
        timing_csv = str(timing_csv) if timing_csv is not None else None               

        data_csv = meta_path.parent / f"{session_id}.csv"
        assert(data_csv.exists()), f"Expected data CSV not found: {data_csv}"

        # sampling rate
        fs_hz = (meta_dict.get("timing") or {}).get("per_channel_rate_hz")
        if (timing_csv is not None and sweeps_per_block == 1):
            timing_csv_df = pd.read_csv(timing_csv, index_col=0)
            gaps = np.diff(timing_csv_df["block_end_us"].to_numpy())
            fs_hz, median_gap = get_approx_sampling_rate(gaps, truncate=5)
            # drop out ratio
            drop_out, ratio = get_dropout_idx(gaps, median_gap, threshold_r=1.8)
            assert(ratio < 0.01), f"High dropout ratio {ratio} in session {session_id}"

        elif fs_hz_channel is not None:
            fs_hz = fs_hz_channel.get("1")  # assuming all channels have same rate

        fd = meta_dict.get("force_data") or {}
        force_available = fd.get("available")
        x_force_available = fd.get("x_force_available")
        z_force_available = fd.get("z_force_available")
        total_force_samples = fd.get("total_force_samples")
        calibration_offset_x = fd.get("calibration_offset_x")
        calibration_offset_z = fd.get("calibration_offset_z")
        force_note = fd.get("note")

        rows.append({
            "session_id": session_id,
            "x_um": x_um,
            "y_um": y_um,
            "date_yyyymmdd": date_yyyymmdd,
            "time_hhmm": time_hhmm,
            "total_captured_sweeps": total_captured_sweeps,
            "saved_sweeps": saved_sweeps,
            "sweeps_per_block": sweeps_per_block,
            "samples_per_block": samples_per_block,
            "csv_path": str(data_csv),
            "meta_path": str(meta_path) if meta_path.exists() else None,
            "timestamp": meta_dict.get("timestamp"),
            "notes": meta_dict.get("notes"),
            "fs_hz": fs_hz,
            "force_available": force_available,
            "x_force_available": x_force_available,
            "z_force_available": z_force_available,
            "total_force_samples": total_force_samples,
            "calibration_offset_x": calibration_offset_x,
            "calibration_offset_z": calibration_offset_z,
            "force_note": force_note,
            "timing_csv": timing_csv
        })

    # If nothing matched, return an empty df WITH schema so downstream code doesn't explode
    if not rows:
        return pl.DataFrame(schema={
            "session_id": pl.Utf8,
            "x_um": pl.Int64,
            "y_um": pl.Int64,
            "date_yyyymmdd": pl.Utf8,
            "time_hhmm": pl.Utf8,
            "total_captured_sweeps": pl.Int64,
            "saved_sweeps": pl.Int64,
            "sweeps_per_block": pl.Int64,
            "samples_per_block": pl.Int64,
            "csv_path": pl.Utf8,
            "meta_path": pl.Utf8,
            "timestamp": pl.Utf8,
            "notes": pl.Utf8,
            "fs_hz": pl.Float64,
            "force_available": pl.Boolean,
            "x_force_available": pl.Boolean,
            "z_force_available": pl.Boolean,
            "total_force_samples": pl.Int64,
            "calibration_offset_x": pl.Float64,
            "calibration_offset_z": pl.Float64,
            "force_note": pl.Utf8,
            "timing_csv": pl.Utf8
        })

    return pl.DataFrame(rows).sort(
        ["date_yyyymmdd", "time_hhmm"],
        nulls_last=True
    )

# given a timestamp (hhmm) return the last row before that time
def get_row_by_time(idx: pl.DataFrame, time_hhmm: str) -> dict | None:
    sub = (
        idx
        .filter(pl.col("time_hhmm") <= time_hhmm)
        .sort("time_hhmm")
        .tail(1)
    )
    return sub.to_dicts()[0] if sub.height else None

# given a location, get the session info (latest)
def get_session_for_x(idx: pl.DataFrame, x_um: int):
    sub = idx.filter(pl.col("x_um") == x_um)
    if sub.height == 0:
        raise ValueError(f"No session found for x_um={x_um}")
    # already sorted, so last row is "latest"
    return sub.tail(1).to_dicts()[0]

# load session csv
def load_session_csv(session_row: dict, columns=None):
    if columns is None:
        columns = ["CH1","CH2","CH3","CH4","CH5","Force_Z"]

    csv_path = session_row["csv_path"]

    # Polars read: fast, and can select columns
    df = pl.read_csv(csv_path, columns=columns, ignore_errors=True)

    # If some columns missing, you can handle it here:
    # df = df.select([c for c in columns if c in df.columns])

    return df

# load session data as pandas
def load_session_pandas(row: dict, columns=None, flip=True):
    if columns is None:
        columns = ["CH1","CH2","CH3","CH4","CH5","Force_Z"]
    df_pl = pl.read_csv(row["csv_path"], columns=columns, ignore_errors=True)
    # now that pyarrow is installed, this should work
    df_pd = df_pl.to_pandas()
    if flip: # flip the sign of the signals
        for c in ["CH1","CH2","CH3","CH4","CH5"]:
            if c in df_pd.columns:
                df_pd[c] = -df_pd[c]
    return df_pd
