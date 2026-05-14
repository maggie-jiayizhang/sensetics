#!/usr/bin/env python3
"""
RC Decay Constant Analyzer

Processes sensor decay data across multiple channels, detects falling edges,
fits exponential decays, and aggregates results into a single CSV file.

Usage:
    python rc_decay_analyzer.py --data_dir /path/to/data --channels 1 2 3 4 5
    python rc_decay_analyzer.py  # Uses defaults
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from scipy.signal import iirnotch, filtfilt
from scipy.optimize import curve_fit

# ======================================================================
# ADC to Voltage Conversion
# ======================================================================
MIN_VAL = 0
MAX_VAL = 4095   # 12-bit ADC
MIN_VOLT = 0.0
MAX_VOLT = 3.3


def adc_to_voltage(adc_value, adc_min=MIN_VAL, adc_max=MAX_VAL,
                   min_volt=MIN_VOLT, max_volt=MAX_VOLT):
    """
    Convert ADC integer(s) to voltage.
    Accepts scalars or numpy arrays.
    """
    scale = (max_volt - min_volt) / float(adc_max - adc_min)
    return min_volt + (np.array(adc_value) - adc_min) * scale


# ======================================================================
# Data Loading Functions
# ======================================================================
def scan_channel_files(data_dir, channels=range(1, 6)):
    """
    Scan a folder for each channel's data CSV and metadata JSON.

    Expected filenames:
        ch1_20260506_1408.csv
        ch1_20260506_1408_metadata.json

    Parameters
    ----------
    data_dir : str or Path
        Folder containing decay-factor CSV and metadata files.
    channels : iterable[int]
        Channel numbers to scan, e.g. [1, 2, 3, 4, 5].

    Returns
    -------
    summary_df : pd.DataFrame
        One row per channel/session candidate, with paths and existence flags.
    """
    data_dir = Path(data_dir)

    rows = []

    for ch in channels:
        # Data CSVs: include ch{ch}_*.csv, exclude metadata and block timing files
        data_csvs = sorted([
            p for p in data_dir.glob(f"ch{ch}_*.csv")
            if "metadata" not in p.name.lower()
            and "block_timing" not in p.name.lower()
        ])
        print(f"Channel {ch}: Found {len(data_csvs)} data CSV(s).")

        # Metadata files
        metadata_jsons = sorted(data_dir.glob(f"ch{ch}_*_metadata.json"))

        # Make rows based on discovered data CSVs
        for csv_path in data_csvs:
            expected_metadata = csv_path.with_name(csv_path.stem + "_metadata.json")

            rows.append({
                "channel": ch,
                "session": csv_path.stem.replace(f"ch{ch}_", ""),
                "data_csv_exists": True,
                "data_csv": csv_path,
                "metadata_exists": expected_metadata.exists(),
                "metadata_json": expected_metadata if expected_metadata.exists() else None,
            })

        # Also report metadata files that do not have a matching CSV
        matched_metadata = {
            row["metadata_json"]
            for row in rows
            if row["channel"] == ch and row["metadata_json"] is not None
        }

        for metadata_path in metadata_jsons:
            if metadata_path not in matched_metadata:
                session = metadata_path.stem.replace(f"ch{ch}_", "").replace("_metadata", "")

                expected_csv = metadata_path.with_name(
                    metadata_path.stem.replace("_metadata", "") + ".csv"
                )

                rows.append({
                    "channel": ch,
                    "session": session,
                    "data_csv_exists": expected_csv.exists(),
                    "data_csv": expected_csv if expected_csv.exists() else None,
                    "metadata_exists": True,
                    "metadata_json": metadata_path,
                })

    summary_df = pd.DataFrame(rows)

    if len(summary_df) == 0:
        return pd.DataFrame(columns=[
            "channel",
            "session",
            "data_csv_exists",
            "data_csv",
            "metadata_exists",
            "metadata_json",
        ])

    return summary_df.sort_values(["channel", "session"]).reset_index(drop=True)


def get_sampling_rate_from_summary_row(row, adc_channel="5"):
    """
    Given one row from file_summary, load its metadata JSON and return
    the per-channel sampling rate.

    Parameters
    ----------
    row : pd.Series
        One row from file_summary.
    adc_channel : str
        ADC channel key in metadata. Since your signal column is always CH5,
        this should usually be "5".

    Returns
    -------
    fs : float
        Sampling rate in Hz.
    """
    metadata_path = row["metadata_json"]

    if metadata_path is None or pd.isna(metadata_path):
        raise FileNotFoundError("This row has no metadata_json path.")

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    try:
        fs = metadata["filtering"]["per_channel_sample_rates_hz"][str(adc_channel)]
    except KeyError as e:
        raise KeyError(
            f"Could not find sampling rate for ADC channel {adc_channel!r} in metadata."
        ) from e

    return float(fs)


def load_data_from_summary_row(row, adc_channel="5"):
    """
    Given one row from file_summary, load the corresponding CSV data and
    sampling rate from metadata.

    Parameters
    ----------
    row : pd.Series
        One row from file_summary.
    adc_channel : str
        ADC channel key to use for sampling rate in metadata.
        For your files, this is usually "5".

    Returns
    -------
    data : dict
        Contains dataframe, adc counts, voltage, time, sampling rate, and paths.
    """
    csv_path = row["data_csv"]

    if csv_path is None or pd.isna(csv_path):
        raise FileNotFoundError("This row has no data_csv path.")

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)

    signal_col = f"CH{row['channel']}"

    fs = get_sampling_rate_from_summary_row(row, adc_channel=adc_channel)

    adc = df[signal_col].to_numpy()
    voltage = adc_to_voltage(adc)
    t = np.arange(len(voltage)) / fs

    return {
        "channel": row.get("channel", None),
        "session": row.get("session", None),
        "data_csv": csv_path,
        "metadata_json": Path(row["metadata_json"]),
        "df": df,
        "adc": adc,
        "voltage": voltage,
        "t": t,
        "fs": fs,
        "signal_col": signal_col,
        "adc_channel": adc_channel,
    }


# ======================================================================
# Signal Processing Functions
# ======================================================================
def notch_filter_60hz(x, fs, f0=60.0, quality_factor=30.0):
    """
    Apply a zero-phase IIR notch filter centered at f0.

    Parameters
    ----------
    x : array
        Signal.
    fs : float
        Sampling rate in Hz.
    f0 : float
        Notch frequency, usually 60 Hz in the US.
    quality_factor : float
        Higher values make the notch narrower.
    """
    b, a = iirnotch(w0=f0, Q=quality_factor, fs=fs)
    return filtfilt(b, a, x)


def detect_falling_edges(x, fs, min_spacing_s=0.2):
    """
    Detect falling edges in signal by comparing to midpoint threshold.

    Parameters
    ----------
    x : array
        Filtered signal.
    fs : float
        Sampling rate in Hz.
    min_spacing_s : float
        Minimum time between edges in seconds.

    Returns
    -------
    falling_edge_indices : array
        Sample indices of detected falling edges.
    threshold : float
        Threshold used for detection.
    """
    # Robust estimates of low/high states.
    low_level = np.percentile(x, 10)
    high_level = np.percentile(x, 90)

    # Midpoint threshold for detecting high -> low transitions.
    threshold = 0.5 * (low_level + high_level)

    # Boolean signal: above threshold or not
    above = x > threshold

    # Falling edges occur when above goes True -> False
    falling_edge_indices = np.where((above[:-1] == True) & (above[1:] == False))[0] + 1

    # Enforce minimum spacing between events
    min_spacing_samples = int(min_spacing_s * fs)

    filtered_edges = []
    last_edge = -np.inf

    for idx in falling_edge_indices:
        if idx - last_edge >= min_spacing_samples:
            filtered_edges.append(idx)
            last_edge = idx

    return np.array(filtered_edges), threshold


# ======================================================================
# Fitting Functions
# ======================================================================
def exp_decay(t_rel, baseline, amplitude, tau):
    """
    Single exponential decay.

    V(t) = baseline + amplitude * exp(-t / tau)
    """
    return baseline + amplitude * np.exp(-t_rel / tau)


def fit_decay_pulses(x, t, fs, falling_edge_indices, fit_window_s=0.15):
    """
    Fit exponential decays to each detected falling edge.

    Parameters
    ----------
    x : array
        Filtered signal.
    t : array
        Time array.
    fs : float
        Sampling rate in Hz.
    falling_edge_indices : array
        Sample indices of falling edges.
    fit_window_s : float
        Duration of window to fit after each edge (seconds).

    Returns
    -------
    fit_df : pd.DataFrame
        DataFrame with fitted parameters for each pulse.
    """
    fit_window_n = int(fit_window_s * fs)
    results = []

    for pulse_idx, edge_idx in enumerate(falling_edge_indices, start=1):
        start = edge_idx
        stop = min(edge_idx + fit_window_n, len(x))

        y_fit = x[start:stop]
        t_fit = t[start:stop] - t[start]

        if len(y_fit) < 10:
            continue

        # Estimate baseline from the end of the fitting window.
        baseline_guess = np.median(y_fit[-max(5, len(y_fit)//10):])

        # Initial amplitude estimate.
        amplitude_guess = y_fit[0] - baseline_guess

        # Tau initial guess: around 20 ms.
        tau_guess = 0.020

        p0 = [baseline_guess, amplitude_guess, tau_guess]

        # Bounds
        lower_bounds = [np.min(y_fit) - 1.0, 0.0, 1e-4]
        upper_bounds = [np.max(y_fit) + 1.0, 5.0, 1.0]

        try:
            popt, pcov = curve_fit(
                exp_decay,
                t_fit,
                y_fit,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000
            )

            baseline_hat, amplitude_hat, tau_hat = popt

            y_pred = exp_decay(t_fit, *popt)

            # R^2
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2 = 1 - ss_res / ss_tot

            results.append({
                "pulse": pulse_idx,
                "edge_sample": edge_idx,
                "edge_time_s": t[edge_idx],
                "baseline_V": baseline_hat,
                "amplitude_V": amplitude_hat,
                "tau_s": tau_hat,
                "tau_ms": tau_hat * 1000,
                "tau_samples": tau_hat * fs,
                "r2": r2
            })

        except RuntimeError as e:
            print(f"  Fit failed for pulse {pulse_idx}: {e}")

    return pd.DataFrame(results)


# ======================================================================
# Main Analysis Function
# ======================================================================
def analyze_channel(row, data_dir, verbose=False):
    """
    Analyze a single channel: filter, detect edges, fit decays.

    Parameters
    ----------
    row : pd.Series
        One row from the file summary.
    data_dir : Path
        Data directory path.
    verbose : bool
        Print intermediate values.

    Returns
    -------
    fit_df : pd.DataFrame or None
        DataFrame with results, or None if analysis failed.
    """
    try:
        data = load_data_from_summary_row(row)
        voltage = data["voltage"]
        fs = data["fs"]
        t = data["t"]
        channel = data["channel"]
        session = data["session"]

        if verbose:
            print(f"  Loaded: CH{channel}, session {session}")
            print(f"    Sampling rate: {fs} Hz")

        # Apply 60 Hz notch filter
        voltage_filt = notch_filter_60hz(voltage, fs=fs)

        # Detect falling edges
        falling_edge_indices, _ = detect_falling_edges(voltage_filt, fs)

        if len(falling_edge_indices) == 0:
            if verbose:
                print(f"  No falling edges detected for CH{channel}")
            return None

        if verbose:
            print(f"    Detected {len(falling_edge_indices)} falling edge(s)")

        # Fit exponential decays
        fit_df = fit_decay_pulses(voltage_filt, t, fs, falling_edge_indices)

        if len(fit_df) == 0:
            if verbose:
                print(f"  No successful fits for CH{channel}")
            return None

        # Add channel and session info
        fit_df.insert(0, "channel", channel)
        fit_df.insert(1, "session", session)

        if verbose:
            mean_tau_ms = fit_df["tau_ms"].mean()
            std_tau_ms = fit_df["tau_ms"].std()
            print(f"    Mean tau: {mean_tau_ms:.3f} ± {std_tau_ms:.3f} ms")

        return fit_df

    except Exception as e:
        print(f"  Error analyzing row: {e}")
        return None


# ======================================================================
# Main Script
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze RC decay constants across multiple channels"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/jiayi/Princeton Dropbox/Jiayi Zhang/sensetics/05_06_decay_factors",
        help="Path to data directory"
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Channel numbers to analyze (default: 1 2 3 4 5)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="rc_decay_analysis_all_channels.csv",
        help="Output CSV filename"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    channels = args.channels
    output_csv = args.output_csv

    print(f"RC Decay Analysis")
    print(f"Data directory: {data_dir}")
    print(f"Channels: {channels}")
    print()

    # Scan for available files
    print("Scanning for channel files...")
    session_index = scan_channel_files(data_dir, channels)

    if len(session_index) == 0:
        print("No data files found!")
        return

    print(f"Found {len(session_index)} file(s) to process.\n")

    # Process each file
    all_results = []

    for idx, row in session_index.iterrows():
        ch = row["channel"]
        session = row["session"]
        print(f"[{idx+1}/{len(session_index)}] Processing CH{ch}, session {session}...")

        fit_df = analyze_channel(row, data_dir, verbose=args.verbose)

        if fit_df is not None:
            all_results.append(fit_df)

    # Combine all results
    if len(all_results) == 0:
        print("\nNo results to save.")
        return

    combined_df = pd.concat(all_results, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"SUMMARY: Combined {len(combined_df)} pulses from {len(all_results)} file(s)")
    print(f"{'='*60}\n")

    # Print per-channel statistics
    for ch in combined_df["channel"].unique():
        ch_df = combined_df[combined_df["channel"] == ch]
        mean_tau_ms = ch_df["tau_ms"].mean()
        std_tau_ms = ch_df["tau_ms"].std()
        n_pulses = len(ch_df)
        mean_r2 = ch_df["r2"].mean()

        print(f"CH{int(ch)}: {n_pulses} pulses")
        print(f"  tau: {mean_tau_ms:.3f} ± {std_tau_ms:.3f} ms")
        print(f"  R²:  {mean_r2:.4f}")

    # Save to CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to: {output_csv}")

if __name__ == "__main__":
    main()
