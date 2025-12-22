import pandas as pd
from dataclasses import dataclass
import json
import numpy as np
import os

# create a class to hold a session's data + metadata
@dataclass
class Sesssion:
    session_id: str
    metadata: dict
    data: pd.DataFrame
    fs_hz: float

# load session from a directory given session id
def load_session(session_id: str, session_dir="resolution_study_data",
                 calibration_t_s=1):
    full_dir = os.path.join(session_dir, session_id)

    if (not os.path.exists(full_dir)):
        raise ValueError(f"Session directory {full_dir} does not exist.")
    
    meta_path = os.path.join(full_dir, f"{session_id}_metadata.json")
    data_path = os.path.join(full_dir, f"{session_id}.csv")

    if (not os.path.exists(meta_path) or not os.path.exists(data_path)):
        raise ValueError(f"Session files for {session_id} are missing in {full_dir}.")

    meta = json.load(open(meta_path, 'r'))
    data = pd.read_csv(data_path)

    # remove the calibration period
    calibration_rows = int(calibration_t_s * meta["timing"]["per_channel_rate_hz"])
    data = data.iloc[calibration_rows:, :].reset_index(drop=True)

    expected_chs = [f"CH{i}" for i in meta["configuration"]["channels"]]
    missing = [c for c in expected_chs if c not in data.columns]
    if missing:
        raise ValueError(f"CSV missing expected channel columns: {missing}. "
                         f"Found columns: {list(data.columns)}")
    
    saved_sweeps = int(meta["saved_sweeps"]) # includes calibration period
    if data.shape[0] != saved_sweeps-calibration_rows:
        raise ValueError(f"Data rows {data.shape[0]} does not match saved sweeps {saved_sweeps}.")
    
    fs = meta["timing"]["per_channel_rate_hz"]
    assert(fs > 0), "Sampling rate must be positive."

    metadata = dict(meta)

    return Sesssion(session_id=session_id, metadata=metadata, 
                    data=data, fs_hz=fs)
