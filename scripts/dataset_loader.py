import pandas as pd
import json
from pathlib import Path
import os

def is_session_dir(d):
    expected_meta = os.path.join(d, f"{os.path.basename(d)}_metadata.json")
    expected_data = os.path.join(d, f"{os.path.basename(d)}.csv")
    return os.path.exists(expected_meta) and os.path.exists(expected_data)

def load_dataset(dataset_dir="resolution_study_data"):
    root_dir = Path(dataset_dir)
    if (not root_dir.is_dir()):
        raise ValueError(f"Dataset directory {dataset_dir} is not a valid directory.")
    
    session_dirs = [d for d in root_dir.iterdir() if is_session_dir(d)]
    if (len(session_dirs) == 0):
        raise ValueError(f"No valid session directories found in {dataset_dir}.")
    
    rows = []
    for d in session_dirs:
        session_id = d.name
        try:
            meta_path = os.path.join(d, f"{session_id}_metadata.json")
            meta = json.loads(open(meta_path, 'r').read())

            rows.append({
                "session_id": session_id,
                "notes": meta.get("notes"),
                "timestamp": meta.get("timestamp"),
                "n_channels": len(meta.get("configuration", {}).get("channels", [])),
                "mcu_type": meta.get("mcu_type"),
                "saved_sweeps": meta.get("saved_sweeps"),
                "fs_hz": meta.get("timing", {}).get("per_channel_rate_hz"),
                "session_dir": str(d),
                "csv_path": str(d / f"{session_id}.csv"),
                "meta_path": str(meta_path),
                })
        except Exception as e:
            print(f"Warning: Could not read metadata for session {session_id}: {e}")
            continue

    return pd.DataFrame(rows)