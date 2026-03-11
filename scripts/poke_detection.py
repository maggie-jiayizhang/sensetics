import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

CHS = ["CH1", "CH2", "CH3", "CH4", "CH5"]

def _mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12

def force_peak_detection(df_trim, fs, height=50, min_sep_ms=100):
    # 1) robust baseline removal (median)
    x = df_trim["Force_Z"].to_numpy(dtype=float)
    x = x - np.median(x, keepdims=True)
    x = np.maximum(0, x)

    # 2) detect peaks
    candidate_peaks, _ = find_peaks(
        x,
        height=height,
        distance=int((min_sep_ms / 1000.0) * fs)
    )

    return candidate_peaks, x

def poke_detection(pdf, fs, N_cap=None, discard_s=0.1, height=50,
                   min_sep_ms=100, CHS=CHS):
    """
    Returns:
      det_full: detector array aligned to the trimmed dataframe (after N_cap truncation)
      n_discard: number of samples discarded at the start
      pdf_trim: pdf truncated to N_cap (if provided)
    """
    # 1) truncate to true captured sweeps (avoid duplicated tail)
    if N_cap is not None:
        pdf_trim = pdf.iloc[:N_cap].copy()
    else:
        pdf_trim = pdf

    # 2) discard initial settling period
    n_discard = int(discard_s * fs)
    df = pdf_trim.iloc[n_discard:].copy()

    # 3) robust baseline removal per channel (median)
    X = np.vstack([df[ch].to_numpy(dtype=float) for ch in CHS])
    X = X - np.median(X, axis=1, keepdims=True)
    X = np.maximum(0, X)

    # 4) combine channels into one activity signal
    #    only summing the channels but not force
    mix = np.sum(np.abs(X), axis=0)  # exclude Force_Z

    candidate_peaks, _ = find_peaks(
        mix,
        height=height,
        distance=int((min_sep_ms / 1000.0) * fs)
    )

    return n_discard, df, mix, candidate_peaks

def extract_poke_windows(pdf, peaks, fs, pre_ms=10.0, post_ms=10.0):
    """
    Returns a list of dicts, one per poke:
      {
        'peak_idx': int,
        'window': DataFrame,
        'baseline': dict per channel
      }
    """
    pre = int((pre_ms / 1000.0) * fs)
    post = int((post_ms / 1000.0) * fs)

    pokes = []
    for p in peaks:
        a = max(0, p - pre)
        b = min(len(pdf), p + post)

        win = pdf.iloc[a:b].copy()

        # baseline per channel from pre region
        base = {}
        for ch in CHS:
            base[ch] = pdf[ch].iloc[max(0, p - pre):p].median()
            win[ch] -= base[ch]

        pokes.append({
            "peak_idx": p,
            "start_idx": a,
            "end_idx": b,
            "window": win,
            "baseline": base
        })

    return pokes

def annotate_pokes_from_peaks(candidate_peaks, mix, fs,
                              search_back_ms=120.0, search_fwd_ms=160.0,baseline_ms=80.0, guard_ms=30.0, frac=0.10,min_quiet_ms=3.0,
    ):
    """
    Annotate detected poke peaks with onset/offset and baseline window.
    Returns
    -------
    events : list of dict
        Each dict has:
            peak_idx, onset_idx, offset_idx,
            baseline_start, baseline_end,
            valid, notes
    """
    mix = np.asarray(mix, dtype=float)
    n = len(mix)

    # max search range for onset/offset
    search_back = int(search_back_ms * fs / 1000.0)
    search_fwd = int(search_fwd_ms * fs / 1000.0)
    # baseline window length
    baseline_n = int(baseline_ms * fs / 1000.0)
    guard_n = int(guard_ms * fs / 1000.0)
    quiet_n = max(1, int(min_quiet_ms * fs / 1000.0))

    events = []

    for p in candidate_peaks:
        p = int(p)

        left = max(0, p - search_back)
        right = min(n, p + search_fwd)

        peak_val = float(mix[p])
        local_floor = float(np.median(mix[left:p])) if p > left else 0.0
        # frac % * peak above local floor
        thr = local_floor + frac * max(0.0, peak_val - local_floor)

        # onset: walk backward until quiet
        onset_idx = left
        found_onset = False
        for i in range(p, left + quiet_n - 1, -1):
            seg = mix[i - quiet_n:i]
            if len(seg) == quiet_n and np.all(seg <= thr):
                onset_idx = i
                found_onset = True
                break

        # offset: walk forward until quiet
        offset_idx = right
        found_offset = False
        for i in range(p, right - quiet_n + 1):
            seg = mix[i:i + quiet_n]
            if len(seg) == quiet_n and np.all(seg <= thr):
                offset_idx = i
                found_offset = True
                break

        baseline_end = max(0, onset_idx - guard_n)
        baseline_start = max(0, baseline_end - baseline_n)

        notes = []
        valid = True

        if not found_onset:
            notes.append("onset_not_found")
            valid = False
        if not found_offset:
            notes.append("offset_not_found")
        if baseline_end <= baseline_start:
            notes.append("baseline_window_empty")
            valid = False
        if offset_idx <= onset_idx:
            notes.append("offset_before_onset")
            valid = False

        events.append({
            "peak_idx": p,
            "onset_idx": int(onset_idx),
            "offset_idx": int(offset_idx),
            "baseline_start": int(baseline_start),
            "baseline_end": int(baseline_end),
            "valid": valid,
            "notes": notes,
        })

    return events

def extract_poke_windows_from_annotations(pdf, events, fs, 
                                          pad_pre_ms=10.0, pad_post_ms=10.0,chs=CHS,baseline_stat="median",
    ):
    """
    Extract poke windows using annotated events.

    Parameters
    ----------
    pdf : needs to be pdf_trim (aligned with annotation)
    pre_, post_ms: window around annotated peak (onset-offset)

    Returns
    -------
    pokes : list of dict
        One per poke, with fields:
            peak_idx, onset_idx, offset_idx,
            start_idx, end_idx,
            baseline_start, baseline_end,
            window, baseline, valid, notes
    """
    pokes = []
    pre = int((pad_pre_ms / 1000.0) * fs)
    post = int((pad_post_ms / 1000.0) * fs)

    for e in events:
        # peak, onset, offset from annotation
        onset, offset = int(e["onset_idx"]), int(e["offset_idx"])
        p = int(e["peak_idx"])

        a = max(0, p - pre)
        b = min(len(pdf), p + post)

        win = pdf.iloc[a:b].copy()

        # baseline per channel from annotated baseline window
        bs = int(e["baseline_start"])
        be = int(e["baseline_end"])

        base = {}
        for ch in chs:
            seg = pdf[ch].iloc[bs:be]

            if len(seg) == 0:
                val = 0.0
            elif baseline_stat == "mean":
                val = float(seg.mean())
            else:
                val = float(seg.median())

            base[ch] = val
            win[ch] -= val

        pokes.append({
            "peak_idx": p,
            "onset_idx": onset,
            "offset_idx": offset,
            "start_idx": a,
            "end_idx": b,
            "baseline_start": bs,
            "baseline_end": be,
            "window": win,
            "baseline": base,
            "valid": e.get("valid", True),
            "notes": e.get("notes", []),
        })

    return pokes

def plot_poke_windows(pokes, fs, ch="CH3", n=30, title=None,
                      show_peak=True, show_onset_offset=True, show_baseline_window=True,):
    
    f = plt.figure(figsize=(6,3))

    for i in range(min(n, len(pokes))):
        pk = pokes[i]
        w = pk["window"][ch].to_numpy()
        p = pk["peak_idx"]
        t = np.arange(len(w)) / fs * 1000  # ms
        
        plt.plot(t, w, color="black", alpha=0.25, linewidth=1)

        if show_peak and "peak_idx" in pk:
            plt.axvline(x=(p - pk["start_idx"]) / fs * 1000, color='r', linestyle='--', alpha=0.5)

        if show_onset_offset and "onset_idx" in pk and "offset_idx" in pk:
            onset_rel = (pk["onset_idx"] - pk["start_idx"]) / fs * 1000
            offset_rel = (pk["offset_idx"] - pk["start_idx"]) / fs * 1000
            plt.axvline(x=onset_rel, color="0.4", linestyle=":", linewidth=1)
            plt.axvline(x=offset_rel, color="0.4", linestyle=":", linewidth=1)

        if show_baseline_window and \
            ("baseline_start" in pk) and ("baseline_end" in pk):
            bs_rel = (pk["baseline_start"] - pk["start_idx"]) / fs * 1000
            be_rel = (pk["baseline_end"] - pk["start_idx"]) / fs * 1000
            plt.axvspan(bs_rel, be_rel, color="#b0c4de", alpha=0.15)

    plt.xlabel("Time (ms)")
    plt.ylabel("baseline-subtracted")

    if (title is not None):
        plt.title(title)
    else:
        plt.title(f"First {n} poke windows, channel {ch}")

    handles, labels = [], []
    if show_peak:
        handles.append(plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.5))
        labels.append("peak")
    if show_onset_offset:
        handles.append(plt.Line2D([0], [0], color='0.4', linestyle=':', alpha=0.5))
        labels.append("onset/offset")
    if show_baseline_window:
        handles.append(plt.Rectangle((0,0),1,1, color="#b0c4de", alpha=0.6))
        labels.append("baseline window")
    if handles:
        plt.legend(handles, labels)
    plt.axhline(0, color="0.8", linestyle="--", linewidth=1)
    plt.show()
    return f

CHS = ["CH1","CH2","CH3","CH4","CH5"]

def robust_p2p(x, lo=1, hi=99):
    return float(np.percentile(x, hi) - np.percentile(x, lo))

def poke_features_from_window_robust(win, fs, t_ref_ms=12.0, pre_ms=3.0, post_ms=8.0):
    i_ref = int((t_ref_ms/1000.0) * fs)
    i0 = max(0, i_ref - int((pre_ms/1000.0)*fs))
    i1 = min(len(win), i_ref + int((post_ms/1000.0)*fs))

    feats = {}
    pk_list = []

    for ch in CHS:
        x = win[ch].iloc[i0:i1].to_numpy(dtype=float)

        pk98 = float(np.percentile(x, 98))               # robust peak
        p2p = robust_p2p(x, lo=1, hi=99)                 # robust peak-to-peak
        eng = float(np.mean(x*x))                        # mean energy (scale-invariant to window length)
        pos_area = float(np.mean(np.maximum(0.0, x)))    # mean positive area

        feats[f"{ch}_pk98"] = pk98
        feats[f"{ch}_p2p99"] = p2p
        feats[f"{ch}_eng"] = eng
        feats[f"{ch}_posmean"] = pos_area

        pk_list.append(pk98)

    pk = np.array(pk_list)
    s = float(pk.sum()) + 1e-12
    prof = pk / s
    for i, ch in enumerate(CHS):
        feats[f"{ch}_pk98_norm"] = float(prof[i])

    return feats

def poke_features_for_session(x, pokes, fs, t_ref_ms=12.0, pre_ms=3.0, post_ms=8.0):
    feature_list = []
    for p in pokes:
        feats = poke_features_from_window_robust(p["window"], fs, t_ref_ms, pre_ms, post_ms)
        feats["location"] = x//100
        feature_list.append(feats)
    return pd.DataFrame(feature_list)

### force channel processing
def force_peak_detection(df_trim, fs, height=50, min_sep_ms=100):
    # 1) robust baseline removal (median)
    x = df_trim["Force_Z"].to_numpy(dtype=float)
    x = x - np.median(x, keepdims=True)
    x = np.maximum(0, x)

    # 2) detect peaks
    candidate_peaks, _ = find_peaks(
        x,
        height=height,
        distance=int((min_sep_ms / 1000.0) * fs)
    )

    return candidate_peaks, x

def extract_force_peak_strength(df_trim, candidate_peaks, fs,
                                pre_ms=100, post_ms=100, quant=0.98):
    
    pre_samples = int((pre_ms / 1000.0) * fs)
    post_samples = int((post_ms / 1000.0) * fs)

    peak_strengths = []
    for p in candidate_peaks:
        start = max(0, p - pre_samples)
        end = min(len(df_trim), p + post_samples)

        # define peak as 98% percentile of force in the window
        peak_strength = df_trim["Force_Z"].iloc[start:end].quantile(quant)
        peak_strengths.append(peak_strength)

    return np.array(peak_strengths)

CHS = ["CH1","CH2","CH3","CH4","CH5"]
def extract_sensor_peak_strength(win, fs, t_ref_ms=50.0, 
                                 pre_ms=50.0, post_ms=100.0, 
                                 ch_list=CHS):
    
    i_ref = int((t_ref_ms/1000.0) * fs)
    i0 = max(0, i_ref - int((pre_ms/1000.0)*fs))
    i1 = min(len(win), i_ref + int((post_ms/1000.0)*fs))

    feats = {}
    pk_list = []

    for ch in ch_list:
        x = win[ch].iloc[i0:i1].to_numpy(dtype=float)

        pk98 = float(np.percentile(x, 98))               # robust peak
        p2p = robust_p2p(x, lo=1, hi=99)                 # robust peak-to-peak
        eng = float(np.mean(x*x))                        # mean energy (scale-invariant to window length)
        pos_area = float(np.mean(np.maximum(0.0, x)))    # mean positive area

        feats[f"{ch}_pk98"] = pk98
        feats[f"{ch}_p2p99"] = p2p
        feats[f"{ch}_eng"] = eng
        feats[f"{ch}_posmean"] = pos_area

        pk_list.append(pk98)

    pk = np.array(pk_list)
    s = float(pk.sum()) + 1e-12
    prof = pk / s
    for i, ch in enumerate(CHS):
        feats[f"{ch}_pk98_norm"] = float(prof[i])

    return feats

def extract_sensor_peak_strength_session(poke_windows, fs, t_ref_ms=50.0, pre_ms=50.0, post_ms=100.0):
    all_feats = []
    for w in poke_windows:
        feats = extract_sensor_peak_strength(w["window"], fs, t_ref_ms=t_ref_ms, pre_ms=pre_ms, post_ms=post_ms)
        all_feats.append(feats)
    
    all_feats = pd.DataFrame(all_feats)
    return all_feats


from scipy.signal import butter, sosfiltfilt

def integrate_poke_window(win, fs, ch, cutoff_hz=60.0, filt_order=4):
    dt = 1 / fs
    x = win[ch].to_numpy(dtype=float)   # already baseline-subtracted

    # slow, fast split
    sos = butter(filt_order, cutoff_hz, btype='low', fs=fs, output='sos')
    x_slow = sosfiltfilt(sos, x)
    x_fast = x - x_slow

    Xs = [x, x_slow, x_fast]
    labels = ["raw", "slow", "fast"]

    # integration
    out = win.copy()

    for i, X in enumerate(Xs):
        integrated_X = np.cumsum(X) * dt
        out[f"{ch}_dFdT_{labels[i]}"] = X
        out[f"{ch}_F_{labels[i]}"] = integrated_X
        
    return out

def integrate_poke_session(pokes, fs, ch="CH5", cutoff_hz=40.0, filt_order=4):
    out = []
    for pk in pokes:
        w_int = integrate_poke_window(
            pk["window"], fs, ch=ch, cutoff_hz=cutoff_hz, filt_order=filt_order
        )
        pk_new = pk.copy()
        pk_new["force_window"] = w_int
        out.append(pk_new)
    return out

def plot_force_windows(
    pokes,
    fs,
    col,
    n=30,
    align_to="peak",
    show_mean=True,
    title=None,
):
    f = plt.figure(figsize=(6,3))

    traces = []
    time_axes = []

    for i in range(min(n, len(pokes))):
        pk = pokes[i]
        fw = pk["force_window"]

        if col not in fw.columns:
            continue

        y = fw[col].to_numpy(dtype=float)

        if align_to == "peak":
            ref = pk["peak_idx"] - pk["start_idx"]
        elif align_to == "onset" and "onset_idx" in pk:
            ref = pk["onset_idx"] - pk["start_idx"]
        else:
            ref = 0

        t = (np.arange(len(y)) - ref) / fs * 1000

        plt.plot(t, y, color="steelblue", alpha=0.18)
        traces.append(y)
        time_axes.append(t)

    if show_mean and traces:
        min_len = min(len(y) for y in traces)
        Y = np.array([y[:min_len] for y in traces])
        t_mean = time_axes[0][:min_len]
        plt.plot(t_mean, Y.mean(axis=0), color="black", linewidth=2)

    if align_to is not None:
        plt.axvline(0, color="red", linestyle="--")

    plt.xlabel("Time (ms)")
    plt.ylabel(col)

    if title is None:
        title = col + (f" aligned to {align_to}" if align_to else "")
    plt.title(title)

    plt.show()
    return f