import numpy as np

def get_approx_sampling_rate(gap_us, truncate=5):
    gap_us = gap_us[truncate:]
    median_gap = np.median(gap_us)
    fs_hz = 1e6 / median_gap
    return fs_hz, median_gap

def get_dropout_idx(gaps, median_gap, threshold_r=1.8):
    threshold = median_gap * threshold_r
    dropout_idx = np.where(gaps > threshold)[0]
    return dropout_idx, len(dropout_idx)/len(gaps)