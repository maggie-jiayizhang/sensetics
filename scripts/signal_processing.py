import numpy as np
from scipy.signal import correlation_lags, correlate

def estimate_lag(x, y):
    x = np.asarray(x) - np.nanmean(x)
    y = np.asarray(y) - np.nanmean(y)
    c = correlate(x, y, mode="full")
    lags = correlation_lags(len(x), len(y), mode="full")
    return lags[np.argmax(c)]