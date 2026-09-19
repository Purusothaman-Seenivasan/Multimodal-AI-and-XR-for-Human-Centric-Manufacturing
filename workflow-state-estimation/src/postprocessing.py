"""Temporal post-processing on per-step probabilities.

These three operations are shared by both approaches. They take an array of raw
per-step probabilities P[N, 9] and turn it into a cleaner binary prediction.
"""

import numpy as np
import pandas as pd


def probability_smoothing(P, window):
    """Causal rolling mean of each step's probability over `window` frames.

    A single noisy spike gets diluted by its neighbours, so a lone confident
    frame no longer flips the prediction.
    """
    return pd.DataFrame(P).rolling(window, min_periods=1).mean().values


def monotonic_clamp(Y):
    """Cumulative max along time so a step bit can never revert 1 -> 0."""
    return np.maximum.accumulate(Y, axis=0)


def sustained_commitment(P, threshold, K):
    """Commit a step to 1 only after K consecutive frames above `threshold`.

    A dip below threshold resets the streak. Once committed the bit stays 1, so
    the output is monotonic by construction (no separate clamp needed).
    """
    N, S = P.shape
    Y         = np.zeros((N, S), dtype=int)
    streak    = np.zeros(S, dtype=int)
    committed = np.zeros(S, dtype=bool)
    for t in range(N):
        above  = P[t] >= threshold
        streak = np.where(above, streak + 1, 0)
        committed |= streak >= K
        Y[t] = committed
    return Y
