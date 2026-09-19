"""Feature helpers for the sensor baseline (Approach A).

The raw 118-dim sensor vector is already assembled in data_utils.load_recording.
Here we just add the temporal sliding window used before training.
"""

import pandas as pd

from constants import A_WINDOW_SIZE


def sliding_window(X, window=A_WINDOW_SIZE):
    """Replace each feature by its causal rolling mean over `window` frames.

    Smooths high-frequency sensor noise so adjacent frames look more similar.
    Applied per recording (call once per recording before concatenating).
    """
    return pd.DataFrame(X).rolling(window, min_periods=1).mean().values
