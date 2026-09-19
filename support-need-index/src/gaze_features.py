"""Gaze sub-signals: spatial entropy and instruction-AOI dwell."""

import numpy as np

from constants import (
    IMAGE_WIDTH, IMAGE_HEIGHT, WINDOW_SIZE, GRID_SIZE,
    INSTRUCTION_AOI_X_MAX_FRAC, INSTRUCTION_AOI_Y_MIN_FRAC,
)


def valid_gaze_mask(gaze_df):
    """A gaze sample is valid unless it is the (0, 0) no-detection marker."""
    x = gaze_df["x"].values
    y = gaze_df["y"].values
    return ~((x == 0) & (y == 0))


def compute_gaze_entropy(gaze_df, window=WINDOW_SIZE, grid_size=GRID_SIZE):
    """Rolling Shannon entropy of gaze over a grid_size x grid_size grid.

    High entropy = gaze scattered (visual search); low = gaze settled.
    Invalid samples are dropped; windows with <3 valid samples are NaN.
    """
    x = gaze_df["x"].values
    y = gaze_df["y"].values
    valid = valid_gaze_mask(gaze_df)

    cx = np.clip((x / IMAGE_WIDTH * grid_size).astype(int), 0, grid_size - 1)
    cy = np.clip((y / IMAGE_HEIGHT * grid_size).astype(int), 0, grid_size - 1)
    cell = cy * grid_size + cx

    n = len(x)
    entropy = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window + 1)
        cells = cell[lo:t + 1][valid[lo:t + 1]]
        if len(cells) < 3:
            continue
        counts = np.bincount(cells, minlength=grid_size * grid_size).astype(float)
        p = counts[counts > 0] / counts.sum()
        entropy[t] = -(p * np.log2(p)).sum()
    return entropy


def in_instruction_aoi(gaze_df):
    """Boolean mask: valid gaze falling in the lower-left instruction AOI."""
    x = gaze_df["x"].values
    y = gaze_df["y"].values
    valid = valid_gaze_mask(gaze_df)
    return valid & (x < INSTRUCTION_AOI_X_MAX_FRAC * IMAGE_WIDTH) \
        & (y > INSTRUCTION_AOI_Y_MIN_FRAC * IMAGE_HEIGHT)


def compute_instruction_aoi_dwell(gaze_df, window=WINDOW_SIZE):
    """Fraction of valid gaze samples inside the instruction AOI over the window.

    High = the operator keeps looking at the instruction sheet (consulting).
    Windows with <3 valid samples are NaN.
    """
    valid = valid_gaze_mask(gaze_df).astype(float)
    in_aoi = in_instruction_aoi(gaze_df).astype(float)

    n = len(valid)
    dwell = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - window + 1)
        v = valid[lo:t + 1].sum()
        if v < 3:
            continue
        dwell[t] = in_aoi[lo:t + 1].sum() / v
    return dwell
