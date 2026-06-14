"""Hand sub-signal: hand stall from joint-level motion."""

import numpy as np
import pandas as pd

from constants import WINDOW_SIZE, MIN_VALID_JOINTS, HAND_MOTION_PERCENTILE


def extract_valid_hand_joints(hands_df):
    """Reshape the 104 joint columns into left/right (N, 26, 2) arrays.

    Returns (left, right). A joint at (0, 0) is a missing/invalid joint.
    """
    vals = hands_df.iloc[:, 1:].astype(float).values  # (N, 104)
    n = vals.shape[0]
    left = vals[:, 0:52].reshape(n, 26, 2)
    right = vals[:, 52:104].reshape(n, 26, 2)
    return left, right


def _single_hand_motion(joints, min_valid_joints):
    """Per-frame joint speed (percentile over joints valid at both t-1 and t)."""
    x, y = joints[:, :, 0], joints[:, :, 1]
    valid = ~((x == 0) & (y == 0))                       # (N, 26)
    dx = np.diff(x, axis=0, prepend=x[:1])
    dy = np.diff(y, axis=0, prepend=y[:1])
    speed = np.sqrt(dx ** 2 + dy ** 2)
    prev_valid = np.vstack([np.zeros((1, 26), bool), valid[:-1]])
    valid_pair = valid & prev_valid                      # valid at t-1 and t

    n = joints.shape[0]
    motion = np.full(n, np.nan)
    for t in range(n):
        s = speed[t, valid_pair[t]]
        if s.size >= min_valid_joints:
            motion[t] = np.percentile(s, HAND_MOTION_PERCENTILE)
    return motion


def compute_hand_motion(hands_df, min_valid_joints=MIN_VALID_JOINTS):
    """Per-frame motion of the more active hand (NaN if neither is scorable).

    Using joint-level speed (not the hand centroid) avoids labelling fine work
    such as screw tightening as a stall.
    """
    left, right = extract_valid_hand_joints(hands_df)
    # fmax: keeps the active hand even when the other hand drops out (NaN).
    return np.fmax(
        _single_hand_motion(left, min_valid_joints),
        _single_hand_motion(right, min_valid_joints),
    )


def compute_hand_stall(hands_df, window=WINDOW_SIZE):
    """Negative rolling-mean hand motion: high = hands paused / barely moving."""
    motion = compute_hand_motion(hands_df)
    smooth = pd.Series(motion).rolling(window, min_periods=3).mean().values
    return -smooth
