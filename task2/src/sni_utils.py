"""SNI fusion, z-scoring and episode detection."""

import numpy as np
import pandas as pd

from constants import EPISODE_SIGMA


def safe_zscore(x):
    """Per-recording z-score. Returns zeros if std is zero/NaN (flat signal)."""
    x = np.asarray(x, dtype=float)
    std = np.nanstd(x)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(x)
    z = (x - np.nanmean(x)) / std
    return np.nan_to_num(z, nan=0.0)   # missing frame -> neutral contribution


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_sni(gaze_entropy, hand_stall, aoi_dwell):
    """Equal-weight fusion of the three z-scored signals through a sigmoid.

    All three inputs are oriented so higher = more support need.
    """
    z_gaze = safe_zscore(gaze_entropy)
    z_stall = safe_zscore(hand_stall)
    z_dwell = safe_zscore(aoi_dwell)
    sni = sigmoid((z_gaze + z_stall + z_dwell) / 3.0)
    return sni, z_gaze, z_stall, z_dwell


def detect_support_episodes(sni, threshold=None):
    """Flag frames above the per-recording threshold (mean + 1 std by default).

    Returns (binary_mask, threshold).
    """
    if threshold is None:
        threshold = np.nanmean(sni) + EPISODE_SIGMA * np.nanstd(sni)
    mask = (sni > threshold).astype(int)
    return mask, threshold


def extract_episode_table(frame_names, sni, episode_mask, recording):
    """Group contiguous flagged frames into one row per support-need episode."""
    rows = []
    episode_id, start = 0, None
    n = len(episode_mask)
    for t in range(n):
        flagged = episode_mask[t] == 1
        if flagged and start is None:
            start = t
        if (not flagged or t == n - 1) and start is not None:
            end = t if (flagged and t == n - 1) else t - 1
            seg = sni[start:end + 1]
            rows.append(dict(
                recording=recording,
                episode_id=episode_id,
                start_frame=frame_names[start],
                end_frame=frame_names[end],
                start_index=start,
                end_index=end,
                duration_frames=end - start + 1,
                max_sni=float(np.nanmax(seg)),
                mean_sni=float(np.nanmean(seg)),
            ))
            episode_id += 1
            start = None
    return pd.DataFrame(rows)
