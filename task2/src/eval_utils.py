"""Evaluation against the weak instruction-consulting proxy."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from constants import INSTRUCTION_PROXY_LABELS


def _frame_to_int(frame):
    return int(str(frame).replace(".jpg", ""))


def build_instruction_proxy(recording_dir, frames):
    """Weak per-frame proxy from AR_labels check/browse_instruction intervals.

    This is used for EVALUATION ONLY and is never an input to the SNI.
    Returns a boolean array aligned to `frames`.
    """
    ar = pd.read_csv(os.path.join(recording_dir, "AR_labels.csv"), header=None)
    frame_ints = np.array([_frame_to_int(f) for f in frames])
    proxy = np.zeros(len(frames), dtype=bool)
    for _, row in ar.iterrows():
        if str(row[2]).strip() in INSTRUCTION_PROXY_LABELS:
            start, end = _frame_to_int(row[3]), _frame_to_int(row[4])
            proxy |= (frame_ints >= start) & (frame_ints <= end)
    return proxy


def compute_auc(y_proxy, score):
    """ROC-AUC of `score` against the boolean proxy. NaN if proxy is one class."""
    y = np.asarray(y_proxy, dtype=bool)
    if not y.any() or y.all():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def compute_ablation_auc(proxy, z_gaze, z_stall, z_dwell, sni):
    """Pooled AUC for each single z-scored signal and the fused SNI."""
    return {
        "Instruction-AOI dwell only": compute_auc(proxy, z_dwell),
        "Fused SNI": compute_auc(proxy, sni),
        "Gaze entropy only": compute_auc(proxy, z_gaze),
        "Hand stall only": compute_auc(proxy, z_stall),
    }
