"""Evaluation metrics for cumulative multi-hot workflow-state prediction.

The small per-metric helpers are kept separate so they read clearly; `evaluate`
bundles them into the summary dict used everywhere, and `per_step_f1` /
`transition_metrics` produce the per-step and transition tables saved as CSV.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, hamming_loss, accuracy_score,
)

from constants import EXPECTED_STEPS, N_STEPS


def compute_hamming(y_true, y_pred):
    """Fraction of step bits predicted wrong (lower is better)."""
    return float(hamming_loss(y_true, y_pred))


def compute_exact_match(y_true, y_pred):
    """Fraction of frames whose full 9-bit state vector is exactly correct."""
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true, y_pred):
    """Unweighted mean of per-step F1 (every step counts equally)."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_micro_f1(y_true, y_pred):
    """F1 pooled over all step bits (dominated by the common early steps)."""
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def compute_progress_mae(y_true, y_pred):
    """Mean absolute error of progress (= fraction of steps complete)."""
    pt = y_true.sum(axis=1) / N_STEPS
    pp = y_pred.sum(axis=1) / N_STEPS
    return float(np.abs(pt - pp).mean())


def count_mono_violations(y_pred):
    """Number of 1 -> 0 reversals across time (should be 0 for a clean run)."""
    return int(((y_pred[:-1] - y_pred[1:]) > 0).sum())


def _transition_flags(Y):
    """Per-frame flag: 1 when the state vector changed from the previous frame."""
    flags = np.zeros(len(Y), dtype=int)
    flags[1:] = (Y[1:] != Y[:-1]).any(axis=1).astype(int)
    return flags


def transition_metrics(y_true, y_pred):
    """Precision / recall / F1 of detecting *when* the state changes."""
    tt, tp = _transition_flags(y_true), _transition_flags(y_pred)
    return {
        "true_count": int(tt.sum()),
        "pred_count": int(tp.sum()),
        "precision": float(precision_score(tt, tp, zero_division=0)),
        "recall":    float(recall_score(tt, tp, zero_division=0)),
        "f1":        float(f1_score(tt, tp, zero_division=0)),
    }


def per_step_f1(y_true, y_pred):
    """Per-step precision / recall / F1 -> list of dicts (one row per step)."""
    p = precision_score(y_true, y_pred, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, zero_division=0)
    return [
        {"step": s, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
        for i, s in enumerate(EXPECTED_STEPS)
    ]


def evaluate(y_true, y_pred):
    """Summary metric dict for one prediction (one row of summary_metrics.csv)."""
    trans = transition_metrics(y_true, y_pred)
    return {
        "macro_f1":       compute_macro_f1(y_true, y_pred),
        "micro_f1":       compute_micro_f1(y_true, y_pred),
        "hamming_loss":   compute_hamming(y_true, y_pred),
        "exact_match":    compute_exact_match(y_true, y_pred),
        "progress_mae":   compute_progress_mae(y_true, y_pred),
        "pred_transitions": trans["pred_count"],
        "mono_violations":  count_mono_violations(y_pred),
    }
