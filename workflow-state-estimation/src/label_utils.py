"""Building the cumulative PSR labels.

PSR_labels.csv only annotates 8-9 keyframes per recording. We expand them into a
per-frame cumulative multi-hot vector: once a step is completed it stays completed.
"""

import numpy as np

from constants import EXPECTED_STEPS, STEP_TO_IDX, N_STEPS, STEP_ALIASES


def apply_step_aliases(step_id):
    """Remap optional-variant steps (e.g. step 12 short rear chassis -> step 9)."""
    return STEP_ALIASES.get(step_id, step_id)


def build_cumulative_labels(psr_df, frame_names):
    """Expand sparse PSR keyframes into per-frame cumulative labels [N, 9].

    Walk frames in order; when a keyframe is reached, set its step bit to 1 and
    keep it 1 for all later frames (completion is irreversible).
    """
    # frame_name -> list of step indices annotated at that frame
    frame_events = {}
    for _, row in psr_df.iterrows():
        sid = apply_step_aliases(int(row["step_id"]))
        if sid in STEP_TO_IDX:
            frame_events.setdefault(row["frame"], []).append(STEP_TO_IDX[sid])

    completed = np.zeros(N_STEPS, dtype=int)
    labels    = np.zeros((len(frame_names), N_STEPS), dtype=int)
    for t, f in enumerate(frame_names):
        for idx in frame_events.get(f, []):
            completed[idx] = 1
        labels[t] = completed
    return labels


def compute_progress(labels):
    """Fraction of steps completed per frame: sum(y_t) / 9 -> [N]."""
    return labels.sum(axis=1) / N_STEPS
