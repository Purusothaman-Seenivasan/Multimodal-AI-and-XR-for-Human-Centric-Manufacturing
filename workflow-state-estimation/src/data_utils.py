"""Loading the raw recording files.

Each recording folder holds gaze.csv, hands.csv, pose.csv, PSR_labels.csv and an
rgb/ image folder. These helpers read the sensor streams and list the RGB frames.
All files are headerless; column 0 is the frame name (e.g. "000123.jpg").
"""

import numpy as np
import pandas as pd

from constants import IMG_W, IMG_H, TRAIN_DIR, TEST_DIR, FRAME_STRIDE


def recording_dir(split, name):
    """Path to one recording folder ('train' or 'test')."""
    base = TRAIN_DIR if split == "train" else TEST_DIR
    return base / name


def load_gaze(rec_dir):
    """Gaze stream -> (frame_names, features[N, 3]) = gaze_x/W, gaze_y/H, gaze_valid.

    A (0, 0) gaze sample means no detection, so gaze_valid is 0 there.
    """
    df = pd.read_csv(rec_dir / "gaze.csv", header=None,
                     names=["frame", "gaze_x", "gaze_y"])
    frames = df["frame"].values
    valid  = ((df["gaze_x"] != 0) | (df["gaze_y"] != 0)).astype(float).values
    feats  = np.stack([df["gaze_x"].values / IMG_W,
                       df["gaze_y"].values / IMG_H, valid], axis=1)
    return frames, feats


def load_hands(rec_dir):
    """Hand stream -> features[N, 106] = 104 normalised joint coords + 2 valid flags.

    104 values = 2 hands x 26 joints x (x, y), interleaved. Left = cols 0..51,
    right = 52..103. An all-zero half means that hand was not tracked.
    """
    vals = pd.read_csv(rec_dir / "hands.csv", header=None).iloc[:, 1:].values.astype(float)
    left_valid  = (vals[:, :52] != 0).any(axis=1).astype(float)
    right_valid = (vals[:, 52:] != 0).any(axis=1).astype(float)
    norm = np.array([IMG_W, IMG_H] * 52)
    return np.concatenate([vals / norm, left_valid[:, None], right_valid[:, None]], axis=1)


def load_pose(rec_dir):
    """Head pose stream -> features[N, 9] (forward, position, up vectors)."""
    return pd.read_csv(rec_dir / "pose.csv", header=None).iloc[:, 1:].values.astype(float)


def load_psr(rec_dir):
    """Sparse PSR keyframe annotations -> DataFrame[frame, step_id, step_name]."""
    return pd.read_csv(rec_dir / "PSR_labels.csv", header=None,
                       names=["frame", "step_id", "step_name"])


def load_recording(split, name):
    """Sensor view of one recording -> (X[N, 118], frame_names, psr_df).

    X concatenates gaze (3) + hands (106) + pose (9). Labels are built separately
    in label_utils so the loading and labelling steps stay independent.
    """
    rec_dir = recording_dir(split, name)
    frames, gaze = load_gaze(rec_dir)
    X = np.concatenate([gaze, load_hands(rec_dir), load_pose(rec_dir)], axis=1)
    return X, frames, load_psr(rec_dir)


# ── RGB helpers (Approach B) ───────────────────────────────────────────────

def list_rgb_frames(rec_dir, stride=FRAME_STRIDE):
    """Sorted RGB frame names sampled every `stride` frames.

    Returns the list of file names (e.g. "000005.jpg") in chronological order.
    """
    files = sorted(p.name for p in (rec_dir / "rgb").glob("*.jpg"))
    return files[::stride]


def load_hands_imgspace(rec_dir):
    """frame_name -> dict(left[26,2], right[26,2], left_valid, right_valid).

    Image-space pixel joints (same convention as gaze), used by the hand crop.
    """
    df     = pd.read_csv(rec_dir / "hands.csv", header=None)
    frames = df.iloc[:, 0].values
    vals   = df.iloc[:, 1:].values.astype(np.float32)
    out = {}
    for i, fname in enumerate(frames):
        row = vals[i]
        out[fname] = dict(
            left=row[:52].reshape(26, 2), right=row[52:].reshape(26, 2),
            left_valid=bool(np.any(row[:52] != 0)),
            right_valid=bool(np.any(row[52:] != 0)),
        )
    return out
