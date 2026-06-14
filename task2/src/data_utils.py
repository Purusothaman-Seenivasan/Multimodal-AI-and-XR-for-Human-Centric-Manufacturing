"""Loaders for the per-recording gaze, hand and instruction-label files."""

import os
import pandas as pd

from constants import DATA_ROOT, TEST_RECORDING


def recording_dir(name):
    """Return the folder for a recording, searching the train and test splits."""
    for split in ("train", "test"):
        path = os.path.join(DATA_ROOT, split, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"recording '{name}' not found under {DATA_ROOT}")


def split_of(name):
    """'test' for the held-out recording, otherwise 'train'."""
    return "test" if name == TEST_RECORDING else "train"


def load_gaze(recording_dir):
    """Load gaze.csv as columns: frame, x, y. (x, y) = (0, 0) means no detection."""
    df = pd.read_csv(os.path.join(recording_dir, "gaze.csv"), header=None)
    df.columns = ["frame", "x", "y"]
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df


def load_hands(recording_dir):
    """Load hands.csv: frame + 104 joint coords (2 hands x 26 joints x x,y)."""
    df = pd.read_csv(os.path.join(recording_dir, "hands.csv"), header=None)
    df.columns = ["frame"] + [f"j{i}" for i in range(df.shape[1] - 1)]
    return df


def list_recording_frames(recording_dir):
    """Frame names for a recording, taken from the gaze file (one row per frame)."""
    return load_gaze(recording_dir)["frame"].astype(str).tolist()


def load_recording(recording_dir):
    """Convenience loader returning (frames, gaze_df, hands_df)."""
    gaze = load_gaze(recording_dir)
    hands = load_hands(recording_dir)
    frames = gaze["frame"].astype(str).tolist()
    return frames, gaze, hands
