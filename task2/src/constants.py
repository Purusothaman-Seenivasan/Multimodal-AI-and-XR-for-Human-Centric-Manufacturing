"""Shared constants for the Support-Need Index (SNI) pipeline."""

import os

# Frame geometry (HoloLens RGB stream).
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Rolling-window length (~1 s at 30 fps) and gaze grid resolution.
WINDOW_SIZE = 30
GRID_SIZE = 8

# Used wherever a seed is relevant; the method itself is deterministic.
RANDOM_STATE = 42

# Recording splits (4 train + 1 held-out test, see report).
TRAIN_RECORDINGS = [
    "22_assy_0_1",
    "22_assy_2_3",
    "25_assy_0_1",
    "25_assy_2_1",
]
TEST_RECORDING = "27_assy_0_1"
ALL_RECORDINGS = TRAIN_RECORDINGS + [TEST_RECORDING]

# Instruction-sheet AOI: lower-left of the frame (x < 0.35W and y > 0.25H).
INSTRUCTION_AOI_X_MAX_FRAC = 0.35
INSTRUCTION_AOI_Y_MIN_FRAC = 0.25

# Hand-stall details: a joint speed only counts if the joint is present in both
# frames, and a hand needs at least this many such joints to be scored.
MIN_VALID_JOINTS = 5
HAND_MOTION_PERCENTILE = 75

# Episode threshold rule: SNI > mean + EPISODE_SIGMA * std (per recording).
EPISODE_SIGMA = 1.0

# AR_labels step names treated as the weak instruction-consulting proxy.
INSTRUCTION_PROXY_LABELS = {"check_instruction", "browse_instruction"}

# Locate the dataset relative to this file: task2/src -> AI_XR -> Data folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DATA_ROOT = os.path.join(_PROJECT_ROOT, "Data for technical task", "Data for technical task")
