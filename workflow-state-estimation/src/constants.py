"""Shared constants for both approaches.

Paths, the PSR step list, the train/test split, and the hyperparameters that
match the report. Importing this from one place keeps the two notebooks in sync.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
# Repo layout:  <root>/src, <root>/notebooks, <root>/outputs
# The dataset sits next to the repo (see README for where to drop it).
ROOT       = Path(__file__).resolve().parents[1]
DATA_ROOT  = ROOT.parent / "Data for technical task" / "Data for technical task"
TRAIN_DIR  = DATA_ROOT / "train"
TEST_DIR   = DATA_ROOT / "test"

OUTPUT_DIR        = ROOT / "outputs"
SENSOR_OUT_DIR    = OUTPUT_DIR / "sensor"
DINOV2_OUT_DIR    = OUTPUT_DIR / "dinov2"
DINOV2_CACHE_DIR  = DINOV2_OUT_DIR / "cache"

# ── PSR steps ──────────────────────────────────────────────────────────────
# The 9 assembly steps we predict, in order. Labels are a cumulative multi-hot
# vector over these steps.
EXPECTED_STEPS = [3, 6, 9, 15, 18, 21, 24, 27, 30]
STEP_TO_IDX    = {s: i for i, s in enumerate(EXPECTED_STEPS)}
N_STEPS        = len(EXPECTED_STEPS)

# Step 12 (install short rear chassis) is an optional variant of step 9
# (install rear chassis). Remap it so both count as the same step.
STEP_ALIASES = {12: 9}

# Late steps that complete near the very end of the test recording (hardest).
LATE_STEPS = [21, 24, 27, 30]

# ── Data split ─────────────────────────────────────────────────────────────
# Only 4 train recordings + 1 test recording are available, so we keep models
# small and avoid anything over-parameterised.
TRAIN_RECS = ["22_assy_0_1", "22_assy_2_3", "25_assy_0_1", "25_assy_2_1"]
TEST_RECS  = ["27_assy_0_1"]

# Native HoloLens RGB resolution; gaze and hand joints are in this pixel space.
IMG_W, IMG_H = 1280, 720

RANDOM_STATE = 42

# ── Approach A (sensor) hyperparameters ────────────────────────────────────
A_WINDOW_SIZE = 15     # rolling-mean window on the 118-dim sensor features
A_PROB_WINDOW = 15     # probability-smoothing window (A3)
A_THRESHOLD   = 0.5    # threshold applied after smoothing
A_SUSTAIN_K   = 15     # sustained-commitment streak length (A5)

# ── Approach B (DINOv2) hyperparameters ────────────────────────────────────
DINO_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"  # ViT-S/14, 384-dim CLS
INPUT_SIZE      = 224
FRAME_STRIDE    = 5     # sample every 5th frame
BATCH_SIZE      = 32
B_WINDOW_SIZE   = 10    # rolling-mean window on embeddings (sampled-frame units)
CROP_SIZE       = 448   # square crop side length for the crop ablation

# Selected final prototype (matches the report).
B_PROB_WINDOW = 15
B_THRESHOLD   = 0.6
B_SUSTAIN_K   = 3

CROP_MODES = ["full_frame", "center_crop", "gaze_crop", "hand_crop"]
