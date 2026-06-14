"""Frozen DINOv2 RGB embeddings (Approach B).

Extracts a 384-dim CLS embedding per sampled frame with a frozen DINOv2 ViT-S/14,
optionally cropping the frame first (full / center / gaze / hand). Embeddings are
cached to .npz so the heavy extraction runs once.

torch / timm / Pillow are imported lazily inside the functions that need them, so
the downstream notebook can load a cached .npz without a GPU stack installed.
"""

import numpy as np
import pandas as pd

from constants import (
    IMG_W, IMG_H, INPUT_SIZE, BATCH_SIZE, FRAME_STRIDE, CROP_SIZE,
    DINO_MODEL_NAME, B_WINDOW_SIZE, DINOV2_CACHE_DIR,
)
import data_utils
import label_utils


# ── Model + image transforms ───────────────────────────────────────────────

def load_dino_model():
    """Load the frozen DINOv2 ViT-S/14; returns (model, device). Needs timm+torch."""
    import timm, torch
    model = timm.create_model(DINO_MODEL_NAME, pretrained=True,
                              num_classes=0, img_size=INPUT_SIZE)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device


def full_frame_transform():
    """Standard ImageNet preprocessing: resize 256 -> center-crop 224 -> normalise."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def crop_transform():
    """Preprocessing for an already-square crop: resize straight to 224 (no crop)."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Crop geometry ──────────────────────────────────────────────────────────

def is_valid_gaze(gx, gy, W=IMG_W, H=IMG_H):
    """A gaze sample is valid only if finite, non-(0,0), and in bounds."""
    if gx is None or gy is None:
        return False
    gx, gy = float(gx), float(gy)
    if not (np.isfinite(gx) and np.isfinite(gy)):
        return False
    return not (gx == 0 and gy == 0) and 0 <= gx < W and 0 <= gy < H


def square_crop_box(cx, cy, crop=CROP_SIZE, W=IMG_W, H=IMG_H):
    """(left, top, right, bottom) for a `crop`x`crop` square centred at (cx, cy),
    shifted to stay inside the image (no black padding)."""
    half = crop / 2.0
    left, top = cx - half, cy - half
    if crop <= W:
        left = min(max(left, 0.0), W - crop)
    else:
        left = (W - crop) / 2.0
    if crop <= H:
        top = min(max(top, 0.0), H - crop)
    else:
        top = (H - crop) / 2.0
    left, top = int(round(left)), int(round(top))
    return (left, top, left + crop, top + crop)


def hand_crop_center(hand_info, W=IMG_W, H=IMG_H):
    """Centre of the union bbox over the valid hand joints; image centre if none."""
    pts = []
    for side, valid in [("left", "left_valid"), ("right", "right_valid")]:
        if hand_info[valid]:
            j = hand_info[side]
            j = j[~((j[:, 0] == 0) & (j[:, 1] == 0))]   # drop (0,0) missing joints
            if len(j):
                pts.append(j)
    if not pts:
        return W / 2.0, H / 2.0
    allp = np.concatenate(pts, axis=0)
    return float((allp[:, 0].min() + allp[:, 0].max()) / 2.0), \
           float((allp[:, 1].min() + allp[:, 1].max()) / 2.0)


def crop_image(img, mode, gaze_xy, hand_info):
    """Apply one crop mode to a PIL image. full_frame is returned unchanged."""
    if mode == "full_frame":
        return img
    if mode == "center_crop":
        cx, cy = IMG_W / 2.0, IMG_H / 2.0
    elif mode == "gaze_crop":
        gx, gy = gaze_xy
        cx, cy = (gx, gy) if is_valid_gaze(gx, gy) else (IMG_W / 2.0, IMG_H / 2.0)
    elif mode == "hand_crop":
        cx, cy = hand_crop_center(hand_info)
    else:
        raise ValueError(f"unknown crop mode: {mode}")
    return img.crop(square_crop_box(cx, cy))


# ── Extraction + cache ─────────────────────────────────────────────────────

def _cache_path(mode):
    DINOV2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if mode == "full_frame" else f"_crop{CROP_SIZE}"
    return DINOV2_CACHE_DIR / f"dinov2_vits14_{mode}{tag}_stride{FRAME_STRIDE}_size{INPUT_SIZE}.npz"


def load_embeddings(mode):
    """Load a cached embedding set as a dict of arrays, or None if not cached."""
    path = _cache_path(mode)
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def extract_embeddings(mode, splits=(("train", None), ("test", None)), batch_size=BATCH_SIZE):
    """Extract (and cache) DINOv2 embeddings for one crop mode over all recordings.

    Returns the same dict shape as `load_embeddings`. Requires torch+timm+Pillow.
    Re-extraction is skipped when a cache file already exists.
    """
    cached = load_embeddings(mode)
    if cached is not None:
        return cached

    import torch
    from PIL import Image
    from constants import TRAIN_RECS, TEST_RECS

    model, device = load_dino_model()
    transform = full_frame_transform() if mode == "full_frame" else crop_transform()

    X, Y, split_col, rec_col, frame_col = [], [], [], [], []
    for split, recs in [("train", TRAIN_RECS), ("test", TEST_RECS)]:
        for rec in recs:
            rec_dir = data_utils.recording_dir(split, rec)
            frames  = data_utils.list_rgb_frames(rec_dir)
            # Raw pixel-space gaze for the gaze crop centre.
            gaze_df  = pd.read_csv(rec_dir / "gaze.csv", header=None,
                                   names=["frame", "gaze_x", "gaze_y"])
            gaze_map = {r.frame: (r.gaze_x, r.gaze_y) for r in gaze_df.itertuples()}
            hands = data_utils.load_hands_imgspace(rec_dir)
            labels = label_utils.build_cumulative_labels(data_utils.load_psr(rec_dir), frames)

            for i in range(0, len(frames), batch_size):
                chunk = frames[i:i + batch_size]
                tensors = []
                for f in chunk:
                    img = Image.open(rec_dir / "rgb" / f).convert("RGB")
                    img = crop_image(img, mode, gaze_map.get(f, (0, 0)), hands.get(f))
                    tensors.append(transform(img))
                with torch.no_grad():
                    feats = model(torch.stack(tensors).to(device)).cpu().numpy()
                X.append(feats)
            Y.append(labels)
            split_col += [split] * len(frames)
            rec_col   += [rec] * len(frames)
            frame_col += list(frames)

    out = dict(
        X_embeddings=np.concatenate(X).astype(np.float32),
        Y_labels=np.concatenate(Y).astype(np.int32),
        split_names=np.array(split_col), recording_names=np.array(rec_col),
        frame_names=np.array(frame_col),
    )
    np.savez(_cache_path(mode), **out)
    return out


def rolling_mean_per_recording(X, rec_names, window=B_WINDOW_SIZE):
    """Causal rolling mean of embeddings, applied independently per recording."""
    X_out = np.empty_like(X)
    for rec in pd.unique(rec_names):
        idx = np.where(rec_names == rec)[0]
        s, e = idx[0], idx[-1] + 1
        X_out[s:e] = pd.DataFrame(X[s:e]).rolling(window, min_periods=1).mean().values
    return X_out
