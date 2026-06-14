"""Simple matplotlib plots, all driven by the saved CSV tables.

Each function takes a DataFrame (loaded from outputs/) and an optional save path.
Plain matplotlib only: bar charts, line plots, and imshow heatmaps.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from constants import EXPECTED_STEPS, LATE_STEPS


def _save(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---- Shared style for the final Part A summary plots ----
PARTA_METHODS = [
    "A3 sensor smooth",
    "A5 sensor K=15",
    "B1 DINOv2 MLP raw",
    "B2 DINOv2 smooth",
    "Candidate K=3",
]
METHOD_COLORS = {
    "A3 sensor smooth": "#9e9e9e",
    "A5 sensor K=15": "#dd8452",
    "B1 DINOv2 MLP raw": "#4c72b0",
    "B2 DINOv2 smooth": "#55a868",
    "Candidate K=3": "#c44e52",
}
STATE_STEP_LABELS = [
    "step 3", "step 6", "step 9", "step 15", "step 18",
    "step 21", "step 24", "step 27", "step 30",
]


def plot_progress_curve(progress_df, save_path=None):
    """Line plot of true vs predicted progress over frames.

    progress_df: columns 'frame_idx', 'true', and one column per prediction.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    x = progress_df["frame_idx"]
    for col in progress_df.columns:
        if col == "frame_idx":
            continue
        style = "k--" if col == "true" else "-"
        ax.plot(x, progress_df[col], style, label=col, alpha=0.85)
    ax.set_xlabel("frame index")
    ax.set_ylabel("progress (steps complete / 9)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_per_step_f1(per_step_df, label_col="config", save_path=None):
    """Grouped bar chart of F1 per step, one group of bars per config.

    per_step_df: long format with columns 'step', 'f1', and `label_col`.
    """
    configs = per_step_df[label_col].unique()
    x = np.arange(len(EXPECTED_STEPS))
    w = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cfg in enumerate(configs):
        sub = per_step_df[per_step_df[label_col] == cfg].set_index("step")
        f1s = [sub.loc[s, "f1"] if s in sub.index else 0.0 for s in EXPECTED_STEPS]
        ax.bar(x + i * w - 0.4 + w / 2, f1s, w, label=str(cfg))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in EXPECTED_STEPS], rotation=45, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_late_step_f1(late_df, label_col="config", save_path=None):
    """Bar chart of F1 on the late steps (21/24/27/30) per config."""
    configs = late_df[label_col].unique()
    x = np.arange(len(LATE_STEPS))
    w = 0.8 / len(configs)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, cfg in enumerate(configs):
        sub = late_df[late_df[label_col] == cfg].set_index("step")
        f1s = [sub.loc[s, "f1"] if s in sub.index else 0.0 for s in LATE_STEPS]
        ax.bar(x + i * w - 0.4 + w / 2, f1s, w, label=str(cfg))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in LATE_STEPS])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Late-step F1")
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_state_heatmap(Y, title="", save_path=None):
    """Heatmap of a [N, 9] cumulative state matrix (steps on y, frames on x)."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(Y.T, aspect="auto", cmap="Greys", interpolation="nearest",
              vmin=0, vmax=1)
    ax.set_yticks(range(len(EXPECTED_STEPS)))
    ax.set_yticklabels([f"Step {s}" for s in EXPECTED_STEPS])
    ax.set_xlabel("frame index")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_metric_bars(summary_df, metric, label_col="config", save_path=None):
    """Single-metric bar chart across configs (e.g. macro_f1)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary_df[label_col].astype(str), summary_df[metric], color="#4C72B0")
    ax.set_ylabel(metric)
    ax.set_xticklabels(summary_df[label_col].astype(str), rotation=30, ha="right")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    return _save(fig, save_path)


# =====================================================================
# Final Part A summary plots (all driven by the saved CSV tables).
# =====================================================================

def plot_final_state_heatmap(frame_idx, Y_true, Y_pred, save_path=None):
    """Two-panel cumulative-state heatmap: true (top) vs predicted (bottom).

    frame_idx: 1-D array of the original frame index for each column.
    Y_true, Y_pred: [N, 9] 0/1 cumulative-state matrices.
    """
    state_cmap = ListedColormap(["#f7fcf5", "#00441b"])  # light bg / dark green
    # extent puts the 9 rows at integer y-centres 0..8 and uses real frame ids on x.
    extent = [float(frame_idx[0]), float(frame_idx[-1]), 8.5, -0.5]
    panels = [
        (Y_true, "TRUE cumulative workflow state"),
        (Y_pred, "PREDICTED (candidate: MLP smooth+sustained K=3)"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for ax, (Y, title) in zip(axes, panels):
        ax.imshow(np.asarray(Y).T, aspect="auto", cmap=state_cmap,
                  interpolation="nearest", vmin=0, vmax=1, extent=extent)
        ax.set_yticks(range(9))
        ax.set_yticklabels(STATE_STEP_LABELS, fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.tick_params(labelsize=10)
    axes[1].set_xlabel("original frame index", fontsize=11)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_late_step_f1_summary(late_df, save_path=None):
    """Grouped late-step (21/24/27/30) F1 bar chart, one group per step.

    late_df: long format with columns 'method', 'step', 'f1'.
    """
    steps = [21, 24, 27, 30]
    methods = [m for m in PARTA_METHODS if m in set(late_df["method"])]
    x = np.arange(len(steps))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, m in enumerate(methods):
        sub = late_df[late_df["method"] == m].set_index("step")
        f1s = [float(sub.loc[s, "f1"]) if s in sub.index else 0.0 for s in steps]
        ax.bar(x + i * w - 0.4 + w / 2, f1s, w, label=m, color=METHOD_COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"step {s}" for s in steps])
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.08)

    # summary annotation: mean F1 over steps 21/24/27, and step 30 alone.
    lines = []
    for m in methods:
        sub = late_df[late_df["method"] == m].set_index("step")
        block = np.mean([float(sub.loc[s, "f1"]) for s in (21, 24, 27) if s in sub.index])
        s30 = float(sub.loc[30, "f1"]) if 30 in sub.index else float("nan")
        short = m.split(" ")[0]
        lines.append(f"{short}: block21/24/27={block:.2f}, step30={s30:.2f}")
    ax.text(0.01, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=len(methods), frameon=False)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_metric_progression(metric_df, save_path=None):
    """1x4 bar panels showing per-frame metric progression across methods.

    metric_df: columns 'method', 'macro_f1', 'hamming_loss',
               'exact_match', 'progress_mae'.
    """
    methods = [m for m in PARTA_METHODS if m in set(metric_df["method"])]
    df = metric_df.set_index("method").loc[methods]
    bar_colors = [METHOD_COLORS[m] for m in methods]
    xlabels = [
        "A3 sensor\nsmooth", "A5 sensor\nK=15", "B1 DINOv2\nMLP raw",
        "B2 DINOv2\nsmooth", "Candidate\nK=3",
    ][:len(methods)]
    panels = [
        ("macro_f1", "↑ Macro F1"),
        ("hamming_loss", "↓ Hamming loss"),
        ("exact_match", "↑ Exact Match"),
        ("progress_mae", "↓ Progress MAE"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    x = np.arange(len(methods))
    for ax, (col, title) in zip(axes, panels):
        vals = df[col].to_numpy()
        ax.bar(x, vals, color=bar_colors)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        offset = 0.02 * (vals.max() if vals.max() > 0 else 1.0)
        for xi, v in zip(x, vals):
            ax.text(xi, v + offset, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_ylim(0, vals.max() * 1.18 if vals.max() > 0 else 1.0)
    fig.suptitle(
        "Plot 1 — Experimental decision path: per-frame metric progression\n"
        "(two sensor baselines: A3 frame-state + A5 K=15 temporal "
        "→ DINOv2 raw → smooth → candidate K=3)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return _save(fig, save_path)
