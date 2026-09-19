"""Simple matplotlib plots, all driven by the saved CSV files."""

import numpy as np
import matplotlib.pyplot as plt


def plot_sni_timeline(per_frame_df, recording, threshold=None, save_path=None):
    """SNI trace for one recording with episode and proxy bands shaded."""
    df = per_frame_df[per_frame_df["recording"] == recording].reset_index(drop=True)
    x = np.arange(len(df))
    sni = df["sni"].values
    if threshold is None:
        threshold = df["episode_threshold"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(x, sni, lw=0.8, color="#1f77b4", label="SNI")
    ax.axhline(threshold, color="red", ls="--", lw=0.8,
               label=f"episode threshold ({threshold:.2f})")
    ax.fill_between(x, 0, 1, where=df["support_episode"].values.astype(bool),
                    color="red", alpha=0.12, transform=ax.get_xaxis_transform(),
                    label="support-need episode")
    ax.fill_between(x, 0, 1, where=df["instruction_proxy"].values.astype(bool),
                    color="orange", alpha=0.20, transform=ax.get_xaxis_transform(),
                    label="instruction-consulting proxy (weak)")
    ax.set_xlabel("frame index")
    ax.set_ylabel("Support-Need Index")
    ax.set_ylim(0, 1)
    ax.set_title(f"Support-Need Index over time - {recording}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig


def plot_signal_components(per_frame_df, recording, save_path=None):
    """The three z-scored components and the fused SNI for one recording."""
    df = per_frame_df[per_frame_df["recording"] == recording].reset_index(drop=True)
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1]))
    ax1.plot(x, df["z_gaze_entropy"], lw=0.7, color="#4C72B0", label="z gaze entropy")
    ax1.plot(x, df["z_hand_stall"], lw=0.7, color="#DD8452", label="z hand stall")
    ax1.plot(x, df["z_instruction_aoi_dwell"], lw=0.7, color="#55A868",
             label="z instruction-AOI dwell")
    ax1.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax1.set_ylabel("z-score (higher = more support need)")
    ax1.set_title(f"SNI components - {recording}")
    ax1.legend(fontsize=8, ncol=3, loc="upper right")

    ax2.plot(x, df["sni"], lw=0.8, color="#1f77b4", label="fused SNI")
    ax2.axhline(df["episode_threshold"].iloc[0], color="red", ls="--", lw=0.8,
                label="threshold")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("frame index")
    ax2.set_ylabel("SNI")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig


def plot_recording_auc(recording_auc_df, save_path=None):
    """Bar chart of per-recording AUC (with the pooled row highlighted)."""
    df = recording_auc_df.copy()
    colors = ["#C44E52" if r == "pooled" else "#4C72B0" for r in df["recording"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["recording"], df["auc"], color=colors)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC vs proxy")
    ax.set_title("Fused SNI AUC per recording")
    ax.set_xticklabels(df["recording"], rotation=40, ha="right", fontsize=8)
    for i, v in enumerate(df["auc"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig


def plot_ablation_auc(ablation_df, save_path=None):
    """Bar chart of pooled AUC for each single signal vs the fused SNI."""
    df = ablation_df.copy()
    colors = ["#C44E52" if s == "Fused SNI" else "#4C72B0" for s in df["signal"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["signal"], df["pooled_auc"], color=colors)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax.set_ylim(0, 1)
    ax.set_ylabel("pooled ROC-AUC vs proxy")
    ax.set_title("Ablation: single signal vs fused SNI")
    ax.set_xticklabels(df["signal"], rotation=20, ha="right", fontsize=8)
    for i, v in enumerate(df["pooled_auc"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig
