# Multimodal Support-Need Index

This component estimates when an assembly operator may benefit from XR guidance. It
is an interpretable, training-free signal built from HoloLens gaze and hand tracking.

## Method

Three signals are computed over a 30-frame rolling window:

1. **Gaze entropy** — dispersed gaze as a cue for visual search.
2. **Hand stall** — reduced joint motion as a cue for hesitation or pausing.
3. **Instruction-area dwell** — sustained gaze on the instruction sheet.

Each signal is standardized per recording. Their equal-weight mean is passed through
a sigmoid to produce a continuous score in `[0, 1]`. Frames above the recording mean
plus one standard deviation are grouped into support-need episodes.

Because IndustReal has no ground-truth support-need annotation, the pipeline uses
instruction-consulting actions as a weak evaluation proxy. The fused index reaches
0.716 pooled ROC-AUC and 0.804 ROC-AUC on the held-out recording.

## Structure

```text
support-need-index/
├── notebooks/
│   └── 01_support_need_index.ipynb
├── src/                            # signal extraction, fusion, evaluation, and plots
├── outputs/
│   ├── episodes_*.csv
│   ├── sni_per_frame.csv
│   ├── ablation_auc.csv
│   ├── recording_auc.csv
│   └── plots/
├── README.md
└── requirements.txt
```

## Run

From the repository root:

```bash
pip install -r support-need-index/requirements.txt
jupyter notebook support-need-index/notebooks/
```

Set `DATA_ROOT` in `src/constants.py` to the extracted IndustReal dataset, then run
the notebook from top to bottom. Result tables and figures are written to `outputs/`.

For the full formulation and limitations, see
[`report_task2.pdf`](../report_task2.pdf).
