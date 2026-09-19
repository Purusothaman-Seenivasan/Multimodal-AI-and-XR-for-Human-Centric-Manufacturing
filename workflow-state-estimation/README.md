# Workflow-State Estimation

This component estimates cumulative assembly progress from Microsoft HoloLens 2
data. It compares a multimodal sensor baseline with a visual representation pipeline
based on frozen DINOv2 embeddings.

## Approaches

### Sensor baseline

Gaze, hand-joint, and head-pose measurements are combined into a 118-dimensional
feature vector. A multi-output Random Forest predicts completed procedure steps,
followed by probability smoothing or sustained temporal commitment.

### DINOv2 visual pipeline

RGB frames are encoded with a frozen DINOv2 ViT-S/14 model. Logistic Regression,
Random Forest, and MLP heads are compared, after which the selected MLP prediction
stream is smoothed and constrained to remain monotonic.

The strongest recorded configuration (`mlp_k3`) achieves 0.813 macro F1, 0.639
exact-match accuracy, and zero monotonicity violations on the held-out recording.

## Structure

```text
workflow-state-estimation/
├── notebooks/
│   ├── 01_sensor_baseline.ipynb
│   └── 02_dinov2_workflow_state.ipynb
├── src/                            # loading, labels, features, metrics, and plots
├── outputs/
│   ├── sensor/                     # sensor-baseline tables and figures
│   ├── dinov2/                     # visual-model tables, figures, and embedding cache
│   └── plots/                      # combined comparison figures
├── README.md
└── requirements.txt
```

## Run

From the repository root:

```bash
pip install -r workflow-state-estimation/requirements.txt
jupyter notebook workflow-state-estimation/notebooks/
```

Set `DATA_ROOT` in `src/constants.py` to the extracted IndustReal dataset before
running. Execute the notebooks in numeric order and run all cells from top to bottom.

The cached embeddings in `outputs/dinov2/cache/` allow the classifier and temporal
post-processing experiments to run without a GPU. Remove the cache only when you
want to re-extract embeddings from RGB frames.

## Outputs

- `outputs/sensor/` contains sensor-baseline metrics, predictions, and plots.
- `outputs/dinov2/` contains classifier comparisons, crop ablations, predictions,
  transition metrics, and plots.
- `outputs/plots/` contains the combined experiment summary figures.

For the full formulation and analysis, see
[`report_task1.pdf`](../report_task1.pdf).
