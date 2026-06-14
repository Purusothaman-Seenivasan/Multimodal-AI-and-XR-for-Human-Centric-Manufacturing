# Technical Assignment Submission

This repository contains the code for the technical assignment on the IndustReal
dataset, split into two tasks:

- **Task 1 — Cumulative workflow-state estimation** (`task1/`)
- **Task 2 — Multimodal support-need index** (`task2/`)

The full explanation, methodology, results, and limitations for each task are
provided in the submitted reports (`report_task1.pdf`, `report_task2.pdf`). This
README only covers the folder layout and how to run the code.

## Folder Structure

```text
final_submission/
  README.md                 # this file
  report_task1.pdf          # Task 1 report (methodology, results, discussion)
  report_task2.pdf          # Task 2 report (methodology, results, discussion)

  task1/                    # Task 1 — workflow-state estimation
    notebooks/
      01_sensor_baseline.ipynb        # Approach A — sensor baseline
      02_dinov2_workflow_state.ipynb  # Approach B — DINOv2 RGB
    src/                              # shared utilities (constants, data, labels, metrics, ...)
    outputs/
      sensor/                         # Approach A CSVs + plots
      dinov2/                         # Approach B CSVs + plots (+ cached embeddings)
      plots/                          # combined summary figures
    README.md                         # Task 1 run instructions
    requirements.txt

  task2/                    # Task 2 — multimodal support-need index
    notebooks/
      01_support_need_index.ipynb     # Support-Need Index (SNI) pipeline
    src/                              # feature + eval utilities (gaze, hand, SNI, ...)
    outputs/
      episodes_*.csv                  # per-recording episode tables
      sni_per_frame.csv               # per-frame SNI signal
      ablation_auc.csv                # component ablation results
      recording_auc.csv              # per-recording evaluation
      plots/                          # SNI timeline, component, and AUC figures
    requirements.txt
```

## Task 1 — Workflow-State Estimation

Cumulative workflow-state estimation on IndustReal. Two approaches are compared:

- **Approach A** — a sensor-feature + sliding-window baseline.
- **Approach B** — DINOv2 RGB embeddings with downstream smoothing/postprocessing.

The DINOv2 embeddings ship pre-extracted in `task1/outputs/dinov2/cache/*.npz`,
so notebook 02 runs the full downstream pipeline without a GPU. Delete the cache
to force re-extraction (this needs `torch`, `torchvision`, `timm`, and `Pillow`).

See `task1/README.md` for detailed run instructions and `report_task1.pdf` for
the methodology and results.

## Task 2 — Multimodal Support-Need Index

A multimodal support-seeking proxy (the Support-Need Index, SNI) built from gaze
entropy, hand-motion stall, and instruction-AOI gaze dwell. Notebook
`01_support_need_index.ipynb` builds the per-frame signal, runs the component
ablation, and evaluates it per recording.

See `report_task2.pdf` for the methodology and results.

## Setup

Each task has its own `requirements.txt`. Install from the relevant task folder,
e.g.:

```bash
pip install -r task1/requirements.txt   # or task2/requirements.txt
```

Place the dataset next to this folder so the layout is:

```text
<parent>/
  final_submission/                   # this repo
  Data for technical task/
    Data for technical task/
      train/{22_assy_0_1, 22_assy_2_3, 25_assy_0_1, 25_assy_2_1}/
      test/{27_assy_0_1}/
```

The data path is set in each task's `src/constants.py` (`DATA_ROOT`) — adjust it
if your dataset lives elsewhere.

## Running

```bash
jupyter notebook task1/notebooks/    # Task 1
jupyter notebook task2/notebooks/    # Task 2
```

Run each notebook top to bottom. Each one trains its models / builds its signals,
writes its result tables to the task's `outputs/`, and draws its plots from those
saved CSVs.
