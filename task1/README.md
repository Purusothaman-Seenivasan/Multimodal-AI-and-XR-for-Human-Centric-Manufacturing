# Workflow-State Estimation Submission

This repository contains the code for the technical assignment (Part A:
cumulative workflow-state estimation on IndustReal).

The full explanation, methodology, results, and limitations are provided in the
submitted report. This README only covers how to run the code.

## Repository Structure

```text
task1/
  notebooks/
    01_sensor_baseline.ipynb        # Approach A — sensor baseline
    02_dinov2_workflow_state.ipynb  # Approach B — DINOv2 RGB
  src/                              # shared utilities (constants, data, labels, metrics, ...)
  outputs/
    sensor/                         # Approach A CSVs + plots
    dinov2/                         # Approach B CSVs + plots (+ cached embeddings)
  README.md
  requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset next to this folder so the layout is:

```text
<parent>/
  task1/                            # this folder
  Data for technical task/
    Data for technical task/
      train/{22_assy_0_1, 22_assy_2_3, 25_assy_0_1, 25_assy_2_1}/
      test/{27_assy_0_1}/
```

The data path is set in `src/constants.py` (`DATA_ROOT`) — adjust it if your
dataset lives elsewhere.

## Running

```bash
jupyter notebook notebooks/
```

Run each notebook top to bottom. Each one trains its models, writes its result
tables to `outputs/`, and draws its plots from those saved CSVs.

The DINOv2 embeddings ship pre-extracted in `outputs/dinov2/cache/*.npz`, so
notebook 02 runs the full downstream pipeline without a GPU. Delete the cache to
force re-extraction (this needs `torch`, `torchvision`, `timm`, and `Pillow`).

## Outputs

All generated tables and figures are written to:

- `outputs/sensor/` — Approach A results
- `outputs/dinov2/` — Approach B results

See the submitted report for the methodology, result tables, and discussion.
