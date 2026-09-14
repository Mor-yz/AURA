# AURA: Actuator Degradation-aware Unified Temporal Adaptation for Humanoid Locomotion

[![Release stage](https://img.shields.io/badge/release-phase--1-blue)](MANIFEST.md)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This staged release contains the AURA actuator-degradation identification network, its self-supervised training entry point, and a trained alpha/kappa checkpoint. 

## Release roadmap

- [x] Release AURA network architecture
- [x] Release alpha/kappa training code
- [x] Release trained alpha/kappa checkpoint
- [x] Release standalone smoke test
- [ ] Release locomotion policy checkpoints
- [ ] Release training/evaluation datasets
- [ ] Release robot deployment and hardware integration

## Contents

- `aura_model.py`: temporal convolution, cross-motor attention, and alpha/kappa predictor.
- `aura_data.py`: CSV data contract, contiguous-window construction, session-level splitting, and normalization.
- `train_aura.py`: reproducible training and evaluation entry point.
- `checkpoints/aura.pt`: legacy trained checkpoint with normalization statistics; see compatibility notes below.

## Checkpoint compatibility and tests

The bundled `checkpoints/aura.pt` predates the current network definition.
It lacks the version-2 `format_version` and `model_config` fields, and its
temporal-encoder weight keys differ from the current architecture. It cannot
currently be loaded by `aura_model.py`; adding metadata alone is insufficient.
A compatible checkpoint or a matching legacy loader is pending.

Run the smoke tests from this directory:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests/test_smoke.py
```

Current validation: two tests pass and the packaged-checkpoint test fails.
That failure is retained to make the incompatibility visible. The training
entry point writes version-2 checkpoints when supplied with aligned recordings;
training data is not included in this release.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Training

Supply your own aligned recordings; no recordings are included here:

```bash
python train_aura.py --csv-dir /path/to/csv_aligned --out-dir checkpoints/retrained
```

The loader requires complete contiguous frames for all 12 motors, keeps recording sessions separated before splitting, and fits normalization statistics on the training split only. The training objective is self-supervised torque reconstruction; simulation metadata labels are optional and used only for evaluation.

## Input contract

The network expects `[batch, 12, 30, 4]` tensors, ordered as `cmd_effort`, `fb_effort`, `position`, and `velocity` for each motor. Use `torch.load(..., weights_only=True)` when loading the checkpoint.

## Reproducibility and scope

Record the random seed, window length, stride, torque limits, session IDs in each split, and held-out torque reconstruction RMSE. Do not construct windows across gaps or recording boundaries. This is research code, not a robot deployment package. Policies, benchmark data, and hardware deployment integration are reserved for later staged releases.
