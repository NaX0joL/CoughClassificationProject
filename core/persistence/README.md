# core/persistence — Experiment Packages ("mpkg")

Saves and reloads the complete artifacts of one experiment run as a self-contained
directory tree called an **mpkg**, and coordinates the sub-persistence processes
that write each artifact type.

## Position in the framework

```
core/experiment.py
  ├─ ExperimentPersistence.create(config, persistence_config, experiment_id)
  │     → allocates run_directory, scaffolds layout, writes config.pkl/config.txt
  ├─ per fold: ExperimentPersistence.save_fold(...) → figures + json + weights
  └─ ExperimentPersistence.save_cross_validation_summary(folds_metrics)

load path:
  ExperimentOrchestrator.load(mpkg_path)
    └─ ExperimentPersistence.load(run_directory) → PersistedExperimentArtifacts
```

## Key classes

| Class | Role |
|---|---|
| `ExperimentPersistence` | Orchestrator: `create`, `load`, `save_fold`, `save_cross_validation_summary` |
| `PersistenceConfig` | Tuning for figures/output rendering |
| `PersistedExperimentArtifacts` | Loaded artifacts: config, folds, CV summary, run_directory |
| `PersistedFoldArtifacts` | One loaded fold: `state_dict`, `loss_log`, `validation_metrics` |
| `processes/` | Sub-persistence modules, one per artifact type |

## The mpkg layout

```
<run_directory>/                       e.g. outputs/mpkg/tmp/mfcc_mlp/
├── __mpkg__.py                        marker; presence validates a run directory
├── config.pkl                         pickled experiment config dict
├── config.txt                         human-readable formatted config
├── figures/
│   ├── loss/loss-fold_<N>.png         train/validation loss curves
│   ├── confusion_matrix/confusion_matrix-fold_<N>.png
│   ├── output_train/fold_<N>-train.pdf        per-sample model-output PDFs
│   └── output_validation/fold_<N>-validation.pdf
├── json/
│   ├── loss/loss-fold_<N>.json        losses + best loss/epoch
│   └── metrics/
│       ├── metrics-fold_<N>.json      per-fold validation metrics
│       └── cross_validation_summary.json   per-metric mean ± std over folds
└── weights/
    └── fold_<N>.pth                   model state_dict per fold
```

Run directories are named by `experiment_id` (validated; no path separators) or a
timestamp `run_<date>_<time>`, with `_2`, `_3`, ... appended on collisions.

## Core concepts

- **Config** — the full experiment config dict (keyed `data_pipeline`, `model`,
  `training`, `metrics`, `persistence`) is pickled to `config.pkl` and formatted
  for humans to `config.txt`.
- **Weights** — per-fold `state_dict`s saved via `torch.save`; loaded with
  `weights_only=True` on CPU, then restored into fresh models via
  `FullModel.create_from_state_dict`.
- **JSON** — per-fold losses and metrics plus the cross-validation summary
  (`mean`, `standard_deviation` per metric over folds).
- **Figures** — loss-curve and confusion-matrix PNGs (dpi=150) per fold.
- **Model outputs** — per-sample PDF pages (features + optional GradCAM/LeGrad
  attributions) for a sampled subset of train and validation examples.
- **`__mpkg__.py` marker** — its presence is what `load` checks to validate that a
  directory is a saved run.

## Usage

Saving is handled automatically by `ExperimentOrchestrator.train_model()`. To
reload a run:

```python
from pathlib import Path
from core.experiment import ExperimentOrchestrator

orchestrator = ExperimentOrchestrator.load(Path("outputs/mpkg/tmp/mfcc_mlp"))
```

## Key parameters (`PersistenceConfig.default()`)

- `number_of_train_model_outputs=12`, `number_of_validation_model_outputs=12`
- `feature_color_percentile=99.0`
- `include_grad_cam=True`, `include_legrad=True` (subject to model support)
- Axis/colorbar labels default to `"Frame"` / `"Feature bin"` / `"Feature value"`

Pre-built variants in `config_plan/persistence/` toggle GradCAM/LeGrad and labels
per feature type (MFCC vs MelBand).

## Gotchas

- Loading requires exactly the five expected config keys; anything else raises
  `ValueError`.
- `save_fold` must be called for every fold and `save_cross_validation_summary`
  requires at least one fold with identical metric names across folds.
- Figures and PDFs are only produced if the relevant options and model support
  allow it (e.g., GradCAM needs a Conv1d-based architecture).

## Tests

`tests/persistence/` covers config formatting and the JSON persistence round-trip.