# core — Framework Library

`core/` is the reusable framework of the Cough Classification Project. It wires the
data pipeline, models, training, metrics, persistence, and explainability into a
single reproducible experiment workflow, driven entirely by configuration.

## Components

| Directory / file | Component | README |
|---|---|---|
| `experiment.py`, `experiment_config.py` | Experiment orchestration (this page) | — |
| `data_pipeline/` | Audio → feature extraction → dataset splitting | [`core/data_pipeline/README.md`](data_pipeline/README.md) |
| `model/` | Neural-network architectures + behaviors | [`core/model/README.md`](model/README.md) |
| `training/` | Training loop, checkpointing, config | [`core/training/README.md`](training/README.md) |
| `metrics/` | Model evaluation and classification metrics | [`core/metrics/README.md`](metrics/README.md) |
| `persistence/` | "mpkg" experiment package save/load | [`core/persistence/README.md`](persistence/README.md) |
| `gallery/` | Explainability (GradCAM / LeGrad) and model outputs | [`core/gallery/README.md`](gallery/README.md) |

Pre-built configuration instances for every component live in `config_plan/` and
are what scripts and `main.py` actually use.

## Experiment Orchestration

The top-level workflow is `ExperimentOrchestrator` in `core/experiment.py`.

### Position in the framework

```
ExperimentOrchestrator.train_model()
│
├── ExperimentConfig  (data_pipeline + model + training + metrics + persistence)
│
├── DataPipeline.create(config) ──► get_data_split()   → DataSplit (test + 5 dev folds)
│
├── for each development fold:
│   ├── FullModel.create(model_config)                → fresh model on device
│   ├── Trainer(config, model, train_ds, val_ds).fit() → LossLog
│   ├── ModelEvaluator.evaluate(model, val_ds)         → ModelEvaluation
│   └── ExperimentPersistence.save_fold(...)           → per-fold artifacts
│
└── ExperimentPersistence.save_cross_validation_summary(folds_metrics)
```

### Key classes

- **`ExperimentOrchestrator`** (`experiment.py`) — runs the whole workflow.
  - `train_model()` — resolves device, seeds RNG, creates persistence, gets the
    data split, trains + evaluates one model per fold, saves everything.
  - `load(mpkg_path)` — classmethod that reconstructs an orchestrator from a saved
    mpkg package (rebuilds each fold's `FullModel` from saved weights).
  - `test_model()` — placeholder; hold-out test evaluation is not yet implemented.
- **`ExperimentConfig`** (`experiment_config.py`) — a dataclass bundling the five
  sub-configs: `data_pipeline_config`, `model_config`, `training_config`,
  `metrics_config`, `persistence_config`.
  - `default()` — builds every sub-config from its `default()` factory.
  - `from_persisted_config()` — rebuilds from an mpkg config dict, requiring exactly
    the keys `data_pipeline`, `model`, `training`, `metrics`, `persistence`.

### Key parameters

`ExperimentConfig` has no parameters of its own — all knobs live in the five
sub-configs it bundles.

### Gotchas

- `test_model()` is a stub; the hold-out `test_dataset` produced by the data
  pipeline is currently unused during training.
- Per-fold models are created fresh via `FullModel.create()`, which deepcopies the
  shared `model_config` so folds never share weights.
- The config dict handed to persistence contains the config **instances** (not just
  values); they are pickled into the mpkg and unpickled on load.

## Related

- Project overview and setup: [`README.md`](../README.md)
- Coding conventions: [`AGENTS.md`](../AGENTS.md)