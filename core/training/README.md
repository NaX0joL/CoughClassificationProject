# core/training — Training Loop

Runs the training loop for one model on one development fold: build dataloaders,
optimizer, and criterion, iterate epochs, track the best model, and return a
`LossLog`.

## Position in the framework

```
core/experiment.py
  └─ _train_development_fold()
       └─ Trainer(config, model, train_dataset, validation_dataset).fit()  → LossLog
            ├─ build_criterion / build_optimizer
            ├─ do_train_logic(epochs, model, criterion, optimizer, loaders, checkpoint)
            │    ├─ train phase:  model.training_step(batch, criterion) → loss.backward() → optimizer.step()
            │    └─ val phase:    model.validation_step(batch, criterion)
            │    └─ BestModelCheckpoint.update(model, val_loss, epoch)
            └─ loss_log (persisted by core/persistence)
```

## Key classes

| Class | Role |
|---|---|
| `Trainer` | Public entry point; builds loaders/criterion/optimizer, delegates to `do_train_logic`, returns `LossLog` |
| `do_train_logic` | Epoch loop with train/val phases and batch-size-weighted loss averaging |
| `BestModelCheckpoint` | In-memory tracker of the best validation-loss state |
| `TrainDisplay` | tqdm epoch progress bar (train + validation loss in the postfix) |
| `TrainingConfig` | Dataclass of all training hyperparameters |
| `LossLog` | Records per-epoch train/val losses + best loss/epoch |

## Core concepts

- **Batch contract** — batches are dicts with `"value"` (features) and `"label"`
  (class index), produced by `ExampleDataset`. The loop moves tensors to the
  model's device and passes the criterion *into* the model's step methods.
- **Loss averaging** — per-epoch loss is the batch-size-weighted mean of
  per-batch losses (weighted so the reported value reflects example count, not
  batch count).
- **Best-model tracking** — `BestModelCheckpoint` records a `deepcopy` of the
  `state_dict` only when validation loss is *strictly* lower; ties are ignored.
  Training always runs the full epoch count (there is **no early termination**),
  then optionally restores the best state via `load_best_model`.
- **Determinism** — only the DataLoader is seeded (from `random_seed`); model
  weight init and the global torch RNG are not.

## Usage

`Trainer` is constructed by the orchestrator per fold; it never writes files
(persistence is external). For a standalone run:

```python
from core.training import Trainer, TrainingConfig

loss_log = Trainer(
    config=TrainingConfig.default(),
    model=model,                       # already on target device
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
).fit()
```

## Key parameters (`TrainingConfig.default()`)

- `num_epochs=100`, `criterion_name="cross_entropy"` (only supported value)
- `optimizer_name="adamw"` (`"adam"` or `"adamw"`), `learning_rate=0.0001`, `weight_decay=0.001`
- `batch_size=32`, `num_workers=1`, `drop_last=False`
- `random_seed=42`, `load_best_model=True`

Pre-built variants: `config_plan/training/` (`default_training_config`,
`normal_batch_training_config`, `small_batch_training_config`).

## Gotchas

- **No early stopping / no scheduler / no gradient clipping** — the checkpoint
  only selects the best state; the loop runs all `num_epochs`.
- The device is always derived from the model (`get_model_device`), never from a
  config.
- Validation runs under `torch.inference_mode()`; train under `torch.enable_grad()`.
- An empty train or validation loader raises `ValueError`.

## Tests

`tests/training/test_checkpoint.py` covers the checkpoint's strictly-lower-loss
behavior.