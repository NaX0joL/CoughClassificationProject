# core/model — Architectures and Behaviors

Defines the neural-network models. A model is a composition of an **architecture**
(feature extractor → logits) and a **behavior** (task semantics: loss and
predictions), wrapped in a single `nn.Module`.

## Position in the framework

```
config_plan/model/*.py            ModelConfig(architecture=..., behavior=ClassificationBehavior())
core/experiment.py                FullModel.create(model_config).to(device)     # per fold
core/training/train_logic.py      model.training_step(batch, criterion)          # train phase
                                  model.validation_step(batch, criterion)        # val phase
core/metrics/evaluation.py        model.predict_probabilities(values) → softmax → argmax
core/gallery/                     GradCam (Conv1d), LeGrad (PatchTST attention)
core/persistence                  save/load fold state_dicts
```

## Key classes

| Class | Role |
|---|---|
| `FullModel` | The single `nn.Module` used everywhere; composes `architecture` + `behavior` |
| `ModelArchitecture`, `ModelBehavior`, `StepResult` | ABCs / result type that define the plug-in contract |
| `ClassificationBehavior` | Multi-class semantics: softmax, argmax, external cross-entropy criterion |
| `MLP`, `LeNet`, `ResNet`, `PatchTST` | Concrete architectures (+ their config dataclasses) |
| `ModelConfig` | Bundles an instantiated `architecture` + `behavior` |

## Core concepts

- **`FullModel`** wraps two submodules. `forward` delegates to the architecture.
  `training_step` / `validation_step` extract `batch["value"]` / `batch["label"]`,
  run the architecture, and delegate to the behavior. `predict_probabilities` /
  `predict_classes` return softmax / argmax.
- **`FullModel.create(config)`** **deepcopies** both submodules, so every fold gets
  independent weights from the shared config. `create_from_state_dict` rebuilds a
  model from saved weights (used when loading mpkg packages).
- **`StepResult`** (`loss`, `logits`, `predictions`, `labels`) is the uniform
  return type of every train/val step.
- **`ClassificationBehavior`** is stateless — the loss function is injected
  externally by the `Trainer`, not owned by the behavior.

### Input format

The data pipeline produces 2D per-sample feature matrices `[820, features]`
(820 padded frames × 40 feature bins). Architectures handle this differently:

| Architecture | View of input | First layer |
|---|---|---|
| `MLP` | flattened "bag of frames" (no temporal structure) | `Flatten` → `LazyLinear` |
| `LeNet` | `[seq_len, features]`; feature bins = Conv1d channels | `LazyConv1d` + pooling |
| `ResNet` | `[seq_len, features]`; feature bins = Conv1d channels | stem `LazyConv1d` + residual stages |
| `PatchTST` | `[seq_len, features]`; frames = token sequence | patching → encoder |

### PatchTST specifics

PatchTST is a time-series transformer adapted for classification:

- **Patching** — windows of `patch_len` timesteps with `stride` become tokens
  (each embedded by one `Linear(patch_len, d_model)`), cutting attention cost.
- **Channel independence** — the encoder (`TSTiEncoder`) embeds each feature
  channel independently; batch and variable dims are merged through the encoder.
- **Residual attention** — attention scores from layer *N-1* feed layer *N*
  (`res_attention=True`).
- **Optional extras** — RevIN instance normalization, BatchNorm-in-place-of-LayerNorm,
  series decomposition (trend + residual backbones), multiple head types.
- **Classification trick** — `pred_len` is repurposed as the number of classes, and
  the wrapper collapses the per-variable dimension with `mean(dim=1)` to produce
  logits. `enc_in_feature` must equal the data's feature count and `seq_len` the
  padded length (820).

## Usage

Pre-built model configs live in `config_plan/model/` (`mlp_config`,
`lenet_config`, `resnet_config`, `patchtst_config`, `transformer_config`). The
"transformer" experiment is PatchTST with `patch_len=1, stride=1` (one token per
timestep). Each config pairs an architecture with `ClassificationBehavior()`.

## Key parameters

- `ModelConfig.default()` → MLP with `linear_dims=[256, 256, 256]`, `output_dim=2`.
- Per-architecture configs (`MLPConfig`, `LeNetConfig`, `ResNetConfig`,
  `PatchTSTConfig`) hold architecture-specific fields; PatchTST also accepts flat
  kwargs (used by `config_plan`).

## Gotchas

- **`Classification_Head` is a broken stub** — selecting `head_type="classification"`
  constructs a non-functional object; use the default `"flatten"` head.
- **`Bottleneck_Head` is implemented but unreachable** — no `head_type ==
  "bottleneck"` branch exists in the backbone.
- **`legacy_model.py`** is an older PatchTST wrapper (references an undefined
  `ArchitectureConfig` type) and is not used by any config.
- **`architectures/Transformer/` is empty** — the transformer experiment reuses
  PatchTST.
- The `PatchTSTConfig` dataclass is mainly a defaults source for the legacy model;
  modern configs pass flat kwargs to `PatchTST(...)`.

## Tests

No tests exist for `core/model/`.