# core/gallery — Explainability and Model Outputs

Produces per-example explanations for a trained model: the predicted class and
confidence, plus 1D attribution heatmaps over the input frames from two
complementary methods — GradCAM (convolutional models) and LeGrad (PatchTST).

## Position in the framework

```
core/persistence/figures (model output PDFs)
  └─ create_model_output(model, example, include_grad_cam, include_legrad)  → ModelOutput
       ├─ GradCam.create(model, values, target_class)   → np.ndarray   (last Conv1d layer)
       └─ LeGrad.create(model, values, target_class)    → np.ndarray   (PatchTST attention maps)
```

## Key classes

| Class | Role |
|---|---|
| `GradCam` | Gradient-weighted activations for Conv1d architectures (LeNet1D, ResNet) |
| `LeGrad` | Layer-gradient attribution over PatchTST attention maps |
| `ModelOutput` | Dataclass: `prediction`, `confidence`, `grad_cam`, `legrad` |
| `create_model_output` | Factory that runs prediction + both explanation methods for one example |

## Core concepts

Both methods share a duck-typed interface:

- `supports(model) -> bool` — whether the model can be explained by this method.
- `create(model, values, target_class) -> np.ndarray` — a normalized `[0, 1]`
  heatmap aligned to the input's frame count.

- **GradCAM** — hooks the **last `nn.Conv1d`** in the model, runs the target-class
  score backward, global-average-pools the gradient over the time axis, weights the
  activations, applies ReLU, and linearly upsamples back to the input frame count.
  Works on LeNet1D and ResNet; `MLP` and `PatchTST` are unsupported (raises if used).
- **LeGrad** — runs the target-class score backward through PatchTST, reads the
  per-layer `attention_maps` captured by `_ScaledDotProductAttention` modules,
  averages the ReLU'd gradients over heads/patches and layers, and upsamples the
  resulting token scores to the frame count. Only works when the architecture is a
  `PatchTST`.
- **`create_model_output`** — for a single `Example`, runs the model in eval mode
  (restoring training mode in a `finally`), computes softmax probabilities →
  `prediction`/`confidence`, then generates whichever attributions are enabled and
  supported.

## Usage

Used automatically by persistence when rendering model-output PDFs. Standalone:

```python
from core.gallery import create_model_output

output = create_model_output(model, example, include_grad_cam=True, include_legrad=True)
# output.prediction, output.confidence, output.grad_cam, output.legrad
```

## Key parameters

- Toggled via `PersistenceConfig.include_grad_cam` / `include_legrad`
  (`config_plan/persistence/*` per feature type). Rendering falls back gracefully
  when the model is unsupported.

## Gotchas

- Attribution methods return `None` when disabled or unsupported for the model.
- LeGrad relies on PatchTST's stored `attention_maps`; if none are captured (e.g.
  attention not exercised), it raises.
- Both methods normalize by max value, so outputs are relative saliency maps, not
  absolute scores.

## Tests

No dedicated tests exist for `core/gallery/`.
