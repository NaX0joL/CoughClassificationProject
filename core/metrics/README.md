# core/metrics — Model Evaluation

Runs a trained model over a dataset and computes a configurable set of
classification metrics, returning labels, predictions, and class names alongside
the metric values.

## Position in the framework

```
core/experiment.py
  └─ train_model(): for each fold
       └─ ModelEvaluator.evaluate(model, validation_dataset)  → ModelEvaluation
            ├─ model.predict_probabilities(values)  (softmax)
            ├─ predictions = probabilities.argmax(dim=1)
            └─ ClassificationMetricsCalculator.calculate(labels, predictions, probabilities)
                 → ClassificationMetrics (name → value)
       └─ persisted via ExperimentPersistence.save_fold(...)
```

## Key classes

| Class | Role |
|---|---|
| `ModelEvaluator` | Public entry point; `evaluate(model, dataset)` → `ModelEvaluation` |
| `ModelEvaluation` | Frozen result: `metrics`, `labels`, `predictions`, `class_names` |
| `ClassificationMetric` | ABC: a `name` attribute + `calculate(ClassificationMetricInput) -> float` |
| `ClassificationMetricInput` | Frozen bundle of `labels`, `predictions`, `probabilities`, `class_labels` |
| `ClassificationMetricsCalculator` | Runs the configured metrics and validates inputs |
| `ClassificationMetrics` | Dict-like metric results; attribute access + `to_dict()` |
| `MetricsConfig` | Ordered tuple of metric instances (order = compute & output order) |

## Core concepts

- **Metrics only need a model + dataset** — `evaluate` collects softmax
  probabilities and argmax predictions in eval/inference mode over a non-shuffled
  DataLoader, then feeds them to the calculator.
- **Class names** — numeric labels resolve to human-readable names: 0 →
  `"non-infectious"`, 1 → `"infectious"`, otherwise the dataset's `original_label`.
- **Selection by construction** — there is no string-based selection; a
  `MetricsConfig` is built by instantiating the metric objects you want.
  `MetricsConfig.default()` returns all 11.

### The 11 metrics

| Metric | Name | Notes |
|---|---|---|
| `RocAucMetric` | `roc_auc` | needs probabilities (scores), not just labels |
| `PRAucMetric` | `pr_auc` | needs probabilities |
| `PrecisionMetric` | `precision` | weighted average |
| `RecallMetric` | `recall` | weighted average |
| `SpecificityMetric` | `specificity` | macro-style mean over classes |
| `F1ScoreMetric` | `f1_score` | weighted average |
| `AccuracyMetric` | `accuracy` | simple accuracy |
| `MacroAccuracyMetric` | `macro_accuracy` | sklearn `balanced_accuracy` (mean per-class recall) |
| `MacroF1ScoreMetric` | `macro_f1_score` | unweighted mean over classes |
| `MacroPrecisionMetric` | `macro_precision` | unweighted mean over classes |
| `MacroRecallMetric` | `macro_recall` | unweighted mean over classes |

Non-macro precision/recall/F1 use **weighted** averaging (class-size weighted);
macro variants use unweighted class means. `roc_auc`/`pr_auc` read
`probabilities` (column 1 for binary), so they cannot be computed from labels alone.

## Usage

```python
from core.metrics import ModelEvaluator, MetricsConfig

evaluation = ModelEvaluator(MetricsConfig.default()).evaluate(model, dataset)
metrics_dict = evaluation.metrics.to_dict()   # {"roc_auc": 0.85, ...}
```

## Key parameters

- `MetricsConfig.default()` → all 11 metrics in a fixed order.
- `ModelEvaluator.evaluate(model, dataset, batch_size=32)`.

## Gotchas

- `ClassificationMetrics` attribute access on a metric that wasn't selected raises
  `AttributeError("metric was not calculated: <name>")`.
- The calculator infers `class_labels` from the distinct labels when not passed;
  a mismatch between label classes and probability columns raises `ValueError`.
- Duplicate metric names in a config raise `ValueError`.

## Tests

`tests/metrics/` covers the metric implementations and the calculator.