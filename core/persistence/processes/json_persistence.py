import json
from pathlib import Path

import numpy as np

from ...training import LossLog


def save_fold_json(
    json_directory:Path,
    fold_index:int,
    loss_log:LossLog,
    validation_metrics:dict[str, float],
) -> None:
    _write_json(
        json_directory / "loss" / f"loss-fold_{fold_index}.json",
        {
            "training": loss_log.training_losses,
            "validation": loss_log.validation_losses,
            "best_validation_loss": loss_log.best_validation_loss,
            "best_epoch": loss_log.best_epoch,
        },
    )
    _write_json(
        json_directory / "metrics" / f"metrics-fold_{fold_index}.json",
        validation_metrics,
    )
    return


def load_fold_json(
    json_directory:Path,
    fold_index:int,
) -> tuple[LossLog, dict[str, float]]:
    loss_data = _read_json(
        json_directory / "loss" / f"loss-fold_{fold_index}.json",
    )
    metrics_data = _read_json(
        json_directory / "metrics" / f"metrics-fold_{fold_index}.json",
    )
    loss_log = LossLog(
        training_losses=[float(loss) for loss in loss_data["training"]],
        validation_losses=[float(loss) for loss in loss_data["validation"]],
        best_validation_loss=_get_optional_float(
            loss_data.get("best_validation_loss"),
        ),
        best_epoch=_get_optional_int(loss_data.get("best_epoch")),
    )
    validation_metrics = {
        name: float(value)
        for name, value in metrics_data.items()
    }
    return loss_log, validation_metrics


def save_cross_validation_summary_json(
    json_directory:Path,
    folds_metrics:list[dict[str, float]],
) -> None:
    if not folds_metrics:
        raise ValueError("cross-validation summary requires at least one fold")

    metric_names = folds_metrics[0].keys()
    if any(metrics.keys() != metric_names for metrics in folds_metrics[1:]):
        raise ValueError("each fold must contain the same metrics")

    summary = {
        metric_name: {
            "mean": float(np.mean([
                metrics[metric_name]
                for metrics in folds_metrics
            ])),
            "standard_deviation": float(np.std([
                metrics[metric_name]
                for metrics in folds_metrics
            ])),
        }
        for metric_name in metric_names
    }
    _write_json(
        json_directory / "metrics" / "cross_validation_summary.json",
        {
            "number_of_folds": len(folds_metrics),
            "metrics": summary,
        },
    )
    return


def load_cross_validation_summary_json(
    json_directory:Path,
) -> dict[str, object]:
    return _read_json(
        json_directory / "metrics" / "cross_validation_summary.json",
    )


def _write_json(path:Path, data:dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)
        json_file.write("\n")
    return


def _read_json(path:Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"mpkg JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    if not isinstance(data, dict):
        raise ValueError(f"mpkg JSON data must be a dictionary: {path}")

    return data


def _get_optional_float(value:object) -> float|None:
    if value is None:
        return None
    return float(value)


def _get_optional_int(value:object) -> int|None:
    if value is None:
        return None
    return int(value)
