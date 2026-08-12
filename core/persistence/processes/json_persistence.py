import json
from pathlib import Path

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
        },
    )
    _write_json(
        json_directory / "metrics" / f"metrics-fold_{fold_index}.json",
        validation_metrics,
    )
    return


def _write_json(path:Path, data:dict[str, float|list[float]]) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)
        json_file.write("\n")
    return
