from datetime import datetime
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
from torch import Tensor

from ..model import FullModel
from ..data_pipeline.dataset import ExampleDataset
from ..training import LossLog
from .persistence_config import PersistenceConfig
from .processes.configuration_persistence import load_configuration, save_configuration
from .processes.figures_persistence import save_fold_figures
from .processes.json_persistence import (
    load_cross_validation_summary_json,
    load_fold_json,
    save_cross_validation_summary_json,
    save_fold_json,
)
from .processes.weights_persistence import load_fold_weights, save_fold_weights


DEFAULT_OUTPUT_DIRECTORY = Path("outputs/mpkg/tmp")


@dataclass
class PersistedFoldArtifacts:
    fold_index:int
    state_dict:dict[str, Tensor]
    loss_log:LossLog
    validation_metrics:dict[str, float]


@dataclass
class PersistedExperimentArtifacts:
    run_directory:Path
    config:dict[str, Any]
    folds:list[PersistedFoldArtifacts]
    cross_validation_summary:dict[str, object]



class ExperimentPersistence:
    """Coordinate persistence for the artifacts of one experiment-training call."""

    def __init__(
        self,
        run_directory:Path,
        config:PersistenceConfig,
    ) -> None:
        self.run_directory = run_directory
        self.config = config
        self.figures_directory = run_directory / "figures"
        self.json_directory = run_directory / "json"
        self.weights_directory = run_directory / "weights"
        return

    @classmethod
    def create(
        cls,
        config:dict[str, Any],
        persistence_config:PersistenceConfig,
        output_directory:Path=DEFAULT_OUTPUT_DIRECTORY,
        experiment_id:str="",
    ) -> "ExperimentPersistence":
        run_directory = _create_run_directory(output_directory, experiment_id)
        persistence = cls(run_directory, persistence_config)
        persistence._create_directory_layout()
        save_configuration(run_directory, config)
        return persistence

    @classmethod
    def load(cls, run_directory:Path) -> PersistedExperimentArtifacts:
        run_directory = Path(run_directory)
        if not (run_directory / "__mpkg__.py").is_file():
            raise ValueError(f"mpkg run directory is invalid: {run_directory}")

        config = load_configuration(run_directory)
        folds = _load_fold_artifacts(run_directory)
        cross_validation_summary = load_cross_validation_summary_json(
            run_directory / "json",
        )
        return PersistedExperimentArtifacts(
            run_directory=run_directory,
            config=config,
            folds=folds,
            cross_validation_summary=cross_validation_summary,
        )

    def save_fold(
        self,
        fold_index:int,
        model:FullModel,
        loss_log:LossLog,
        validation_metrics:dict[str, float],
        labels:np.ndarray,
        predictions:np.ndarray,
        class_names:dict[int, str],
        train_dataset:ExampleDataset,
        validation_dataset:ExampleDataset,
    ) -> None:
        save_fold_figures(
            self.figures_directory,
            fold_index,
            loss_log,
            labels,
            predictions,
            class_names,
            model,
            train_dataset,
            validation_dataset,
            self.config,
        )
        save_fold_json(
            self.json_directory,
            fold_index,
            loss_log,
            validation_metrics,
        )
        save_fold_weights(self.weights_directory, fold_index, model)
        return

    def save_cross_validation_summary(
        self,
        folds_metrics:list[dict[str, float]],
    ) -> None:
        save_cross_validation_summary_json(
            self.json_directory,
            folds_metrics,
        )
        return

    def _create_directory_layout(self) -> None:
        self.figures_directory.mkdir(parents=True)
        (self.figures_directory / "loss").mkdir()
        (self.figures_directory / "confusion_matrix").mkdir()
        (self.figures_directory / "output_train").mkdir()
        (self.figures_directory / "output_validation").mkdir()
        self.json_directory.mkdir()
        (self.json_directory / "loss").mkdir()
        (self.json_directory / "metrics").mkdir()
        self.weights_directory.mkdir()
        (self.run_directory / "__mpkg__.py").touch()
        return


def _create_run_directory(output_directory:Path, experiment_id:str="") -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)

    if experiment_id:
        _validate_experiment_id(experiment_id)
        return _create_unique_directory(output_directory, experiment_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return _create_unique_directory(output_directory, f"run_{timestamp}")


def _create_unique_directory(parent_directory:Path, name:str) -> Path:
    """Create ``name``, adding ``_2``, ``_3``, etc. when it already exists."""
    suffix = 1
    while True:
        directory_name = name if suffix == 1 else f"{name}_{suffix}"
        run_directory = parent_directory / directory_name
        try:
            run_directory.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return run_directory


def _validate_experiment_id(experiment_id:str) -> None:
    if not experiment_id.strip():
        raise ValueError("experiment_id must not be blank")
    if Path(experiment_id).name != experiment_id:
        raise ValueError("experiment_id must be a folder name, not a path")
    return


def _load_fold_artifacts(run_directory:Path) -> list[PersistedFoldArtifacts]:
    weights_directory = run_directory / "weights"
    fold_indices = sorted(
        int(path.stem.removeprefix("fold_"))
        for path in weights_directory.glob("fold_*.pth")
    )
    if not fold_indices:
        raise FileNotFoundError(f"mpkg weights do not exist: {weights_directory}")

    folds = []
    for fold_index in fold_indices:
        loss_log, validation_metrics = load_fold_json(
            run_directory / "json",
            fold_index,
        )
        folds.append(PersistedFoldArtifacts(
            fold_index=fold_index,
            state_dict=load_fold_weights(weights_directory, fold_index),
            loss_log=loss_log,
            validation_metrics=validation_metrics,
        ))

    return folds
