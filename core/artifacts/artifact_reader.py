import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from core.artifacts.processes.configuration_persistence import load_configuration
from core.data_pipeline import StandardDataPipeline
from core.model import FullModel



_FOLD_FILENAME_PATTERN = re.compile(r"fold_(\d+)\.pth\Z")
_REQUIRED_CONFIGURATION_KEYS = {
    "experiment_id",
    "data_pipeline",
    "model",
    "training",
    "evaluation",
}



class ExperimentArtifactReader:

    def __init__(self,directory:Path) -> None:
        self._directory = Path(directory)
        self._configuration:dict[str, Any]|None = None
        self._fold_paths:dict[int, Path] = {}
        self._metadata:dict[str, Any] = {}
        self._index_artifact()
        return

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def experiment_id(self) -> str:
        return self._metadata["experiment_id"]

    @property
    def elapsed_seconds(self) -> float:
        return self._metadata["elapsed_seconds"]

    @property
    def fold_indices(self) -> tuple[int, ...]:
        return tuple(self._fold_paths)

    def load_configuration(self) -> dict[str, Any]:
        if self._configuration is None:
            configuration = load_configuration(self._directory)
            missing_keys = _REQUIRED_CONFIGURATION_KEYS - configuration.keys()
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(
                    f"mpkg configuration is missing required keys: {missing}",
                )
            self._configuration = configuration
        return self._configuration

    def load_fold_state_dict(self,fold_index:int) -> dict[str, torch.Tensor]:
        fold_path = self._get_fold_path(fold_index)
        try:
            state_dict = torch.load(
                fold_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise ValueError(
                f"could not load state dict for fold {fold_index}: {fold_path}",
            ) from error

        if not isinstance(state_dict, Mapping):
            raise ValueError(
                f"mpkg weights for fold {fold_index} must contain a state dict",
            )
        if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
            raise ValueError(
                f"mpkg weights for fold {fold_index} must contain tensors",
            )
        return dict(state_dict)

    def create_data_pipeline(self) -> StandardDataPipeline:
        configuration = self.load_configuration()
        pipeline_configuration = configuration.get("data_pipeline")
        if not isinstance(pipeline_configuration, dict):
            raise ValueError("mpkg data_pipeline configuration must be a dictionary")
        if pipeline_configuration.get("type") != "StandardDataPipeline":
            raise ValueError(
                "mpkg data_pipeline configuration must describe "
                "StandardDataPipeline",
            )

        required_components = (
            "source_reader",
            "partitioner",
            "example_constructor",
        )
        missing_components = [
            name for name in required_components if name not in pipeline_configuration
        ]
        if missing_components:
            missing = ", ".join(missing_components)
            raise ValueError(
                f"mpkg data_pipeline configuration is missing: {missing}",
            )

        return StandardDataPipeline(
            source_reader=pipeline_configuration["source_reader"],
            partitioner=pipeline_configuration["partitioner"],
            example_constructor=pipeline_configuration["example_constructor"],
            oversampler=pipeline_configuration.get("oversampler"),
        )

    def create_fold_model(self,fold_index:int) -> FullModel:
        configuration = self.load_configuration()
        if "model" not in configuration:
            raise ValueError("mpkg configuration is missing the model configuration")
        return FullModel.create_from_state_dict(
            configuration["model"],
            self.load_fold_state_dict(fold_index),
        )

    def load_fold_metrics(self) -> pd.DataFrame:
        metrics = self._load_metrics_sheet("Fold Metrics")
        if "fold" not in metrics.columns:
            raise ValueError("mpkg Fold Metrics sheet must contain a 'fold' column")
        return metrics

    def load_metrics_summary(self) -> pd.DataFrame:
        summary = self._load_metrics_sheet("Summary")
        required_columns = {"metric", "mean", "standard_deviation"}
        missing_columns = required_columns - set(summary.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"mpkg Summary sheet is missing required columns: {missing}",
            )
        return summary

    def validate(self) -> None:
        self._index_artifact()
        self.load_configuration()
        fold_metrics = self.load_fold_metrics()
        fold_values = fold_metrics["fold"].tolist()
        if fold_values != list(self.fold_indices):
            raise ValueError(
                "mpkg Fold Metrics rows must contain each fold exactly once in "
                f"one-based order; found {fold_values}",
            )
        self.load_metrics_summary()
        return

    def _index_artifact(self) -> None:
        if not self._directory.is_dir():
            raise FileNotFoundError(
                f"mpkg directory does not exist: {self._directory}",
            )
        self._require_file("__mpkg__.py", "finalized mpkg marker")
        self._require_file("run.json", "run metadata")
        self._require_file("config.pkl", "mpkg configuration")

        self._metadata = self._load_metadata()
        self._index_fold_paths()
        return

    def _index_fold_paths(self) -> None:
        weights_directory = self._directory / "weights"
        if not weights_directory.is_dir():
            raise FileNotFoundError(
                f"mpkg weights directory does not exist: {weights_directory}",
            )

        fold_paths:dict[int, Path] = {}
        malformed_names:list[str] = []
        for path in weights_directory.iterdir():
            match = _FOLD_FILENAME_PATTERN.fullmatch(path.name)
            if not path.is_file() or match is None:
                malformed_names.append(path.name)
                continue
            fold_index = int(match.group(1))
            if path.name != f"fold_{fold_index}.pth" or fold_index in fold_paths:
                malformed_names.append(path.name)
                continue
            fold_paths[fold_index] = path

        if malformed_names:
            names = ", ".join(sorted(malformed_names))
            raise ValueError(f"malformed mpkg fold filename(s): {names}")
        if not fold_paths:
            raise ValueError("mpkg must contain at least one fold weight file")

        indices = sorted(fold_paths)
        expected_indices = list(range(1, len(indices) + 1))
        if indices != expected_indices:
            raise ValueError(
                "mpkg fold weight filenames must use contiguous one-based indices; "
                f"found {indices}",
            )
        self._fold_paths = {index: fold_paths[index] for index in indices}
        return

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self._directory / "run.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid mpkg run metadata: {metadata_path}") from error

        if not isinstance(metadata, dict):
            raise ValueError("mpkg run metadata must be a JSON object")
        experiment_id = metadata.get("experiment_id")
        elapsed_seconds = metadata.get("elapsed_seconds")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ValueError("mpkg run metadata must contain a non-empty experiment_id")
        if isinstance(elapsed_seconds, bool) or not isinstance(
            elapsed_seconds,
            (int, float),
        ):
            raise ValueError(
                "mpkg run metadata must contain numeric elapsed_seconds",
            )
        return {
            "experiment_id": experiment_id,
            "elapsed_seconds": float(elapsed_seconds),
        }

    def _load_metrics_sheet(self,sheet_name:str) -> pd.DataFrame:
        metrics_path = self._require_file("metrics.xlsx", "metrics workbook")
        try:
            return pd.read_excel(metrics_path, sheet_name=sheet_name)
        except (OSError, ValueError, ImportError) as error:
            raise ValueError(
                f"mpkg metrics workbook is missing or invalid sheet: {sheet_name}",
            ) from error

    def _get_fold_path(self,fold_index:int) -> Path:
        if not isinstance(fold_index, int) or isinstance(fold_index, bool):
            raise TypeError("fold_index must be a one-based integer")
        try:
            return self._fold_paths[fold_index]
        except KeyError as error:
            raise IndexError(f"mpkg fold index is out of range: {fold_index}") from error

    def _require_file(self,name:str,description:str) -> Path:
        path = self._directory / name
        if not path.is_file():
            raise FileNotFoundError(f"mpkg {description} does not exist: {path}")
        return path
