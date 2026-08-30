import argparse
from pathlib import Path
from typing import Any

import yaml

from core.data_pipeline import DataPipelineConfig
from core.data_pipeline.preprocessing import (
    CoughSegmenter,
    DownSampler,
    FeatureWiseNormalization,
    FeatureWiseStandardization,
    LogMelSpectrogram,
    MFCC,
    SlidingWindowSegmenter,
    ZeroPadder,
)
from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.stratifier import DataSplitter
from core.experiment import ExperimentOrchestrator
from core.experiment_config import ExperimentConfig
from core.metrics import (
    AccuracyMetric,
    F1ScoreMetric,
    MacroAccuracyMetric,
    MacroF1ScoreMetric,
    MacroPrecisionMetric,
    MacroRecallMetric,
    MetricsConfig,
    PRAucMetric,
    PrecisionMetric,
    RecallMetric,
    RocAucMetric,
    SpecificityMetric,
)
from core.model import ModelConfig
from core.model.architectures.LeNet1D import LeNet1D
from core.model.architectures.LeNet2D import LeNet2D
from core.model.architectures.MLP import MLP
from core.model.architectures.PatchTST import PatchTST
from core.model.architectures.ResNet import ResNet
from core.model.behavior.classification_behavior import ClassificationBehavior
from core.persistence import PersistenceConfig
from core.training import TrainingConfig



class YamlExperimentError(ValueError):
    pass



class YamlToExperimentConverter:

    _TYPE_CONSTRUCTORS:dict[str, type[Any]] = {
        "Path": Path,
        "ExperimentConfig": ExperimentConfig,
        "DataPipelineConfig": DataPipelineConfig,
        "ElderlyCoughAudioSourceReader": ElderlyCoughAudioSourceReader,
        "CoughSegmenter": CoughSegmenter,
        "SlidingWindowSegmenter": SlidingWindowSegmenter,
        "MFCC": MFCC,
        "LogMelSpectrogram": LogMelSpectrogram,
        "DownSampler": DownSampler,
        "FeatureWiseNormalization": FeatureWiseNormalization,
        "FeatureWiseStandardization": FeatureWiseStandardization,
        "ZeroPadder": ZeroPadder,
        "DataSplitter": DataSplitter,
        "ModelConfig": ModelConfig,
        "MLP": MLP,
        "LeNet1D": LeNet1D,
        "LeNet2D": LeNet2D,
        "PatchTST": PatchTST,
        "ResNet": ResNet,
        "ClassificationBehavior": ClassificationBehavior,
        "TrainingConfig": TrainingConfig,
        "MetricsConfig": MetricsConfig,
        "RocAucMetric": RocAucMetric,
        "PRAucMetric": PRAucMetric,
        "PrecisionMetric": PrecisionMetric,
        "RecallMetric": RecallMetric,
        "SpecificityMetric": SpecificityMetric,
        "F1ScoreMetric": F1ScoreMetric,
        "AccuracyMetric": AccuracyMetric,
        "MacroAccuracyMetric": MacroAccuracyMetric,
        "MacroF1ScoreMetric": MacroF1ScoreMetric,
        "MacroPrecisionMetric": MacroPrecisionMetric,
        "MacroRecallMetric": MacroRecallMetric,
        "PersistenceConfig": PersistenceConfig,
    }

    def convert(self, yaml_path:Path) -> ExperimentOrchestrator:
        yaml_path = self._resolve_yaml_path(yaml_path)
        yaml_config = self._read_yaml_file(yaml_path)
        self._validate_yaml_config(yaml_config, yaml_path)
        experiment_config = self._create_experiment_config(
            yaml_config["config"],
            yaml_path,
        )
        experiment = self._create_experiment(
            experiment_config,
            experiment_id=yaml_config["experiment_id"],
        )
        return experiment

    def _resolve_yaml_path(self, yaml_path:Path) -> Path:
        return yaml_path.resolve()

    def _read_yaml_file(self, yaml_path:Path) -> dict[str, Any]:
        try:
            with yaml_path.open("r", encoding="utf-8") as yaml_file:
                yaml_config = yaml.safe_load(yaml_file)
        except OSError as error:
            raise YamlExperimentError(f"cannot read {yaml_path}: {error}") from error
        except yaml.YAMLError as error:
            raise YamlExperimentError(f"invalid YAML in {yaml_path}: {error}") from error

        if not isinstance(yaml_config, dict):
            raise YamlExperimentError(f"{yaml_path} must contain a mapping")
        return yaml_config

    def _validate_yaml_config(
        self,
        yaml_config:dict[str, Any],
        yaml_path:Path,
    ) -> None:
        required_keys = {"experiment_id", "config"}
        if yaml_config.keys() != required_keys:
            raise YamlExperimentError(
                f"{yaml_path} must contain exactly {sorted(required_keys)}",
            )

        experiment_id = yaml_config["experiment_id"]
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise YamlExperimentError(
                f"{yaml_path} experiment_id must be a non-blank string",
            )
        return

    def _create_experiment_config(
        self,
        yaml_config:Any,
        yaml_path:Path,
    ) -> ExperimentConfig:
        experiment_config = self._create_config_object(
            yaml_config,
            yaml_path,
            field_path="config",
        )
        if not isinstance(experiment_config, ExperimentConfig):
            raise YamlExperimentError(
                f"{yaml_path} config must have type ExperimentConfig",
            )
        return experiment_config

    def _create_config_object(
        self,
        yaml_value:Any,
        yaml_path:Path,
        field_path:str,
    ) -> Any:
        if isinstance(yaml_value, list):
            return self._create_config_list(
                yaml_value,
                yaml_path,
                field_path,
            )

        if isinstance(yaml_value, dict):
            return self._create_config_dictionary(
                yaml_value,
                yaml_path,
                field_path,
            )

        return yaml_value

    def _create_config_list(
        self,
        yaml_values:list[Any],
        yaml_path:Path,
        field_path:str,
    ) -> list[Any]:
        return [
            self._create_config_object(
                yaml_value,
                yaml_path,
                field_path=f"{field_path}[{index}]",
            )
            for index, yaml_value in enumerate(yaml_values)
        ]

    def _create_config_dictionary(
        self,
        yaml_config:dict[str, Any],
        yaml_path:Path,
        field_path:str,
    ) -> Any:
        if "type" in yaml_config:
            return self._create_registered_object(
                yaml_config,
                yaml_path,
                field_path,
            )

        return self._create_plain_dictionary(
            yaml_config,
            yaml_path,
            field_path,
        )

    def _create_plain_dictionary(
        self,
        yaml_config:dict[str, Any],
        yaml_path:Path,
        field_path:str,
    ) -> dict[str, Any]:
        return {
            key: self._create_config_object(
                value,
                yaml_path,
                field_path=f"{field_path}.{key}",
            )
            for key, value in yaml_config.items()
        }

    def _create_registered_object(
        self,
        yaml_config:dict[str, Any],
        yaml_path:Path,
        field_path:str,
    ) -> Any:
        type_name = self._get_type_name(yaml_config, yaml_path, field_path)
        constructor = self._get_constructor(type_name)
        parameters = self._create_constructor_parameters(
            yaml_config,
            yaml_path,
            field_path,
        )
        config_object = self._call_constructor(
            constructor,
            parameters,
            type_name,
            yaml_path,
            field_path,
        )
        return config_object

    def _get_type_name(
        self,
        yaml_config:dict[str, Any],
        yaml_path:Path,
        field_path:str,
    ) -> str:
        type_name = yaml_config["type"]
        if not isinstance(type_name, str) or type_name not in self._TYPE_CONSTRUCTORS:
            raise YamlExperimentError(
                f"{yaml_path} {field_path}.type has unknown type {type_name!r}",
            )
        return type_name

    def _get_constructor(self, type_name:str) -> type[Any]:
        return self._TYPE_CONSTRUCTORS[type_name]

    def _create_constructor_parameters(
        self,
        yaml_config:dict[str, Any],
        yaml_path:Path,
        field_path:str,
    ) -> dict[str, Any]:
        return {
            name: self._create_config_object(
                value,
                yaml_path,
                field_path=f"{field_path}.{name}",
            )
            for name, value in yaml_config.items()
            if name != "type"
        }

    def _call_constructor(
        self,
        constructor:type[Any],
        parameters:dict[str, Any],
        type_name:str,
        yaml_path:Path,
        field_path:str,
    ) -> Any:
        try:
            if constructor is Path:
                return self._create_path(parameters)

            if constructor is MetricsConfig:
                parameters["metrics"] = tuple(parameters["metrics"])

            config_object = constructor(**parameters)

        except Exception as error:
            raise YamlExperimentError(
                f"{yaml_path} cannot create {field_path} as {type_name}: {error}",
            ) from error

        return config_object

    def _create_path(self, parameters:dict[str, Any]) -> Path:
        if parameters.keys() != {"value"}:
            raise TypeError("Path requires exactly one value")

        path = Path(parameters["value"])
        return path

    def _create_experiment(
        self,
        experiment_config:ExperimentConfig,
        experiment_id:str,
    ) -> ExperimentOrchestrator:
        experiment = ExperimentOrchestrator(
            config=experiment_config,
            experiment_id=experiment_id,
        )
        return experiment



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=Path)
    arguments = parser.parse_args()
    return arguments


def do_experiment(experiment:ExperimentOrchestrator) -> None:
    experiment.train_model()
    return


def main() -> None:
    arguments = get_arguments()
    experiment = YamlToExperimentConverter().convert(arguments.yaml_path)
    do_experiment(experiment)
    return



if __name__ == "__main__":
    main()
    print("DONE!")
