
from dataclasses import dataclass

from .data_pipeline import DataPipelineConfig
from .metrics import MetricsConfig
from .model import ModelConfig
from .persistence import PersistenceConfig
from .training import TrainingConfig



@dataclass
class ExperimentConfig:
    data_pipeline_config:DataPipelineConfig
    model_config:ModelConfig
    training_config:TrainingConfig
    metrics_config:MetricsConfig
    persistence_config:PersistenceConfig

    @classmethod
    def from_persisted_config(
        cls,
        persisted_config:dict[str, object],
    ) -> "ExperimentConfig":
        required_config_keys = {
            "data_pipeline",
            "model",
            "training",
            "metrics",
            "persistence",
        }
        if persisted_config.keys() != required_config_keys:
            raise ValueError("mpkg configuration has invalid keys")

        experiment_config = cls(
            data_pipeline_config=persisted_config["data_pipeline"],
            model_config=persisted_config["model"],
            training_config=persisted_config["training"],
            metrics_config=persisted_config["metrics"],
            persistence_config=persisted_config["persistence"],
        )
        return experiment_config

    @classmethod
    def default(cls) -> "ExperimentConfig":
        experiment_config = cls(
            data_pipeline_config=DataPipelineConfig.default(),
            model_config=ModelConfig.default(),
            training_config=TrainingConfig.default(),
            metrics_config=MetricsConfig.default(),
            persistence_config=PersistenceConfig.default(),
        )
        return experiment_config
