from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from modules.randomness import set_random_seed
from modules.resolve_pytorch_device import get_optimal_device

from core.data_pipeline.abstract import AbstractDataPipeline
from core.data_pipeline.loading.data_module import DataModule
from core.artifacts.artifact_writer import ExperimentArtifactWriter
from core.metrics import MetricsConfig, ModelEvaluator
from core.metrics.label_mapping import BinaryInfectionLabelMapping
from core.model import FullModel, ModelConfig
from core.trainer import Trainer, TrainingConfig



@dataclass(frozen=True)
class ExperimentOrchestratorConfig:
    experiment_id:str
    model_config:ModelConfig
    training_config:TrainingConfig
    metrics_config:MetricsConfig=field(
        default_factory=MetricsConfig.default
    )
    label_mapping:BinaryInfectionLabelMapping=field(
        default_factory=BinaryInfectionLabelMapping,
    )



class ExperimentOrchestrator:

    def __init__(
        self,
        config:ExperimentOrchestratorConfig,
        data_pipeline:AbstractDataPipeline,
        output_directory:Path|None=None,
    ) -> None:
        self.config = config
        self.data_pipeline = data_pipeline
        self.output_directory = output_directory
        self.artifact_writer:ExperimentArtifactWriter|None = None
        return

    def run(self) -> None:
        start_time = perf_counter()
        self.artifact_writer = ExperimentArtifactWriter(
            config=self.config,
            data_pipeline=self.data_pipeline,
            output_directory=self.output_directory,
        )
        
        self._set_experiment_seed()
        #print("preparing data pipeline...")
        self._initialize_data_pipeline()
        print("begin training")
        evaluator = ModelEvaluator(
            self.config.metrics_config,
            self.config.label_mapping,
        )
        device = get_optimal_device()
        print(f"using device {device}")
        
        for fold_index in range(len(self.data_pipeline)):
            print(f"fold-{fold_index + 1}")
            self._run_fold(fold_index, evaluator)
        
        save_directory = self.artifact_writer.get_save_directory()
        print(f"mpkg saved to {save_directory}")        
        
        self.artifact_writer.finalize(perf_counter() - start_time)
        print("training finished")
        return

    def _set_experiment_seed(self) -> None:
        random_seed = getattr(
            self.config.training_config,
            "random_seed",
            None,
        )
        if random_seed is not None:
            set_random_seed(random_seed)
        return

    def _initialize_data_pipeline(self) -> None:
        self.data_pipeline.initialize()
        return

    def _get_fold_data_module(self, fold_index:int) -> DataModule:
        training_config = self.config.training_config
        random_seed = getattr(training_config, "random_seed", None)
        data_module = self.data_pipeline.get_data_module(
            index=fold_index,
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            drop_last_batch=training_config.drop_last,
            pin_memory=get_optimal_device().type == "cuda",
            random_seed=random_seed,
        )
        return data_module

    def _run_fold(
        self,
        fold_index:int,
        evaluator:ModelEvaluator|None=None,
    ) -> None:
        data_module = self._get_fold_data_module(fold_index)
        self._validate_fold_labels(data_module)
        
        model = FullModel.create(self.config.model_config)
        model = model.to(get_optimal_device())
        
        trainer = Trainer(
            config=self.config.training_config,
            model=model,
            data_module=data_module,
        )
        loss_log = trainer.fit()
        
        if evaluator is None:
            evaluator = ModelEvaluator(
                self.config.metrics_config,
                self.config.label_mapping,
            )
        evaluation = evaluator.evaluate(model, data_module.test_loader)
        
        if self.artifact_writer is None:
            raise RuntimeError("experiment artifact writer is not initialized")
        
        self.artifact_writer.save_fold(
            fold_index=fold_index + 1,
            model=model,
            loss_log=loss_log,
            evaluation=evaluation,
        )
        return

    def _validate_fold_labels(self, data_module:DataModule) -> None:
        architecture = getattr(self.config.model_config, "architecture", None)
        
        output_dim = getattr(architecture, "output_dim", None)
        if not isinstance(output_dim, int):
            return

        for split_name in ("train", "validation"):
            loader = getattr(data_module, f"{split_name}_loader", None)
            dataset = getattr(loader, "dataset", None)
            examples = getattr(dataset, "examples", None)
            if examples is None:
                continue

            invalid_labels = sorted({
                example.label
                for example in examples
                if example.label < 0 or example.label >= output_dim
            })
            if invalid_labels:
                raise ValueError(
                    f"{split_name} dataset contains invalid labels "
                    f"{invalid_labels} for model.output_dim={output_dim}; "
                    f"expected labels in [0, {output_dim})",
                )
                
        return
