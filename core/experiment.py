from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from modules.resolve_pytorch_device import get_optimal_device
from modules.randomness import set_random_seed

from .data_pipeline_3.pipeline import DataPipeline
from .experiment_config import ExperimentConfig
from .gallery import (
    ClassDistributionGenerator,
    ExampleGalleryGenerator,
    GalleryDataConfig,
)
from .metrics import ModelEvaluator
from .model import FullModel
from .persistence import ExperimentPersistence
from .training import LossLog, Trainer



GALLERY_CLASS_NAMES = {
    0: "label 0",
    1: "label 1",
    2: "label 2"
}
PERSISTED_CLASS_NAMES = {
    0: "0",
    1: "1",
    2: "2",
}
GALLERY_REGENERATION = False
GALLERY_VERSION = 2



@dataclass
class PersistedFold:
    fold_index:int
    model:FullModel
    loss_log:LossLog
    validation_metrics:dict[str, float]



class ExperimentOrchestrator:

    def __init__(
        self,
        config:ExperimentConfig,
        experiment_id:str="",
    ) -> None:
        self.config = config
        self.experiment_id = experiment_id
        self.run_directory:Path|None = None
        self.persisted_folds:list[PersistedFold] = []
        self.cross_validation_summary:dict[str, object]|None = None
        self.device = get_optimal_device()
        return

    @classmethod
    def load(cls, mpkg_path:Path) -> "ExperimentOrchestrator":
        persisted_experiment = ExperimentPersistence.load(mpkg_path)
        experiment_config = ExperimentConfig.from_persisted_config(
            persisted_experiment.config,
        )
        experiment = cls(
            experiment_config,
            experiment_id=persisted_experiment.run_directory.name,
        )
        experiment.run_directory = persisted_experiment.run_directory
        experiment.persisted_folds = [
            PersistedFold(
                fold_index=fold.fold_index,
                model=FullModel.create_from_state_dict(
                    experiment_config.model_config,
                    fold.state_dict,
                ),
                loss_log=fold.loss_log,
                validation_metrics=fold.validation_metrics,
            )
            for fold in persisted_experiment.folds
        ]
        experiment.cross_validation_summary = (
            persisted_experiment.cross_validation_summary
        )
        return experiment

    def test_model(self) -> None:
        raise NotImplementedError

    def train_model(self) -> None:
        print(f"begin training {self.experiment_id}")
        print(f"using device {self.device}")

        data_pipeline_config:object = self.config.data_pipeline_config
        if not isinstance(data_pipeline_config, DataPipeline):
            raise TypeError("experiment requires a data pipeline version 3")
        data_pipeline = data_pipeline_config

        training_config = self.config.training_config
        if training_config.random_seed is not None:
            set_random_seed(training_config.random_seed)

        persistence = ExperimentPersistence.create(
            config={
                "data_pipeline": {
                    "source_reader": data_pipeline.source_reader,
                    "partitioner": data_pipeline.partitioner,
                    "example_constructor": data_pipeline.example_constructor,
                    "batch_size": data_pipeline.batch_size,
                    "oversampler": data_pipeline.oversampler,
                },
                "model": self.config.model_config,
                "training": training_config,
                "metrics": self.config.metrics_config,
                "persistence": self.config.persistence_config,
            },
            persistence_config=self.config.persistence_config,
            experiment_id=self.experiment_id,
        )
        self.run_directory = persistence.run_directory

        gallery_config = GalleryDataConfig(
            components={
                "source_reader": data_pipeline.source_reader,
                "partitioner": data_pipeline.partitioner,
                "example_constructor": data_pipeline.example_constructor,
                "batch_size": data_pipeline.batch_size,
                "oversampler": data_pipeline.oversampler,
                "gallery_version": GALLERY_VERSION,
            },
        )
        gallery = ExampleGalleryGenerator(
            data_pipeline_config=gallery_config,
            random_seed=training_config.random_seed,
            num_examples=100,
            class_names=GALLERY_CLASS_NAMES,
            feature_colormap=self.config.persistence_config.feature_colormap,
            regenerate=GALLERY_REGENERATION,
        )
        distribution_generator = ClassDistributionGenerator(
            data_pipeline_config=gallery_config,
            class_names=GALLERY_CLASS_NAMES,
            regenerate=GALLERY_REGENERATION,
        )
        model_evaluator = ModelEvaluator(self.config.metrics_config)
        folds_metrics = []
        total_training_seconds = 0.0

        for fold_index in range(len(data_pipeline)):
            persisted_fold_index = fold_index + 1
            print(f"fold-{persisted_fold_index}")

            data_module = data_pipeline.get_data_module(
                index=fold_index,
                num_workers=training_config.num_workers,
                drop_last_batch=training_config.drop_last,
                pin_memory=self.device.type == "cuda",
                random_seed=training_config.random_seed,
            )

            gallery.generate_fold_from_dataloaders(
                fold_index=persisted_fold_index,
                train_loader=data_module.train_loader,
                validation_loader=data_module.validation_loader,
                test_loader=data_module.test_loader,
            )

            distribution_generator.collect_from_dataloaders(
                fold_index=persisted_fold_index,
                train_loader=data_module.train_loader,
                validation_loader=data_module.validation_loader,
                test_loader=data_module.test_loader,
            )

            model = FullModel.create(self.config.model_config).to(self.device)
            
            trainer = Trainer(
                config=training_config,
                model=model,
                data_module=data_module,
            )
            loss_log, fold_training_seconds = self._time_trainer_fit(trainer)
            total_training_seconds += fold_training_seconds
            self._print_training_time(
                f"fold-{persisted_fold_index}",
                fold_training_seconds,
            )
            
            train_evaluation = model_evaluator.evaluate_dataloader(
                model=model,
                data_loader=data_module.train_loader,
            )
            validation_evaluation = model_evaluator.evaluate_dataloader(
                model=model,
                data_loader=data_module.validation_loader,
            )
            test_evaluation = model_evaluator.evaluate_dataloader(
                model=model,
                data_loader=data_module.test_loader,
            )
            fold_metrics = test_evaluation.metrics.to_dict()
            folds_metrics.append(fold_metrics)

            persistence.save_fold_from_dataloaders(
                fold_index=persisted_fold_index,
                model=model,
                loss_log=loss_log,
                validation_metrics=fold_metrics,
                labels=test_evaluation.labels,
                predictions=test_evaluation.predictions,
                class_names=PERSISTED_CLASS_NAMES,
                train_loader=data_module.train_loader,
                validation_loader=data_module.validation_loader,
                additional_confusion_matrices={
                    "train": (
                        train_evaluation.labels,
                        train_evaluation.predictions,
                    ),
                    "validation": (
                        validation_evaluation.labels,
                        validation_evaluation.predictions,
                    ),
                },
            )

            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        distribution_generator.generate_collected()
        persistence.save_cross_validation_summary(folds_metrics)

        print("training finished")
        self._print_training_time("total", total_training_seconds)
        print(f"mpkg stored in {persistence.run_directory}")
        return

    def _time_trainer_fit(self, trainer:Trainer) -> tuple[LossLog, float]:
        start_time = perf_counter()
        loss_log = trainer.fit()
        training_seconds = perf_counter() - start_time
        return loss_log, training_seconds

    def _print_training_time(self, name:str, training_seconds:float) -> None:
        formatted_time = _format_elapsed_time(training_seconds)
        print(f"{name} training time: {formatted_time}")
        return



def _format_elapsed_time(elapsed_seconds:float) -> str:
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
