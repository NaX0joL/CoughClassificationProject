"""Legacy experiment orchestrator retained for the pre-v3 data pipeline."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from modules.resolve_pytorch_device import get_optimal_device
from modules.randomness import set_random_seed

from .data_pipeline import (
    DataPipeline,
    DevelopmentFold,
)
from .data_pipeline_3.data_module import DataModule as DataModule3
from .data_pipeline_3.pipeline import DataPipeline as DataPipeline3
from .experiment_config import ExperimentConfig
from .model import FullModel
from .training import LossLog, Trainer
from .metrics import ModelEvaluator
from .persistence import ExperimentPersistence
from .gallery import (
    ClassDistributionGenerator,
    ExampleGalleryGenerator,
    GalleryDataConfig,
)



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
        experiment_id:str = "",
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
        
        random_seed = self.config.training_config.random_seed
        if random_seed is not None:
            set_random_seed(random_seed)
        
        persistence = ExperimentPersistence.create(
            config={
                "data_pipeline": self.config.data_pipeline_config,
                "model": self.config.model_config,
                "training": self.config.training_config,
                "metrics": self.config.metrics_config,
                "persistence": self.config.persistence_config,
            },
            persistence_config=self.config.persistence_config,
            experiment_id=self.experiment_id,
        )
        self.run_directory = persistence.run_directory

        data_pipeline = DataPipeline.create(self.config.data_pipeline_config)
        data_split = data_pipeline.get_data_split()

        gallery = ExampleGalleryGenerator(
            data_pipeline_config=self.config.data_pipeline_config,
            random_seed=self.config.training_config.random_seed,
            num_examples=50,
            feature_colormap=self.config.persistence_config.feature_colormap,
            regenerate=False,
        )
        gallery.generate(data_pipeline.get_examples())

        distribution_generator = ClassDistributionGenerator(
            data_pipeline_config=self.config.data_pipeline_config,
        )
        distribution_generator.generate(data_split)

        model_evaluator = ModelEvaluator(self.config.metrics_config)

        loss_logs = []
        folds_metrics = []
        total_training_seconds = 0.0

        for fold_index, development_fold in enumerate(data_split.development_folds, start=1):
            print(f"fold-{fold_index}")
            
            model = FullModel.create(self.config.model_config).to(self.device)
            loss_log, fold_training_seconds = self._time_development_fold_training(
                model,
                development_fold,
            )
            total_training_seconds += fold_training_seconds
            
            print(f" time: {_format_elapsed_time(fold_training_seconds)}")
            
            evaluation = model_evaluator.evaluate(
                model,
                development_fold.validation_dataset,
                batch_size=self.config.training_config.batch_size,
            )
            fold_metrics = evaluation.metrics.to_dict()
            
            loss_logs.append(loss_log)
            folds_metrics.append(fold_metrics)
            persistence.save_fold(
                fold_index=fold_index,
                model=model,
                loss_log=loss_log,
                validation_metrics=fold_metrics,
                labels=evaluation.labels,
                predictions=evaluation.predictions,
                class_names=evaluation.class_names,
                train_dataset=development_fold.train_dataset,
                validation_dataset=development_fold.validation_dataset,
            )
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        persistence.save_cross_validation_summary(folds_metrics)
        
        print("training finished")
        print(f"total training time: {_format_elapsed_time(total_training_seconds)}")
        print(f"mpkg stored in {persistence.run_directory}")
        return

    def train_model_v3(
        self,
        data_pipeline:DataPipeline3,
    ) -> None:
        print(f"begin training {self.experiment_id}")
        print(f"using device {self.device}")

        training_config = self.config.training_config
        if training_config.random_seed is not None:
            set_random_seed(training_config.random_seed)

        persistence = self._create_v3_persistence(data_pipeline)
        model_evaluator = ModelEvaluator(self.config.metrics_config)
        folds_metrics = [
            self._train_model_v3_fold(
                fold_index,
                data_pipeline,
                model_evaluator,
                persistence,
            )
            for fold_index in range(len(data_pipeline))
        ]
        persistence.save_cross_validation_summary(folds_metrics)

        print("training finished")
        print(f"mpkg stored in {persistence.run_directory}")
        return

    def _create_v3_persistence(
        self,
        data_pipeline:DataPipeline3,
    ) -> ExperimentPersistence:
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
                "training": self.config.training_config,
                "metrics": self.config.metrics_config,
                "persistence": self.config.persistence_config,
            },
            persistence_config=self.config.persistence_config,
            experiment_id=self.experiment_id,
        )
        self.run_directory = persistence.run_directory
        return persistence

    def _train_model_v3_fold(
        self,
        fold_index:int,
        data_pipeline:DataPipeline3,
        model_evaluator:ModelEvaluator,
        persistence:ExperimentPersistence,
    ) -> dict[str, float]:
        persisted_fold_index = fold_index + 1
        print(f"fold-{persisted_fold_index}")
        training_config = self.config.training_config
        data_module = data_pipeline.get_data_module(
            index=fold_index,
            num_workers=training_config.num_workers,
            drop_last_batch=training_config.drop_last,
            pin_memory=self.device.type == "cuda",
            random_seed=training_config.random_seed,
        )

        if fold_index == 0:
            self._visualize_v3_data(data_pipeline, data_module)

        model = FullModel.create(self.config.model_config).to(self.device)
        trainer = Trainer(
            config=training_config,
            model=model,
            data_module=data_module,
        )
        loss_log = trainer.fit()
        evaluation = model_evaluator.evaluate_dataloader(
            model,
            data_module.validation_loader,
        )
        fold_metrics = evaluation.metrics.to_dict()
        persistence.save_fold_from_dataloaders(
            fold_index=persisted_fold_index,
            model=model,
            loss_log=loss_log,
            validation_metrics=fold_metrics,
            labels=evaluation.labels,
            predictions=evaluation.predictions,
            class_names=evaluation.class_names,
            train_loader=data_module.train_loader,
            validation_loader=data_module.validation_loader,
        )

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return fold_metrics

    def _visualize_v3_data(
        self,
        data_pipeline:DataPipeline3,
        data_module:DataModule3,
    ) -> None:
        gallery_config = GalleryDataConfig(
            components={
                "source_reader": data_pipeline.source_reader,
                "partitioner": data_pipeline.partitioner,
                "example_constructor": data_pipeline.example_constructor,
                "batch_size": data_pipeline.batch_size,
                "oversampler": data_pipeline.oversampler,
            },
            name=f"{self.experiment_id}_v3",
        )
        gallery = ExampleGalleryGenerator(
            data_pipeline_config=gallery_config,
            random_seed=self.config.training_config.random_seed,
            num_examples=50,
            feature_colormap=self.config.persistence_config.feature_colormap,
            regenerate=False,
        )
        distribution_generator = ClassDistributionGenerator(
            data_pipeline_config=gallery_config,
        )
        gallery.generate_from_dataloader(data_module.train_loader)
        distribution_generator.generate_from_dataloaders(
            train_loader=data_module.train_loader,
            validation_loader=data_module.validation_loader,
            test_loader=data_module.test_loader,
        )
        return

    def _time_development_fold_training(
        self,
        model:FullModel,
        development_fold:DevelopmentFold,
    ) -> tuple[LossLog, float]:
        start_time = perf_counter()
        loss_log = self._train_development_fold(model, development_fold)
        training_seconds = perf_counter() - start_time
        return loss_log, training_seconds

    def _train_development_fold(
        self,
        model:FullModel,
        development_fold:DevelopmentFold,
    ) -> LossLog:
        trainer = Trainer(
            config=self.config.training_config,
            model=model,
            train_dataset=development_fold.train_dataset,
            validation_dataset=development_fold.validation_dataset,
        )
        loss_log = trainer.fit()
        return loss_log



def _format_elapsed_time(elapsed_seconds:float) -> str:
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = elapsed_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
