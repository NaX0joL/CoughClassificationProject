from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from modules.resolve_pytorch_device import get_optimal_device
from modules.randomness import set_random_seed

from .data_pipeline import DataPipeline, DevelopmentFold
from .experiment_config import ExperimentConfig
from .model import FullModel
from .training import LossLog, Trainer
from .metrics import ModelEvaluator
from .persistence import ExperimentPersistence
from .gallery import ClassDistributionGenerator, ExampleGalleryGenerator



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

        persistence.save_cross_validation_summary(folds_metrics)
        
        print("training finished")
        print(f"total training time: {_format_elapsed_time(total_training_seconds)}")
        print(f"mpkg stored in {persistence.run_directory}")
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
