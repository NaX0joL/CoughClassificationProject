import numpy as np

from .data_pipeline import DataPipeline, DataPipelineConfig, DevelopmentFold
from .model import FullModel, ModelConfig
from .training import LossLog, Trainer, TrainingConfig
from .metrics import MetricsConfig, ModelEvaluator
from .persistence import ExperimentPersistence



class ExperimentOrchestrator:

    def __init__(
        self,
        data_pipeline_config: DataPipelineConfig,
        model_config:ModelConfig,
        training_config:TrainingConfig,
    ) -> None:
        self.data_pipeline_config = data_pipeline_config
        self.model_config = model_config
        self.training_config = training_config
        self.metrics_config = MetricsConfig.default()
        self.model_evaluator = ModelEvaluator(self.metrics_config)
        return
    
    def train_model(self) -> None:
        persistence = ExperimentPersistence.create(
            config={
                "data_pipeline": self.data_pipeline_config,
                "model": self.model_config,
                "training": self.training_config,
                "metrics": self.metrics_config,
            },
        )
        data_pipeline = DataPipeline.create(self.data_pipeline_config)
        data_split = data_pipeline.get_data_split()

        loss_logs = []
        folds_metrics = []
        
        for fold_index, development_fold in enumerate(data_split.development_folds, start=1):

            model = FullModel.create(self.model_config)
            loss_log = self._train_development_fold(model, development_fold)
            
            fold_metrics = self.model_evaluator.evaluate(
                model, 
                development_fold.validation_dataset, 
            ).to_dict()
            
            loss_logs.append(loss_log)
            folds_metrics.append(fold_metrics)
            persistence.save_fold(
                fold_index=fold_index,
                model=model,
                loss_log=loss_log,
                validation_metrics=fold_metrics,
            )
        
        # test_metrics = self.model_evaluator.evaluate(
        #     model,
        #     data_split.test_dataset,
        # ).to_dict()
        
        return

    def _train_development_fold(
        self,
        model:FullModel,
        development_fold:DevelopmentFold,
    ) -> LossLog:
        trainer = Trainer(
            config=self.training_config,
            model=model,
            train_dataset=development_fold.train_dataset,
            validation_dataset=development_fold.validation_dataset,
        )
        loss_log = trainer.fit()
        return loss_log
