import numpy as np

from .data_pipeline import DataPipeline, DataPipelineConfig, DevelopmentFold
from .model import FullModel, ModelConfig
from .training import LossLog, Trainer, TrainingConfig
from .metrics import evaluate_model, calculate_classification_metrics



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
        return
    
    def train_model(self) -> None:
        data_pipeline = DataPipeline.create(self.data_pipeline_config)
        data_split = data_pipeline.get_data_split()

        loss_logs = []
        folds_metrics = []
        
        for development_fold in data_split.development_folds:

            model = FullModel.create(self.model_config)
            loss_log = self._train_development_fold(model, development_fold)
            
            fold_metrics = evaluate_model(
                model, 
                development_fold.validation_dataset, 
                calculate_classification_metrics
            ).to_dict()
            
            loss_logs.append(loss_log)
            folds_metrics.append(fold_metrics)
        
        # test_metrics = evaluate_model(
        #     model,
        #     data_split.test_dataset,
        #     calculate_classification_metrics,
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
