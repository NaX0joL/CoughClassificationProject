from dataclasses import asdict
from pathlib import Path

from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline import DataPipeline, DataPipelineConfig
from core.data_pipeline.preprocessing import CoughSegmenter, MFCC, ZeroPadder
from core.data_pipeline.stratifier import DataSplitter

from core.model import FullModel, ModelConfig
from core.model.architectures.MLP import MLP, MLPConfig
from core.model.behavior.classification_behavior import ClassificationBehavior

from core.training import TrainingConfig

from core.metrics import MetricsConfig
from core.persistence import PersistenceConfig

from core.experiment import ExperimentOrchestrator
from core.experiment_config import ExperimentConfig
from config_plan import (
    mfcc_data_pipeline_config,
    mlp_config,
    lenet_config,
    patchtst_config,
    default_training_config,
    default_metrics_config,
    default_persistence_config,
)



def main():
    
    exp_config = ExperimentConfig(
        data_pipeline_config=mfcc_data_pipeline_config,
        model_config=patchtst_config,
        training_config=default_training_config,
        metrics_config=default_metrics_config,
        persistence_config=default_persistence_config,
    )
    
    exp = ExperimentOrchestrator(
        config=exp_config,
        experiment_id="test_run",
    )
    exp.train_model()
    
    load_path = Path("outputs/mpkg/tmp/test_run")
    exp.load(mpkg_path=load_path)
    
    return



if __name__ == "__main__":
    main()
    print("DONE!")
