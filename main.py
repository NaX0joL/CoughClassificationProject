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



# def main():
#     print("Hello from coughclassificationproject!")
    
#     data_pipeline_config = DataPipelineConfig(
#         source_reader=ElderlyCoughAudioSourceReader(),
#         segmenter=CoughSegmenter(
#             kept_metadata_key=["patient_id", "cough_audio"],
#         ),
#         transformer=MFCC(),
#         padder=ZeroPadder(
#             target_length=820,
#             padding_type="random",
#             random_seed=42,
#         ),
#         splitter=DataSplitter(
#             group_metadata_key="patient_id",
#             test_ratio=0.1,
#             number_of_folds=5,
#             random_seed=42,
#         ),
#     )
#     # pipeline = DataPipeline.create(data_pipeline_config)
#     # dataset = pipeline.get_dataset()
#     # data_split = pipeline.get_data_split()
    
#     model_config = ModelConfig(
#         architecture=MLP(**asdict(MLPConfig(linear_dims=[64, 64], output_dim=2))),
#         behavior=ClassificationBehavior(),
#     )
#     # model = FullModel.create(model_config)
#     # out = model(dataset[0]["value"])
    
#     training_config = TrainingConfig(
#         num_epochs=2,
#         criterion_name="cross_entropy",
#         optimizer_name="adamw",
#         learning_rate=0.0001,
#         weight_decay=0.001,
#         batch_size=32,
#         num_workers=1,
#         drop_last=False,
#         load_best_model=True,
#     )
    
#     exp = ExperimentOrchestrator(
#         config=ExperimentConfig(
#             data_pipeline_config=data_pipeline_config,
#             model_config=model_config,
#             training_config=training_config,
#             metrics_config=MetricsConfig.default(),
#             persistence_config=PersistenceConfig.default(),
#         ),
#     )
#     exp.train_model()
    
#     return


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
