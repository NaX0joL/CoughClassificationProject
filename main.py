from dataclasses import asdict

from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline import DataPipeline, DataPipelineConfig
from core.data_pipeline.preprocessing import CoughSegmenter, MFCC, ZeroPadder
from core.data_pipeline.stratifier import DataSplitter

from core.model import FullModel, ModelConfig
from core.model.architectures.MLP import MLP, MLPConfig
from core.model.behavior.classification_behavior import ClassificationBehavior

from core.training import TrainingConfig

from core.experiment import ExperimentOrchestrator



def main():
    print("Hello from coughclassificationproject!")
    
    data_pipeline_config = DataPipelineConfig(
        source_reader=ElderlyCoughAudioSourceReader(),
        segmenter=CoughSegmenter(
            kept_metadata_key=["patient_id", "cough_audio"],
        ),
        transformer=MFCC(),
        padder=ZeroPadder(
            target_length=820,
            padding_type="random",
            random_seed=42,
        ),
        splitter=DataSplitter(
            group_metadata_key="patient_id",
            test_ratio=0.1,
            number_of_folds=5,
            random_seed=42,
        ),
    )
    # pipeline = DataPipeline.create(data_pipeline_config)
    # dataset = pipeline.get_dataset()
    # data_split = pipeline.get_data_split()
    
    model_config = ModelConfig(
        architecture=MLP(**asdict(MLPConfig(linear_dims=[64, 64], output_dim=2))),
        behavior=ClassificationBehavior(),
    )
    # model = FullModel.create(model_config)
    # out = model(dataset[0]["value"])
    
    training_config = TrainingConfig(
        num_epochs=3,
        criterion_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.0001,
        weight_decay=0.001,
        batch_size=32,
        num_workers=1,
        drop_last=False,
    )
    
    exp = ExperimentOrchestrator(
        data_pipeline_config=data_pipeline_config,
        model_config=model_config,
        training_config=training_config,
    )
    exp.train_model()
    
    # print(dataset[0])
    # print(out)
    return



if __name__ == "__main__":
    main()
