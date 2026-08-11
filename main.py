from dataclasses import asdict

from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.pipeline import DataPipeline
from core.data_pipeline.preprocessing import CoughSegmenter, MFCC, ZeroPadder

from core.model.full_model import FullModel
from core.model.architectures.MLP import MLP, MLPConfig
from core.model.behavior.classification_behavior import ClassificationBehavior




def main():
    print("Hello from coughclassificationproject!")
    
    pipeline = DataPipeline(
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
    )
    dataset = pipeline.get_dataset()
    
    model = FullModel(
        architecture=MLP(**asdict(MLPConfig.default())),
        behavior=ClassificationBehavior(),
    )
    out = model(dataset[0]["value"])
    
    print(dataset[0])
    print(out)
    return



if __name__ == "__main__":
    main()
