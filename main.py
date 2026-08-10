from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.pipeline import DataPipeline
from core.data_pipeline.preprocessing import CoughSegmenter, MFCC, ZeroPadder



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

    print(dataset[0])
    
    return



if __name__ == "__main__":
    main()
