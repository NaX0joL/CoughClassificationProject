
from core.data_pipeline import DataPipelineConfig
from core.data_pipeline.preprocessing import CoughSegmenter, MFCC, Resampler, ZeroPadder
from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.stratifier import DataSplitter


mfcc_resampled_data_pipeline_config = DataPipelineConfig(
    name="mfcc_resampled",
    source_reader=ElderlyCoughAudioSourceReader(),
    segmenter=CoughSegmenter(
        kept_metadata_key=["patient_id", "cough_audio"],
    ),
    transformer=[
        MFCC(
            sample_rate=16_000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40,
            n_mfcc=40,
        ),
        Resampler(
            target_length=500,
            method="linear",
        ),
    ],
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
