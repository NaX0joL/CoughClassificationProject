from core.data_pipeline import DataPipelineConfig
from core.data_pipeline.preprocessing import SlidingWindowSegmenter, MFCC, ZeroPadder
from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.stratifier import DataSplitter



mfcc_sliding_window_data_pipeline_config = DataPipelineConfig(
    name="mfcc_sliding_window",
    source_reader=ElderlyCoughAudioSourceReader(),
    segmenter=SlidingWindowSegmenter(
        window_size=8200,
        stride=4100,
        overlap_threshold=0.7,
        negative_label=0,
        keep_short_segments=False,
        kept_metadata_key=["patient_id", "cough_audio"],
    ),
    transformer=MFCC(
        sample_rate=16_000,
        n_fft=512,              # 512, 1024
        win_length=400,
        hop_length=200,
        n_mels=64,              # 64, 128
        n_mfcc=40,
    ),
    padder=None,
    splitter=DataSplitter(
        group_metadata_key="patient_id",
        test_ratio=0.1,
        number_of_folds=5,
        random_seed=42,
    ),
)
