from core.data_pipeline import DataPipelineConfig
from core.data_pipeline.preprocessing import LogMelSpectogram, SlidingWindowSegmenter, ZeroPadder
from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.stratifier import DataSplitter



melband_sliding_window_data_pipeline_config = DataPipelineConfig(
    name="mel_band_sliding_window",
    source_reader=ElderlyCoughAudioSourceReader(),
    segmenter=SlidingWindowSegmenter(
        window_size=8200,
        stride=1025,
        overlap_threshold=0.5,
        negative_label=0,
        keep_short_segments=False,
        kept_metadata_key=["patient_id", "cough_audio"],
    ),
    transformer=LogMelSpectogram(
        sample_rate=16_000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=40,
    ),
    padder=None,
    splitter=DataSplitter(
        group_metadata_key="patient_id",
        test_ratio=0.1,
        number_of_folds=5,
        random_seed=42,
    ),
)
