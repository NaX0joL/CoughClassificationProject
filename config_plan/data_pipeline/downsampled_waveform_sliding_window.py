from core.data_pipeline import DataPipelineConfig
from core.data_pipeline.preprocessing import DownSampler, SlidingWindowSegmenter
from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.stratifier import DataSplitter



downsampled_waveform_sliding_window_data_pipeline_config = DataPipelineConfig(
    name="downsampled_waveform_sliding_window",
    source_reader=ElderlyCoughAudioSourceReader(),
    segmenter=SlidingWindowSegmenter(
        window_size=8200,
        stride=1025,
        overlap_threshold=0.5,
        negative_label=0,
        keep_short_segments=False,
        kept_metadata_key=["patient_id", "cough_audio"],
    ),
    transformer=DownSampler(
        original_sampling_rate=16000,
        target_sampling_rate=16000,
        resampling_method="sinc_interp_hann",
        lowpass_filter_width=6,
    ),
    padder=None,
    splitter=DataSplitter(
        group_metadata_key="patient_id",
        test_ratio=0.1,
        number_of_folds=5,
        random_seed=42,
    ),
)
