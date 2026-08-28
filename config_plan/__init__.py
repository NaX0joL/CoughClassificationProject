from .data_pipeline.mfcc_annotated_segments import mfcc_annotated_segments_data_pipeline_config
from .data_pipeline.log_mel_spectrogram_annotated_segments import log_mel_spectrogram_annotated_segments_data_pipeline_config
from .data_pipeline.mfcc_sliding_windows import mfcc_sliding_windows_data_pipeline_config
from .data_pipeline.log_mel_spectrogram_sliding_windows import log_mel_spectrogram_sliding_windows_data_pipeline_config
from .data_pipeline import (
    downsampled_waveform_sliding_window_data_pipeline_config,
)

from .metrics.default_metrics import default_metrics_config

from .model import (
    mlp_config,
    lenet_config,
    patchtst_config,
    transformer_config,
    resnet_config,
)

from .persistence import (
    default_persistence_config,
    log_mel_spectrogram_gradcam_persistence_config,
    log_mel_spectrogram_legrad_persistence_config,
    mfcc_gradcam_persistence_config,
    mfcc_legrad_persistence_config,
    mfcc_persistence_config,
    log_mel_spectrogram_persistence_config,
)

from .training.default_training import default_training_config
from .training.normal_batch_training import normal_batch_training_config
from .training.small_batch_training import small_batch_training_config


# __all__ = [
#     "mfcc_annotated_segments_data_pipeline_config",
#     "log_mel_spectrogram_annotated_segments_data_pipeline_config",
#     "mfcc_sliding_windows_data_pipeline_config",
#     "log_mel_spectrogram_sliding_windows_data_pipeline_config",
#     "default_metrics_config",
    
#     "lenet_config",
#     "mlp_config",
#     "patchtst_config",
#     "transformer_config",
#     "resnet_config",
    
#     "default_persistence_config",
#     "log_mel_spectrogram_gradcam_persistence_config",
#     "log_mel_spectrogram_legrad_persistence_config",
#     "mfcc_gradcam_persistence_config",
#     "mfcc_legrad_persistence_config",
    
#     "default_training_config",
#     "normal_batch_training_config",
#     "small_batch_training_config",
# ]
