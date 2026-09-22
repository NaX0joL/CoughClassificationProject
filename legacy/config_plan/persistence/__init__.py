from .default_persistence import default_persistence_config
from .log_mel_spectrogram_legrad_persistence import log_mel_spectrogram_legrad_persistence_config
from .log_mel_spectrogram_gradcam_persistence import log_mel_spectrogram_gradcam_persistence_config
from .mfcc_gradcam_persistence import mfcc_gradcam_persistence_config
from .mfcc_legrad_persistence import mfcc_legrad_persistence_config


# Backward-compatible feature-plan names for the Grad-CAM variants.
mfcc_persistence_config = mfcc_gradcam_persistence_config
log_mel_spectrogram_persistence_config = log_mel_spectrogram_gradcam_persistence_config


__all__ = [
    "default_persistence_config",
    "log_mel_spectrogram_legrad_persistence_config",
    "log_mel_spectrogram_gradcam_persistence_config",
    "mfcc_gradcam_persistence_config",
    "mfcc_legrad_persistence_config",
    "mfcc_persistence_config",
    "log_mel_spectrogram_persistence_config",
]
