import numpy as np
import torch
import torchaudio.transforms as audio_transforms

from ...intermediary import SeriesSegment



SCALING_EPSILON = 1e-8



class SegmentTransformer():

    def transform(self, segments:list[SeriesSegment]) -> list[SeriesSegment]:
        transformed_segments = []

        for segment in segments:
            transformed_segments.append(SeriesSegment(
                value=self._transform_value(segment.value),
                original_index=segment.original_index,
            ))

        return transformed_segments

    def _transform_value(self, value:np.ndarray) -> np.ndarray:
        raise NotImplementedError



class MFCC(SegmentTransformer):

    def __init__(
        self,
        sampling_rate:int=16_000,
        n_fft:int=400,
        win_length:int=400,
        hop_length:int=160,
        n_mels:int=40,
        n_mfcc:int=40,
    ) -> None:
        self.transformer = audio_transforms.MFCC(
            sample_rate=sampling_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "n_mels": n_mels,
            },
        )
        return

    def _transform_value(self, value:np.ndarray) -> np.ndarray:
        waveform = torch.as_tensor(value, dtype=torch.float32)
        coefficients = self.transformer(waveform)
        return coefficients.transpose(0, 1).numpy()



class LogMelSpectrogram(SegmentTransformer):

    def __init__(
        self,
        sampling_rate:int=16_000,
        n_fft:int=400,
        win_length:int=400,
        hop_length:int=160,
        n_mels:int=40,
        log_offset:float=1e-6,
    ) -> None:
        self.log_offset = log_offset
        self.transformer = audio_transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        return

    def _transform_value(self, value:np.ndarray) -> np.ndarray:
        waveform = torch.as_tensor(value, dtype=torch.float32)
        mel_bands = self.transformer(waveform)
        log_mel_bands = torch.log(mel_bands + self.log_offset)
        return log_mel_bands.transpose(0, 1).numpy()



class FeatureWiseStandardization(SegmentTransformer):

    def _transform_value(self, value:np.ndarray) -> np.ndarray:
        feature_means = value.mean(axis=0, keepdims=True)
        feature_standard_deviations = value.std(axis=0, keepdims=True)
        safe_standard_deviations = np.maximum(
            feature_standard_deviations,
            SCALING_EPSILON,
        )
        return (value - feature_means) / safe_standard_deviations



class FeatureWiseNormalization(SegmentTransformer):

    def _transform_value(self, value:np.ndarray) -> np.ndarray:
        feature_minimums = value.min(axis=0, keepdims=True)
        feature_ranges = value.max(axis=0, keepdims=True) - feature_minimums
        safe_feature_ranges = np.maximum(feature_ranges, SCALING_EPSILON)
        return (value - feature_minimums) / safe_feature_ranges
