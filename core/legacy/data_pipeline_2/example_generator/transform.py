import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from ..abstract import Transformer
from ..intermediary import Example


SCALING_EPSILON = 1e-8



class MFCC(Transformer):

    def __init__(
        self,
        sample_rate:int=16_000,
        n_fft:int=400,
        win_length:int=400,
        hop_length:int=160,
        n_mels:int=40,
        n_mfcc:int=40,
    ) -> None:
        _validate_feature_parameters(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        if n_mfcc <= 0:
            raise ValueError("n_mfcc must be positive")
        if n_mfcc > n_mels:
            raise ValueError("n_mfcc cannot exceed n_mels")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.transformer = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            log_mels=True,
            melkwargs={
                "n_fft": self.n_fft,
                "win_length": self.win_length,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
            },
        )
        return

    def transform(self, examples:list[Example]) -> list[Example]:
        transformed_examples:list[Example] = []

        for example in examples:
            transformed_examples.append(Example(
                value=self._transform_value(example.value),
                label=example.label,
                metadata=example.metadata,
            ))

        return transformed_examples

    def _transform_value(self, value:np.ndarray) -> np.ndarray:
        if value.size == 0:
            raise ValueError("MFCC requires at least one audio sample")

        waveform = torch.as_tensor(value, dtype=torch.float32)
        coefficients = self.transformer(waveform)
        return coefficients.transpose(0, 1).numpy()



class LogMelSpectrogram(Transformer):

    def __init__(
        self,
        sample_rate:int=16_000,
        n_fft:int=400,
        win_length:int=400,
        hop_length:int=160,
        n_mels:int=40,
        log_offset:float=1e-6,
    ) -> None:
        _validate_feature_parameters(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        if log_offset <= 0:
            raise ValueError("log_offset must be positive")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.log_offset = log_offset
        self.transformer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        return

    def transform(self, examples:list[Example]) -> list[Example]:
        transformed_examples:list[Example] = []

        for example in examples:
            transformed_examples.append(Example(
                value=self._transform_value(example.value),
                label=example.label,
                metadata=example.metadata,
            ))

        return transformed_examples

    def _transform_value(self, value:np.ndarray) -> np.ndarray:
        if value.size == 0:
            raise ValueError(
                "LogMelSpectrogram requires at least one audio sample"
            )

        waveform = torch.as_tensor(value, dtype=torch.float32)
        mel_bands = self.transformer(waveform)
        log_mel_bands = torch.log(mel_bands + self.log_offset)
        return log_mel_bands.transpose(0, 1).numpy()



class FeatureWiseStandardization(Transformer):

    def transform(self, examples:list[Example]) -> list[Example]:
        standardized_examples:list[Example] = []

        for example in examples:
            standardized_examples.append(Example(
                value=self._standardize_value(example.value),
                label=example.label,
                metadata=example.metadata,
            ))

        return standardized_examples

    @staticmethod
    def _standardize_value(value:np.ndarray) -> np.ndarray:
        if value.size == 0:
            raise ValueError(
                "FeatureWiseStandardization requires at least one value"
            )

        feature_means = value.mean(axis=0, keepdims=True)
        feature_standard_deviations = value.std(axis=0, keepdims=True)
        safe_standard_deviations = np.maximum(
            feature_standard_deviations,
            SCALING_EPSILON,
        )
        return (value - feature_means) / safe_standard_deviations



class FeatureWiseNormalization(Transformer):

    def transform(self, examples:list[Example]) -> list[Example]:
        normalized_examples:list[Example] = []

        for example in examples:
            normalized_examples.append(Example(
                value=self._normalize_value(example.value),
                label=example.label,
                metadata=example.metadata,
            ))

        return normalized_examples

    @staticmethod
    def _normalize_value(value:np.ndarray) -> np.ndarray:
        if value.size == 0:
            raise ValueError(
                "FeatureWiseNormalization requires at least one value"
            )

        feature_minimums = value.min(axis=0, keepdims=True)
        feature_ranges = value.max(axis=0, keepdims=True) - feature_minimums
        safe_feature_ranges = np.maximum(feature_ranges, SCALING_EPSILON)
        return (value - feature_minimums) / safe_feature_ranges



class DownSampler(Transformer):

    RESAMPLING_METHOD = ("sinc_interp_hann", "sinc_interp_kaiser")

    def __init__(
        self,
        original_sampling_rate:int,
        target_sampling_rate:int,
        resampling_method:str="sinc_interp_hann",
        lowpass_filter_width:int=6,
    ) -> None:
        if original_sampling_rate <= 0 or target_sampling_rate <= 0:
            raise ValueError("Sample rates must be positive integers.")
        if resampling_method not in self.RESAMPLING_METHOD:
            raise ValueError(f"method must be one of {self.RESAMPLING_METHOD}")

        self.original_sampling_rate = original_sampling_rate
        self.lowpass_filter_width = lowpass_filter_width
        if original_sampling_rate == target_sampling_rate:
            self.resampler:T.Resample|None = None
        else:
            self.resampler = T.Resample(
                orig_freq=original_sampling_rate,
                new_freq=target_sampling_rate,
                lowpass_filter_width=lowpass_filter_width,
                resampling_method=resampling_method,
            )
        return

    def transform(self, examples:list[Example]) -> list[Example]:
        transformed_examples:list[Example] = []

        for example in examples:
            transformed_examples.append(Example(
                value=self._resample_value(example.value),
                label=example.label,
                metadata=example.metadata,
            ))

        return transformed_examples

    def _resample_value(self, value:np.ndarray) -> np.ndarray:
        if value.size == 0:
            raise ValueError("DownSampler requires a non-empty audio array.")

        if self.resampler is None:
            return value.copy()

        waveform = torch.as_tensor(value, dtype=torch.float32)
        is_multi_channel = waveform.ndim > 1
        if is_multi_channel:
            waveform = waveform.mT

        resampled = self.resampler(waveform)
        if is_multi_channel:
            resampled = resampled.mT
        return resampled.numpy()


def _validate_feature_parameters(
    sample_rate:int,
    n_fft:int,
    win_length:int,
    hop_length:int,
    n_mels:int,
) -> None:
    parameters = {
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
        "n_mels": n_mels,
    }
    for name, value in parameters.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if win_length > n_fft:
        raise ValueError("win_length cannot exceed n_fft")
    return
