
import numpy as np
import torch
import torchaudio

from ..abstract import Transformer
from ..intermediary import Example

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
        transformed_examples = []

        for example in examples:
            transformed_example = Example(
                value=self._transform_value(example.value),
                label=example.label,
                metadata=example.metadata,
            )
            transformed_examples.append(transformed_example)

        return transformed_examples

    def _transform_value(self, value:np.ndarray) -> np.ndarray:
        if value.size == 0:
            raise ValueError("MFCC requires at least one audio sample")

        waveform = torch.as_tensor(value, dtype=torch.float32)
        coefficients = self.transformer(waveform)
        return coefficients.transpose(0, 1).numpy()


class MelBand(Transformer):
    """Convert waveforms to log-mel band energies."""

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
        transformed_examples = []

        for example in examples:
            transformed_example = Example(
                value=self._transform_value(example.value),
                label=example.label,
                metadata=example.metadata,
            )
            transformed_examples.append(transformed_example)

        return transformed_examples

    def _transform_value(self, value:np.ndarray) -> np.ndarray:
        if value.size == 0:
            raise ValueError("MelBand requires at least one audio sample")

        waveform = torch.as_tensor(value, dtype=torch.float32)
        mel_bands = self.transformer(waveform)
        log_mel_bands = torch.log(mel_bands + self.log_offset)
        return log_mel_bands.transpose(0, 1).numpy()


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
