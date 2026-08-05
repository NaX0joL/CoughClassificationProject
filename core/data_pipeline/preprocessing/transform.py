
import numpy as np
import torch
import torchaudio

from ..intermediary import Example



SAMPLE_RATE = 16_000
FRAME_LENGTH = 400
FRAME_STEP = 160
MEL_FILTER_COUNT = 40
MFCC_COEFFICIENT_COUNT = 40



class MFCC():

    def __init__(self) -> None:
        self.transformer = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=MFCC_COEFFICIENT_COUNT,
            log_mels=True,
            melkwargs={
                "n_fft": FRAME_LENGTH,
                "win_length": FRAME_LENGTH,
                "hop_length": FRAME_STEP,
                "n_mels": MEL_FILTER_COUNT,
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
