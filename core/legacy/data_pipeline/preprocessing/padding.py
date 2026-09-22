
import random
from typing import Literal

import numpy as np

from ..abstract import Padder
from ..intermediary import Example



PADDING_TYPE = Literal["left", "right", "balanced", "random"]



class ZeroPadder(Padder):

    def __init__(self, target_length:int, padding_type:PADDING_TYPE="balanced", random_seed:int=42) -> None:
        self.target_length = target_length
        self.padding_type = padding_type
        self.random_seed = random_seed
        
        if padding_type == "random":
            self.rng = random.Random(self.random_seed)
        return

    def pad(self, examples:list[Example]) -> list[Example]:
        if not examples:
            return []

        if self.target_length <= 0:
            raise ValueError("target length must be positive")

        padded_examples = []
        
        for example in examples:
            padding_length = self.target_length - len(example.value)
            
            if padding_length < 0:
                raise ValueError("example is longer than target length")

            left_padding_length, right_padding_length = self._get_each_padding_length(padding_length)
            padded_value = np.pad(
                example.value,
                (
                    (left_padding_length, right_padding_length),
                    (0, 0),
                ),
            )
            
            padded_example = Example(
                value=padded_value,
                label=example.label,
                metadata=example.metadata,
            )
            padded_examples.append(padded_example)

        return padded_examples

    def _get_each_padding_length(self, padding_length:int) -> tuple[int, int]:
        if self.padding_type == "left":
            left_pad, right_pad = padding_length, 0
        
        elif self.padding_type == "right":
            left_pad, right_pad = 0, padding_length
            
        elif self.padding_type == "balanced":
            left_pad = padding_length // 2
            right_pad = padding_length - left_pad
            
        elif self.padding_type == "random":
            left_pad = self.rng.randint(0, padding_length)
            right_pad = padding_length - left_pad
        
        else:
            raise ValueError(f"invalid padding type, got: {self.padding_type}")
        
        return left_pad, right_pad
