
import random
from typing import Literal

import numpy as np

from ..intermediary import Example



padding_type = Literal["left", "right", "balanced", "random"]



class Padder():

    def __init__(self, target_length:int, padding_type:padding_type="balanced", random_seed:int=42) -> None:
        self.target_length = target_length
        self.padding_type = padding_type
        self.random_seed = random_seed
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
            
            padded_example = Example(
                value=np.pad(
                    example.value,
                    (left_padding_length, right_padding_length),
                ),
                label=example.label,
                metadata=example.metadata,
            )
            padded_examples.append(padded_example)

        return padded_examples

    def _get_each_padding_length(self, padding_length:int) -> int:
        if self.padding_type == "left":
            left_pad, right_pad = padding_length, 0
        
        if self.padding_type == "right":
            left_pad, right_pad = 0, padding_length
            
        if self.padding_type == "balanced":
            left_pad = padding_length // 2
            right_pad = padding_length - left_pad
            
        if self.padding_type == "random":
            left_pad = random.Random(self.random_seed).randint(0, padding_length)
            right_pad = padding_length - left_pad
        
        return left_pad, right_pad
