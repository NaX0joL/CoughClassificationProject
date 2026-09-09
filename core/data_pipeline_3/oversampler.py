import numpy as np
from imblearn.over_sampling import RandomOverSampler

from .intermediary import Example



class UniformOversampler():

    def __init__(self, random_seed:int|None=None) -> None:
        self.random_seed = random_seed
        self.oversampler = RandomOverSampler(random_state=random_seed)
        return

    def oversample(self, examples:list[Example]) -> list[Example]:
        labels = np.asarray(
            [example.label for example in examples],
            dtype=np.int64,
        )
        if np.unique(labels).size < 2:
            return list(examples)

        indices = np.arange(len(examples)).reshape(-1, 1)
        self.oversampler.fit_resample(
            indices,
            labels,
        )

        return [
            examples[int(index)]
            for index in self.oversampler.sample_indices_
        ]
