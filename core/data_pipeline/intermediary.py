from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass
class SourceSeries:
    value: np.ndarray
    label: int
    metadata: dict[str, object]


@dataclass
class Example:
    value: np.ndarray
    label: int
    metadata: dict[str, object]


@dataclass
class DataSplit:
    test_set: Dataset
    development_folds: list[dict[str, Dataset]]