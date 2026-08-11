from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .dataset import ExampleDataset


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
class DevelopmentFold:
    train_dataset:ExampleDataset
    validation_dataset:ExampleDataset



@dataclass
class DataSplit:
    test_dataset:ExampleDataset
    development_folds:list[DevelopmentFold]
