from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .dataset import ExampleDataset


ORIGINAL_LABEL_METADATA_KEY = "original_label"


@dataclass
class SourceSeries:
    value:np.ndarray
    metadata:dict[str, object]



@dataclass
class Segment:
    source_series:SourceSeries
    value:np.ndarray
    original_index:tuple[int, int]
    cough_annotations:list[tuple[int, int]]=field(default_factory=list)



@dataclass
class FoldPartition:
    train:list[SourceSeries]
    validation:list[SourceSeries]
    test:list[SourceSeries]



@dataclass
class Example:
    value:np.ndarray
    label:int
    metadata:dict[str, object]



@dataclass
class ExampleBundle:
    train:list[Example]
    validation:list[Example]
    test:list[Example]



@dataclass
class DevelopmentFold:
    train_dataset:ExampleDataset
    validation_dataset:ExampleDataset



@dataclass
class DataSplit:
    test_dataset:ExampleDataset
    development_folds:list[DevelopmentFold]
