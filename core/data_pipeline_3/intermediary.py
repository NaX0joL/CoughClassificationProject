from dataclasses import dataclass
from pathlib import Path
import numpy as np



@dataclass(frozen=True)
class SourceRecord:
    metadata:dict[str, object]
    audio_path:Path


@dataclass(frozen=True)
class FoldPartition:
    train:list[SourceRecord]
    validation:list[SourceRecord]
    test:list[SourceRecord]


@dataclass(frozen=True)
class SeriesSegment:
    value:np.ndarray
    original_index:tuple[int, int]


@dataclass(frozen=True)
class Example:
    value:np.ndarray
    label:int
    metadata:dict[str, object]
