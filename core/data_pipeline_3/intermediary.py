from dataclasses import dataclass
from pathlib import Path



@dataclass(frozen=True)
class SourceRecord:
    metadata:dict[str, object]
    audio_path:Path


@dataclass(frozen=True)
class FoldPartition:
    train:list[SourceRecord]
    validation:list[SourceRecord]
    test:list[SourceRecord]