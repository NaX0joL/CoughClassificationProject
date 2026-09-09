from pathlib import Path
import json

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .intermediary import Example



class ExampleDataset(Dataset):

    def __init__(self, examples:list[Example]) -> None:
        self.examples = examples
        return

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index:int) -> dict[str, object]:
        example = self.examples[index]

        return {
            "value": torch.as_tensor(example.value, dtype=torch.float32),
            "label": torch.as_tensor(example.label, dtype=torch.long),
            "metadata": MetadataSanitizer.sanitize(example.metadata),
        }



class MetadataSanitizer:
    
    @classmethod
    def sanitize(cls, metadata:dict[str, object]) -> dict[str, str]:
        sanitized = {
            str(key): cls._serialize_metadata_value(value)
            for key, value in metadata.items()
        }
        return sanitized
    
    @staticmethod
    def _serialize_metadata_value(value:object) -> str:
        if value is None:
            return ""

        if isinstance(value, (str, Path)):
            return str(value)

        if isinstance(value, (list, tuple, dict)):
            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )

        return str(value)