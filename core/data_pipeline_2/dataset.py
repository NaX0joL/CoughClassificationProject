import torch
from torch import Tensor
from torch.utils.data import Dataset

from .intermediary import ORIGINAL_LABEL_METADATA_KEY, Example



class ExampleDataset(Dataset):

    def __init__(self, examples:list[Example]) -> None:
        self.examples = examples
        return

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index:int) -> dict[str, Tensor|dict[str, str]]:
        example = self.examples[index]
        original_label = example.metadata.get(ORIGINAL_LABEL_METADATA_KEY)
        metadata = {
            ORIGINAL_LABEL_METADATA_KEY: (
                str(original_label)
                if original_label is not None
                else str(example.label)
            ),
        }

        return {
            "value": torch.tensor(example.value, dtype=torch.float32),
            "label": torch.tensor(example.label, dtype=torch.long),
            "metadata": metadata,
        }
