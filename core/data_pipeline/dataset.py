import torch
from torch import Tensor
from torch.utils.data import Dataset

from .intermediary import Example


class ExampleDataset(Dataset):
    
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples
        return
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, index:int) -> dict[str, Tensor]:
        example = self.examples[index]
        
        return {
            "value": torch.tensor(
                example.value,
                dtype=torch.float32,
            ),
            "label": torch.tensor(
                example.label,
                dtype=torch.long,
            ),
        }
