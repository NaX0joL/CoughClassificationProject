from dataclasses import dataclass

from torch.utils.data import DataLoader




@dataclass(frozen=True)
class DataModule:
    train_loader:DataLoader
    validation_loader:DataLoader
    test_loader:DataLoader
