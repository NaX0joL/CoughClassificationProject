import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from modules.resolve_pytorch_device import get_optimal_device

from ..data_pipeline.intermediary import DataSplit
from ..model.full_model import FullModel
from .training_config import TrainingConfig
from .training_logic import train_model



class Trainer():
    
    def __init__(self, config:TrainingConfig, model:FullModel, data_split:DataSplit) -> None:
        self.config = config
        self.model = model
        self.data_split = data_split
        
        self.device = get_optimal_device()
        return
    
    def train_model(self) -> None:
        # create fresh model
        # train and compute metrics on each folds
        # compute metrics on test split
        # save all relevant data (
        #   config.pkl
        #   config.txt
        #   loss.json               [per fold]
        #   loss.png                [per fold]
        #   confusion_matrix.png    [per fold and test]
        #   metrics.json            [per fold and test]
        #   train_outputs.pdf       [per fold]
        #   validation_outputs.pdf  [per fold]
        # )
        return 
    
    def train_model(self, train_dataset:Dataset, validation_dataset:Dataset) -> None:
        train_loader = self._dataset_to_dataloader(train_dataset, shuffle=True)
        validation_loader = self._dataset_to_dataloader(validation_dataset, shuffle=False)
        
        self.model = self.model.to(self.device)
        criterion = build_criterion(self.config).to(self.device)
        optimizer = build_optimizer(self.config, self.model)
        
        train_model(
            epochs=self.config.num_epochs,
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )
        
        return
    
    def _dataset_to_dataloader(self, dataset:Dataset, shuffle:bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=self.config.drop_last,
            pin_memory=True if self.device.type == "cuda" else False,
        )



def build_criterion(config:TrainingConfig) -> nn.Module:
    
    if config.criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    
    raise ValueError(f"unsupported criterion, got: {config.criterion_name}")


def build_optimizer(config:TrainingConfig, model:FullModel) -> Optimizer:
    
    if config.optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    
    raise ValueError(f"unsupported optimizer, got: {config.optimizer_name}")
