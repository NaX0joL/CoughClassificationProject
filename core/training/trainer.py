import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from modules.resolve_pytorch_device import get_optimal_device

from ..data_pipeline.dataset import ExampleDataset
from ..model.full_model import FullModel
from .training_config import TrainingConfig
from .train_logic import LossLog, do_train_logic



class Trainer:
    
    def __init__(
        self,
        config:TrainingConfig,
        model:FullModel,
        train_dataset:ExampleDataset,
        validation_dataset:ExampleDataset,
    ) -> None:
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        
        self.device = get_optimal_device()
        return
    
    def fit(self) -> LossLog:
        print(f"started training on device {self.device}")
        
        train_loader = self._dataset_to_dataloader(self.train_dataset, shuffle=True)
        validation_loader = self._dataset_to_dataloader(self.validation_dataset, shuffle=False)
        
        self.model = self.model.to(self.device)
        criterion = build_criterion(self.config).to(self.device)
        optimizer = build_optimizer(self.config, self.model)
        
        loss_log = do_train_logic(
            epochs=self.config.num_epochs,
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=validation_loader,
        )
        return loss_log
    
    def _dataset_to_dataloader(
        self,
        dataset:ExampleDataset,
        shuffle:bool,
    ) -> DataLoader:
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
