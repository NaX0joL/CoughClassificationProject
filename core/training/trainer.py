import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from modules.resolve_pytorch_device import get_model_device

from ..data_pipeline.dataset import ExampleDataset
from ..model.full_model import FullModel
from .checkpoint import BestModelCheckpoint
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
        
        self.device = get_model_device(model)
        return
    
    def fit(self) -> LossLog:
        
        train_loader = self._dataset_to_dataloader(self.train_dataset, shuffle=True)
        validation_loader = self._dataset_to_dataloader(self.validation_dataset, shuffle=False)
        
        criterion = build_criterion(self.config, self.train_dataset).to(self.device)
        optimizer = build_optimizer(self.config, self.model)
        checkpoint = BestModelCheckpoint()
        
        loss_log = do_train_logic(
            epochs=self.config.num_epochs,
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=validation_loader,
            checkpoint=checkpoint,
            load_best_model=self.config.load_best_model,
            early_stopping_patience=self.config.early_stopping_patience,
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
            generator=self._create_generator(),
        )

    def _create_generator(self) -> torch.Generator|None:
        if self.config.random_seed is None:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.config.random_seed)
        return generator



def build_criterion(
    config:TrainingConfig,
    train_dataset:ExampleDataset,
) -> nn.Module:
    
    if config.criterion_name == "cross_entropy":
        class_weights = _build_class_weights(config, train_dataset)
        return nn.CrossEntropyLoss(weight=class_weights)
    
    raise ValueError(f"unsupported criterion, got: {config.criterion_name}")


def _build_class_weights(
    config:TrainingConfig,
    train_dataset:ExampleDataset,
) -> Tensor|None:
    
    if config.class_weighting == "none":
        return None

    if config.class_weighting == "balanced":
        labels = torch.tensor(
            [example.label for example in train_dataset.examples],
            dtype=torch.long,
        )
        class_counts = torch.bincount(labels, minlength=2)
        if len(class_counts) != 2 or torch.any(class_counts == 0):
            raise ValueError(
                "balanced class weighting requires both binary classes "
                "in the training dataset"
            )

        total_examples = len(train_dataset.examples)
        return total_examples / (2 * class_counts.float())

    raise ValueError(f"unsupported class weighting, got: {config.class_weighting}")


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
