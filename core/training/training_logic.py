from typing import Literal, TypeAlias

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from modules.resolve_pytorch_device import get_model_device

from ..model.full_model import FullModel



EPOCH_TYPE: TypeAlias = Literal["train", "validation"]



def train_model(
    epochs: int,
    model: FullModel,
    criterion: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> None:
    
    for _ in range(epochs):
        
        _epoch_logic(
            epoch_type="train",
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loader=train_loader,
        )
        _epoch_logic(
            epoch_type="validation",
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loader=validation_loader,
        )
    
    return


def _epoch_logic(
    epoch_type: EPOCH_TYPE,
    model: FullModel,
    criterion: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
) -> None:
    
    if epoch_type == "train":
        model.train()
        context_manager = torch.enable_grad()
        step_function = model.training_step
        
    elif epoch_type == "validation":
        model.eval()
        context_manager = torch.inference_mode()
        step_function = model.validation_step
        
    else:
        raise ValueError(f"invalid epoch type, got: {epoch_type}")
    
    with context_manager:
        for batch in loader:
            
            if epoch_type == "train":
                optimizer.zero_grad()
                
            batch = _move_batch_to_device(batch, device=get_model_device(model))
            step_result = step_function(batch, criterion)
            
            if epoch_type == "train":
                step_result.loss.backward()
                optimizer.step()
    
    return


def _move_batch_to_device(batch:dict[str, Tensor], device:torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }