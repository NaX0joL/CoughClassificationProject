from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from modules.resolve_pytorch_device import get_model_device

from ..model.full_model import FullModel
from .train_display import TrainDisplay



EPOCH_TYPE: TypeAlias = Literal["train", "validation"]



@dataclass
class LossLog:
    training_losses:list[float]
    validation_losses:list[float]



def do_train_logic(
    epochs: int,
    model: FullModel,
    criterion: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> LossLog:
    loss_log = LossLog(
        training_losses=[],
        validation_losses=[],
    )
    train_display = TrainDisplay(number_of_epochs=epochs)

    try:
        for _ in range(epochs):
            training_loss = _epoch_logic(
                epoch_type="train",
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loader=train_loader,
            )
            validation_loss = _epoch_logic(
                epoch_type="validation",
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loader=validation_loader,
            )

            loss_log.training_losses.append(training_loss)
            loss_log.validation_losses.append(validation_loss)
            
            train_display.update(training_loss, validation_loss)

    finally:
        train_display.close()

    return loss_log


def _epoch_logic(
    epoch_type: EPOCH_TYPE,
    model: FullModel,
    criterion: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
) -> float:
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
    
    total_loss = 0.0
    total_examples = 0

    with context_manager:
        for batch in loader:
            if epoch_type == "train":
                optimizer.zero_grad()

            batch = _move_batch_to_device(batch, device=get_model_device(model))
            step_result = step_function(batch, criterion)
            batch_size = batch["label"].numel()
            total_loss += step_result.loss.item() * batch_size
            total_examples += batch_size

            if epoch_type == "train":
                step_result.loss.backward()
                optimizer.step()

    if total_examples == 0:
        raise ValueError(f"{epoch_type} loader must contain at least one example")

    mean_loss = total_loss / total_examples
    return mean_loss


def _move_batch_to_device(batch:dict[str, Tensor], device:torch.device) -> dict[str, Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }
