from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BestModelCheckpoint:
    best_validation_loss:float|None = None
    best_epoch:int|None = None
    state_dict:dict[str, torch.Tensor]|None = None

    def update(
        self,
        model:nn.Module,
        validation_loss:float,
        epoch:int,
    ) -> None:
        if (
            self.best_validation_loss is not None
            and validation_loss >= self.best_validation_loss
        ):
            return

        self.best_validation_loss = validation_loss
        self.best_epoch = epoch
        self.state_dict = deepcopy(model.state_dict())
        return

    def load_best_model(self, model:nn.Module) -> None:
        if self.state_dict is None:
            raise ValueError("best model checkpoint has not been created")

        model.load_state_dict(self.state_dict)
        return
