from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn


CHECKPOINT_DIRECTORY = Path("outputs/checkpoint")


@dataclass
class BestModelCheckpoint:
    checkpoint_path: Path = field(
        default_factory=lambda: CHECKPOINT_DIRECTORY / "checkpoint.pth",
    )
    best_validation_loss:float|None = None
    best_epoch:int|None = None

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
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        return

    def load_best_model(self, model:nn.Module) -> None:
        if not self.checkpoint_path.is_file():
            raise ValueError("best model checkpoint has not been created")

        state_dict = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return
