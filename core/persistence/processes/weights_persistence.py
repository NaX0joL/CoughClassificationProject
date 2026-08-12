from pathlib import Path

import torch

from ...model import FullModel


def save_fold_weights(weights_directory:Path, fold_index:int, model:FullModel) -> None:
    torch.save(model.state_dict(), weights_directory / f"fold_{fold_index}.pth")
    return
