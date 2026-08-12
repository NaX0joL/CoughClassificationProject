from pathlib import Path

import torch
from torch import Tensor

from ...model import FullModel


def save_fold_weights(weights_directory:Path, fold_index:int, model:FullModel) -> None:
    torch.save(model.state_dict(), weights_directory / f"fold_{fold_index}.pth")
    return


def load_fold_weights(
    weights_directory:Path,
    fold_index:int,
) -> dict[str, Tensor]:
    weights_path = weights_directory / f"fold_{fold_index}.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(f"mpkg weights do not exist: {weights_path}")

    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"mpkg weights must be a state dictionary: {weights_path}")

    return state_dict
