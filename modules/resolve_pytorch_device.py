import torch
from torch import nn, device



def get_optimal_device() -> device:
    return device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_device(model:nn.Module) -> device:
    return next(model.parameters()).device
