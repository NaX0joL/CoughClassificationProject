import torch
from torch import nn

from core.training import BestModelCheckpoint


def test_best_model_checkpoint_loads_lowest_validation_loss_model() -> None:
    model = nn.Linear(1, 1, bias=False)
    checkpoint = BestModelCheckpoint()

    with torch.no_grad():
        model.weight.fill_(1.0)
    checkpoint.update(model, validation_loss=0.5, epoch=1)

    with torch.no_grad():
        model.weight.fill_(2.0)
    checkpoint.update(model, validation_loss=0.75, epoch=2)

    checkpoint.load_best_model(model)

    assert model.weight.item() == 1.0
    assert checkpoint.best_validation_loss == 0.5
    assert checkpoint.best_epoch == 1
