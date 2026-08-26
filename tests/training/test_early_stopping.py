import pytest

from core.training.early_stopping import EarlyStopping


def test_early_stopping_stops_after_patience_without_improvement() -> None:
    early_stopping = EarlyStopping(patience=2)

    assert early_stopping.update(validation_loss=0.5) is False
    assert early_stopping.update(validation_loss=0.6) is False
    assert early_stopping.update(validation_loss=0.7) is True


def test_early_stopping_resets_after_an_improvement() -> None:
    early_stopping = EarlyStopping(patience=2)

    early_stopping.update(validation_loss=0.5)
    early_stopping.update(validation_loss=0.6)

    assert early_stopping.update(validation_loss=0.4) is False
    assert early_stopping.epochs_without_improvement == 0


def test_early_stopping_rejects_invalid_patience() -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        EarlyStopping(patience=0)
