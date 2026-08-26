import numpy as np
import pytest

from core.data_pipeline.dataset import ExampleDataset
from core.data_pipeline.intermediary import Example
from core.training.trainer import build_criterion
from core.training.training_config import TrainingConfig


def _make_config(class_weighting:str="none") -> TrainingConfig:
    return TrainingConfig(
        random_seed=42,
        num_epochs=1,
        criterion_name="cross_entropy",
        optimizer_name="adamw",
        learning_rate=0.0001,
        weight_decay=0.001,
        class_weighting=class_weighting,
        batch_size=2,
        num_workers=0,
        drop_last=False,
    )


def _make_dataset(labels:list[int]) -> ExampleDataset:
    examples = [
        Example(
            value=np.zeros((2, 2), dtype=np.float32),
            label=label,
            metadata={},
        )
        for label in labels
    ]
    return ExampleDataset(examples)


def test_build_criterion_without_class_weights() -> None:
    criterion = build_criterion(
        _make_config(class_weighting="none"),
        _make_dataset([0, 0, 1]),
    )

    assert criterion.weight is None


def test_build_criterion_with_balanced_class_weights() -> None:
    criterion = build_criterion(
        _make_config(class_weighting="balanced"),
        _make_dataset([0, 0, 0, 0, 1, 1]),
    )

    assert criterion.weight is not None
    assert criterion.weight.dtype.is_floating_point
    assert criterion.weight.tolist() == pytest.approx([0.75, 1.5])


def test_balanced_class_weights_require_both_classes() -> None:
    with pytest.raises(ValueError, match="requires both binary classes"):
        build_criterion(
            _make_config(class_weighting="balanced"),
            _make_dataset([0, 0, 0]),
        )
