from unittest.mock import Mock

import numpy as np
from torch.utils.data import DataLoader

from core.data_pipeline.dataset import ExampleDataset
from core.persistence import ExperimentPersistence



def test_save_fold_from_dataloaders_delegates_their_datasets(
    monkeypatch,
) -> None:
    train_loader = DataLoader(ExampleDataset([]), batch_size=1)
    validation_loader = DataLoader(ExampleDataset([]), batch_size=1)
    persistence = ExperimentPersistence.__new__(ExperimentPersistence)
    save_fold = Mock()
    monkeypatch.setattr(persistence, "save_fold", save_fold)

    persistence.save_fold_from_dataloaders(
        fold_index=1,
        model=Mock(),
        loss_log=Mock(),
        validation_metrics={"accuracy": 1.0},
        labels=np.asarray([0]),
        predictions=np.asarray([0]),
        class_names={0: "non-infectious"},
        train_loader=train_loader,
        validation_loader=validation_loader,
        additional_confusion_matrices={
            "train": (np.asarray([0]), np.asarray([0])),
        },
    )

    assert save_fold.call_args.kwargs["train_dataset"] is train_loader.dataset
    assert (
        save_fold.call_args.kwargs["validation_dataset"]
        is validation_loader.dataset
    )
    assert "train" in save_fold.call_args.kwargs["additional_confusion_matrices"]
