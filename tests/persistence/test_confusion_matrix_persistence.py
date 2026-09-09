from pathlib import Path

import numpy as np

from core.persistence.processes.figures_persistence import (
    _save_confusion_matrix,
    save_fold_figures,
)



def test_save_confusion_matrix_uses_split_name_in_file(
    tmp_path:Path,
) -> None:
    confusion_matrix_directory = tmp_path / "confusion_matrix"
    confusion_matrix_directory.mkdir()

    _save_confusion_matrix(
        figures_directory=tmp_path,
        fold_index=2,
        labels=np.asarray([0, 1, 2]),
        predictions=np.asarray([0, 2, 2]),
        class_names={0: "0", 1: "1", 2: "2"},
        split_name="train",
    )

    assert (
        confusion_matrix_directory / "confusion_matrix-train-fold_2.png"
    ).is_file()


def test_save_fold_figures_names_primary_confusion_matrix_as_test(
    monkeypatch,
    tmp_path:Path,
) -> None:
    saved_split_names = []
    monkeypatch.setattr(
        "core.persistence.processes.figures_persistence._save_loss_figure",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "core.persistence.processes.figures_persistence._save_confusion_matrix",
        lambda *args: saved_split_names.append(args[-1]),
    )
    monkeypatch.setattr(
        "core.persistence.processes.figures_persistence.save_model_output_pdfs",
        lambda *args: None,
    )

    save_fold_figures(
        figures_directory=tmp_path,
        fold_index=1,
        loss_log=None,
        labels=np.asarray([1]),
        predictions=np.asarray([1]),
        class_names={0: "0", 1: "1", 2: "2"},
        model=None,
        train_dataset=None,
        validation_dataset=None,
        persistence_config=None,
    )

    assert saved_split_names == ["test"]
