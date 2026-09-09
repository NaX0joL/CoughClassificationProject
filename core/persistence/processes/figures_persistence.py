from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from ...data_pipeline.dataset import ExampleDataset
from ...model import FullModel
from ...training import LossLog
from .model_output_persistence import save_model_output_pdfs
from ..persistence_config import PersistenceConfig


def save_fold_figures(
    figures_directory:Path,
    fold_index:int,
    loss_log:LossLog,
    labels:np.ndarray,
    predictions:np.ndarray,
    class_names:dict[int, str],
    model:FullModel,
    train_dataset:ExampleDataset,
    validation_dataset:ExampleDataset,
    persistence_config:PersistenceConfig,
    additional_confusion_matrices:dict[
        str,
        tuple[np.ndarray, np.ndarray],
    ]|None=None,
) -> None:
    _save_loss_figure(figures_directory, fold_index, loss_log)
    _save_confusion_matrix(
        figures_directory,
        fold_index,
        labels,
        predictions,
        class_names,
        "test",
    )
    for split_name, (split_labels, split_predictions) in (
        additional_confusion_matrices or {}
    ).items():
        _save_confusion_matrix(
            figures_directory,
            fold_index,
            split_labels,
            split_predictions,
            class_names,
            split_name,
        )
    save_model_output_pdfs(
        figures_directory,
        fold_index,
        model,
        train_dataset,
        validation_dataset,
        class_names,
        persistence_config,
    )
    return


def _save_loss_figure(
    figures_directory:Path,
    fold_index:int,
    loss_log:LossLog,
) -> None:
    figure, axis = plt.subplots()
    axis.plot(loss_log.training_losses, label="train")
    axis.plot(loss_log.validation_losses, label="validation")
    axis.set(title=f"Loss: fold {fold_index}", xlabel="Epoch", ylabel="Loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figures_directory / "loss" / f"loss-fold_{fold_index}.png",
        dpi=150,
    )
    plt.close(figure)
    return


def _save_confusion_matrix(
    figures_directory:Path,
    fold_index:int,
    labels:np.ndarray,
    predictions:np.ndarray,
    class_names:dict[int, str],
    split_name:str|None=None,
) -> None:
    configured_labels = np.asarray(list(class_names))
    class_labels = np.unique(
        np.concatenate((configured_labels, labels, predictions)),
    )
    matrix = confusion_matrix(labels, predictions, labels=class_labels)
    display_labels = [class_names.get(int(label), str(label)) for label in class_labels]

    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Count")
    axis.set(
        title=_confusion_matrix_title(fold_index, split_name),
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(len(class_labels)),
        yticks=range(len(class_labels)),
        xticklabels=display_labels,
        yticklabels=display_labels,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index, column_index in np.ndindex(matrix.shape):
        axis.text(
            column_index,
            row_index,
            str(matrix[row_index, column_index]),
            ha="center",
            va="center",
            color="white" if matrix[row_index, column_index] > threshold else "black",
        )
    figure.tight_layout()
    figure.savefig(
        _confusion_matrix_path(figures_directory, fold_index, split_name),
        dpi=150,
    )
    plt.close(figure)
    return


def _confusion_matrix_title(fold_index:int, split_name:str|None) -> str:
    if split_name is None:
        return f"Confusion matrix: fold {fold_index}"
    return f"Confusion matrix ({split_name}): fold {fold_index}"


def _confusion_matrix_path(
    figures_directory:Path,
    fold_index:int,
    split_name:str|None,
) -> Path:
    if split_name is None:
        file_name = f"confusion_matrix-fold_{fold_index}.png"
    else:
        file_name = f"confusion_matrix-{split_name}-fold_{fold_index}.png"
    return figures_directory / "confusion_matrix" / file_name
