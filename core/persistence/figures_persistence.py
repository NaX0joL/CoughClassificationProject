from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt

from ..training import LossLog


def save_fold_figures(
    figures_directory:Path,
    fold_index:int,
    loss_log:LossLog,
    validation_metrics:dict[str, float],
) -> None:
    _save_loss_figure(figures_directory, fold_index, loss_log)
    _save_confusion_matrix_placeholder(figures_directory, fold_index)
    _save_train_report_placeholder(figures_directory, fold_index)
    _save_validation_report(figures_directory, fold_index, validation_metrics)
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


def _save_confusion_matrix_placeholder(figures_directory:Path, fold_index:int) -> None:
    _save_text_figure(
        figures_directory / "confusion_matrix" / f"confusion_matrix-fold_{fold_index}.png",
        "Confusion matrix\nPrediction-level results are not yet persisted.",
    )
    return


def _save_train_report_placeholder(figures_directory:Path, fold_index:int) -> None:
    _save_text_figure(
        figures_directory / "output_train" / f"fold_{fold_index}-train.pdf",
        "Training result\nTraining metrics are not yet collected.",
    )
    return


def _save_validation_report(
    figures_directory:Path,
    fold_index:int,
    validation_metrics:dict[str, float],
) -> None:
    _save_text_figure(
        figures_directory / "output_validation" / f"fold_{fold_index}-validation.pdf",
        f"Validation result\n\n{pformat(validation_metrics, sort_dicts=False)}",
    )
    return


def _save_text_figure(path:Path, text:str) -> None:
    figure, axis = plt.subplots(figsize=(8.27, 11.69))
    axis.axis("off")
    axis.text(0.05, 0.95, text, va="top", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return
