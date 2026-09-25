import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch

from core.artifacts.processes.metrics_writing import write_metrics_workbook
from core.metrics.evaluation import ModelEvaluation
from core.metrics.label_mapping import BinaryInfectionLabelMapping
from core.metrics.metrics_config import MetricsConfig
from core.artifacts.processes.configuration_persistence import save_configuration
from core.trainer import LossLog



DEFAULT_OUTPUT_DIRECTORY = Path("outputs/mpkg/tmp")



class ExperimentArtifactWriter:

    def __init__(
        self,
        config:object,
        data_pipeline:object,
        output_directory:Path|None=None,
    ) -> None:
        self.config = config
        self.configuration = self._build_configuration(config, data_pipeline)
        self.experiment_id = self._get_experiment_id(config)
        self.root_directory = output_directory or DEFAULT_OUTPUT_DIRECTORY
        self.experiment_directory = self._create_experiment_directory()
        self.figures_directory = self.experiment_directory / "figures"
        self.loss_directory = self.figures_directory / "loss"
        self.confusion_matrix_directory = self.figures_directory / "confusion_matrix"
        self.weights_directory = self.experiment_directory / "weights"
        self.loss_directory.mkdir(parents=True)
        self.confusion_matrix_directory.mkdir()
        self.weights_directory.mkdir()
        self._fold_metrics:list[tuple[int, dict[str, float|None]]] = []
        self._finalized = False
        self._write_config()
        return

    def get_save_directory(self) -> Path:
        return self.root_directory
    
    def save_fold(
        self,
        fold_index:int,
        model:torch.nn.Module,
        loss_log:LossLog,
        evaluation:ModelEvaluation,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot save a fold after finalization")
        if fold_index < 1:
            raise ValueError("fold_index must be one-based and positive")

        report_class_names = evaluation.reported_class_names

        torch.save(
            model.state_dict(),
            self.weights_directory / f"fold_{fold_index}.pth",
        )
        self._save_loss_figure(fold_index, loss_log)
        self._save_confusion_matrix_figure(
            fold_index,
            report_class_names,
            evaluation.reported_labels,
            evaluation.reported_predictions,
        )
        self._fold_metrics.append((fold_index, evaluation.metrics.to_dict()))
        return

    def _save_confusion_matrix_figure(
        self,
        fold_index:int,
        class_names:tuple[str, ...],
        reported_labels:np.ndarray,
        reported_predictions:np.ndarray,
    ) -> None:
        class_labels = np.arange(len(class_names))
        matrix = confusion_matrix(
            reported_labels,
            reported_predictions,
            labels=class_labels,
        )
        figure, axis = plt.subplots()
        try:
            image = axis.imshow(matrix, cmap="Blues")
            axis.set(
                title=f"Confusion Matrix: fold {fold_index}",
                xlabel="Predicted class",
                ylabel="True class",
                xticks=class_labels,
                yticks=class_labels,
                xticklabels=class_names,
                yticklabels=class_names,
            )
            figure.colorbar(image, ax=axis)
            for row_index in range(len(class_names)):
                for column_index in range(len(class_names)):
                    axis.text(
                        column_index,
                        row_index,
                        str(matrix[row_index, column_index]),
                        ha="center",
                        va="center",
                    )
            figure.tight_layout()
            figure.savefig(
                self.confusion_matrix_directory
                / f"fold_{fold_index}.png",
            )
        finally:
            plt.close(figure)
        return

    def _save_loss_figure(self, fold_index:int, loss_log:LossLog) -> None:
        figure, axis = plt.subplots()
        try:
            axis.plot(loss_log.training_losses, label="train")
            axis.plot(loss_log.validation_losses, label="validation")
            axis.set(
                title=f"Loss: fold {fold_index}",
                xlabel="Epoch",
                ylabel="Loss",
            )
            axis.legend()
            figure.tight_layout()
            figure.savefig(self.loss_directory / f"fold_{fold_index}.png")
        finally:
            plt.close(figure)
        return

    def finalize(self, elapsed_seconds:float) -> None:
        if self._finalized:
            raise RuntimeError("artifact writer is already finalized")
        write_metrics_workbook(
            self.experiment_directory / "metrics.xlsx",
            self._fold_metrics,
        )
        run_metadata = {
            "experiment_id": self.experiment_id,
            "elapsed_seconds": elapsed_seconds,
        }
        (self.experiment_directory / "run.json").write_text(
            json.dumps(run_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.experiment_directory / "__mpkg__.py").write_text(
            "# X experiment artifact\n",
            encoding="utf-8",
        )
        self._finalized = True
        return

    def _create_experiment_directory(self) -> Path:
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("experiment_id must be a non-empty string")

        self.root_directory.mkdir(parents=True, exist_ok=True)
        base_directory = self.root_directory / self.experiment_id
        candidate = base_directory
        suffix = 1
        while True:
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                candidate = self.root_directory / (
                    f"{self.experiment_id}-{suffix}"
                )
                suffix += 1

    def _write_config(self) -> None:
        save_configuration(self.experiment_directory, self.configuration)
        return

    @staticmethod
    def _build_configuration(config:object, data_pipeline:object) -> dict[str, object]:
        return {
            "experiment_id": getattr(config, "experiment_id", ""),
            "data_pipeline": {
                "type": data_pipeline.__class__.__name__,
                "source_reader": data_pipeline.source_reader,
                "partitioner": data_pipeline.partitioner,
                "example_constructor": data_pipeline.example_constructor,
                "oversampler": getattr(data_pipeline, "oversampler", None),
            },
            "model": config.model_config,
            "training": config.training_config,
            "evaluation": {
                "metrics": getattr(config, "metrics_config", MetricsConfig.default()),
                "label_mapping": getattr(
                    config,
                    "label_mapping",
                    BinaryInfectionLabelMapping(),
                ),
            },
        }

    @staticmethod
    def _get_experiment_id(config:object) -> str:
        return getattr(config, "experiment_id", "")
