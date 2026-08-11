from datetime import datetime
from pathlib import Path
from typing import Any

from ..model import FullModel
from ..training import LossLog
from .configuration_persistence import save_configuration
from .figures_persistence import save_fold_figures
from .json_persistence import save_fold_json
from .weights_persistence import save_fold_weights


DEFAULT_OUTPUT_DIRECTORY = Path("outputs/mpkg")



class ExperimentPersistence:
    """Coordinate persistence for the artifacts of one experiment-training call."""

    def __init__(self, run_directory:Path) -> None:
        self.run_directory = run_directory
        self.figures_directory = run_directory / "figures"
        self.json_directory = run_directory / "json"
        self.weights_directory = run_directory / "weights"
        return

    @classmethod
    def create(
        cls,
        config:dict[str, Any],
        output_directory:Path=DEFAULT_OUTPUT_DIRECTORY,
    ) -> "ExperimentPersistence":
        run_directory = _create_run_directory(output_directory)
        persistence = cls(run_directory)
        persistence._create_directory_layout()
        save_configuration(run_directory, config)
        return persistence

    def save_fold(
        self,
        fold_index:int,
        model:FullModel,
        loss_log:LossLog,
        validation_metrics:dict[str, float],
    ) -> None:
        save_fold_figures(
            self.figures_directory,
            fold_index,
            loss_log,
            validation_metrics,
        )
        save_fold_json(
            self.json_directory,
            fold_index,
            loss_log,
            validation_metrics,
        )
        save_fold_weights(self.weights_directory, fold_index, model)
        return

    def _create_directory_layout(self) -> None:
        self.figures_directory.mkdir(parents=True)
        (self.figures_directory / "loss").mkdir()
        (self.figures_directory / "confusion_matrix").mkdir()
        (self.figures_directory / "output_train").mkdir()
        (self.figures_directory / "output_validation").mkdir()
        self.json_directory.mkdir()
        (self.json_directory / "loss").mkdir()
        (self.json_directory / "metrics").mkdir()
        self.weights_directory.mkdir()
        (self.run_directory / "__mpkg__.py").touch()
        return


def _create_run_directory(output_directory:Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = output_directory / f"run_{timestamp}"
    run_directory.mkdir()
    return run_directory
