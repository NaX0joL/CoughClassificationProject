from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.experiment import ExperimentOrchestrator



def test_cross_validation_reuses_training_batch_size_and_releases_each_fold(
    monkeypatch:pytest.MonkeyPatch,
) -> None:
    events = []
    evaluation_batch_sizes = []
    created_model_count = 0

    class FoldModel:

        def __init__(self, fold_index:int) -> None:
            self.fold_index = fold_index

        def to(self, device:torch.device) -> "FoldModel":
            events.append(f"move-{self.fold_index}")
            return self

        def __del__(self) -> None:
            events.append(f"release-{self.fold_index}")

    class Persistence:

        run_directory = Path("outputs/test")

        def save_fold(self, fold_index:int, **kwargs) -> None:
            events.append(f"save-{fold_index}")
            return

        def save_cross_validation_summary(self, folds_metrics) -> None:
            return

    class Evaluator:

        def evaluate(self, model, dataset, batch_size:int):
            evaluation_batch_sizes.append(batch_size)
            return SimpleNamespace(
                metrics=SimpleNamespace(to_dict=lambda: {"accuracy": 1.0}),
                labels=[],
                predictions=[],
                class_names={},
            )

    def create_model(config):
        nonlocal created_model_count
        created_model_count += 1
        events.append(f"create-{created_model_count}")
        return FoldModel(created_model_count)

    folds = [
        SimpleNamespace(train_dataset=object(), validation_dataset=object()),
        SimpleNamespace(train_dataset=object(), validation_dataset=object()),
    ]
    pipeline = SimpleNamespace(
        get_data_split=lambda: SimpleNamespace(development_folds=folds),
        get_examples=lambda: [],
    )
    persistence = Persistence()
    experiment = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
    experiment.config = SimpleNamespace(
        data_pipeline_config=object(),
        model_config=object(),
        training_config=SimpleNamespace(random_seed=None, batch_size=7),
        metrics_config=object(),
        persistence_config=SimpleNamespace(
            feature_colormap="inferno",
        ),
    )
    experiment.experiment_id = "test"
    experiment.device = torch.device("cuda")
    experiment._time_development_fold_training = (
        lambda model, fold: (object(), 1.0)
    )

    monkeypatch.setattr(
        "core.experiment.ExperimentPersistence.create",
        lambda **kwargs: persistence,
    )
    monkeypatch.setattr(
        "core.experiment.DataPipeline.create",
        lambda config: pipeline,
    )
    monkeypatch.setattr(
        "core.experiment.ExampleGalleryGenerator",
        lambda **kwargs: SimpleNamespace(generate=lambda examples: None),
    )
    monkeypatch.setattr(
        "core.experiment.ClassDistributionGenerator",
        lambda **kwargs: SimpleNamespace(generate=lambda data_split: None),
    )
    monkeypatch.setattr(
        "core.experiment.ModelEvaluator",
        lambda config: Evaluator(),
    )
    monkeypatch.setattr(
        "core.experiment.FullModel",
        SimpleNamespace(create=create_model),
    )
    monkeypatch.setattr(
        "core.experiment.torch.cuda.empty_cache",
        lambda: events.append("empty-cache"),
    )

    experiment.train_model()

    assert evaluation_batch_sizes == [7, 7]
    assert events == [
        "create-1",
        "move-1",
        "save-1",
        "release-1",
        "empty-cache",
        "create-2",
        "move-2",
        "save-2",
        "release-2",
        "empty-cache",
    ]
