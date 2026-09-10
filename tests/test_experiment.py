from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

from core.experiment import ExperimentOrchestrator, _format_elapsed_time
from core.experiment_old import ExperimentOrchestrator as OldExperimentOrchestrator
from core.data_pipeline_3.pipeline import DataPipeline



def test_format_elapsed_time() -> None:
    formatted_time = _format_elapsed_time(3723.456)

    assert formatted_time == "01:02:03.46"


def test_time_development_fold_training(
    monkeypatch:pytest.MonkeyPatch,
) -> None:
    times = iter([10.0, 13.5])
    loss_log = Mock()
    experiment = OldExperimentOrchestrator.__new__(OldExperimentOrchestrator)
    monkeypatch.setattr(
        "core.experiment_old.perf_counter",
        lambda: next(times),
    )
    monkeypatch.setattr(
        experiment,
        "_train_development_fold",
        lambda model, development_fold: loss_log,
    )

    returned_loss_log, training_seconds = (
        experiment._time_development_fold_training(Mock(), Mock())
    )

    assert returned_loss_log is loss_log
    assert training_seconds == 3.5


def test_print_training_time(
    capsys:pytest.CaptureFixture[str],
) -> None:
    experiment = ExperimentOrchestrator.__new__(ExperimentOrchestrator)

    experiment._print_training_time("fold-2", 65.5)

    assert capsys.readouterr().out == "fold-2 training time: 00:01:05.50\n"


def test_time_trainer_fit(monkeypatch:pytest.MonkeyPatch) -> None:
    times = iter([10.0, 13.5])
    loss_log = Mock()
    trainer = Mock()
    trainer.fit.return_value = loss_log
    experiment = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
    monkeypatch.setattr(
        "core.experiment.perf_counter",
        lambda: next(times),
    )

    returned_loss_log, training_seconds = experiment._time_trainer_fit(trainer)

    trainer.fit.assert_called_once_with()
    assert returned_loss_log is loss_log
    assert training_seconds == 3.5


def test_train_model_builds_and_trains_each_pipeline_fold(
    monkeypatch:pytest.MonkeyPatch,
) -> None:
    data_modules = [
        SimpleNamespace(
            train_loader=SimpleNamespace(dataset=object()),
            validation_loader=SimpleNamespace(dataset=object()),
            test_loader=SimpleNamespace(dataset=object()),
        ),
        SimpleNamespace(
            train_loader=SimpleNamespace(dataset=object()),
            validation_loader=SimpleNamespace(dataset=object()),
            test_loader=SimpleNamespace(dataset=object()),
        ),
    ]
    data_module_calls = []
    created_models = [Mock(), Mock()]
    loss_logs = [Mock(), Mock()]
    trainer_calls = []
    saved_folds = []

    class Persistence:

        run_directory = Path("outputs/test")

        def save_fold_from_dataloaders(self, **kwargs) -> None:
            saved_folds.append(kwargs)
            return

        def save_cross_validation_summary(self, folds_metrics) -> None:
            self.folds_metrics = folds_metrics
            return

    class Evaluator:

        def evaluate_dataloader(self, model, data_loader):
            return SimpleNamespace(
                metrics=SimpleNamespace(to_dict=lambda: {"accuracy": 1.0}),
                labels=np.asarray([0]),
                predictions=np.asarray([0]),
                class_names={0: "non-infectious", 1: "infectious"},
            )

    class StubDataPipeline(DataPipeline):

        source_reader = object()
        partitioner = object()
        example_constructor = SimpleNamespace(
            train_segmenter=object(),
            transformers=[],
        )
        batch_size = 8
        oversampler = None

        def __init__(self) -> None:
            return

        def __len__(self) -> int:
            return len(data_modules)

        def get_data_module(self, **kwargs):
            data_module_calls.append(kwargs)
            return data_modules[kwargs["index"]]

    def create_model(config):
        model = created_models[len(trainer_calls)]
        model.to.return_value = model
        return model

    def create_trainer(**kwargs):
        trainer_calls.append(kwargs)
        return SimpleNamespace(
            fit=lambda: loss_logs[len(trainer_calls) - 1],
        )

    persistence = Persistence()
    training_config = SimpleNamespace(
        random_seed=42,
        batch_size=8,
        num_workers=2,
        drop_last=True,
    )
    experiment = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
    experiment.config = SimpleNamespace(
        data_pipeline_config=StubDataPipeline(),
        training_config=training_config,
        model_config=object(),
        metrics_config=object(),
        persistence_config=SimpleNamespace(feature_colormap="inferno"),
    )
    experiment.experiment_id = "test"
    experiment.device = torch.device("cuda")

    monkeypatch.setattr("core.experiment.set_random_seed", Mock())
    monkeypatch.setattr("core.experiment.FullModel.create", create_model)
    monkeypatch.setattr("core.experiment.Trainer", create_trainer)
    monkeypatch.setattr(
        "core.experiment.ExperimentPersistence.create",
        lambda **kwargs: persistence,
    )
    monkeypatch.setattr(
        "core.experiment.ModelEvaluator",
        lambda config: Evaluator(),
    )
    example_gallery = Mock()
    class_distribution = Mock()
    example_gallery_generator = Mock(return_value=example_gallery)
    monkeypatch.setattr(
        "core.experiment.ExampleGalleryGenerator",
        example_gallery_generator,
    )
    class_distribution_generator = Mock(return_value=class_distribution)
    monkeypatch.setattr(
        "core.experiment.ClassDistributionGenerator",
        class_distribution_generator,
    )
    monkeypatch.setattr("core.experiment.torch.cuda.empty_cache", Mock())

    experiment.train_model()

    assert data_module_calls == [
        {
            "index": 0,
            "num_workers": 2,
            "drop_last_batch": True,
            "pin_memory": True,
            "random_seed": 42,
        },
        {
            "index": 1,
            "num_workers": 2,
            "drop_last_batch": True,
            "pin_memory": True,
            "random_seed": 42,
        },
    ]
    assert [call["data_module"] for call in trainer_calls] == data_modules
    assert [fold["fold_index"] for fold in saved_folds] == [1, 2]
    assert [fold["loss_log"] for fold in saved_folds] == loss_logs
    assert all(
        set(fold["additional_confusion_matrices"]) == {"train", "validation"}
        for fold in saved_folds
    )
    assert [fold["class_names"] for fold in saved_folds] == [
        {0: "0", 1: "1", 2: "2"},
        {0: "0", 1: "1", 2: "2"},
    ]
    assert persistence.folds_metrics == [
        {"accuracy": 1.0},
        {"accuracy": 1.0},
    ]
    assert example_gallery.generate_fold_from_dataloaders.call_args_list == [
        call(
            fold_index=1,
            train_loader=data_modules[0].train_loader,
            validation_loader=data_modules[0].validation_loader,
            test_loader=data_modules[0].test_loader,
        ),
        call(
            fold_index=2,
            train_loader=data_modules[1].train_loader,
            validation_loader=data_modules[1].validation_loader,
            test_loader=data_modules[1].test_loader,
        ),
    ]
    assert example_gallery_generator.call_args.kwargs["class_names"] == {
        0: "label 0",
        1: "label 1",
        2: "label 2",
    }
    gallery_config = example_gallery_generator.call_args.kwargs[
        "data_pipeline_config"
    ]
    assert gallery_config.name is None
    assert class_distribution_generator.call_args.kwargs["class_names"] == {
        0: "label 0",
        1: "label 1",
        2: "label 2",
    }
    assert class_distribution.collect_from_dataloaders.call_args_list == [
        call(
            fold_index=1,
            train_loader=data_modules[0].train_loader,
            validation_loader=data_modules[0].validation_loader,
            test_loader=data_modules[0].test_loader,
        ),
        call(
            fold_index=2,
            train_loader=data_modules[1].train_loader,
            validation_loader=data_modules[1].validation_loader,
            test_loader=data_modules[1].test_loader,
        ),
    ]
    class_distribution.generate_collected.assert_called_once_with()
    assert experiment.run_directory == persistence.run_directory
