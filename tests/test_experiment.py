from unittest.mock import Mock

import pytest

from core.experiment import ExperimentOrchestrator, _format_elapsed_time



def test_format_elapsed_time() -> None:
    formatted_time = _format_elapsed_time(3723.456)

    assert formatted_time == "01:02:03.46"


def test_time_development_fold_training(
    monkeypatch:pytest.MonkeyPatch,
) -> None:
    times = iter([10.0, 13.5])
    loss_log = Mock()
    experiment = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
    monkeypatch.setattr(
        "core.experiment.perf_counter",
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
