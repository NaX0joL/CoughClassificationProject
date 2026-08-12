import json

import pytest

from core.persistence.processes.json_persistence import (
    save_cross_validation_summary_json,
)


def test_save_cross_validation_summary_json_calculates_mean_and_standard_deviation(
    tmp_path,
) -> None:
    json_directory = tmp_path / "json"
    (json_directory / "metrics").mkdir(parents=True)

    save_cross_validation_summary_json(
        json_directory,
        [
            {"accuracy": 0.5, "f1_score": 0.6},
            {"accuracy": 0.7, "f1_score": 0.8},
        ],
    )

    summary_path = json_directory / "metrics" / "cross_validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["number_of_folds"] == 2
    assert summary["metrics"]["accuracy"] == {
        "mean": pytest.approx(0.6),
        "standard_deviation": pytest.approx(0.1),
    }
    assert summary["metrics"]["f1_score"] == {
        "mean": pytest.approx(0.7),
        "standard_deviation": pytest.approx(0.1),
    }
