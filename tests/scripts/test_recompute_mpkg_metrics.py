import json
import sys

import pytest

from scripts.analysis import recompute_mpkg_metrics
from scripts.analysis.recompute_mpkg_metrics import extract_one_mpkg


def test_extract_one_mpkg_returns_saved_metrics_in_fold_order(tmp_path) -> None:
    mpkg_folder = tmp_path / "experiment"
    metrics_directory = mpkg_folder / "json" / "metrics"
    metrics_directory.mkdir(parents=True)
    (metrics_directory / "metrics-fold_10.json").write_text(
        json.dumps({"accuracy": 0.8, "f1_score": 0.7}),
        encoding="utf-8",
    )
    (metrics_directory / "metrics-fold_2.json").write_text(
        json.dumps({"accuracy": 0.6, "f1_score": 0.5}),
        encoding="utf-8",
    )

    fold_rows = extract_one_mpkg(mpkg_folder)

    assert fold_rows == [
        {
            "mpkg": "experiment",
            "folder": str(mpkg_folder),
            "fold": 2,
            "accuracy": 0.6,
            "f1_score": 0.5,
        },
        {
            "mpkg": "experiment",
            "folder": str(mpkg_folder),
            "fold": 10,
            "accuracy": 0.8,
            "f1_score": 0.7,
        },
    ]


def test_extract_one_mpkg_filters_selected_saved_metrics(tmp_path) -> None:
    mpkg_folder = tmp_path / "experiment"
    metrics_directory = mpkg_folder / "json" / "metrics"
    metrics_directory.mkdir(parents=True)
    (metrics_directory / "metrics-fold_1.json").write_text(
        json.dumps({"accuracy": 0.8, "f1_score": 0.7}),
        encoding="utf-8",
    )

    fold_rows = extract_one_mpkg(mpkg_folder, ["f1_score"])

    assert fold_rows == [{
        "mpkg": "experiment",
        "folder": str(mpkg_folder),
        "fold": 1,
        "f1_score": 0.7,
    }]


def test_extract_one_mpkg_rejects_missing_selected_metric(tmp_path) -> None:
    mpkg_folder = tmp_path / "experiment"
    metrics_directory = mpkg_folder / "json" / "metrics"
    metrics_directory.mkdir(parents=True)
    metric_path = metrics_directory / "metrics-fold_1.json"
    metric_path.write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")

    with pytest.raises(ValueError, match="Saved metrics are missing f1_score"):
        extract_one_mpkg(mpkg_folder, ["f1_score"])


def test_main_recomputes_metrics_only_when_requested(monkeypatch, tmp_path) -> None:
    mpkg_folder = tmp_path / "experiment"
    recomputed_folders = []

    def recompute(mpkg_folder, metrics_config):
        recomputed_folders.append(mpkg_folder)
        return [{
            "mpkg": mpkg_folder.name,
            "folder": str(mpkg_folder),
            "fold": 1,
            "accuracy": 0.9,
        }]

    def reject_extraction(mpkg_folder, metric_names=None):
        raise AssertionError("stored metrics should not be extracted")

    monkeypatch.setattr(
        sys,
        "argv",
        ["recompute_mpkg_metrics.py", str(tmp_path), "--recompute"],
    )
    monkeypatch.setattr(
        recompute_mpkg_metrics,
        "find_mpkg_folders",
        lambda folder: [mpkg_folder],
    )
    monkeypatch.setattr(recompute_mpkg_metrics, "recompute_one_mpkg", recompute)
    monkeypatch.setattr(
        recompute_mpkg_metrics,
        "extract_one_mpkg",
        reject_extraction,
    )
    monkeypatch.setattr(
        recompute_mpkg_metrics,
        "save_excel",
        lambda summary_table, fold_table, excel_path: None,
    )

    recompute_mpkg_metrics.main()

    assert recomputed_folders == [mpkg_folder]
