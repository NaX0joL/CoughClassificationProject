"""Recalculate metrics for saved mpkg models and save them in Excel."""

import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

import argparse
import sys
from pathlib import Path

import pandas as pd


# Let this script import project files when run from the project root.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from core.data_pipeline import DataPipeline
from core.experiment import ExperimentOrchestrator
from core.metrics import (
    AccuracyMetric,
    F1ScoreMetric,
    MacroAccuracyMetric,
    MacroF1ScoreMetric,
    MacroPrecisionMetric,
    MacroRecallMetric,
    MetricsConfig,
    ModelEvaluator,
    PRAucMetric,
    PrecisionMetric,
    RecallMetric,
    RocAucMetric,
    SpecificityMetric,
)


METRICS_BY_NAME = {
    "roc_auc": RocAucMetric,
    "pr_auc": PRAucMetric,
    "precision": PrecisionMetric,
    "recall": RecallMetric,
    "specificity": SpecificityMetric,
    "f1_score": F1ScoreMetric,
    "accuracy": AccuracyMetric,
    "macro_accuracy": MacroAccuracyMetric,
    "macro_f1_score": MacroF1ScoreMetric,
    "macro_precision": MacroPrecisionMetric,
    "macro_recall": MacroRecallMetric,
}

# Metrics calculated when --metrics is not supplied. Set a value to False to
# omit that metric from the script-wide default report.
DEFAULT_METRICS = {
    "roc_auc": True,
    "pr_auc": True,
    "precision": True,
    "recall": True,
    "specificity": True,
    "f1_score": True,
    "accuracy": True,
    "macro_accuracy": True,
    "macro_f1_score": True,
    "macro_precision": True,
    "macro_recall": True,
}


def find_mpkg_folders(folder):
    # An mpkg run folder contains this marker file.
    mpkg_folders = sorted(marker.parent for marker in folder.rglob("__mpkg__.py"))
    if not mpkg_folders:
        raise FileNotFoundError(f"No mpkg folders found in: {folder}")
    return mpkg_folders


def recompute_one_mpkg(mpkg_folder, metrics_config):
    """Return one row of metrics for every saved fold in an mpkg folder."""
    print(f"Checking {mpkg_folder.name}...")

    # Load the saved models and the configuration used to train them.
    experiment = ExperimentOrchestrator.load(mpkg_folder)

    # Recreate the exact validation folds from the saved configuration.
    pipeline = DataPipeline.create(experiment.config.data_pipeline_config)
    validation_folds = pipeline.get_data_split().development_folds

    if len(experiment.persisted_folds) != len(validation_folds):
        raise ValueError(f"Saved models and validation folds do not match: {mpkg_folder}")

    evaluator = ModelEvaluator(metrics_config)
    fold_rows = []

    # Evaluate each saved fold model on its corresponding validation fold.
    for saved_fold, validation_fold in zip(
        experiment.persisted_folds,
        validation_folds,
        strict=True,
    ):
        result = evaluator.evaluate(
            saved_fold.model,
            validation_fold.validation_dataset,
        )

        fold_rows.append({
            "mpkg": mpkg_folder.name,
            "folder": str(mpkg_folder),
            "fold": saved_fold.fold_index,
            **result.metrics.to_dict(),
        })
    return fold_rows


def create_summary_table(fold_table):
    """Average every metric across folds, producing one row per mpkg."""
    metric_columns = [
        column for column in fold_table.columns
        if column not in {"mpkg", "folder", "fold"}
    ]
    summary_table = fold_table.groupby(
        ["mpkg", "folder"],
        as_index=False,
    )[metric_columns].mean()
    summary_table.insert(
        2,
        "number_of_folds",
        summary_table["folder"].map(fold_table.groupby("folder")["fold"].nunique()),
    )
    return summary_table


def save_excel(summary_table, fold_table, excel_path):
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_table.to_excel(writer, sheet_name="mpkg_summary", index=False)
        fold_table.to_excel(writer, sheet_name="fold_metrics", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Folder containing mpkg runs")
    parser.add_argument("--output", type=Path, help="Where to save the Excel file")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS_BY_NAME,
        metavar="METRIC",
        help=(
            "Metrics to calculate instead of DEFAULT_METRICS. Choices: "
            + ", ".join(METRICS_BY_NAME)
        ),
    )
    args = parser.parse_args()

    mpkg_folders = find_mpkg_folders(args.folder)
    selected_metric_names = args.metrics or [
        name for name, is_enabled in DEFAULT_METRICS.items() if is_enabled
    ]
    if not selected_metric_names:
        parser.error("Enable at least one metric in DEFAULT_METRICS or use --metrics")
    selected_metrics_config = MetricsConfig(
        metrics=tuple(
            METRICS_BY_NAME[name]() for name in selected_metric_names
        ),
    )

    fold_rows = []
    for mpkg_folder in mpkg_folders:
        fold_rows.extend(recompute_one_mpkg(mpkg_folder, selected_metrics_config))

    # Make a table with one row per fold.
    fold_table = pd.DataFrame(fold_rows).sort_values(["mpkg", "fold"])
    summary_table = create_summary_table(fold_table)

    print("\nMean validation metrics for each mpkg:")
    print(summary_table.to_string(index=False))

    excel_path = args.output or args.folder / "recomputed_metrics.xlsx"
    save_excel(summary_table, fold_table, excel_path)

    print(f"\nSaved Excel file: {excel_path}")


if __name__ == "__main__":
    main()
