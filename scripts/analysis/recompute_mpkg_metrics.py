"""Collect metrics from saved mpkg models and save them in Excel."""

import json
import importlib
import importlib.util
import os
import sys
from types import ModuleType
from typing import Any

print(os.getcwd())
sys.path.append(os.getcwd())

import argparse
import sys
from pathlib import Path

import pandas as pd


# Let this script import project files when run from the project root.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from core.data_pipeline_3.pipeline import DataPipeline
from core.experiment import ExperimentOrchestrator
from core.experiment_config import ExperimentConfig
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


REPLACEMENT_CONFIG_ATTRIBUTE_NAMES = (
    "replacement_config",
    "experiment_config",
    "config",
)


def load_replacement_config(config_reference:str|Path) -> ExperimentConfig:
    """Import an externally defined replacement ``ExperimentConfig``.

    The reference may be a Python module, a Python file, or either followed
    by ``:attribute``.  When the attribute is omitted, a conventional config
    attribute is selected, or a sole ``ExperimentConfig`` in the module is
    used.
    """
    module_reference, attribute_name = _split_config_reference(
        str(config_reference),
    )
    module, inferred_attribute_name = _import_config_module(module_reference)
    replacement_config = _select_config_attribute(
        module,
        attribute_name or inferred_attribute_name,
    )
    if not isinstance(replacement_config, ExperimentConfig):
        raise TypeError(
            "replacement config must be an ExperimentConfig instance"
        )
    return replacement_config


def _split_config_reference(config_reference:str) -> tuple[str, str|None]:
    if ":" not in config_reference:
        return config_reference, None

    module_reference, attribute_name = config_reference.rsplit(":", maxsplit=1)
    if not module_reference or not attribute_name:
        raise ValueError(
            "replacement config must use MODULE[:ATTRIBUTE] or FILE[:ATTRIBUTE]"
        )
    return module_reference, attribute_name


def _import_config_module(
    module_reference:str,
) -> tuple[ModuleType, str|None]:
    config_path = Path(module_reference)
    if config_path.is_file():
        module_name = f"_recompute_replacement_config_{abs(hash(config_path.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import replacement config: {config_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module, None

    try:
        return importlib.import_module(module_reference), None
    except ModuleNotFoundError as import_error:
        if "." not in module_reference:
            raise import_error
        module_name, attribute_name = module_reference.rsplit(".", maxsplit=1)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise import_error
        return module, attribute_name


def _select_config_attribute(
    module:ModuleType,
    attribute_name:str|None,
) -> Any:
    if attribute_name is not None:
        try:
            return getattr(module, attribute_name)
        except AttributeError as error:
            raise AttributeError(
                f"replacement config attribute does not exist: {attribute_name}"
            ) from error

    for conventional_name in REPLACEMENT_CONFIG_ATTRIBUTE_NAMES:
        candidate = getattr(module, conventional_name, None)
        if isinstance(candidate, ExperimentConfig):
            return candidate

    config_instances = [
        value for value in vars(module).values()
        if isinstance(value, ExperimentConfig)
    ]
    if len(config_instances) == 1:
        return config_instances[0]
    if not config_instances:
        raise ValueError(
            "replacement config module must define an ExperimentConfig"
        )
    raise ValueError(
        "replacement config module defines multiple ExperimentConfig instances; "
        "specify one with :ATTRIBUTE"
    )


def find_mpkg_folders(folder):
    # An mpkg run folder contains this marker file.
    mpkg_folders = sorted(marker.parent for marker in folder.rglob("__mpkg__.py"))
    if not mpkg_folders:
        raise FileNotFoundError(f"No mpkg folders found in: {folder}")
    return mpkg_folders


def extract_one_mpkg(mpkg_folder, metric_names=None):
    """Return the metrics already stored for every fold in an mpkg folder."""
    print(f"Extracting {mpkg_folder.name}...")

    metric_paths = sorted(
        (mpkg_folder / "json" / "metrics").glob("metrics-fold_*.json"),
        key=_get_fold_index,
    )
    if not metric_paths:
        raise FileNotFoundError(f"No saved fold metrics found in: {mpkg_folder}")

    fold_rows = []
    for metric_path in metric_paths:
        with metric_path.open(encoding="utf-8") as metric_file:
            stored_metrics = json.load(metric_file)
        if not isinstance(stored_metrics, dict):
            raise ValueError(f"Saved fold metrics must be a dictionary: {metric_path}")

        selected_metrics = stored_metrics
        if metric_names is not None:
            missing_metric_names = set(metric_names) - stored_metrics.keys()
            if missing_metric_names:
                missing_metrics = ", ".join(sorted(missing_metric_names))
                raise ValueError(
                    f"Saved metrics are missing {missing_metrics}: {metric_path}"
                )
            selected_metrics = {
                name: stored_metrics[name]
                for name in metric_names
            }

        fold_rows.append({
            "mpkg": mpkg_folder.name,
            "folder": str(mpkg_folder),
            "fold": _get_fold_index(metric_path),
            **selected_metrics,
        })
    return fold_rows


def recompute_one_mpkg(
    mpkg_folder,
    metrics_config=None,
    replacement_config:ExperimentConfig|None=None,
):
    """Return one row of metrics for every saved fold in an mpkg folder."""
    print(f"Checking {mpkg_folder.name}...")

    if replacement_config is None and isinstance(metrics_config, ExperimentConfig):
        replacement_config = metrics_config
        metrics_config = None

    # Load the saved models and the configuration used to train them.
    experiment = ExperimentOrchestrator.load(mpkg_folder)

    # A replacement config can change data and metric settings, but never the
    # model config.  The loaded experiment and its saved fold models therefore
    # remain the source of truth for model construction and evaluation.
    data_pipeline_config = experiment.config.data_pipeline_config
    if replacement_config is not None:
        data_pipeline_config = replacement_config.data_pipeline_config

    if metrics_config is None:
        metrics_config = (
            replacement_config.metrics_config
            if replacement_config is not None
            else experiment.config.metrics_config
        )

    # Current experiments persist the version-3 pipeline as its components,
    # rather than as a DataPipelineConfig.  Rebuild that pipeline directly.
    pipeline = _create_current_data_pipeline(data_pipeline_config)

    persisted_fold_indices = [
        saved_fold.fold_index for saved_fold in experiment.persisted_folds
    ]
    expected_fold_indices = list(range(1, len(pipeline) + 1))
    if sorted(persisted_fold_indices) != expected_fold_indices:
        raise ValueError(f"Saved models and data folds do not match: {mpkg_folder}")

    evaluator = ModelEvaluator(metrics_config)
    fold_rows = []

    # Fold metrics are persisted from test loaders, so recompute against the
    # test population for the matching one-based persisted fold index.
    for saved_fold in experiment.persisted_folds:
        data_module = pipeline.get_data_module(saved_fold.fold_index - 1)
        result = evaluator.evaluate_dataloader(
            saved_fold.model,
            data_module.test_loader,
        )

        fold_rows.append({
            "mpkg": mpkg_folder.name,
            "folder": str(mpkg_folder),
            "fold": saved_fold.fold_index,
            **result.metrics.to_dict(),
        })
    return fold_rows


def _create_current_data_pipeline(data_pipeline_config):
    if isinstance(data_pipeline_config, DataPipeline):
        return data_pipeline_config

    if isinstance(data_pipeline_config, dict):
        try:
            return DataPipeline(**data_pipeline_config)
        except TypeError as error:
            raise ValueError(
                "saved data pipeline configuration has invalid components"
            ) from error

    raise TypeError(
        "data pipeline configuration must be a core.data_pipeline_3.DataPipeline "
        "or persisted component dictionary"
    )


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


def _get_fold_index(metric_path):
    return int(metric_path.stem.removeprefix("metrics-fold_"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Folder containing mpkg runs")
    parser.add_argument("--output", type=Path, help="Where to save the Excel file")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-evaluate saved models instead of extracting stored metrics",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS_BY_NAME,
        metavar="METRIC",
        help=(
            "Stored metrics to include, or metrics to calculate with --recompute. "
            "Choices: "
            + ", ".join(METRICS_BY_NAME)
        ),
    )
    parser.add_argument(
        "--replacement-config",
        metavar="MODULE[:ATTRIBUTE]",
        help=(
            "Python module or file defining a replacement ExperimentConfig "
            "for --recompute. Its data and metrics settings are used; the "
            "saved mpkg model configuration remains authoritative."
        ),
    )
    args = parser.parse_args()

    if args.replacement_config is not None and not args.recompute:
        parser.error("--replacement-config requires --recompute")

    mpkg_folders = find_mpkg_folders(args.folder)
    fold_rows = []
    if args.recompute:
        replacement_config = None
        if args.replacement_config is not None:
            replacement_config = load_replacement_config(args.replacement_config)

        if args.metrics is not None:
            selected_metrics_config = MetricsConfig(
                metrics=tuple(
                    METRICS_BY_NAME[name]() for name in args.metrics
                ),
            )
        elif replacement_config is not None:
            selected_metrics_config = replacement_config.metrics_config
        else:
            selected_metric_names = [
                name for name, is_enabled in DEFAULT_METRICS.items() if is_enabled
            ]
            if not selected_metric_names:
                parser.error(
                    "Enable at least one metric in DEFAULT_METRICS or use --metrics"
                )
            selected_metrics_config = MetricsConfig(
                metrics=tuple(
                    METRICS_BY_NAME[name]() for name in selected_metric_names
                ),
            )
        for mpkg_folder in mpkg_folders:
            if replacement_config is None:
                fold_rows.extend(
                    recompute_one_mpkg(mpkg_folder, selected_metrics_config)
                )
            else:
                fold_rows.extend(
                    recompute_one_mpkg(
                        mpkg_folder,
                        selected_metrics_config,
                        replacement_config,
                    )
                )
    else:
        for mpkg_folder in mpkg_folders:
            fold_rows.extend(extract_one_mpkg(mpkg_folder, args.metrics))

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
