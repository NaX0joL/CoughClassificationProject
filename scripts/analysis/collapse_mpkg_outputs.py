"""Recalculate saved MPKG test outputs after collapsing classes to binary labels."""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd


# Let this script import project files when run from the project root.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


from core.data_pipeline_3.pipeline import DataPipeline
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


TARGET_CLASSES = (0, 1)
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


def main() -> None:
    parser = _create_argument_parser()
    args = parser.parse_args()

    try:
        class_map = parse_class_map(args.class_map)
        fold_rows = _recompute_mpkg_folders(
            args.folder,
            class_map,
            args.metrics,
        )
        fold_table = pd.DataFrame(fold_rows).sort_values(["mpkg", "fold"])
        summary_table = create_summary_table(fold_table)
        if args.output is not None:
            save_report(summary_table, fold_table, args.output, class_map)
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(summary_table.to_string(index=False))
    if args.output is not None:
        print(f"Saved report: {args.output}")
    return


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recalculate binary metrics from saved MPKG model folds.",
    )
    parser.add_argument("folder", type=Path, help="Folder containing saved MPKG runs")
    parser.add_argument(
        "--class-map",
        required=True,
        help="Source-to-target mapping, for example 0:0,1:0,2:1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Excel (.xlsx) or JSON report path",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS_BY_NAME,
        metavar="METRIC",
        help="Metrics to calculate (default: metrics saved in the MPKG config)",
    )
    return parser


def parse_class_map(value:str) -> dict[int, int]:
    """Parse ``SOURCE:TARGET`` pairs from the command line."""
    if not value.strip():
        raise ValueError("class map must not be empty")

    class_map:dict[int, int] = {}
    for pair in value.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            raise ValueError(
                "class map must use SOURCE:TARGET pairs separated by commas"
            )

        source_text, target_text = pair.split(":", maxsplit=1)
        if not source_text.strip() or not target_text.strip():
            raise ValueError(
                "class map must use SOURCE:TARGET pairs separated by commas"
            )

        try:
            source_class = int(source_text.strip())
            target_class = int(target_text.strip())
        except ValueError as error:
            raise ValueError("class map classes must be integers") from error

        if source_class in class_map:
            raise ValueError(f"source class {source_class} appears more than once")
        class_map[source_class] = target_class

    if set(class_map.values()) != set(TARGET_CLASSES):
        raise ValueError("class map must contain both target classes 0 and 1")
    return class_map


def _recompute_mpkg_folders(
    folder:Path,
    class_map:Mapping[int, int],
    metric_names:list[str]|None,
) -> list[dict[str, object]]:
    mpkg_folders = find_mpkg_folders(folder)
    metrics_config = _create_metrics_config(metric_names)
    fold_rows = []

    print(f"Found {len(mpkg_folders)} mpkg(s)!")
    for index, mpkg_folder in enumerate(mpkg_folders, start=1):
        print(
            f"Processing {index}/{len(mpkg_folders)}: {mpkg_folder.name}",
            flush=True,
        )
        fold_rows.extend(recompute_one_mpkg(
            mpkg_folder,
            class_map,
            metrics_config,
        ))
        
    return fold_rows


def find_mpkg_folders(folder:Path) -> list[Path]:
    """Find MPKG run directories by their marker file."""
    mpkg_folders = sorted(marker.parent for marker in folder.rglob("__mpkg__.py"))
    if not mpkg_folders:
        raise FileNotFoundError(f"no MPKG folders found in: {folder}")
    return mpkg_folders


def _create_metrics_config(metric_names:list[str]|None) -> MetricsConfig|None:
    if metric_names is None:
        return None
    return MetricsConfig(metrics=tuple(
        METRICS_BY_NAME[name]() for name in metric_names
    ))


def recompute_one_mpkg(
    mpkg_folder:Path,
    class_map:Mapping[int, int],
    metrics_config:MetricsConfig|None=None,
) -> list[dict[str, object]]:
    """Rerun every saved fold and return its binary metric row."""
    experiment = ExperimentOrchestrator.load(mpkg_folder)
    pipeline = _create_current_data_pipeline(experiment.config.data_pipeline_config)
    saved_fold_indices = [
        saved_fold.fold_index for saved_fold in experiment.persisted_folds
    ]
    expected_fold_indices = list(range(1, len(pipeline) + 1))
    if sorted(saved_fold_indices) != expected_fold_indices:
        raise ValueError(f"saved models and data folds do not match: {mpkg_folder}")

    if metrics_config is None:
        metrics_config = experiment.config.metrics_config
    evaluator = ModelEvaluator(metrics_config)
    fold_rows = []

    for saved_fold in experiment.persisted_folds:
        data_module = pipeline.get_data_module(saved_fold.fold_index - 1)
        labels, predictions, probabilities, _ = evaluator._collect_outputs(
            saved_fold.model,
            data_module.test_loader,
        )
        collapsed_labels, collapsed_predictions, collapsed_probabilities = collapse_predictions(
            labels,
            predictions,
            probabilities,
            class_map,
        )
        metrics = evaluator.metrics_calculator.calculate(
            collapsed_labels,
            collapsed_predictions,
            collapsed_probabilities,
            np.asarray(TARGET_CLASSES),
        )
        fold_rows.append({
            "mpkg": mpkg_folder.name,
            "fold": saved_fold.fold_index,
            **metrics.to_dict(),
        })

    return fold_rows


def collapse_predictions(
    labels:np.ndarray,
    predictions:np.ndarray,
    probabilities:np.ndarray,
    class_map:Mapping[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse labels and probabilities, then predict from collapsed scores.

    ``predictions`` is accepted to make the distinction from the original
    model output explicit.  It is deliberately not remapped: summing source
    probabilities can change the winning target class.
    """
    probability_array = np.asarray(probabilities)
    if probability_array.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    validated_map = validate_class_map(class_map, probability_array.shape[1])

    collapsed_labels = collapse_labels(
        labels,
        validated_map,
        source_class_count=probability_array.shape[1],
    )
    collapsed_probabilities = collapse_probabilities(
        probability_array,
        validated_map,
    )
    prediction_array = np.asarray(predictions)
    if prediction_array.ndim != 1:
        raise ValueError("predictions must be a one-dimensional array")
    if len(prediction_array) != len(collapsed_probabilities):
        raise ValueError(
            "labels, predictions, and probabilities must have equal lengths"
        )

    collapsed_predictions = collapsed_probabilities.argmax(axis=1)
    return collapsed_labels, collapsed_predictions, collapsed_probabilities


def validate_class_map(
    class_map:Mapping[int, int],
    source_class_count:int,
) -> dict[int, int]:
    """Validate that every model output column has one binary destination."""
    if source_class_count < 2:
        raise ValueError("the model must have at least two output classes")

    normalized_map = dict(class_map)
    expected_classes = set(range(source_class_count))
    actual_classes = set(normalized_map)
    missing_classes = expected_classes - actual_classes
    extra_classes = actual_classes - expected_classes
    if missing_classes or extra_classes:
        details = []
        if missing_classes:
            details.append(f"missing source classes {sorted(missing_classes)}")
        if extra_classes:
            details.append(f"unknown source classes {sorted(extra_classes)}")
        detail_text = "; ".join(details)
        raise ValueError(
            "class map must map every original output column exactly once ("
            + detail_text
            + ")"
        )

    if set(normalized_map.values()) != set(TARGET_CLASSES):
        raise ValueError("class map must map to both target classes 0 and 1")
    if any(target not in TARGET_CLASSES for target in normalized_map.values()):
        raise ValueError("class map targets must be 0 or 1")
    return normalized_map


def collapse_probabilities(
    probabilities:np.ndarray,
    class_map:Mapping[int, int],
) -> np.ndarray:
    """Sum source probability columns into the two target-class columns."""
    probability_array = np.asarray(probabilities)
    if probability_array.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array")
    validated_map = validate_class_map(class_map, probability_array.shape[1])
    collapsed = np.zeros(
        (probability_array.shape[0], len(TARGET_CLASSES)),
        dtype=probability_array.dtype,
    )

    for source_class, target_class in validated_map.items():
        collapsed[:, target_class] += probability_array[:, source_class]
    return collapsed


def collapse_labels(
    labels:np.ndarray,
    class_map:Mapping[int, int],
    source_class_count:int|None=None,
) -> np.ndarray:
    """Remap labels using the same source-to-target class map."""
    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if source_class_count is None:
        if not class_map:
            raise ValueError("class map must not be empty")
        source_class_count = max(class_map) + 1
    validated_map = validate_class_map(class_map, source_class_count)

    try:
        integer_labels = label_array.astype(int)
    except (TypeError, ValueError) as error:
        raise ValueError("labels must contain integer source classes") from error
    if not np.equal(label_array, integer_labels).all():
        raise ValueError("labels must contain integer source classes")
    if not np.isin(integer_labels, list(validated_map)).all():
        raise ValueError("labels contain a source class absent from the class map")
    return np.asarray([validated_map[int(label)] for label in integer_labels])


def _create_current_data_pipeline(data_pipeline_config:object) -> DataPipeline:
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
        "data pipeline configuration must be a DataPipeline or component dictionary"
    )


def create_summary_table(fold_table:pd.DataFrame) -> pd.DataFrame:
    """Average binary metrics across folds."""
    metric_columns = [
        column for column in fold_table.columns
        if column not in {"mpkg", "fold"}
    ]
    summary_table = fold_table.groupby(
        "mpkg",
        as_index=False,
    )[metric_columns].mean()
    fold_counts = fold_table.groupby("mpkg")["fold"].nunique()
    summary_table.insert(
        2,
        "number_of_folds",
        summary_table["mpkg"].map(fold_counts),
    )
    return summary_table


def save_report(
    summary_table:pd.DataFrame,
    fold_table:pd.DataFrame,
    output_path:Path,
    class_map:Mapping[int, int],
) -> None:
    """Write a new Excel or JSON report without touching the source MPKG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        report = {
            "class_map": {
                str(source): target for source, target in class_map.items()
            },
            "summary": summary_table.to_dict(orient="records"),
            "fold_metrics": fold_table.to_dict(orient="records"),
        }
        with output_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
            report_file.write("\n")
        return

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_table.to_excel(writer, sheet_name="mpkg_summary", index=False)
        fold_table.to_excel(writer, sheet_name="fold_metrics", index=False)
    return


if __name__ == "__main__":
    main()
