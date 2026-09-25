from pathlib import Path

import numpy as np
import pandas as pd



def write_metrics_workbook(
    path:Path,
    fold_metrics:list[tuple[int, dict[str, float|None]]],
) -> None:
    if not fold_metrics:
        raise ValueError("metrics workbook requires at least one fold")

    metric_names = tuple(fold_metrics[0][1])
    if any(tuple(metrics) != metric_names for _, metrics in fold_metrics[1:]):
        raise ValueError("each fold must contain the same metrics")

    fold_rows = [
        {"fold": fold_index, **metrics}
        for fold_index, metrics in fold_metrics
    ]
    fold_metrics_frame = pd.DataFrame(
        fold_rows,
        columns=["fold", *metric_names],
    )
    summary_rows = [
        {
            "metric": metric_name,
            "mean": _get_metric_mean(fold_metrics, metric_name),
            "standard_deviation": _get_metric_standard_deviation(
                fold_metrics,
                metric_name,
            ),
        }
        for metric_name in metric_names
    ]
    summary_frame = pd.DataFrame(
        summary_rows,
        columns=["metric", "mean", "standard_deviation"],
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        fold_metrics_frame.to_excel(
            writer,
            sheet_name="Fold Metrics",
            index=False,
        )
        summary_frame.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )
    return


def _get_metric_values(
    fold_metrics:list[tuple[int, dict[str, float|None]]],
    metric_name:str,
) -> list[float]:
    return [
        value
        for _, metrics in fold_metrics
        if (value := metrics[metric_name]) is not None
    ]


def _get_metric_mean(
    fold_metrics:list[tuple[int, dict[str, float|None]]],
    metric_name:str,
) -> float|None:
    values = _get_metric_values(fold_metrics, metric_name)
    return None if not values else float(np.mean(values))


def _get_metric_standard_deviation(
    fold_metrics:list[tuple[int, dict[str, float|None]]],
    metric_name:str,
) -> float|None:
    values = _get_metric_values(fold_metrics, metric_name)
    return None if not values else float(np.std(values))
