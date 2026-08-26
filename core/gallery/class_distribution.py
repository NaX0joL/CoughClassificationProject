import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..data_pipeline.data_pipeline_config import DataPipelineConfig
from ..data_pipeline.intermediary import DataSplit, Example
from ..metrics.evaluation import BINARY_CLASS_NAMES

from .gallery_directory import GALLERY_DIRECTORY, resolve_gallery_directory



CLASS_DISTRIBUTION_FILE_NAME = "class_distribution.png"
CLASS_DISTRIBUTION_JSON_FILE_NAME = "class_distribution.json"
OVERALL_SPLIT_NAME = "overall"
TEST_SPLIT_NAME = "test"
TRAIN_SPLIT_SUFFIX = "-train"
VALIDATION_SPLIT_SUFFIX = "-val"
CLASS_BAR_COLORS = ["tab:blue", "tab:orange"]
COUNT_ANNOTATION_FONTSIZE = 8



class ClassDistributionGenerator:

    def __init__(
        self,
        data_pipeline_config:DataPipelineConfig,
        gallery_directory:Path=GALLERY_DIRECTORY,
        class_names:dict[int, str]|None=None,
        regenerate:bool=False,
    ) -> None:
        self.data_pipeline_config = data_pipeline_config
        self.gallery_directory = gallery_directory
        self.class_names = class_names
        self.regenerate = regenerate
        return

    def generate(self, data_split:DataSplit) -> Path:
        gallery_dir = resolve_gallery_directory(
            self.data_pipeline_config,
            self.gallery_directory,
        )
        gallery_dir.mkdir(parents=True, exist_ok=True)

        figure_path = gallery_dir / CLASS_DISTRIBUTION_FILE_NAME
        if not self.regenerate and figure_path.exists():
            return figure_path

        class_names = (
            self.class_names
            if self.class_names is not None
            else BINARY_CLASS_NAMES
        )
        counts = collect_class_counts(data_split)

        save_class_distribution_figure(
            counts=counts,
            class_names=class_names,
            path=figure_path,
        )
        save_class_distribution_json(
            counts=counts,
            class_names=class_names,
            path=gallery_dir / CLASS_DISTRIBUTION_JSON_FILE_NAME,
        )
        return figure_path



def collect_class_counts(data_split:DataSplit) -> dict[str, dict[int, int]]:
    counts = {
        OVERALL_SPLIT_NAME: _count_labels(_collect_all_examples(data_split)),
    }

    for fold_index, fold in enumerate(data_split.development_folds, start=1):
        fold_prefix = f"fold_{fold_index}"
        counts[f"{fold_prefix}{TRAIN_SPLIT_SUFFIX}"] = _count_labels(
            fold.train_dataset.examples,
        )
        counts[f"{fold_prefix}{VALIDATION_SPLIT_SUFFIX}"] = _count_labels(
            fold.validation_dataset.examples,
        )

    counts[TEST_SPLIT_NAME] = _count_labels(data_split.test_dataset.examples)

    if sum(counts[OVERALL_SPLIT_NAME].values()) == 0:
        raise ValueError("data_split must contain at least one example")

    return counts


def save_class_distribution_figure(
    counts:dict[str, dict[int, int]],
    class_names:dict[int, str]|None,
    path:Path,
) -> None:
    figure, (overall_axis, splits_axis) = plt.subplots(
        nrows=2,
        figsize=(12.0, 9.0),
        layout="constrained",
    )
    figure.suptitle("Cough segment class distribution", fontsize=16)
    _plot_overall_counts(overall_axis, counts[OVERALL_SPLIT_NAME], class_names)
    _plot_per_split_counts(splits_axis, counts, class_names)
    figure.savefig(path)
    plt.close(figure)
    return


def save_class_distribution_json(
    counts:dict[str, dict[int, int]],
    class_names:dict[int, str]|None,
    path:Path,
) -> None:
    labels = sorted({label for split in counts.values() for label in split})
    payload = {
        "class_names": {
            str(label): _label_name(label, class_names)
            for label in labels
        },
        "splits": {
            name: _summarize_split(split_counts, class_names)
            for name, split_counts in counts.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return


def _collect_all_examples(data_split:DataSplit) -> list[Example]:
    examples:list[Example] = []
    for fold in data_split.development_folds:
        examples.extend(fold.train_dataset.examples)
        examples.extend(fold.validation_dataset.examples)
    examples.extend(data_split.test_dataset.examples)
    return examples


def _count_labels(examples:list[Example]) -> dict[int, int]:
    counts:dict[int, int] = {}
    for example in examples:
        counts[example.label] = counts.get(example.label, 0) + 1
    return counts


def _summarize_split(
    split_counts:dict[int, int],
    class_names:dict[int, str]|None,
) -> dict[str, object]:
    total = sum(split_counts.values())
    names = {label: _label_name(label, class_names) for label in split_counts}
    ordered_items = sorted(split_counts.items())
    return {
        "total": total,
        "counts": {names[label]: count for label, count in ordered_items},
        "percentages": {
            names[label]: round(100.0 * count / total, 2) if total else 0.0
            for label, count in ordered_items
        },
    }


def _plot_overall_counts(
    axis,
    counts:dict[int, int],
    class_names:dict[int, str]|None,
) -> None:
    labels = sorted(counts)
    values = [counts[label] for label in labels]
    total = sum(values)
    positions = np.arange(len(labels))
    bars = axis.bar(
        positions,
        values,
        color=[_bar_color(label, labels) for label in labels],
    )
    axis.bar_label(bars, labels=_overall_bar_annotations(values, total))
    axis.set_xticks(positions, [_label_name(label, class_names) for label in labels])
    axis.set(ylabel="Segments", title=f"Overall ({total} segments)")
    axis.grid(axis="y", alpha=0.25)
    axis.margins(y=0.15)
    return


def _plot_per_split_counts(
    axis,
    counts:dict[str, dict[int, int]],
    class_names:dict[int, str]|None,
) -> None:
    split_names = list(counts)
    labels = sorted({label for split in counts.values() for label in split})
    positions = np.arange(len(split_names))
    bar_width = 0.8 / max(len(labels), 1)

    for offset, label in enumerate(labels):
        values = [counts[name].get(label, 0) for name in split_names]
        shift = (offset - (len(labels) - 1) / 2.0) * bar_width
        bars = axis.bar(
            positions + shift,
            values,
            width=bar_width,
            color=_bar_color(label, labels),
            label=_label_name(label, class_names),
        )
        axis.bar_label(
            bars,
            rotation=90,
            padding=2,
            fontsize=COUNT_ANNOTATION_FONTSIZE,
        )

    axis.set_xticks(positions, split_names, rotation=30, ha="right")
    axis.set(ylabel="Segments", title="Per split")
    axis.grid(axis="y", alpha=0.25)
    axis.margins(y=0.2)
    axis.legend()
    return


def _overall_bar_annotations(values:list[int], total:int) -> list[str]:
    if total == 0:
        return [str(value) for value in values]
    return [
        f"{value} ({100.0 * value / total:.1f}%)"
        for value in values
    ]


def _bar_color(label:int, labels:list[int]) -> str:
    index = labels.index(label)
    return CLASS_BAR_COLORS[index % len(CLASS_BAR_COLORS)]


def _label_name(label:int, class_names:dict[int, str]|None) -> str:
    if class_names and label in class_names:
        return class_names[label]
    return str(label)
