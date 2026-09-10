"""Plot per-recording prediction timelines from a saved v3 experiment."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# Let this script import project files when run from the project root.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


from core.data_pipeline_3.example_constructor.example_constructor import (
    ExampleConstructor,
)
from core.data_pipeline_3.example_constructor._utils.series_segmenter import (
    SlidingWindowSegmenter,
)
from core.data_pipeline_3.intermediary import SeriesSegment, SourceRecord
from core.data_pipeline_3.pipeline import DataPipeline
from core.data_pipeline_3.source_reader.elderly_cough_audio.source_reader import (
    SourceReader,
)
from core.experiment import ExperimentOrchestrator, PersistedFold
from core.model import FullModel



CLASS_NAMES = {
    0: "No cough",
    1: "Non-infectious",
    2: "Infectious",
}
DEFAULT_OUTPUT_PARENT = Path("outputs/analysis/audio_timelines")



def main() -> None:
    parser = _create_argument_parser()
    args = parser.parse_args()

    try:
        plot_mpkg_audio_timelines(
            mpkg_path=args.mpkg_path,
            stride=args.stride,
            fold_selection=args.fold_selection,
            output_directory=args.output_dir,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))

    return


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot predicted labels over time for every eligible audio file.",
    )
    parser.add_argument("mpkg_path", type=Path, help="Path to one saved mpkg")
    parser.add_argument(
        "--stride",
        type=int,
        required=True,
        help="Sliding-window stride in audio samples",
    )
    parser.add_argument(
        "--fold-selection",
        choices=("validation", "test"),
        default="validation",
        help=(
            "Best fold by lowest validation loss or most test-metric wins "
            "(default: validation)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder for PNG files (default: outputs/analysis/...)",
    )
    return parser


def plot_mpkg_audio_timelines(
    mpkg_path:Path,
    stride:int,
    fold_selection:str="validation",
    output_directory:Path|None=None,
) -> None:
    experiment = ExperimentOrchestrator.load(mpkg_path)
    example_constructor, source_reader, batch_size = _get_v3_components(
        experiment,
    )
    window_size = example_constructor.validation_segmenter.window_size
    _validate_stride(stride, window_size)

    selected_fold, selection_reason = select_best_fold(
        experiment.persisted_folds,
        fold_selection,
    )
    output_directory = output_directory or _get_default_output_directory(
        mpkg_path,
        fold_selection,
        stride,
    )
    source_records = source_reader.get_source_records()
    if not source_records:
        raise ValueError("the saved source reader found no eligible audio files")

    model = selected_fold.model.to(experiment.device)
    model.eval()

    print(f"selected fold: {selected_fold.fold_index} ({selection_reason})")
    plotted_count, skipped_count = _plot_source_records(
        source_records=source_records,
        source_root=source_reader.source_data_parent_directory,
        output_directory=output_directory,
        example_constructor=example_constructor,
        window_size=window_size,
        stride=stride,
        model=model,
        device=experiment.device,
        batch_size=batch_size,
    )

    print(f"output directory: {output_directory}")
    print(f"files plotted: {plotted_count}")
    print(f"files skipped: {skipped_count}")
    return


def _get_v3_components(
    experiment:ExperimentOrchestrator,
) -> tuple[ExampleConstructor, SourceReader, int]:
    persisted_pipeline = experiment.config.data_pipeline_config
    if isinstance(persisted_pipeline, DataPipeline):
        example_constructor = persisted_pipeline.example_constructor
        source_reader = persisted_pipeline.source_reader
        batch_size = persisted_pipeline.batch_size
    elif isinstance(persisted_pipeline, dict):
        example_constructor = persisted_pipeline.get("example_constructor")
        source_reader = persisted_pipeline.get("source_reader")
        batch_size = persisted_pipeline.get("batch_size")
    else:
        raise TypeError("mpkg does not contain a v3 data pipeline")

    if not isinstance(example_constructor, ExampleConstructor):
        raise TypeError("mpkg must use the v3 example constructor")
    if not isinstance(
        example_constructor.validation_segmenter,
        SlidingWindowSegmenter,
    ):
        raise TypeError("mpkg validation segmenter must use sliding windows")

    if not isinstance(source_reader, SourceReader):
        raise TypeError("mpkg must use the v3 elderly cough audio source reader")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("mpkg batch size must be greater than 0")
    if example_constructor.audio_file_reader.sampling_rate <= 0:
        raise ValueError("mpkg sampling rate must be greater than 0")

    return example_constructor, source_reader, batch_size


def _validate_stride(stride:int, window_size:int) -> None:
    if stride <= 0:
        raise ValueError("stride must be greater than 0")
    if stride > window_size:
        raise ValueError(
            f"stride must not exceed the saved window size ({window_size})",
        )
    return


def select_best_fold(
    folds:list[PersistedFold],
    fold_selection:str,
) -> tuple[PersistedFold, str]:
    if not folds:
        raise ValueError("mpkg does not contain any saved folds")

    if fold_selection == "validation":
        return _select_fold_by_validation_loss(folds)
    if fold_selection == "test":
        return _select_fold_by_test_metrics(folds)
    raise ValueError(f"unknown fold selection: {fold_selection}")


def _select_fold_by_validation_loss(
    folds:list[PersistedFold],
) -> tuple[PersistedFold, str]:
    fold_losses = []
    for fold in folds:
        loss = fold.loss_log.best_validation_loss
        if loss is None or not math.isfinite(loss):
            raise ValueError(
                f"fold {fold.fold_index} has no finite best validation loss",
            )
        fold_losses.append((loss, fold.fold_index, fold))

    loss, _, selected_fold = min(fold_losses)
    return selected_fold, f"best validation loss {loss:.6g}"


def _select_fold_by_test_metrics(
    folds:list[PersistedFold],
) -> tuple[PersistedFold, str]:
    metric_names = tuple(folds[0].validation_metrics)
    if not metric_names:
        raise ValueError("saved folds do not contain test metrics")

    expected_metric_names = set(metric_names)
    for fold in folds:
        if set(fold.validation_metrics) != expected_metric_names:
            raise ValueError("saved folds must contain the same test metrics")

    metric_names = tuple(
        metric_name
        for metric_name in metric_names
        if all(
            math.isfinite(fold.validation_metrics[metric_name])
            for fold in folds
        )
    )
    if not metric_names:
        raise ValueError("saved folds have no finite test metrics to compare")

    win_counts = [0 for fold in folds]
    rank_totals = [0.0 for fold in folds]
    for metric_name in metric_names:
        metric_values = [
            fold.validation_metrics[metric_name]
            for fold in folds
        ]
        highest_value = max(metric_values)
        for index, metric_value in enumerate(metric_values):
            if metric_value == highest_value:
                win_counts[index] += 1

        metric_ranks = _get_descending_average_ranks(metric_values)
        rank_totals = [
            total + rank
            for total, rank in zip(rank_totals, metric_ranks, strict=True)
        ]

    mean_ranks = [total / len(metric_names) for total in rank_totals]
    selected_index = min(
        range(len(folds)),
        key=lambda index: (
            -win_counts[index],
            mean_ranks[index],
            folds[index].fold_index,
        ),
    )
    selected_fold = folds[selected_index]
    reason = (
        f"{win_counts[selected_index]} metric wins, "
        f"mean rank {mean_ranks[selected_index]:.2f}"
    )
    return selected_fold, reason


def _get_descending_average_ranks(values:list[float]) -> list[float]:
    ranks = [0.0 for value in values]
    next_rank = 1

    for value in sorted(set(values), reverse=True):
        tied_indices = [
            index
            for index, candidate in enumerate(values)
            if candidate == value
        ]
        final_rank = next_rank + len(tied_indices) - 1
        average_rank = (next_rank + final_rank) / 2
        for index in tied_indices:
            ranks[index] = average_rank
        next_rank = final_rank + 1

    return ranks


def _get_default_output_directory(
    mpkg_path:Path,
    fold_selection:str,
    stride:int,
) -> Path:
    return (
        DEFAULT_OUTPUT_PARENT
        / mpkg_path.name
        / f"{fold_selection}_stride_{stride}"
    )


def _plot_source_records(
    source_records:list[SourceRecord],
    source_root:Path,
    output_directory:Path,
    example_constructor:ExampleConstructor,
    window_size:int,
    stride:int,
    model:FullModel,
    device:torch.device,
    batch_size:int,
) -> tuple[int, int]:
    plotted_count = 0
    skipped_count = 0

    for source_record in source_records:
        try:
            waveform = example_constructor.audio_file_reader.read(
                source_record.audio_path,
            )
        except ValueError as error:
            warning = f"warning: skipped {source_record.audio_path}: {error}"
            print(warning, file=sys.stderr)
            skipped_count += 1
            continue

        segments = _segment_around_time_steps(
            waveform,
            window_size,
            stride,
        )
        transformed_segments = _transform_segments(
            segments,
            example_constructor,
        )
        class_probabilities = _predict_segment_probabilities(
            model,
            transformed_segments,
            device,
            batch_size,
        )
        sampling_rate = example_constructor.audio_file_reader.sampling_rate
        center_times = _get_window_center_times(
            segments,
            sampling_rate,
        )
        output_path = _get_audio_output_path(
            source_record.audio_path,
            source_root,
            output_directory,
        )
        _save_prediction_timeline(
            waveform=waveform,
            sampling_rate=sampling_rate,
            center_times=center_times,
            class_probabilities=class_probabilities,
            duration_seconds=len(waveform) / sampling_rate,
            audio_name=source_record.audio_path.name,
            actual_label=source_record.label,
            output_path=output_path,
        )
        plotted_count += 1

    return plotted_count, skipped_count


def _segment_around_time_steps(
    waveform:np.ndarray,
    window_size:int,
    stride:int,
) -> list[SeriesSegment]:
    half_window_size = window_size // 2
    segments = []

    for center_index in range(0, len(waveform), stride):
        segment_start = center_index - half_window_size
        segment_end = segment_start + window_size
        source_start = max(0, segment_start)
        source_end = min(len(waveform), segment_end)
        segment_value = waveform[source_start:source_end].copy()
        segment_value = np.pad(
            segment_value,
            pad_width=(
                max(0, -segment_start),
                max(0, segment_end - len(waveform)),
            ),
            mode="constant",
            constant_values=0,
        )
        segments.append(SeriesSegment(
            value=segment_value,
            original_index=(segment_start, segment_end),
        ))

    return segments


def _transform_segments(
    segments:list[SeriesSegment],
    example_constructor:ExampleConstructor,
) -> list[SeriesSegment]:
    transformed_segments = segments
    for transformer in example_constructor.transformers:
        transformed_segments = transformer.transform(transformed_segments)
    return transformed_segments


def _predict_segment_classes(
    model:FullModel,
    segments:list[SeriesSegment],
    device:torch.device,
    batch_size:int,
) -> list[int]:
    predictions = []

    with torch.inference_mode():
        for start in range(0, len(segments), batch_size):
            batch_segments = segments[start:start + batch_size]
            values = torch.as_tensor(
                np.stack([segment.value for segment in batch_segments]),
                dtype=torch.float32,
                device=device,
            )
            batch_predictions = model.predict_classes(values)
            predictions.extend(batch_predictions.cpu().tolist())

    return predictions


def _predict_segment_probabilities(
    model:FullModel,
    segments:list[SeriesSegment],
    device:torch.device,
    batch_size:int,
) -> np.ndarray:
    probability_batches = []

    with torch.inference_mode():
        for start in range(0, len(segments), batch_size):
            batch_segments = segments[start:start + batch_size]
            values = torch.as_tensor(
                np.stack([segment.value for segment in batch_segments]),
                dtype=torch.float32,
                device=device,
            )
            batch_probabilities = model.predict_probabilities(values)
            probability_batches.append(batch_probabilities.cpu().numpy())

    if not probability_batches:
        return np.empty((0, 0), dtype=np.float32)

    return np.concatenate(probability_batches)


def _get_window_center_times(
    segments:list[SeriesSegment],
    sampling_rate:int,
) -> np.ndarray:
    return np.asarray([
        (
            segment.original_index[0]
            + len(segment.value) // 2
        ) / sampling_rate
        for segment in segments
    ])


def _get_audio_output_path(
    audio_path:Path,
    source_root:Path,
    output_directory:Path,
) -> Path:
    try:
        relative_audio_path = audio_path.resolve().relative_to(
            source_root.resolve(),
        )
    except ValueError as error:
        message = f"audio file is outside the saved source folder: {audio_path}"
        raise ValueError(message) from error

    flattened_name = "__".join(relative_audio_path.parts)
    return output_directory / f"{flattened_name}.png"


def _save_prediction_timeline(
    waveform:np.ndarray,
    sampling_rate:int,
    center_times:np.ndarray,
    class_probabilities:np.ndarray,
    duration_seconds:float,
    audio_name:str,
    actual_label:int,
    output_path:Path,
) -> None:
    if class_probabilities.size == 0:
        raise ValueError(f"no prediction windows were created for {audio_name}")
    if class_probabilities.ndim != 2:
        raise ValueError("class probabilities must be a two-dimensional array")
    if len(class_probabilities) != len(center_times):
        raise ValueError(
            "class probabilities must match the number of prediction windows",
        )

    step_edges = _get_step_edges(center_times, duration_seconds)
    waveform_times = np.arange(len(waveform)) / sampling_rate
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    waveform_axis, prediction_axis = axes

    waveform_axis.plot(waveform_times, waveform, linewidth=0.6)
    waveform_axis.set_ylabel("Amplitude")
    waveform_axis.grid(axis="x", alpha=0.25)

    for class_index in range(class_probabilities.shape[1]):
        class_name = CLASS_NAMES.get(class_index, f"Class {class_index}")
        prediction_axis.stairs(
            class_probabilities[:, class_index],
            step_edges,
            baseline=None,
            linewidth=1.0,
            alpha=0.9,
            label=class_name,
        )
    prediction_axis.set_xlim(0, duration_seconds)
    prediction_axis.set_ylim(0, 1)
    prediction_axis.set_xlabel("Time (seconds)")
    prediction_axis.set_ylabel("Probability")
    prediction_axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    prediction_axis.grid(alpha=0.25)
    actual_class_name = CLASS_NAMES.get(actual_label, f"Class {actual_label}")
    figure.suptitle(f"{audio_name} — Actual: {actual_class_name}")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return


def _get_step_edges(
    center_times:np.ndarray,
    duration_seconds:float,
) -> np.ndarray:
    if len(center_times) == 1:
        return np.asarray([0.0, duration_seconds])

    middle_edges = (center_times[:-1] + center_times[1:]) / 2
    return np.concatenate((
        np.asarray([0.0]),
        middle_edges,
        np.asarray([duration_seconds]),
    ))


if __name__ == "__main__":
    main()
