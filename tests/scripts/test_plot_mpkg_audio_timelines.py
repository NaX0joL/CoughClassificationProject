from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from core.data_pipeline_3.intermediary import SeriesSegment, SourceRecord
from scripts.analysis import plot_mpkg_audio_timelines as timeline_script
from scripts.analysis.plot_mpkg_audio_timelines import (
    _get_audio_output_path,
    _get_step_edges,
    _get_v3_components,
    _get_window_center_times,
    _plot_source_records,
    _predict_segment_probabilities,
    _predict_segment_classes,
    _save_prediction_timeline,
    _segment_around_time_steps,
    _transform_segments,
    _validate_stride,
    select_best_fold,
)



def test_validation_selection_uses_lowest_best_loss() -> None:
    folds = [
        _make_fold(1, validation_loss=0.4),
        _make_fold(2, validation_loss=0.2),
        _make_fold(3, validation_loss=0.3),
    ]

    selected_fold, reason = select_best_fold(folds, "validation")

    assert selected_fold.fold_index == 2
    assert reason == "best validation loss 0.2"


def test_validation_selection_uses_lowest_fold_index_to_break_tie() -> None:
    folds = [
        _make_fold(2, validation_loss=0.2),
        _make_fold(1, validation_loss=0.2),
    ]

    selected_fold, _ = select_best_fold(folds, "validation")

    assert selected_fold.fold_index == 1


def test_test_selection_uses_metric_wins() -> None:
    folds = [
        _make_fold(1, metrics={"accuracy": 0.9, "f1": 0.6}),
        _make_fold(2, metrics={"accuracy": 0.8, "f1": 0.9}),
        _make_fold(3, metrics={"accuracy": 0.7, "f1": 0.5}),
    ]

    selected_fold, reason = select_best_fold(folds, "test")

    assert selected_fold.fold_index == 1
    assert reason == "1 metric wins, mean rank 1.50"


def test_test_selection_uses_mean_rank_after_equal_wins() -> None:
    folds = [
        _make_fold(3, metrics={"a": 1.0, "b": 0.9, "c": 0.9}),
        _make_fold(1, metrics={"a": 0.9, "b": 1.0, "c": 0.8}),
        _make_fold(2, metrics={"a": 0.8, "b": 0.8, "c": 1.0}),
    ]

    selected_fold, reason = select_best_fold(folds, "test")

    assert selected_fold.fold_index == 3
    assert reason == "1 metric wins, mean rank 1.67"


def test_test_selection_uses_lowest_fold_index_after_full_tie() -> None:
    folds = [
        _make_fold(3, metrics={"accuracy": 0.9}),
        _make_fold(1, metrics={"accuracy": 0.9}),
    ]

    selected_fold, _ = select_best_fold(folds, "test")

    assert selected_fold.fold_index == 1


def test_test_selection_ignores_metrics_that_are_not_finite() -> None:
    folds = [
        _make_fold(1, metrics={"roc_auc": float("nan"), "accuracy": 0.8}),
        _make_fold(2, metrics={"roc_auc": float("nan"), "accuracy": 0.9}),
    ]

    selected_fold, reason = select_best_fold(folds, "test")

    assert selected_fold.fold_index == 2
    assert reason == "1 metric wins, mean rank 1.00"


def test_stride_must_fit_inside_window() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        _validate_stride(0, window_size=8)

    with pytest.raises(ValueError, match="must not exceed"):
        _validate_stride(9, window_size=8)


def test_persisted_v3_pipeline_dictionary_is_accepted() -> None:
    example_constructor = timeline_script.ExampleConstructor.__new__(
        timeline_script.ExampleConstructor,
    )
    example_constructor.validation_segmenter = (
        timeline_script.SlidingWindowSegmenter(8, 4)
    )
    example_constructor.audio_file_reader = SimpleNamespace(sampling_rate=16_000)
    source_reader = timeline_script.SourceReader.__new__(timeline_script.SourceReader)
    experiment = SimpleNamespace(
        config=SimpleNamespace(
            data_pipeline_config={
                "source_reader": source_reader,
                "example_constructor": example_constructor,
                "batch_size": 128,
            },
        ),
    )

    components = _get_v3_components(experiment)

    assert components == (example_constructor, source_reader, 128)


def test_transform_segments_applies_saved_transformers_in_order() -> None:
    segments = [SeriesSegment(np.asarray([1.0, 2.0]), (0, 2))]
    first_transformer = _AddingTransformer(1)
    second_transformer = _AddingTransformer(2)
    example_constructor = SimpleNamespace(
        transformers=[first_transformer, second_transformer],
    )

    transformed = _transform_segments(segments, example_constructor)

    np.testing.assert_array_equal(transformed[0].value, [4.0, 5.0])
    assert transformed[0].original_index == (0, 2)


def test_predict_segment_classes_batches_model_inputs() -> None:
    segments = [
        SeriesSegment(np.asarray([value]), (value, value + 1))
        for value in range(5)
    ]
    model = _ThresholdModel()

    predictions = _predict_segment_classes(
        model=model,
        segments=segments,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert predictions == [0, 0, 1, 1, 1]
    assert model.batch_sizes == [2, 2, 1]


def test_predict_segment_probabilities_batches_model_inputs() -> None:
    segments = [
        SeriesSegment(np.asarray([value]), (value, value + 1))
        for value in range(5)
    ]
    model = _ProbabilityModel()

    probabilities = _predict_segment_probabilities(
        model=model,
        segments=segments,
        device=torch.device("cpu"),
        batch_size=2,
    )

    np.testing.assert_allclose(
        probabilities,
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.5, 0.4, 0.1],
            [0.4, 0.5, 0.1],
        ],
    )
    assert model.batch_sizes == [2, 2, 1]


def test_windows_are_centered_on_time_steps_with_boundary_padding() -> None:
    waveform = np.arange(5)

    segments = _segment_around_time_steps(
        waveform,
        window_size=4,
        stride=2,
    )

    assert [segment.original_index for segment in segments] == [
        (-2, 2),
        (0, 4),
        (2, 6),
    ]
    np.testing.assert_array_equal(segments[0].value, [0, 0, 0, 1])
    np.testing.assert_array_equal(segments[1].value, [0, 1, 2, 3])
    np.testing.assert_array_equal(segments[2].value, [2, 3, 4, 0])


def test_window_centers_begin_at_zero_and_follow_stride() -> None:
    segments = [
        SeriesSegment(np.zeros(8), (-4, 4)),
        SeriesSegment(np.zeros(8), (0, 8)),
        SeriesSegment(np.zeros(8), (4, 12)),
    ]

    center_times = _get_window_center_times(
        segments,
        sampling_rate=2,
    )

    np.testing.assert_array_equal(center_times, [0.0, 2.0, 4.0])


def test_step_edges_cover_entire_audio_duration() -> None:
    edges = _get_step_edges(
        center_times=np.asarray([2.0, 3.5]),
        duration_seconds=5.0,
    )

    np.testing.assert_array_equal(edges, [0.0, 2.75, 5.0])


def test_audio_output_path_flattens_relative_path_without_losing_identity(
    tmp_path:Path,
) -> None:
    source_root = tmp_path / "source"
    audio_path = source_root / "patient-1" / "session-2" / "cough.wav"
    output_directory = tmp_path / "plots"

    output_path = _get_audio_output_path(
        audio_path,
        source_root,
        output_directory,
    )

    assert output_path == (
        output_directory / "patient-1__session-2__cough.wav.png"
    )
    assert output_path.parent == output_directory


def test_timeline_plot_overlays_named_class_probabilities(
    monkeypatch:pytest.MonkeyPatch,
    tmp_path:Path,
) -> None:
    figure = Mock()
    waveform_axis = Mock()
    prediction_axis = Mock()
    subplots = Mock(
        return_value=(figure, np.asarray([waveform_axis, prediction_axis])),
    )
    monkeypatch.setattr(timeline_script.plt, "subplots", subplots)
    monkeypatch.setattr(timeline_script.plt, "close", Mock())
    output_path = tmp_path / "timeline.png"
    waveform = np.asarray([0.0, 0.5, -0.5, 0.0])

    _save_prediction_timeline(
        waveform=waveform,
        sampling_rate=2,
        center_times=np.asarray([0.5, 1.5]),
        class_probabilities=np.asarray([
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
        ]),
        duration_seconds=2.0,
        audio_name="cough.wav",
        actual_label=2,
        output_path=output_path,
    )

    subplots.assert_called_once_with(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": (2, 1)},
    )
    np.testing.assert_array_equal(
        waveform_axis.plot.call_args.args[0],
        [0.0, 0.5, 1.0, 1.5],
    )
    np.testing.assert_array_equal(
        waveform_axis.plot.call_args.args[1],
        waveform,
    )
    assert prediction_axis.stairs.call_count == 3
    assert [
        call.kwargs["label"]
        for call in prediction_axis.stairs.call_args_list
    ] == ["No cough", "Non-infectious", "Infectious"]
    assert all(
        call.kwargs["linewidth"] == 1.0
        and call.kwargs["alpha"] == 0.9
        for call in prediction_axis.stairs.call_args_list
    )
    prediction_axis.set_ylim.assert_called_once_with(0, 1)
    prediction_axis.set_ylabel.assert_called_once_with("Probability")
    prediction_axis.legend.assert_called_once_with(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    figure.suptitle.assert_called_once_with(
        "cough.wav — Actual: Infectious",
    )
    figure.savefig.assert_called_once_with(output_path, dpi=150)


def test_timeline_plot_saves_png(tmp_path:Path) -> None:
    output_path = tmp_path / "timeline.png"

    _save_prediction_timeline(
        waveform=np.asarray([0.0, 0.5, -0.5, 0.0]),
        sampling_rate=2,
        center_times=np.asarray([0.5, 1.5]),
        class_probabilities=np.asarray([
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
        ]),
        duration_seconds=2.0,
        audio_name="cough.wav",
        actual_label=2,
        output_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_source_record_plotting_skips_only_unreadable_audio(
    monkeypatch:pytest.MonkeyPatch,
    tmp_path:Path,
) -> None:
    source_root = tmp_path / "source"
    source_records = [
        SourceRecord({}, source_root / "bad.wav", 1),
        SourceRecord({}, source_root / "good.wav", 2),
    ]
    audio_reader = Mock(sampling_rate=2)
    audio_reader.read.side_effect = [
        ValueError("could not decode"),
        np.asarray([0.0, 1.0, 2.0, 3.0]),
    ]
    example_constructor = SimpleNamespace(
        audio_file_reader=audio_reader,
        transformers=[],
    )
    saved_timelines = []
    monkeypatch.setattr(
        timeline_script,
        "_save_prediction_timeline",
        lambda **kwargs: saved_timelines.append(kwargs),
    )

    plotted_count, skipped_count = _plot_source_records(
        source_records=source_records,
        source_root=source_root,
        output_directory=tmp_path / "plots",
        example_constructor=example_constructor,
        window_size=2,
        stride=1,
        model=_ProbabilityModel(),
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert plotted_count == 1
    assert skipped_count == 1
    assert len(saved_timelines) == 1
    assert saved_timelines[0]["audio_name"] == "good.wav"
    assert saved_timelines[0]["actual_label"] == 2


def test_missing_fold_selection_data_is_rejected() -> None:
    fold = _make_fold(1, validation_loss=None, metrics={})

    with pytest.raises(ValueError, match="validation loss"):
        select_best_fold([fold], "validation")

    with pytest.raises(ValueError, match="test metrics"):
        select_best_fold([fold], "test")


def _make_fold(
    fold_index:int,
    validation_loss:float|None=0.5,
    metrics:dict[str, float]|None=None,
):
    return SimpleNamespace(
        fold_index=fold_index,
        loss_log=SimpleNamespace(best_validation_loss=validation_loss),
        validation_metrics=metrics or {},
        model=Mock(),
    )


class _AddingTransformer:

    def __init__(self, amount:float) -> None:
        self.amount = amount
        return

    def transform(self, segments:list[SeriesSegment]) -> list[SeriesSegment]:
        return [
            SeriesSegment(
                value=segment.value + self.amount,
                original_index=segment.original_index,
            )
            for segment in segments
        ]


class _ThresholdModel:

    def __init__(self) -> None:
        self.batch_sizes = []
        return

    def predict_classes(self, values:torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(len(values))
        return (values[:, 0] >= 2).to(torch.long)


class _ProbabilityModel:

    def __init__(self) -> None:
        self.batch_sizes = []
        return

    def predict_probabilities(self, values:torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(len(values))
        class_one = (values[:, 0] + 1) / 10
        class_two = torch.full_like(class_one, 0.1)
        class_zero = 1 - class_one - class_two
        return torch.stack((class_zero, class_one, class_two), dim=1)
