from pathlib import Path
from unittest.mock import Mock

import numpy as np

from core.data_pipeline_3.example_constructor.example_constructor import (
    DEFAULT_CACHE_DIRECTORY,
    ExampleConstructor,
)
from core.data_pipeline_3.example_constructor._utils.segment_labeler import (
    OverlapLabeler,
)
from core.data_pipeline_3.example_constructor._utils.series_segmenter import (
    SlidingWindowSegmenter,
)



def test_default_cache_directory_identifies_dataset_and_waveform_version() -> None:
    assert DEFAULT_CACHE_DIRECTORY == Path(
        "outputs/cache/elderly_cough_audio_waveform_16khz_v3"
    )


def test_example_constructor_uses_cached_audio(
    tmp_path:Path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"source audio")
    reader_calls = []
    segmenter = SlidingWindowSegmenter(
        window_size=4,
        stride=4,
        drop_last=True,
    )
    constructor = ExampleConstructor(
        sampling_rate=16_000,
        train_segmenter=segmenter,
        validation_segmenter=segmenter,
        test_segmenter=segmenter,
        segment_labeler=OverlapLabeler(
            overlap_threshold=0.7,
            no_overlap_label=0,
        ),
        cache_directory=tmp_path / "cache",
    )

    def read_audio(path:Path) -> np.ndarray:
        reader_calls.append(path)
        return np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    monkeypatch.setattr(constructor.audio_file_reader, "read", read_audio)

    first_audio = constructor._load_audio(audio_path)
    second_audio = constructor._load_audio(audio_path)

    assert len(reader_calls) == 1
    np.testing.assert_array_equal(first_audio, second_audio)


def test_example_constructor_selects_segmenter_by_example_type(
    tmp_path:Path,
    monkeypatch,
) -> None:
    train_segmenter = Mock()
    validation_segmenter = Mock()
    test_segmenter = Mock()
    train_segmenter.segment.return_value = []
    validation_segmenter.segment.return_value = []
    test_segmenter.segment.return_value = []
    constructor = ExampleConstructor(
        sampling_rate=16_000,
        train_segmenter=train_segmenter,
        validation_segmenter=validation_segmenter,
        test_segmenter=test_segmenter,
        segment_labeler=OverlapLabeler(
            overlap_threshold=0.7,
            no_overlap_label=0,
        ),
        cache_directory=tmp_path / "cache",
    )
    monkeypatch.setattr(
        constructor,
        "_load_audio",
        lambda path: np.arange(8),
    )
    raw_audio_visualizer = Mock()
    constructor.raw_audio_visualizer = raw_audio_visualizer
    source_record = Mock(
        audio_path=Path("audio.wav"),
        metadata={
            "DetectedCoughSegments": [(2, 3)],
            "isInfectious": True,
        },
    )

    constructor.construct([source_record], type="validation")

    train_segmenter.segment.assert_not_called()
    validation_segmenter.segment.assert_called_once()
    test_segmenter.segment.assert_not_called()
    raw_audio_visualizer.visualize.assert_called_once()


def test_example_constructor_hides_skipped_record_error_by_default(
    tmp_path:Path,
    monkeypatch,
    capsys,
) -> None:
    segmenter = SlidingWindowSegmenter(window_size=4, stride=4)
    constructor = ExampleConstructor(
        sampling_rate=16_000,
        train_segmenter=segmenter,
        validation_segmenter=segmenter,
        test_segmenter=segmenter,
        segment_labeler=OverlapLabeler(
            overlap_threshold=0.7,
            no_overlap_label=0,
        ),
        cache_directory=tmp_path / "cache",
    )
    monkeypatch.setattr(
        constructor,
        "_construct_from_source_record",
        Mock(side_effect=ValueError("invalid audio")),
    )

    constructor.construct([Mock()], type="train")

    assert capsys.readouterr().out == ""


def test_example_constructor_shows_skipped_record_error_when_verbose(
    tmp_path:Path,
    monkeypatch,
    capsys,
) -> None:
    segmenter = SlidingWindowSegmenter(window_size=4, stride=4)
    constructor = ExampleConstructor(
        sampling_rate=16_000,
        train_segmenter=segmenter,
        validation_segmenter=segmenter,
        test_segmenter=segmenter,
        segment_labeler=OverlapLabeler(
            overlap_threshold=0.7,
            no_overlap_label=0,
        ),
        cache_directory=tmp_path / "cache",
        verbose=True,
    )
    monkeypatch.setattr(
        constructor,
        "_construct_from_source_record",
        Mock(side_effect=ValueError("invalid audio")),
    )

    constructor.construct([Mock()], type="train")

    assert capsys.readouterr().out == "invalid audio\n"
