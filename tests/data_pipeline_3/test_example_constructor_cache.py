from pathlib import Path

import numpy as np

from core.data_pipeline_3.example_constructor.example_constructor import (
    ExampleConstructor,
)



def test_example_constructor_uses_cached_audio(
    tmp_path:Path,
    monkeypatch,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"source audio")
    reader_calls = []
    constructor = ExampleConstructor(
        sampling_rate=16_000,
        window_size=4,
        stride=4,
        drop_last=True,
        overlap_threshold=0.7,
        metadata_to_label_mapping={False: 1, True: 2},
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
