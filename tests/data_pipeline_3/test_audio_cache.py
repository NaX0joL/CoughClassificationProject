from pathlib import Path

import numpy as np

from core.data_pipeline_3.example_constructor._utils.cache import (
    AudioWaveformCache,
)



def test_audio_cache_reads_written_waveform(tmp_path:Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"source audio")
    waveform = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    cache = AudioWaveformCache(
        directory=tmp_path / "cache",
        sampling_rate=16_000,
    )

    assert cache.read(audio_path) is None

    cache.write(audio_path, waveform)

    np.testing.assert_array_equal(cache.read(audio_path), waveform)


def test_audio_cache_invalidates_changed_source(tmp_path:Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"source audio")
    cache = AudioWaveformCache(
        directory=tmp_path / "cache",
        sampling_rate=16_000,
    )
    cache.write(
        audio_path,
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    )

    audio_path.write_bytes(b"changed source audio")

    assert cache.read(audio_path) is None
