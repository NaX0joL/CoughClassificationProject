from pathlib import Path

import numpy as np

from core.data_pipeline_3.example_constructor._utils.raw_audio_visualizer import (
    RawAudioVisualizer,
)



def test_raw_audio_visualizer_saves_waveform_plot(
    tmp_path:Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    visualizer = RawAudioVisualizer(
        gallery_directory=tmp_path / "gallery",
    )

    figure_path = visualizer.visualize(
        waveform=np.linspace(-1, 1, 20),
        audio_path=audio_path,
    )

    assert figure_path == tmp_path / "gallery" / "audio.png"
    assert figure_path.is_file()


def test_raw_audio_visualizer_stops_at_limit_and_resets(
    tmp_path:Path,
) -> None:
    gallery_directory = tmp_path / "gallery"
    visualizer = RawAudioVisualizer(
        gallery_directory=gallery_directory,
        visualization_limit=2,
    )

    figure_paths = [
        visualizer.visualize(
            waveform=np.linspace(-1, 1, 20),
            audio_path=tmp_path / f"audio_{index}.wav",
        )
        for index in range(3)
    ]

    assert figure_paths[0] is not None
    assert figure_paths[1] is not None
    assert figure_paths[2] is None
    assert visualizer.visualization_count == 2
    assert len(list(gallery_directory.glob("*.png"))) == 2

    visualizer.reset()
    figure_path = visualizer.visualize(
        waveform=np.linspace(-1, 1, 20),
        audio_path=tmp_path / "audio_3.wav",
    )

    assert figure_path is not None
    assert visualizer.visualization_count == 1
    assert len(list(gallery_directory.glob("*.png"))) == 3
