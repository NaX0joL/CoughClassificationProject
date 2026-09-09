from pathlib import Path

import numpy as np

from core.data_pipeline_3.example_constructor._utils.example_visualizer import (
    ExampleVisualizer,
)
from core.data_pipeline_3.intermediary import Example



def test_example_visualizer_saves_feature_plot(
    tmp_path:Path,
) -> None:
    visualizer = ExampleVisualizer(
        gallery_directory=tmp_path / "gallery",
    )
    example = Example(
        value=np.arange(12).reshape(3, 4),
        label=1,
        metadata={},
    )

    figure_path = visualizer.visualize(
        example=example,
        audio_path=tmp_path / "audio.wav",
        example_index=2,
    )

    assert figure_path == tmp_path / "gallery" / "audio_2.png"
    assert figure_path.is_file()


def test_example_visualizer_stops_at_limit_and_resets(
    tmp_path:Path,
) -> None:
    visualizer = ExampleVisualizer(
        gallery_directory=tmp_path / "gallery",
        visualization_limit=1,
    )
    example = Example(
        value=np.arange(4),
        label=0,
        metadata={},
    )

    first_path = visualizer.visualize(
        example,
        tmp_path / "audio.wav",
        0,
    )
    skipped_path = visualizer.visualize(
        example,
        tmp_path / "audio.wav",
        1,
    )

    assert first_path is not None
    assert skipped_path is None
    assert visualizer.visualization_count == 1

    visualizer.reset()
    reset_path = visualizer.visualize(
        example,
        tmp_path / "audio.wav",
        1,
    )

    assert reset_path is not None
    assert visualizer.visualization_count == 1
