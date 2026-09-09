from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ...intermediary import Example



DEFAULT_GALLERY_DIRECTORY = Path("outputs/gallery/examples")
DEFAULT_VISUALIZATION_LIMIT = 20



class ExampleVisualizer:

    def __init__(
        self,
        gallery_directory:Path=DEFAULT_GALLERY_DIRECTORY,
        visualization_limit:int=DEFAULT_VISUALIZATION_LIMIT,
    ) -> None:
        self.gallery_directory = gallery_directory
        self.visualization_limit = visualization_limit
        self.visualization_count = 0
        return

    def reset(self) -> None:
        self.visualization_count = 0
        return

    def visualize(
        self,
        example:Example,
        audio_path:Path,
        example_index:int,
    ) -> Path|None:
        if self.visualization_count >= self.visualization_limit:
            return None

        self.gallery_directory.mkdir(parents=True, exist_ok=True)
        figure_path = (
            self.gallery_directory
            / f"{audio_path.stem}_{example_index}.png"
        )
        figure, axis = plt.subplots()

        if example.value.ndim == 1:
            axis.plot(example.value)
        else:
            axis.imshow(example.value.T, aspect="auto", origin="lower")

        figure.savefig(figure_path)
        plt.close(figure)
        self.visualization_count += 1
        return figure_path
