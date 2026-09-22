from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np



DEFAULT_GALLERY_DIRECTORY = Path("outputs/gallery/raw_audio")
DEFAULT_VISUALIZATION_LIMIT = 100



class RawAudioVisualizer:

    def __init__(
        self,
        gallery_directory:Path=DEFAULT_GALLERY_DIRECTORY,
        visualization_limit:int=DEFAULT_VISUALIZATION_LIMIT,
    ) -> None:
        self.gallery_directory = gallery_directory
        self.visualization_limit = visualization_limit
        
        self.reset()
        return

    def reset(self) -> None:
        self.visualization_count = 0
        return

    def visualize(
        self,
        waveform:np.ndarray,
        audio_path:Path,
    ) -> Path|None:
        if self.visualization_count >= self.visualization_limit:
            return None

        self.gallery_directory.mkdir(parents=True, exist_ok=True)
        figure_path = self.gallery_directory / f"{audio_path.stem}.png"
        figure, axis = plt.subplots()
        axis.plot(waveform)
        figure.savefig(figure_path)
        plt.close(figure)
        self.visualization_count += 1
        return figure_path
