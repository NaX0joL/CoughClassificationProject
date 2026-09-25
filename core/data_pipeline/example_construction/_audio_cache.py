import hashlib
from pathlib import Path

import numpy as np



class AudioWaveformCache():

    def __init__(self, directory:Path, sampling_rate:int) -> None:
        self.directory = directory
        self.sampling_rate = sampling_rate
        return

    def read(self, audio_path:Path) -> np.ndarray|None:
        cache_path = self._get_cache_path(audio_path)
        
        if not cache_path.is_file():
            return None

        source_state = audio_path.stat()
        with np.load(cache_path, allow_pickle=False) as cached_audio:
            if (
                int(cached_audio["source_size"]) != source_state.st_size
                or int(cached_audio["source_mtime_ns"])
                != source_state.st_mtime_ns
            ):
                return None

            waveform = cached_audio["waveform"].copy()

        return waveform

    def write(self, audio_path:Path, waveform:np.ndarray) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        source_state = audio_path.stat()

        np.savez(
            self._get_cache_path(audio_path),
            waveform=waveform,
            source_size=source_state.st_size,
            source_mtime_ns=source_state.st_mtime_ns,
        )
        return

    def _get_cache_path(self, audio_path:Path) -> Path:
        cache_identity = (
            f"{audio_path.resolve()}:{self.sampling_rate}"
        )
        cache_key = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
        return self.directory / f"{cache_key}.npz"
