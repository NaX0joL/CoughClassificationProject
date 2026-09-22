import hashlib
import json
from pathlib import Path

import numpy as np



class AudioCache:

    def __init__(
        self,
        directory:Path,
        source_directory:Path,
        sampling_rate:int,
        preprocessing_version:int,
    ) -> None:
        self.directory = directory
        self.source_directory = source_directory
        self.sampling_rate = sampling_rate
        self.preprocessing_version = preprocessing_version
        return

    def read(self, audio_path:Path) -> tuple[np.ndarray, int] | None:
        cache_path = self._get_cache_path(audio_path)
        if not cache_path.is_file():
            return None

        source_size, source_mtime_ns = self._get_source_state(audio_path)

        with np.load(cache_path, allow_pickle=False) as cached_audio:
            cached_relative_audio_path = str(
                cached_audio["relative_audio_path"]
            )
            cached_source_size = int(cached_audio["source_size"])
            cached_source_mtime_ns = int(cached_audio["source_mtime_ns"])
            cached_sample_rate = int(cached_audio["sample_rate"])
            cached_preprocessing_version = int(
                cached_audio["preprocessing_version"]
            )

            if (
                cached_relative_audio_path
                != self._get_relative_audio_path(audio_path)
                or cached_sample_rate != self.sampling_rate
                or cached_preprocessing_version != self.preprocessing_version
                or cached_source_size != source_size
                or cached_source_mtime_ns != source_mtime_ns
            ):
                return None

            samples = cached_audio["samples"]
            sample_rate = int(cached_audio["sample_rate"])

        return samples, sample_rate

    def write(
        self,
        audio_path: Path,
        samples: np.ndarray,
        sample_rate: int,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        source_size, source_mtime_ns = self._get_source_state(audio_path)
        relative_audio_path = self._get_relative_audio_path(audio_path)

        np.savez(
            self._get_cache_path(audio_path),
            samples=samples,
            sample_rate=sample_rate,
            preprocessing_version=self.preprocessing_version,
            relative_audio_path=relative_audio_path,
            source_size=source_size,
            source_mtime_ns=source_mtime_ns,
        )

        return

    def _get_cache_path(self, audio_path:Path) -> Path:
        cache_identity = {
            "relative_audio_path": self._get_relative_audio_path(audio_path),
            "sampling_rate": self.sampling_rate,
            "preprocessing_version": self.preprocessing_version,
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.directory / f"{cache_key}.npz"

    def _get_relative_audio_path(self, audio_path:Path) -> str:
        return audio_path.resolve().relative_to(
            self.source_directory.resolve()
        ).as_posix()

    @staticmethod
    def _get_source_state(audio_path:Path) -> tuple[int, int]:
        source_stat = audio_path.stat()
        return source_stat.st_size, source_stat.st_mtime_ns
