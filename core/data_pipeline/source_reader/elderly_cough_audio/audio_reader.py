from pathlib import Path
import re
import subprocess
from typing import cast
import wave

import numpy as np

from .abstract import MetadataRow
from .audio_cache import AudioCache
from .audio_resampler import AudioResampler



SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
TARGET_SAMPLING_RATE = 16_000
AUDIO_PREPROCESSING_VERSION = 1
CACHE_FOLDER = Path("outputs/cache/audio_16khz_v1")



class RawAudioReader:

    def __init__(self, directory:Path) -> None:
        self.directory = directory
        self.audio_resampler = AudioResampler(
            sampling_rate=TARGET_SAMPLING_RATE,
        )
        project_root = Path(__file__).resolve().parents[4]
        self.audio_cache = AudioCache(
            directory=project_root / CACHE_FOLDER,
            source_directory=self.directory,
            sampling_rate=TARGET_SAMPLING_RATE,
            preprocessing_version=AUDIO_PREPROCESSING_VERSION,
        )
        return

    def read(self, metadata_rows:list[MetadataRow]) -> list[tuple[np.ndarray, int]|None]:
        audio_filenames = self._index_audio_files(self.directory)
        loaded_audio = []

        for metadata_row in metadata_rows:
            
            if metadata_row.audio_exists is False:
                loaded_audio.append(None)
                continue
            
            if metadata_row.usability is False:
                loaded_audio.append(None)
                continue

            audio_path = self._get_audio_path(
                metadata_row,
                audio_filenames,
            )
            audio = self._get_audio(audio_path)
            loaded_audio.append(audio)

        return loaded_audio

    @staticmethod
    def _index_audio_files(audio_files_dir: Path) -> dict[str, Path]:
        audio_filenames = {}

        for audio_path in audio_files_dir.rglob("*"):
            if not audio_path.is_file():
                continue

            if audio_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
                continue

            if audio_path.name in audio_filenames:
                raise ValueError(
                    f"Duplicate audio filename found: {audio_path.name}"
                )

            audio_filenames[audio_path.name] = audio_path

        return audio_filenames

    @staticmethod
    def _get_audio_path(
        metadata_row: MetadataRow,
        audio_filenames: dict[str, Path],
    ) -> Path:
        path_candidates: tuple[Path | None, Path | None] = (
            metadata_row.cough_audio,
            metadata_row.local_path,
        )

        for path in path_candidates:
            if path is None:
                continue

            audio_path = audio_filenames.get(Path(path).name)
            if audio_path is not None:
                return audio_path

        if all(path is None for path in path_candidates):
            raise ValueError(
                f"Metadata row has no audio path: {metadata_row.patient_id}"
            )

        raise FileNotFoundError(
            f"No audio file matches metadata row: {metadata_row.patient_id}"
        )

    @staticmethod
    def _load(path: Path) -> tuple[np.ndarray, int]:
        if path.suffix.lower() == ".wav":
            try:
                samples, sample_rate = WaveBasedReader.decode_audio(path)
                return samples, sample_rate
            
            except wave.Error:
                pass

        samples, sample_rate = FfmpegBasedReader.decode_audio(path)
        return samples, sample_rate

    def _get_audio(self, audio_path:Path) -> tuple[np.ndarray, int]:
        cached_audio = self.audio_cache.read(audio_path)
        if cached_audio is not None:
            return cached_audio

        samples, sample_rate = self._load(audio_path)
        resampled_audio = self.audio_resampler.resample(
            [(samples, sample_rate)],
        )
        
        resampled_samples, resampled_sampling_rate = resampled_audio[0]
        self.audio_cache.write(
            audio_path,
            resampled_samples,
            resampled_sampling_rate,
        )
        return resampled_samples, resampled_sampling_rate



class WaveBasedReader:

    @classmethod
    def decode_audio(cls, path:Path) -> tuple[np.ndarray, int]:
        with wave.open(str(path), "rb") as wave_file:
            
            if wave_file.getnchannels() != 1 or wave_file.getsampwidth() != 2:
                raise ValueError(
                    f"Wave reader requires mono, 16-bit WAV: {path}"
                )

            audio_bytes = wave_file.readframes(wave_file.getnframes())
            sample_rate = wave_file.getframerate()

        samples = np.frombuffer(audio_bytes, dtype="<i2")
        return samples, sample_rate



class FfmpegBasedReader:
    
    @classmethod
    def decode_audio(cls, path: Path) -> tuple[subprocess.CompletedProcess[bytes], re.Match[str]]:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-f",
                "s16le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        ffmpeg_output = result.stderr.decode("utf-8", errors="replace")
        sample_rate_match: re.Match[str] | None = re.search(
            r"Audio:.*?(\d+) Hz",
            ffmpeg_output,
        )
        
        cls._validate_output(
            path,
            result,
            ffmpeg_output,
            sample_rate_match,
        )
        
        samples = np.frombuffer(result.stdout, dtype="<i2")
        sample_rate = int(sample_rate_match.group(1))
        return samples, sample_rate
    
    @staticmethod
    def _validate_output(
        path: Path,
        result: subprocess.CompletedProcess[bytes],
        ffmpeg_output: str,
        sample_rate_match: re.Match[str] | None,
    ) -> None:
        
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg could not decode {path}: {ffmpeg_output}"
            )

        if sample_rate_match is None:
            raise RuntimeError(
                f"FFmpeg did not report a sample rate for {path}"
            )
        return
