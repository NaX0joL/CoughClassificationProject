import subprocess

import numpy as np



class AudioResampler():

    def __init__(self, sampling_rate:int) -> None:
        self.sampling_rate = sampling_rate
        return

    def resample(
        self,
        audios:list[tuple[np.ndarray, int]|None],
    ) -> list[tuple[np.ndarray, int]|None]:
        resampled_audio = []

        for audio in audios:
            if audio is None:
                resampled_audio.append(None)
                continue

            samples, source_sampling_rate = audio
            if source_sampling_rate == self.sampling_rate:
                resampled_audio.append(audio)
                continue

            resampled_samples = self._resample_audio(
                samples,
                source_sampling_rate,
            )
            resampled_audio.append((resampled_samples, self.sampling_rate))

        return resampled_audio

    def _resample_audio(
        self,
        samples:np.ndarray,
        source_sampling_rate:int,
    ) -> np.ndarray:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "s16le",
                "-ar",
                str(source_sampling_rate),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-ar",
                str(self.sampling_rate),
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=samples.astype("<i2", copy=False).tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if result.returncode != 0:
            ffmpeg_output = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg could not resample audio: {ffmpeg_output}")

        resampled_samples = np.frombuffer(result.stdout, dtype="<i2")
        return resampled_samples
