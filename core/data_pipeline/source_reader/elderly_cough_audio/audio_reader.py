from pathlib import Path



class AudioFileTypeDetector:

    @staticmethod
    def detect(path: str | Path) -> str:
        path = Path(path)

        with path.open("rb") as file:
            header = file.read(32)

        if len(header) >= 12:
            if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                return "wav"

            if header[:4] == b"RF64" and header[8:12] == b"WAVE":
                return "wav"

        if header[4:8] == b"ftyp":
            return "mp4/m4a"

        if header[:4] == b"fLaC":
            return "flac"

        if header[:4] == b"OggS":
            return "ogg"

        if header[:3] == b"ID3":
            return "mp3"

        return "unknown"



class RawAudioReader:

    @staticmethod
    def load_wav(path: Path) -> None:
        return
