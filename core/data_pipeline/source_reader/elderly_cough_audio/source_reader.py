from pathlib import Path

from .audio_reader import AudioFileTypeDetector, RawAudioReader
from .abstract import MetadataRecord
from .metadata_reader import METADATA_FILENAME, MetadataReader
from .metadata_type_caster import MetadataTypeCaster



ROOT_PATH = Path("data/Elderly_Cough_Audio")



class ElderlyCoughAudioSourceReader:

    def __init__(self, root_path:Path=ROOT_PATH) -> None:
        self.root_path = root_path

    def get_source_data(self) -> list[MetadataRecord]:
        metadata_path = self.root_path / METADATA_FILENAME
        raw_rows = MetadataReader.read(metadata_path)
        typed_rows = MetadataTypeCaster.cast_records(raw_rows)
        return typed_rows
