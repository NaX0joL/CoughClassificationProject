from pathlib import Path
from typing import TypeGuard

import numpy as np

from ...intermediary import SourceSeries
from ...abstract import SourceReader
from .abstract import MetadataRow
from .metadata_reader import METADATA_FILENAME, MetadataReader
from .audio_reader import RawAudioReader

from .patient_id_exception import (
    EMPTY_DETECTED_COUGH_SEGMENTS_PATIENT_IDS,
    MULTIPLE_MEDICAL_CONDITION_PATIENT_IDS,
)



ROOT_PATH = Path("data/Elderly_Cough_Audio")
METADATA_FILENAME = "metadata.xlsx"
TRANSLATION_FILENAME = "translations.json"
AUDIO_FOLDER = "source_data"



class ElderlyCoughAudioSourceReader(SourceReader):

    def __init__(self, root_path:Path=ROOT_PATH) -> None:
        self.root_path = root_path

    def get_source_series(self) -> list[SourceSeries]:
        metadata_path = self.root_path / METADATA_FILENAME
        translation_path = self.root_path / TRANSLATION_FILENAME
        audio_directory = self.root_path / AUDIO_FOLDER

        translated_metadata = MetadataReader(
            path=metadata_path,
            translation_path=translation_path,
        ).read()

        raw_audio = RawAudioReader(
            directory=audio_directory
        ).read(translated_metadata)

        source_series = SourceSeriesFactory.create(
            translated_metadata,
            raw_audio,
        )
        return source_series



class SourceSeriesFactory:

    @classmethod
    def create(
        cls,
        metadatas:list[MetadataRow],
        audios:list[tuple[np.ndarray, int] | None]
    ) -> list[SourceSeries]:
        source_series_list = []

        for metadata, audio in zip(metadatas, audios):
            if (
                not cls._is_valid_metadata(metadata)
                or not cls._is_valid_audio(audio)
            ):
                continue

            samples, sample_rate = audio
            source_series = SourceSeries(
                value=cls._assign_source_series_value(samples),
                label=cls._assign_source_series_label(metadata),
                metadata=cls._assign_source_series_metadata(
                    metadata, 
                    sample_rate=sample_rate
                ),
            )
            source_series_list.append(source_series)

        return source_series_list

    @staticmethod
    def _is_valid_metadata(metadata:MetadataRow) -> bool:
        if metadata.patient_id in MULTIPLE_MEDICAL_CONDITION_PATIENT_IDS:
            return False
        if metadata.patient_id in EMPTY_DETECTED_COUGH_SEGMENTS_PATIENT_IDS:
            return False
        if not metadata.audio_exists:
            return False
        if not metadata.usability:
            return False
        return True

    @staticmethod
    def _is_valid_audio(
        audio: tuple[np.ndarray, int] | None,
    ) -> TypeGuard[tuple[np.ndarray, int]]:
        if audio is None:
            return False
        if len(audio) != 2:
            return False

        samples, sample_rate = audio
        if not isinstance(samples, np.ndarray) or samples.size == 0:
            return False

        return isinstance(sample_rate, int) and sample_rate > 0
    
    @staticmethod
    def _assign_source_series_value(samples:np.ndarray) -> np.ndarray:
        return samples
    
    @staticmethod
    def _assign_source_series_label(metadata:MetadataRow):
        return "infectious" if metadata.is_infectious else "non-infectious"
    
    @staticmethod
    def _assign_source_series_metadata(
        metadata: MetadataRow,
        **kwargs,
    ) -> dict[str, object]:
        
        metadata_dict = {
            key: value
            for key, value in vars(metadata).items()
        }
        for key, value in kwargs.items():
            metadata_dict[key] = value
        
        return metadata_dict
