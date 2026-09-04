from pathlib import Path

import pandas as pd

from .abstract import MetadataRow, RawMetadataRow
from .metadata_translator import MetadataTranslator
from .metadata_type_caster import MetadataTypeCaster



METADATA_FILENAME = "metadata.xlsx"
METADATA_WORKSHEET = "dynamo"

METADATA_COLUMNS= {
    "PatientID": True,
    "facility": True,
    "normalized_facility": True,
    "location": True,
    "biologicalSex": True,
    "AgeGroup": True,
    "Timestamp": True,
    "CoughAudio": True,
    "Audio_exists": True,
    "currentMedicalCondition": True,
    "isInfectious": True,
    "currentSymptoms": True,
    "Usability": True,
    "local_path": True,
    "DetectedCoughSegments": True,
    "DetectedSeconds": True,
}



class MetadataReader:
    
    def __init__(self, path:Path, translation_path:Path) -> None:
        self.path = path
        self.translation_path = translation_path
        return

    def read(self) -> list[MetadataRow]:
        raw_metadata = self._read_raw_metadata()
        typed_metadata = MetadataTypeCaster.cast(raw_metadata)
        translated_metadata = MetadataTranslator(path=self.translation_path).translate(typed_metadata)
        return translated_metadata

    def _read_raw_metadata(self) -> list[RawMetadataRow]:
        metadata_path = self._validate_path(self.path)
        metadata_table = self._read_excel(metadata_path)
        raw_rows = self._convert_table_to_raw_metadata_rows(metadata_table)
        return raw_rows

    @staticmethod
    def _validate_path(path:str|Path) -> Path:
        metadata_path = Path(path)

        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata workbook does not exist: {metadata_path}")

        return metadata_path

    @staticmethod
    def _read_excel(path:Path) -> pd.DataFrame:
        metadata_table = pd.read_excel(
            path,
            sheet_name=METADATA_WORKSHEET,
            engine="openpyxl",
        )
        return metadata_table

    @staticmethod
    def _convert_table_to_raw_metadata_rows(metadata_table:pd.DataFrame) -> list[RawMetadataRow]:
        
        enabled_columns = {
            column
            for column, enabled in METADATA_COLUMNS.items()
            if enabled
        }
        
        metadata_table = metadata_table.astype(object).where(
            metadata_table.notna(),
            None,
        )
        raw_rows = metadata_table.to_dict(orient="records")
        
        metadata_rows = []
        for row in raw_rows:
            row = {
                str(column): value
                for column, value in row.items()
                if str(column) in enabled_columns
            }
            metadata_rows.append(row)
        
        return metadata_rows
