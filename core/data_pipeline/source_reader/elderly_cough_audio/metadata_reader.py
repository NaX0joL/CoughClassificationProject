from pathlib import Path

import pandas as pd



METADATA_FILENAME = "metadata.xlsx"
METADATA_WORKSHEET = "dynamo"

METADATA_COLUMNS= {
    "PatientID": True,
    "facility": True,
    "normalized_facility": True,
    "location": True,
    "BiologicalSex": True,
    "AgeGroup": True,
    "Timestamp": True,
    "CoughAudio": True,
    "Audio_exists": True,
    "CurrentMedicalCondition": True,
    "IsInfectious": True,
    "CurrentSymptoms": True,
    "Usability": True,
    "local_path": True,
    "DetectedCoughSegments": True,
    "DetectedSeconds": False,
}



class MetadataReader:

    @classmethod
    def read(cls, path:str | Path) -> list[dict[str, object]]:
        metadata_path = cls._validate_path(path)
        metadata_table = cls._read_table(metadata_path)
        records = cls._convert_table_to_records(metadata_table)
        return records

    @staticmethod
    def _validate_path(path:str | Path) -> Path:
        metadata_path = Path(path)

        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata workbook does not exist: {metadata_path}")

        return metadata_path

    @staticmethod
    def _read_table(path:Path) -> pd.DataFrame:
        metadata_table = pd.read_excel(
            path,
            sheet_name=METADATA_WORKSHEET,
            engine="openpyxl",
        )
        return metadata_table

    @staticmethod
    def _convert_table_to_records(
        metadata_table:pd.DataFrame,
    ) -> list[dict[str, object]]:
        
        enabled_columns = {
            column
            for column, enabled in METADATA_COLUMNS.items()
            if enabled
        }
        
        metadata_table = metadata_table.astype(object).where(
            metadata_table.notna(),
            None,
        )
        records = metadata_table.to_dict(orient="records")
        
        metadata_rows = []
        for record in records:
            row = {
                str(column): value
                for column, value in record.items()
                if str(column) in enabled_columns
            }
            metadata_rows.append(row)
        
        return metadata_rows
