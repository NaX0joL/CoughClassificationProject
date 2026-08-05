from dataclasses import asdict
from pathlib import Path
import json

from .abstract import MetadataRow



class MetadataTranslator():
    
    def __init__(self, path:Path) -> None:
        self.path = path
        return
    
    def translate(self, metadata_rows:list[MetadataRow]):
        translation_dict = self._read_json(self.path)
        
        for row in metadata_rows:
            self._translate_row(row, translation_dict)
        
        return metadata_rows
    
    @staticmethod
    def _read_json(path:Path) -> dict[str, dict[str, str]]:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    
    @staticmethod
    def _translate_row(metadata_row:MetadataRow, translation_dict:dict[str, dict[str, str]]) -> None:
        for column, translations in translation_dict.items():
            value = getattr(metadata_row, column, None)

            if isinstance(value, list):
                setattr(
                    metadata_row,
                    column,
                    [translations.get(item, item) for item in value],
                )
                
        return