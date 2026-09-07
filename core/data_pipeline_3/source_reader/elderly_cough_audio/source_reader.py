from pathlib import Path

from ._utils.metadata_reader import MetadataReader
from ._utils.metadata_translator import MetadataTranslator

from ...intermediary import SourceRecord



DEFAULT_DATASET_PATH = Path("data/Elderly_Cough_Audio")
DEFAULT_METADATA_PATH = DEFAULT_DATASET_PATH / "metadata.xlsx"
DEFAULT_TRANSLATION_PATH = DEFAULT_DATASET_PATH / "translations.json"
DEFAULT_SOURCE_DATA_PARENT_DIRECTORY = DEFAULT_DATASET_PATH / "source_data"

DEFAULT_EXCEL_SHEET_NAME = "dynamo"



class SourceReader():
    
    def __init__(
        self,
        metadata_path:Path=DEFAULT_METADATA_PATH,
        excel_sheet_name:str=DEFAULT_EXCEL_SHEET_NAME,
        translation_path:Path=DEFAULT_TRANSLATION_PATH,
        source_data_parent_directory:Path=DEFAULT_SOURCE_DATA_PARENT_DIRECTORY,
    ) -> None:
        self.source_data_parent_directory = source_data_parent_directory
        
        self.metadata_reader = MetadataReader(path=metadata_path, sheet_name=excel_sheet_name)
        self.metadata_translator = MetadataTranslator(path=translation_path)
        return
    
    def get_source_data(self) -> list[SourceRecord]:
        metadata = self.metadata_reader.read()
        translated = self.metadata_translator.translate(metadata)
        audio_files_paths = self._collect_all_audio_file_paths()
        source_records = self._make_source_records(metadatas=translated, audio_paths=audio_files_paths)
        return source_records
    
    def _collect_all_audio_file_paths(self) -> list[Path]:
        audio_files_path = []
        
        for item in self.source_data_parent_directory.rglob("*"):
            if item.is_file():
                audio_files_path.append(item)
                
        return audio_files_path
    
    def _make_source_records(self, metadatas:list[dict], audio_paths:list[Path]) -> list[SourceRecord]:
        name_to_path_mapping = {path.name: path for path in audio_paths}
        
        source_records = []
        for metadata in metadatas:
            original_path = metadata.get("local_path")
            
            if not isinstance(original_path, (str, Path)):
                continue
            
            file_name = Path(original_path).name
            audio_path = name_to_path_mapping.get(file_name)
            
            if audio_path is None:
                continue
            
            source_record = SourceRecord(
                metadata=metadata,
                audio_path=audio_path,
            )
            source_records.append(source_record)
        
        return source_records