from pathlib import Path
import json



class MetadataTranslator():
    
    def __init__(
        self,
        path:Path,
    ) -> None:
        self.path = path
        return
    
    def translate(self, metadatas:list[dict]) -> list[dict]:
        translation = self._load_translation()
        
        for index in range(len(metadatas)):
            metadatas[index] = self._translate_entry(metadatas[index], translation)

        return metadatas
    
    def _load_translation(self) -> dict:
        with self.path.open("r", encoding="utf-8") as file:
            translations = json.load(file)
            
        return translations
    
    def _translate_entry(self, entry:dict, translation:dict) -> dict:
        for key, value in entry.items():
            
            if key in translation.keys():
                
                if isinstance(value, list):
                    for index in range(len(value)):
                        if value[index] in translation[key]:
                            entry[key][index] = translation[key][value[index]]
                
                elif isinstance(value, str):
                    if value in translation[key]:
                        entry[key] = translation[key][value]
                
                else:
                    raise ValueError(f"metadata entry {key} has invalid type: {type(value)}")
        
        return entry