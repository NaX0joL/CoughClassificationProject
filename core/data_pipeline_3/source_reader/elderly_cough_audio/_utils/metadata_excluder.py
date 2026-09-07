


class MetadataExcluder():
    
    def __init__(self) -> None:
        return
    
    def exclude(self, metadatas:list[dict]) -> list[dict]:
        filtered = []
        for metadata in metadatas:
            metadata = self._filter_based_on_Usability(metadata)
            if metadata is not None:
                filtered.append(metadata)
        return filtered
    
    def _filter_based_on_Audio_exists(self, metadata:dict) -> dict|None:
        if metadata.get("Audio_exists"):
            return metadata
        return None
    
    def _filter_based_on_isInfectious(self, metadata:dict) -> dict|None:
        if metadata.get("isInfectious") is not None:
            return metadata
        return None
    
    def _filter_based_on_Usability(self, metadata:dict) -> dict|None:
        if metadata.get("Usability"):
            return metadata
        return None
