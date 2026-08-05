from pathlib import Path
from dataclasses import dataclass
from typing import Optional



type RawMetadataRow = dict[str, object]



@dataclass
class MetadataRow:
    patient_id: Optional[str] = None
    facility: Optional[str] = None
    normalized_facility: Optional[str] = None
    location: Optional[str] = None
    biological_sex: Optional[str] = None
    age_group: Optional[int] = None
    timestamp: Optional[str] = None
    cough_audio: Optional[Path] = None
    audio_exists: Optional[bool] = None
    current_medical_condition: Optional[list[str]] = None
    is_infectious: Optional[bool] = None
    current_symptoms: Optional[list[str]] = None
    usability: Optional[bool] = None
    local_path: Optional[Path] = None
    detected_cough_segments: Optional[list[tuple[int, int]]] = None
    detected_seconds: Optional[list[tuple[float, float]]] = None
