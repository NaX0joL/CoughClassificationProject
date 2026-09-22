import ast
from collections.abc import Callable
from datetime import datetime
from math import isnan
from pathlib import Path
from typing import Any, TypeVar, cast

from .abstract import RawMetadataRow, MetadataRow



Number = TypeVar("Number")



class MetadataTypeCaster:

    @classmethod
    def cast(cls, raw_rows:list[RawMetadataRow]) -> list[MetadataRow]:
        type_casted = [cls._cast(row) for row in raw_rows]
        return type_casted

    @classmethod
    def _cast(cls, raw_row:RawMetadataRow) -> MetadataRow:
        field_casters = {
            "PatientID": ("patient_id", _to_str),
            "facility": ("facility", _to_str),
            "normalized_facility": ("normalized_facility", _to_str),
            "location": ("location", _to_str),
            "biologicalSex": ("biological_sex", _to_str),
            "AgeGroup": ("age_group", _to_int),
            "Timestamp": ("timestamp", _to_str),
            "CoughAudio": ("cough_audio", _to_path),
            "Audio_exists": ("audio_exists", _to_bool),
            "currentMedicalCondition": ("current_medical_condition", _to_str_list),
            "isInfectious": ("is_infectious", _to_bool),
            "currentSymptoms": ("current_symptoms", _to_str_list),
            "Usability": ("usability", _to_bool),
            "local_path": ("local_path", _to_path),
            "DetectedCoughSegments": ("detected_cough_segments", _to_int_ranges),
            "DetectedSeconds": ("detected_seconds", _to_float_ranges),
        }
        typed_values = {}
        
        for source_column, (target_column, caster) in field_casters.items():
            value = raw_row.get(source_column)
            
            if source_column in raw_row:
                typed_values[target_column] = caster(value)

                if source_column == "isInfectious":
                    typed_values["original_label"] = _to_str(value)

        type_casted = MetadataRow(**cast(dict[str, Any], typed_values))
        return type_casted




def _is_invalid(value:object) -> bool:
    if value is None:
        return True

    if isinstance(value, float) and isnan(value):
        return True

    if isinstance(value, str):
        invalid_values = {"", "N/A", "BROKEN"}
        return value.strip().upper() in invalid_values

    return False


def _to_str(value:object) -> str | None:
    if _is_invalid(value):
        return None
    
    if isinstance(value, datetime):
        return value.isoformat()
    
    return str(value).strip()


def _to_path(value:object) -> Path | None:
    str_value = _to_str(value)
    if str_value is None:
        return None
    return Path(str_value)


def _to_int(value:object) -> int|None:
    if _is_invalid(value) or isinstance(value, bool):
        return None
    
    if isinstance(value, int):
        return value
    
    if isinstance(value, float) and value.is_integer():
        return int(value)
    
    if isinstance(value, str):
        try:
            return int(value.strip())
        
        except ValueError:
            return None

    return None


def _to_float(value:object) -> float | None:
    if _is_invalid(value) or isinstance(value, bool):
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        try:
            return float(value.strip())
        
        except ValueError:
            return None

    return None


def _to_bool(value:object) -> bool|None:
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        normalized = value.strip().upper()
        
        if normalized in {
            "POSITIVE",
            "USABLE",
            "TRUE", 
            "YES", 
            "1"
        }:
            return True
        
        if normalized in {
            "NEGATIVE",
            "INVALID",
            "FALSE", 
            "NO", 
            "0"
        }:
            return False
        
        if normalized in {
            "BROKEN",
            "N/A",
        }:
            return None
    
    return None


def _to_str_list(value:object) -> list[str] | None:
    if _is_invalid(value):
        return None

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)

        except (SyntaxError, ValueError):
            value = value.split(",")

    if not isinstance(value, (list, tuple)):
        return None

    str_list = []
    for item in value:
        str_value = _to_str(item)
        if str_value is not None:
            str_list.append(str_value)

    if not str_list:
        return None

    return str_list


def _to_ranges(
    value:object,
    converter:Callable[[object], Number | None],
) -> list[tuple[Number, Number]] | None:
    
    ranges = _read_ranges(value)
    if ranges is None:
        return None

    converted_ranges: list[tuple[Number, Number]] = []
    for start, end in ranges:
        converted_start = converter(start)
        converted_end = converter(end)

        if converted_start is None or converted_end is None:
            return None

        converted_ranges.append((converted_start, converted_end))

    return converted_ranges


def _to_int_ranges(value:object) -> list[tuple[int, int]] | None:
    ranges = _to_ranges(value, _to_int)
    return ranges


def _to_float_ranges(value:object) -> list[tuple[float, float]] | None:
    ranges = _to_ranges(value, _to_float)
    return ranges


def _read_ranges(value:object) -> list[tuple[object, object]] | None:
    if _is_invalid(value):
        return None

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
        
    if not isinstance(value, (list, tuple)):
        return None
    
    ranges = []
    for item in value:
        
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        
        ranges.append((item[0], item[1]))
        
    return ranges
