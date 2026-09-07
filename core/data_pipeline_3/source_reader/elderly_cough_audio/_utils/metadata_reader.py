from pathlib import Path
import pandas as pd
import ast



class MetadataReader():
    
    def __init__(
        self,
        path:Path,
        sheet_name:str,
    ) -> None:
        self.path = path
        self.sheet_name = sheet_name
        return
    
    def read(self) -> list[dict]:
        self._validate_path()
        
        excel_table_df = self._read_excel_file()
        excel_table_df = self._sanitize_nan_values(excel_table_df)
        excel_table_dicts = self._cast_df_into_list_of_dicts(excel_table_df)
        normalized_excel_table_dicts = self._normalize_columns(excel_table_dicts)
        
        return normalized_excel_table_dicts
    
    def _validate_path(self):
        if not self.path.is_file():
            raise FileNotFoundError(f"metadata excel file does not exist: {self.path}")
        return
    
    def _read_excel_file(self) -> pd.DataFrame:
        excel_table_df = pd.read_excel(
            self.path,
            sheet_name=self.sheet_name,
            engine="openpyxl",
        )
        
        # print(type(excel_table_df))
        # print(excel_table_df)
        return excel_table_df
    
    def _sanitize_nan_values(self, excel_table_df:pd.DataFrame):
        excel_table_df = excel_table_df.astype(object).where(
            excel_table_df.notna(),
            None,
        )
        
        # print(excel_table_df)
        return excel_table_df
    
    def _cast_df_into_list_of_dicts(self, excel_table_df:pd.DataFrame) -> list[dict]:
        excel_table_dicts = excel_table_df.to_dict(orient="records")
        
        #print(excel_table_dicts[0])
        return excel_table_dicts
    
    def _normalize_columns(self, excel_table_dicts:list[dict]) -> list[dict]:
        for index in range(len(excel_table_dicts)):
            excel_table_dicts[index] = ManualColumnNormalizer.normalize(excel_table_dicts[index])
        
        #print(excel_table_dicts[index])
        return excel_table_dicts



class ManualColumnNormalizer():
    
    @classmethod
    def normalize(cls, dict) -> dict:
        
        cls._normalize_AgeGroup(dict)
        cls._normalize_Audio_exists(dict)
        cls._normalize_currentMedicalCondition(dict)
        cls._normalize_isInfectious(dict)
        cls._normalize_currentSymptoms(dict)
        cls._normalize_Usability(dict)
        cls._normalize_DetectedCoughSegments(dict)
        cls._normalize_DetectedSeconds(dict)
        
        return dict
    
    @staticmethod
    def _normalize_AgeGroup(dict) -> None:
        key = "AgeGroup"
        value = dict[key]
        
        if not value:
            return
        
        # for now, some of the data are invalid, so ignore for now
        dict[key] = value
        return
    
    @staticmethod
    def _normalize_Audio_exists(dict) -> None:
        key = "Audio_exists"
        value = dict[key]
        
        if not value:
            return
        
        value = value.lower()        
        if value == "true":
            value = True
        elif value == "false":
            value = False
        elif value == "broken":
            value = False
        else:
            raise ValueError(f"metadata key {key} has invalid value: {value}")
        
        dict[key] = value
        return
    
    @staticmethod
    def _normalize_currentMedicalCondition(dict) -> None:
        key = "currentMedicalCondition"
        value = dict[key]
        
        if not value:
            return
        
        parsed_values = ast.literal_eval(value)
        values = [
            str(item).strip()
            for item in parsed_values
        ]
        
        dict[key] = values
        return 
        
    @staticmethod
    def _normalize_isInfectious(dict) -> None:
        key = "isInfectious"
        value = dict[key]
        
        if not value:
            return
        
        value = value.lower()
        if value == "positive":
            value = True
        elif value == "negative":
            value = False
        elif value == "n/a":
            value = None
        else:
            raise ValueError(f"metadata key {key} has invalid value: {value}")
        
        dict[key] = value
        return
    
    @staticmethod
    def _normalize_currentSymptoms(dict) -> None:
        key = "currentSymptoms"
        value = dict[key]
        
        if not value:
            return
        
        parsed_values = ast.literal_eval(value)
        values = [
            str(item).strip()
            for item in parsed_values
        ]
        
        dict[key] = values
        return
    
    @staticmethod
    def _normalize_Usability(dict) -> None:
        key = "Usability"
        value = dict[key]
        
        if not value:
            return
        
        value = value.lower()
        if value == "usable":
            value = True
        elif value == "invalid":
            value = False
        else:
            raise ValueError(f"metadata key {key} has invalid value: {value}")
        
        dict[key] = value
        return
    
    @staticmethod
    def _normalize_DetectedCoughSegments(dict) -> None:
        key = "DetectedCoughSegments"
        value = dict[key]
        
        if not value:
            return
        
        parsed_segments = ast.literal_eval(value)
        new_values = [
            (int(start), int(end))
            for start, end in parsed_segments
        ]
        
        dict[key] = new_values
        return
    
    @staticmethod
    def _normalize_DetectedSeconds(dict) -> None:
        key = "DetectedSeconds"
        value = dict[key]
        
        if not value:
            return
        
        parsed_segments = ast.literal_eval(value)
        new_values = [
            (float(start), float(end))
            for start, end in parsed_segments
        ]
        
        dict[key] = new_values
        return
