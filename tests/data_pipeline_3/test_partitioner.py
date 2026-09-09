from pathlib import Path

from core.data_pipeline_3.intermediary import SourceRecord
from core.data_pipeline_3.partitioner import Partitioner



def _make_source_records() -> list[SourceRecord]:
    source_records = []

    for patient_index in range(20):
        patient_id = f"patient-{patient_index}"
        is_infectious = bool(patient_index % 2)

        for recording_index in range(2):
            source_records.append(SourceRecord(
                metadata={
                    "PatientID": patient_id,
                    "isInfectious": is_infectious,
                },
                audio_path=Path(f"{patient_id}-{recording_index}.wav"),
                label=int(is_infectious),
            ))

    return source_records


def _get_patient_ids(source_records:list[SourceRecord]) -> set[object]:
    return {
        source_record.metadata["PatientID"]
        for source_record in source_records
    }


def test_partitioner_keeps_patients_in_one_split() -> None:
    partitions = Partitioner().partition(_make_source_records())

    assert len(partitions) == 5

    for partition in partitions:
        train_patient_ids = _get_patient_ids(partition.train)
        validation_patient_ids = _get_patient_ids(partition.validation)
        test_patient_ids = _get_patient_ids(partition.test)

        assert train_patient_ids.isdisjoint(validation_patient_ids)
        assert train_patient_ids.isdisjoint(test_patient_ids)
        assert validation_patient_ids.isdisjoint(test_patient_ids)
