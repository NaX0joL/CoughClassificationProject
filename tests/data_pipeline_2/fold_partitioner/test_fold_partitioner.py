import numpy as np
import pytest

from core.data_pipeline_2.intermediary import FoldPartition, SourceSeries
from core.data_pipeline_2.fold_partitioner import FoldPartitioner



def _make_source_series(
    patient_id:str,
    is_infectious:bool,
) -> SourceSeries:
    return SourceSeries(
        value=np.ones(8, dtype=np.int16),
        metadata={
            "patient_id": patient_id,
            "is_infectious": is_infectious,
        },
    )


def _make_balanced_source_series() -> list[SourceSeries]:
    source_series = []

    for patient_index in range(18):
        patient_id = f"patient_{patient_index}"
        source_series.append(_make_source_series(
            patient_id=patient_id,
            is_infectious=bool(patient_index % 2),
        ))

    source_series.extend([
        _make_source_series("mixed_patient", False),
        _make_source_series("mixed_patient", True),
    ])
    return source_series


def _get_patient_ids(source_series:list[SourceSeries]) -> set[str]:
    return {
        str(source.metadata["patient_id"])
        for source in source_series
    }


def _get_fold_patient_ids(
    fold:FoldPartition,
) -> tuple[set[str], set[str], set[str]]:
    train_patient_ids = _get_patient_ids(fold.train)
    validation_patient_ids = _get_patient_ids(fold.validation)
    test_patient_ids = _get_patient_ids(fold.test)
    return train_patient_ids, validation_patient_ids, test_patient_ids



class TestFoldPartitioner:

    def test_assigns_every_source_to_one_role_per_fold_without_modifying_it(
        self,
    ) -> None:
        source_series = _make_balanced_source_series()
        original_metadata = [source.metadata.copy() for source in source_series]
        partitioner = FoldPartitioner(
            group_metadata_key="patient_id",
            label_metadata_key="is_infectious",
            number_of_folds=3,
            validation_ratio=0.5,
        )
        all_patient_ids = _get_patient_ids(source_series)

        folds = partitioner.partition(source_series)

        assert len(folds) == 3
        for fold in folds:
            train_ids, validation_ids, test_ids = _get_fold_patient_ids(fold)
            assigned_sources = fold.train + fold.validation + fold.test
            assert train_ids.isdisjoint(validation_ids)
            assert train_ids.isdisjoint(test_ids)
            assert validation_ids.isdisjoint(test_ids)
            assert train_ids | validation_ids | test_ids == all_patient_ids
            assert len(assigned_sources) == len(source_series)
            assert {id(source) for source in assigned_sources} == {
                id(source)
                for source in source_series
            }

        assert [source.metadata for source in source_series] == original_metadata

    def test_keeps_every_recording_from_a_patient_together(self) -> None:
        source_series = _make_balanced_source_series()
        partitioner = FoldPartitioner(
            group_metadata_key="patient_id",
            label_metadata_key="is_infectious",
            number_of_folds=3,
            validation_ratio=0.5,
        )

        folds = partitioner.partition(source_series)

        for fold in folds:
            roles = [fold.train, fold.validation, fold.test]
            containing_roles = [
                role
                for role in roles
                if "mixed_patient" in _get_patient_ids(role)
            ]
            assert len(containing_roles) == 1

            mixed_patient_sources = [
                source
                for source in containing_roles[0]
                if source.metadata["patient_id"] == "mixed_patient"
            ]
            assert len(mixed_patient_sources) == 2

    def test_uses_every_patient_for_test_exactly_once(self) -> None:
        source_series = _make_balanced_source_series()
        partitioner = FoldPartitioner(
            group_metadata_key="patient_id",
            label_metadata_key="is_infectious",
            number_of_folds=3,
            validation_ratio=0.5,
        )

        folds = partitioner.partition(source_series)

        test_patient_ids = [
            patient_id
            for fold in folds
            for patient_id in _get_patient_ids(fold.test)
        ]
        assert len(test_patient_ids) == len(set(test_patient_ids))
        assert set(test_patient_ids) == _get_patient_ids(source_series)

    def test_reproduces_assignments_with_the_same_seed(self) -> None:
        source_series = _make_balanced_source_series()
        partitioner = FoldPartitioner(
            group_metadata_key="patient_id",
            label_metadata_key="is_infectious",
            number_of_folds=3,
            validation_ratio=0.5,
            random_seed=7,
        )

        first_folds = partitioner.partition(source_series)
        second_folds = partitioner.partition(source_series)

        assert [
            _get_fold_patient_ids(fold)
            for fold in first_folds
        ] == [
            _get_fold_patient_ids(fold)
            for fold in second_folds
        ]

    def test_rejects_missing_partition_metadata(self) -> None:
        source_series = [SourceSeries(
            value=np.ones(8, dtype=np.int16),
            metadata={"patient_id": "patient_1"},
        )]
        partitioner = FoldPartitioner(
            group_metadata_key="patient_id",
            label_metadata_key="is_infectious",
            number_of_folds=2,
            validation_ratio=0.5,
        )

        with pytest.raises(ValueError, match="is_infectious"):
            partitioner.partition(source_series)
