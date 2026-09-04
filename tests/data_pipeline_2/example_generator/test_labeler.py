import numpy as np
import pytest

from core.data_pipeline_2 import (
    INFECTIOUS_LABEL,
    INVALID_LABEL,
    NON_INFECTIOUS_LABEL,
    AnnotatedCoughLabeler,
    OverlapLabeler,
    Segment,
    SourceSeries,
)


def make_segment(
    annotations:object,
    is_infectious:object=False,
    window_start:int=0,
    window_end:int=4,
    centered_annotation:tuple[int, int]|None=None,
) -> Segment:
    source = SourceSeries(
        value=np.arange(8, dtype=np.float32),
        metadata={
            "patient_id": "patient-1",
            "is_infectious": is_infectious,
            "detected_cough_segments": annotations,
        },
    )
    return Segment(
        value=source.value[window_start:window_end],
        source_series=source,
        original_index=(window_start, window_end),
        cough_annotations=(
            [centered_annotation]
            if centered_annotation is not None
            else annotations if isinstance(annotations, list) else []
        ),
    )


def test_overlap_equal_to_threshold_receives_diagnosis_label() -> None:
    segment = make_segment(annotations=[(1, 2)])
    labeler = OverlapLabeler(overlap_threshold=0.5)

    example = labeler.label([segment])[0]

    assert example.label == NON_INFECTIOUS_LABEL
    assert example.metadata["original_label"] == "non-infectious"
    assert example.metadata["cough_annotation_start"] == 1
    assert example.metadata["cough_annotation_end"] == 2
    assert example.value is segment.value


def test_overlap_below_threshold_receives_invalid_label() -> None:
    segment = make_segment(annotations=[(0, 0)])
    labeler = OverlapLabeler(overlap_threshold=0.5)

    example = labeler.label([segment])[0]

    assert example.label == INVALID_LABEL
    assert example.metadata["original_label"] == "invalid"
    assert "cough_annotation_start" not in example.metadata


def test_overlap_labeler_uses_largest_overlap_and_keeps_provenance() -> None:
    segment = make_segment(
        annotations=[(0, 0), (2, 4)],
        is_infectious=True,
        window_start=1,
        window_end=5,
    )
    labeler = OverlapLabeler(
        overlap_threshold=0.5,
        kept_metadata_keys=["patient_id"],
    )

    example = labeler.label([segment])[0]

    assert example.label == INFECTIOUS_LABEL
    assert example.metadata == {
        "patient_id": "patient-1",
        "original_label": "infectious",
        "original_index": (1, 5),
        "cough_annotation_start": 2,
        "cough_annotation_end": 4,
    }


def test_missing_annotations_make_sliding_segment_invalid() -> None:
    segment = make_segment(annotations=[])
    del segment.source_series.metadata["detected_cough_segments"]

    example = OverlapLabeler().label([segment])[0]

    assert example.label == INVALID_LABEL


def test_annotated_cough_labeler_uses_recording_diagnosis() -> None:
    segment = make_segment(
        annotations=[(2, 3)],
        is_infectious=True,
        window_start=1,
        window_end=5,
        centered_annotation=(2, 3),
    )

    example = AnnotatedCoughLabeler().label([segment])[0]

    assert example.label == INFECTIOUS_LABEL
    assert example.metadata["cough_annotation_start"] == 2
    assert example.metadata["cough_annotation_end"] == 3


@pytest.mark.parametrize("is_infectious", [None, 0, 1, "yes"])
def test_labelers_require_boolean_diagnosis(is_infectious:object) -> None:
    segment = make_segment(
        annotations=[(0, 3)],
        is_infectious=is_infectious,
    )

    with pytest.raises(ValueError, match="is_infectious"):
        OverlapLabeler().label([segment])
