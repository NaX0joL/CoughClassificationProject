from ..intermediary import SourceSeries


ANNOTATION_METADATA_KEY = "detected_cough_segments"


def get_cough_annotations(
    source_series:SourceSeries,
) -> list[tuple[int, int]]:
    if ANNOTATION_METADATA_KEY not in source_series.metadata:
        return []

    annotations = source_series.metadata[ANNOTATION_METADATA_KEY]
    if not isinstance(annotations, list):
        raise ValueError(
            f"metadata[{ANNOTATION_METADATA_KEY!r}] must be a list of integer intervals"
        )

    for annotation_index, annotation in enumerate(annotations):
        _validate_annotation(
            annotation=annotation,
            annotation_index=annotation_index,
            audio_length=len(source_series.value),
        )

    return annotations


def _validate_annotation(
    annotation:object,
    annotation_index:int,
    audio_length:int,
) -> None:
    annotation_path = (
        f"metadata[{ANNOTATION_METADATA_KEY!r}][{annotation_index}]"
    )
    if (
        not isinstance(annotation, tuple)
        or len(annotation) != 2
        or not all(type(bound) is int for bound in annotation)
    ):
        raise ValueError(f"{annotation_path} must be a pair of integer bounds")

    start, end = annotation
    if start < 0 or end < start or end >= audio_length:
        raise ValueError(
            f"{annotation_path} must satisfy 0 <= start <= end < audio length"
        )
    return
