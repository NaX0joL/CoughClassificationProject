from ..abstract import Labeler
from ..intermediary import ORIGINAL_LABEL_METADATA_KEY, Example, Segment, SourceSeries


INVALID_LABEL = 0
NON_INFECTIOUS_LABEL = 1
INFECTIOUS_LABEL = 2

CLASS_NAMES = {
    INVALID_LABEL: "invalid",
    NON_INFECTIOUS_LABEL: "non-infectious",
    INFECTIOUS_LABEL: "infectious",
}



class OverlapLabeler(Labeler):

    def __init__(
        self,
        overlap_threshold:float=0.7,
        label_metadata_key:str="is_infectious",
        kept_metadata_keys:list[str]|None=None,
    ) -> None:
        if not 0 < overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be greater than 0 and at most 1")

        self.overlap_threshold = overlap_threshold
        self.label_metadata_key = label_metadata_key
        self.kept_metadata_keys = kept_metadata_keys or []
        return

    def label(self, segments:list[Segment]) -> list[Example]:
        examples:list[Example] = []

        for segment in segments:
            examples.append(self._label_segment(segment))

        return examples

    def _label_segment(self, segment:Segment) -> Example:
        source_label = _get_source_label(
            segment.source_series,
            self.label_metadata_key,
        )
        matched_annotation, overlap_ratio = self._find_best_overlap(
            segment,
            segment.cough_annotations,
        )

        if matched_annotation is None or overlap_ratio < self.overlap_threshold:
            label = INVALID_LABEL
            matched_annotation = None
        else:
            label = source_label

        return _create_example(
            segment=segment,
            label=label,
            matched_annotation=matched_annotation,
            kept_metadata_keys=self.kept_metadata_keys,
        )

    @staticmethod
    def _find_best_overlap(
        segment:Segment,
        annotations:list[tuple[int, int]],
    ) -> tuple[tuple[int, int]|None, float]:
        matched_annotation:tuple[int, int]|None = None
        largest_overlap_ratio = 0.0
        window_start, window_end = segment.original_index
        window_size = window_end - window_start

        for annotation in annotations:
            annotation_start, annotation_end = annotation
            overlap_start = max(window_start, annotation_start)
            overlap_end = min(window_end, annotation_end + 1)
            overlap_length = max(0, overlap_end - overlap_start)
            overlap_ratio = overlap_length / window_size

            if overlap_ratio > largest_overlap_ratio:
                largest_overlap_ratio = overlap_ratio
                matched_annotation = annotation

        return matched_annotation, largest_overlap_ratio



class AnnotatedCoughLabeler(Labeler):

    def __init__(
        self,
        label_metadata_key:str="is_infectious",
        kept_metadata_keys:list[str]|None=None,
    ) -> None:
        self.label_metadata_key = label_metadata_key
        self.kept_metadata_keys = kept_metadata_keys or []
        return

    def label(self, segments:list[Segment]) -> list[Example]:
        examples:list[Example] = []

        for segment in segments:
            examples.append(self._label_segment(segment))

        return examples

    def _label_segment(self, segment:Segment) -> Example:
        if len(segment.cough_annotations) != 1:
            raise ValueError(
                "annotated cough segments require exactly one cough annotation"
            )

        label = _get_source_label(
            segment.source_series,
            self.label_metadata_key,
        )
        return _create_example(
            segment=segment,
            label=label,
            matched_annotation=segment.cough_annotations[0],
            kept_metadata_keys=self.kept_metadata_keys,
        )



def _get_source_label(
    source_series:SourceSeries,
    label_metadata_key:str,
) -> int:
    is_infectious = source_series.metadata.get(label_metadata_key)
    if type(is_infectious) is not bool:
        raise ValueError(f"metadata[{label_metadata_key!r}] must be a boolean")

    if is_infectious:
        return INFECTIOUS_LABEL
    return NON_INFECTIOUS_LABEL


def _create_example(
    segment:Segment,
    label:int,
    matched_annotation:tuple[int, int]|None,
    kept_metadata_keys:list[str],
) -> Example:
    metadata:dict[str, object] = {}
    for metadata_key in kept_metadata_keys:
        if metadata_key not in segment.source_series.metadata:
            raise ValueError(f"metadata is missing kept key: {metadata_key}")
        metadata[metadata_key] = segment.source_series.metadata[metadata_key]

    metadata[ORIGINAL_LABEL_METADATA_KEY] = CLASS_NAMES[label]
    metadata["original_index"] = segment.original_index

    if matched_annotation is not None:
        metadata["cough_annotation_start"] = matched_annotation[0]
        metadata["cough_annotation_end"] = matched_annotation[1]

    return Example(
        value=segment.value,
        label=label,
        metadata=metadata,
    )
