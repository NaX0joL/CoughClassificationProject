from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike



@dataclass(frozen=True)
class BinaryInfectionLabelMapping:
    raw_class_groups:tuple[tuple[int, ...], tuple[int, ...]] = (
        (0, 1),
        (2,),
    )
    class_names:tuple[str, str] = (
        "non-infectious",
        "infectious",
    )

    def __post_init__(self) -> None:
        try:
            groups = tuple(tuple(group) for group in self.raw_class_groups)
            names = tuple(self.class_names)
        except TypeError as error:
            raise ValueError("label mapping groups and names must be sequences") from error
        if len(groups) != 2 or any(not group for group in groups):
            raise ValueError("label mapping requires exactly two nonempty groups")
        if any(
            not isinstance(raw_class, int)
            or isinstance(raw_class, bool)
            or raw_class < 0
            for group in groups
            for raw_class in group
        ):
            raise ValueError(
                "label mapping raw class groups must contain nonnegative integers",
            )
        raw_classes = [raw_class for group in groups for raw_class in group]
        if len(raw_classes) != len(set(raw_classes)):
            raise ValueError("label mapping raw class IDs must be unique")
        if len(names) != 2 or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("label mapping requires exactly two nonempty class names")
        object.__setattr__(self, "raw_class_groups", groups)
        object.__setattr__(self, "class_names", names)
        return

    @property
    def raw_class_count(self) -> int:
        return max(
            raw_class
            for group in self.raw_class_groups
            for raw_class in group
        ) + 1

    @property
    def report_class_names(self) -> tuple[str, str]:
        return self.class_names

    def remap_labels(self, labels:ArrayLike) -> np.ndarray:
        label_array = np.asarray(labels)
        raw_to_reported = {
            raw_class: reported_class
            for reported_class, group in enumerate(self.raw_class_groups)
            for raw_class in group
        }
        if not np.isin(label_array, list(raw_to_reported)).all():
            raise ValueError("labels must be present in the configured raw class groups")
        return np.asarray([
            raw_to_reported[int(label)]
            for label in label_array
        ], dtype=np.int64)

    def aggregate_probabilities(self, probabilities:ArrayLike) -> np.ndarray:
        probability_array = np.asarray(probabilities)
        if probability_array.ndim != 2:
            raise ValueError("probabilities must be a two-dimensional array")
        if probability_array.shape[1] != self.raw_class_count:
            raise ValueError(
                "binary infection label mapping requires three probability columns",
            )
        return np.column_stack([
            probability_array[:, list(group)].sum(axis=1)
            for group in self.raw_class_groups
        ])
