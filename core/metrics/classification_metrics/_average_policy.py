from typing import Any

import numpy as np
from numpy.typing import ArrayLike



_BINARY_CLASS_LABELS = np.asarray([0, 1])



def _get_average_parameters(class_labels:ArrayLike) -> dict[str, Any]:
    class_label_array = np.asarray(class_labels)
    if np.array_equal(class_label_array, _BINARY_CLASS_LABELS):
        return {
            "average": "binary",
            "pos_label": 1,
        }
    return {
        "labels": class_label_array,
        "average": "weighted",
    }
