
from dataclasses import dataclass

import numpy as np



@dataclass
class SourceSeries():
    value: np.ndarray
    label: str
    metadata: dict[str, object]



@dataclass
class Example():
    value: np.ndarray
    label: str
    metadata: dict[str, object]
