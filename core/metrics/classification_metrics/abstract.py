from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetricInput:
    labels:np.ndarray
    predictions:np.ndarray
    probabilities:np.ndarray
    class_labels:np.ndarray



class ClassificationMetric(ABC):
    name:str

    @abstractmethod
    def calculate(self, metric_input:ClassificationMetricInput) -> float:
        """Calculate this metric from validated classification outputs."""
