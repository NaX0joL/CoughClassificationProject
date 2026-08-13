from dataclasses import dataclass

from .classification_metrics import (
    AccuracyMetric,
    ClassificationMetric,
    F1ScoreMetric,
    MacroAccuracyMetric,
    MacroF1ScoreMetric,
    MacroPrecisionMetric,
    MacroRecallMetric,
    PRAucMetric,
    PrecisionMetric,
    RecallMetric,
    RocAucMetric,
    SpecificityMetric,
)


@dataclass(frozen=True)
class MetricsConfig:
    metrics:tuple[ClassificationMetric, ...]

    @classmethod
    def default(cls) -> "MetricsConfig":
        return cls(
            metrics=(
                RocAucMetric(),
                PRAucMetric(),
                PrecisionMetric(),
                RecallMetric(),
                SpecificityMetric(),
                F1ScoreMetric(),
                AccuracyMetric(),
                MacroAccuracyMetric(),
                MacroF1ScoreMetric(),
                MacroPrecisionMetric(),
                MacroRecallMetric(),
            ),
        )
