from core.metrics import (
    AccuracyMetric,
    F1ScoreMetric,
    MacroAccuracyMetric,
    MacroF1ScoreMetric,
    MacroPrecisionMetric,
    MacroRecallMetric,
    MetricsConfig,
    PRAucMetric,
    PrecisionMetric,
    RecallMetric,
    RocAucMetric,
    SpecificityMetric,
)


default_metrics_config = MetricsConfig(
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
