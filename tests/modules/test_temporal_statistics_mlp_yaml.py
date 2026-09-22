from pathlib import Path

from core.model.architectures.TemporalStatisticsMLP import TemporalStatisticsMLP
from modules.yaml_experiment import YamlToExperimentConverter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = (
    PROJECT_ROOT
    / "yaml"
    / "run"
    / "mfcc_centered_validation_temporal_statistics_mlp_v3.yaml"
)


def test_converter_builds_temporal_statistics_mlp_yaml() -> None:
    experiment = YamlToExperimentConverter().convert(YAML_PATH)
    architecture = experiment.config.model_config.architecture

    assert isinstance(architecture, TemporalStatisticsMLP)
    assert architecture.linear_dims == [64, 32]
    assert architecture.dropout == 0.4
    assert architecture.output_dim == 3
