from pathlib import Path

import pytest

from modules.yaml_experiment import YamlToExperimentConverter



PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_DIRECTORY = PROJECT_ROOT / "yaml"
VARIANT_YAML_PATHS = sorted(
    path
    for path in YAML_DIRECTORY.rglob("*.yaml")
    if path.stem.endswith(("_normalized", "_standardized"))
)



@pytest.mark.parametrize("yaml_path", VARIANT_YAML_PATHS)
def test_pipeline_name_identifies_preprocessing_variant(yaml_path:Path) -> None:
    preprocessing_variant = yaml_path.stem.rsplit("_", maxsplit=1)[-1]

    experiment = YamlToExperimentConverter().convert(yaml_path)
    pipeline_name = experiment.config.data_pipeline_config.name

    assert pipeline_name is not None
    assert pipeline_name.endswith(preprocessing_variant)
