from pathlib import Path

import numpy as np
import pytest

from core.data_pipeline.preprocessing import (
    DownSampler,
    FeatureWiseNormalization,
    FeatureWiseStandardization,
)
from core.model.architectures.LeNet1D import LeNet1D
from core.model.architectures.MLP import MLP
from core.model.architectures.PatchTST import PatchTST
from modules.yaml_experiment import YamlToExperimentConverter



PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_DIRECTORY = PROJECT_ROOT / "yaml" / "run"



@pytest.mark.parametrize(
    ("yaml_name", "architecture_type", "batch_size"),
    [
        ("downsampled_waveform_sliding_windows_mlp.yaml", MLP, 16),
        ("downsampled_waveform_sliding_windows_lenet_1d.yaml", LeNet1D, 16),
        ("downsampled_waveform_sliding_windows_transformer.yaml", PatchTST, 1),
    ],
)
def test_converter_builds_downsampled_waveform_experiments(
    yaml_name:str,
    architecture_type:type,
    batch_size:int,
) -> None:
    experiment = YamlToExperimentConverter().convert(
        YAML_DIRECTORY / "raw" / yaml_name,
    )
    pipeline_config = experiment.config.data_pipeline_config
    architecture = experiment.config.model_config.architecture

    assert pipeline_config.name == "downsampled_waveform_sliding_windows"
    assert isinstance(pipeline_config.transformer, DownSampler)
    assert isinstance(architecture, architecture_type)
    assert experiment.config.training_config.batch_size == batch_size

    waveform = np.zeros(8200, dtype=np.float32)
    downsampled_waveform = pipeline_config.transformer._resample_value(waveform)
    assert downsampled_waveform.shape == (2050,)

    if isinstance(architecture, PatchTST):
        assert architecture.seq_len == 2050
        assert architecture.enc_in_feature == 1
        assert architecture.patch_len == 1


@pytest.mark.parametrize(
    ("model_name", "architecture_type", "batch_size"),
    [
        ("mlp", MLP, 16),
        ("lenet_1d", LeNet1D, 16),
        ("transformer", PatchTST, 1),
    ],
)
@pytest.mark.parametrize(
    ("preprocessing_name", "preprocessing_type"),
    [
        ("normalized", FeatureWiseNormalization),
        ("standardized", FeatureWiseStandardization),
    ],
)
def test_converter_builds_preprocessed_waveform_experiments(
    model_name:str,
    architecture_type:type,
    batch_size:int,
    preprocessing_name:str,
    preprocessing_type:type,
) -> None:
    yaml_name = (
        f"downsampled_waveform_sliding_windows_{model_name}_"
        f"{preprocessing_name}.yaml"
    )
    experiment = YamlToExperimentConverter().convert(
        YAML_DIRECTORY / preprocessing_name / yaml_name,
    )
    pipeline_config = experiment.config.data_pipeline_config
    transformers = pipeline_config.transformer
    architecture = experiment.config.model_config.architecture

    assert pipeline_config.name.endswith(preprocessing_name)
    assert isinstance(transformers, list)
    assert isinstance(transformers[0], DownSampler)
    assert isinstance(transformers[1], preprocessing_type)
    assert isinstance(architecture, architecture_type)
    assert experiment.config.training_config.batch_size == batch_size

    waveform = np.zeros(8200, dtype=np.float32)
    downsampled_waveform = transformers[0]._resample_value(waveform)
    assert downsampled_waveform.shape == (2050,)

    if isinstance(architecture, PatchTST):
        assert architecture.seq_len == 2050
        assert architecture.enc_in_feature == 1
        assert architecture.patch_len == 1
