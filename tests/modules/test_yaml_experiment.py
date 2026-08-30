from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.data_pipeline.preprocessing import (
    FeatureWiseNormalization,
    FeatureWiseStandardization,
    LogMelSpectrogram,
    MFCC,
)
from core.experiment import ExperimentOrchestrator
from core.metrics import MetricsConfig
from core.model.architectures.LeNet1D import LeNet1D
from core.model.architectures.LeNet2D import LeNet2D
from core.model.architectures.MLP import MLP
from core.model.architectures.PatchTST import PatchTST
from modules.yaml_experiment import (
    YamlToExperimentConverter,
    YamlExperimentError,
    main,
)



PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_YAML_PATH = (
    PROJECT_ROOT
    / "yaml"
    / "run"
    / "mfcc_sliding_windows_transformer.yaml"
)



def test_converter_builds_complete_sample() -> None:
    experiment = YamlToExperimentConverter().convert(SAMPLE_YAML_PATH)

    assert isinstance(experiment, ExperimentOrchestrator)
    assert experiment.experiment_id == "mfcc_sliding_windows_transformer"
    assert experiment.config.data_pipeline_config.name == "mfcc_sliding_windows"
    assert experiment.config.data_pipeline_config.segmenter.stride == 4100
    assert isinstance(experiment.config.data_pipeline_config.transformer, MFCC)
    assert experiment.config.data_pipeline_config.padder is None
    assert experiment.config.model_config.architecture.seq_len == 52
    assert isinstance(experiment.config.metrics_config, MetricsConfig)
    assert isinstance(experiment.config.metrics_config.metrics, tuple)
    assert len(experiment.config.metrics_config.metrics) == 11


@pytest.mark.parametrize(
    (
        "yaml_name",
        "transformer_type",
        "architecture_type",
        "include_grad_cam",
    ),
    [
        ("run/mfcc_sliding_windows_mlp.yaml", MFCC, MLP, False),
        ("run/mfcc_sliding_windows_lenet_1d.yaml", MFCC, LeNet1D, True),
        (
            "run/log_mel_spectrogram_sliding_windows_mlp.yaml",
            LogMelSpectrogram,
            MLP,
            False,
        ),
        (
            "run/log_mel_spectrogram_sliding_windows_lenet_1d.yaml",
            LogMelSpectrogram,
            LeNet1D,
            True,
        ),
        ("run/mfcc_sliding_windows_lenet_2d.yaml", MFCC, LeNet2D, False),
        (
            "run/log_mel_spectrogram_sliding_windows_lenet_2d.yaml",
            LogMelSpectrogram,
            LeNet2D,
            False,
        ),
        (
            "run/log_mel_spectrogram_sliding_windows_transformer.yaml",
            LogMelSpectrogram,
            PatchTST,
            False,
        ),
    ],
)
def test_converter_builds_additional_architecture_samples(
    yaml_name:str,
    transformer_type:type,
    architecture_type:type,
    include_grad_cam:bool,
) -> None:
    experiment = YamlToExperimentConverter().convert(
        PROJECT_ROOT / "yaml" / yaml_name,
    )

    assert isinstance(
        experiment.config.model_config.architecture,
        architecture_type,
    )
    assert isinstance(
        experiment.config.data_pipeline_config.transformer,
        transformer_type,
    )
    assert (
        experiment.config.persistence_config.include_grad_cam
        is include_grad_cam
    )

    if yaml_name == "run/log_mel_spectrogram_sliding_windows_transformer.yaml":
        assert experiment.config.model_config.architecture.seq_len == 52


@pytest.mark.parametrize(
    "yaml_name",
    [
        "dummy/mfcc_sliding_windows_mlp_dummy.yaml",
        "dummy/mfcc_sliding_windows_lenet_1d_dummy.yaml",
        "dummy/mfcc_sliding_windows_transformer_dummy.yaml",
        "dummy/log_mel_spectrogram_sliding_windows_mlp_dummy.yaml",
        "dummy/log_mel_spectrogram_sliding_windows_lenet_1d_dummy.yaml",
        "dummy/log_mel_spectrogram_sliding_windows_transformer_dummy.yaml",
    ],
)
def test_converter_builds_two_epoch_dummy_samples(yaml_name:str) -> None:
    experiment = YamlToExperimentConverter().convert(
        PROJECT_ROOT / "yaml" / yaml_name,
    )

    assert experiment.experiment_id.endswith("_dummy")
    assert experiment.config.training_config.num_epochs == 2


def test_converter_builds_all_standardized_samples() -> None:
    standardized_run_yaml_paths = sorted(
        (PROJECT_ROOT / "yaml" / "all").glob("*_standardized.yaml")
    )
    standardized_dummy_yaml_paths = sorted(
        (PROJECT_ROOT / "yaml" / "dummy").glob("*_standardized.yaml")
    )
    standardized_yaml_paths = (
        standardized_run_yaml_paths + standardized_dummy_yaml_paths
    )

    assert len(standardized_run_yaml_paths) == 8
    assert len(standardized_dummy_yaml_paths) == 6

    for yaml_path in standardized_yaml_paths:
        experiment = YamlToExperimentConverter().convert(yaml_path)
        transformers = experiment.config.data_pipeline_config.transformer

        assert experiment.experiment_id.endswith("_standardized")
        assert isinstance(transformers, list)
        assert isinstance(transformers[0], (MFCC, LogMelSpectrogram))
        assert isinstance(transformers[1], FeatureWiseStandardization)


def test_converter_builds_all_normalized_run_samples() -> None:
    normalized_yaml_paths = sorted(
        (
            PROJECT_ROOT
            / "yaml"
            / "run"
            / "normalized_examples"
        ).rglob("*.yaml")
    )

    assert len(normalized_yaml_paths) == 8

    for yaml_path in normalized_yaml_paths:
        experiment = YamlToExperimentConverter().convert(yaml_path)
        transformers = experiment.config.data_pipeline_config.transformer

        assert experiment.experiment_id.endswith("_normalized")
        assert isinstance(transformers, list)
        assert isinstance(transformers[0], (MFCC, LogMelSpectrogram))
        assert isinstance(transformers[1], FeatureWiseNormalization)


def test_converter_builds_paths_lists_and_nulls(tmp_path:Path) -> None:
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(_create_list_transformer_yaml(), encoding="utf-8")

    experiment = YamlToExperimentConverter().convert(yaml_path)
    pipeline_config = experiment.config.data_pipeline_config

    assert pipeline_config.source_reader.root_path == Path("data/test")
    assert isinstance(pipeline_config.transformer, list)
    assert isinstance(pipeline_config.transformer[0], MFCC)
    assert isinstance(pipeline_config.transformer[1], LogMelSpectrogram)
    assert pipeline_config.padder is None


def test_converter_builds_lenet_2d_architecture(tmp_path:Path) -> None:
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(
        _create_lenet_2d_yaml(),
        encoding="utf-8",
    )

    experiment = YamlToExperimentConverter().convert(yaml_path)
    architecture = experiment.config.model_config.architecture

    assert isinstance(architecture, LeNet2D)
    assert architecture.conv_channels == [4, 8]
    assert architecture.linear_dims == [16]
    assert architecture.dropout == 0.3
    assert architecture.output_dim == 2


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("- not\n- a\n- mapping\n", "must contain a mapping"),
        (
            "experiment_id: test\nconfig:\n  type: UnknownConfig\n",
            "unknown type 'UnknownConfig'",
        ),
        (
            "experiment_id: test\nconfig:\n  type: ExperimentConfig\n",
            "cannot create config as ExperimentConfig",
        ),
        (
            "experiment_id: test\nconfig: [\n",
            "invalid YAML in",
        ),
    ],
)
def test_converter_rejects_invalid_yaml(
    tmp_path:Path,
    yaml_text:str,
    message:str,
) -> None:
    yaml_path = tmp_path / "invalid.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(YamlExperimentError, match=message):
        YamlToExperimentConverter().convert(yaml_path)


def test_converter_reports_constructor_validation_failure(
    tmp_path:Path,
) -> None:
    yaml_text = _create_list_transformer_yaml().replace(
        "  persistence_config:\n    type: PersistenceConfig\n",
        "  persistence_config:\n"
        "    type: PersistenceConfig\n"
        "    feature_color_percentile: 0\n",
    )
    yaml_path = tmp_path / "invalid_persistence.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(
        YamlExperimentError,
        match=r"config\.persistence_config.*feature_color_percentile",
    ):
        YamlToExperimentConverter().convert(yaml_path)


def test_main_converts_and_trains_experiment(
    monkeypatch:pytest.MonkeyPatch,
) -> None:
    experiment = Mock()
    monkeypatch.setattr(
        "modules.yaml_experiment.get_arguments",
        lambda: SimpleNamespace(yaml_path=Path("experiment.yaml")),
    )
    monkeypatch.setattr(
        "modules.yaml_experiment.YamlToExperimentConverter.convert",
        lambda self, path: experiment,
    )

    main()

    experiment.train_model.assert_called_once_with()


def _create_lenet_2d_yaml() -> str:
    mlp_architecture = """\
    architecture:
      type: MLP
      linear_dims: [8]
      dropout: 0.1
      output_dim: 2
"""
    lenet_2d_architecture = """\
    architecture:
      type: LeNet2D
      conv_channels: [4, 8]
      linear_dims: [16]
      dropout: 0.3
      output_dim: 2
"""
    yaml_text = _create_list_transformer_yaml().replace(
        mlp_architecture,
        lenet_2d_architecture,
    )
    return yaml_text


def _create_list_transformer_yaml() -> str:
    return """\
experiment_id: list_transformer_test
config:
  type: ExperimentConfig
  data_pipeline_config:
    type: DataPipelineConfig
    name: list_transformer
    source_reader:
      type: ElderlyCoughAudioSourceReader
      root_path:
        type: Path
        value: data/test
    segmenter:
      type: CoughSegmenter
      kept_metadata_key: [patient_id]
    transformer:
      - type: MFCC
      - type: LogMelSpectrogram
    padder: null
    splitter:
      type: DataSplitter
      group_metadata_key: patient_id
  model_config:
    type: ModelConfig
    architecture:
      type: MLP
      linear_dims: [8]
      dropout: 0.1
      output_dim: 2
    behavior:
      type: ClassificationBehavior
  training_config:
    type: TrainingConfig
    random_seed: 42
    num_epochs: 1
    criterion_name: cross_entropy
    optimizer_name: adamw
    learning_rate: 0.0001
    weight_decay: 0.001
    batch_size: 2
    num_workers: 0
    drop_last: false
  metrics_config:
    type: MetricsConfig
    metrics:
      - type: AccuracyMetric
  persistence_config:
    type: PersistenceConfig
"""
