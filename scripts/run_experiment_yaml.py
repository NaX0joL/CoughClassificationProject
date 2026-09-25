import argparse
from pathlib import Path
import sys
from typing import Any

import yaml


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data_pipeline.example_construction._segment_labeler import OverlapLabeler
from core.data_pipeline.example_construction._segment_transformer import (
    FeatureWiseNormalization,
    FeatureWiseStandardization,
    LogMelSpectrogram,
    MFCC,
)
from core.data_pipeline.example_construction._series_segmenter import (
    CenteredCoughSegmenter,
    SlidingWindowSegmenter,
)
from core.data_pipeline.example_construction.example_constructor import (
    ExampleConstructor,
)
from core.data_pipeline.loading.oversampler import UniformOversampler
from core.data_pipeline.partitioning.partitioner import Partitioner
from core.data_pipeline.pipelines.standard import StandardDataPipeline
from core.data_pipeline.source_reading.source_reader import (
    DEFAULT_METADATA_PATH,
    DEFAULT_SOURCE_DATA_PARENT_DIRECTORY,
    DEFAULT_TRANSLATION_PATH,
    SourceReader,
)
from core.experiment_orchestrator import (
    ExperimentOrchestrator,
    ExperimentOrchestratorConfig,
)
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
from core.metrics.label_mapping import BinaryInfectionLabelMapping
from core.model.architectures.LeNet1D import LeNet1D
from core.model.architectures.LeNet2D import LeNet2D
from core.model.architectures.MLP import MLP
from core.model.architectures.PatchTST import PatchTST
from core.model.architectures.ResNet import ResNet
from core.model.architectures.TemporalStatisticsMLP import TemporalStatisticsMLP
from core.model.behavior.classification_behavior import ClassificationBehavior
from core.model import ModelConfig
from core.trainer import TrainingConfig
from modules.email_notification.email_sender import send_email


_METRIC_TYPES = {
    "roc_auc": RocAucMetric,
    "pr_auc": PRAucMetric,
    "precision": PrecisionMetric,
    "recall": RecallMetric,
    "specificity": SpecificityMetric,
    "f1_score": F1ScoreMetric,
    "accuracy": AccuracyMetric,
    "macro_accuracy": MacroAccuracyMetric,
    "macro_f1_score": MacroF1ScoreMetric,
    "macro_precision": MacroPrecisionMetric,
    "macro_recall": MacroRecallMetric,
}



def _require_section(
    config:dict[str, Any],
    name:str,
) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"missing required section '{name}'")
    return section


def _get_component_type(
    section:dict[str, Any],
    name:str,
) -> str:
    component_type = section.get("type")
    if not isinstance(component_type, str):
        raise ValueError(f"missing component type for '{name}'")
    return component_type


def _path_value(
    section:dict[str, Any],
    name:str,
    default:Path,
) -> Path:
    value = section.get(name, default)
    return Path(value)


def _build_source_reader(section:dict[str, Any]) -> SourceReader:
    component_type = _get_component_type(section, "source_reader")
    if component_type != "SourceReader":
        raise ValueError(
            f"unsupported source_reader type: {component_type!r}",
        )

    return SourceReader(
        metadata_path=_path_value(section, "metadata_path", DEFAULT_METADATA_PATH),
        excel_sheet_name=section.get("excel_sheet_name", "dynamo"),
        translation_path=_path_value(
            section,
            "translation_path",
            DEFAULT_TRANSLATION_PATH,
        ),
        source_data_parent_directory=_path_value(
            section,
            "source_data_parent_directory",
            DEFAULT_SOURCE_DATA_PARENT_DIRECTORY,
        ),
    )


def _build_partitioner(section:dict[str, Any]) -> Partitioner:
    component_type = _get_component_type(section, "partitioner")
    if component_type != "Partitioner":
        raise ValueError(
            f"unsupported partitioner type: {component_type!r}",
        )

    return Partitioner(
        label_key=section.get("label_key", "isInfectious"),
        group_key=section.get("group_key", "PatientID"),
        number_of_outer_folds=int(section.get("number_of_outer_folds", 5)),
        number_of_inner_folds=int(section.get("number_of_inner_folds", 5)),
        random_seed=section.get("random_seed", 42),
    )


def _build_segmenter(
    section:dict[str, Any],
    name:str,
) -> SlidingWindowSegmenter|CenteredCoughSegmenter:
    component_type = _get_component_type(section, name)
    if component_type == "SlidingWindowSegmenter":
        return SlidingWindowSegmenter(
            window_size=int(section.get("window_size", 8_000)),
            stride=int(section.get("stride", 4_000)),
            drop_last=bool(section.get("drop_last", True)),
        )
    if component_type == "CenteredCoughSegmenter":
        return CenteredCoughSegmenter(
            window_size=int(section.get("window_size", 8_000)),
        )
    raise ValueError(
        f"unsupported {name} type: {component_type!r}",
    )


def _build_segment_labeler(section:dict[str, Any]) -> OverlapLabeler:
    component_type = _get_component_type(section, "segment_labeler")
    if component_type != "OverlapLabeler":
        raise ValueError(
            f"unsupported segment_labeler type: {component_type!r}",
        )
    return OverlapLabeler(
        overlap_threshold=float(section.get("overlap_threshold", 0.7)),
        no_overlap_label=int(section.get("no_overlap_label", 0)),
    )


def _build_transformer(section:dict[str, Any]) -> object:
    component_type = _get_component_type(section, "transformer")
    if component_type == "MFCC":
        return MFCC(
            sampling_rate=int(section.get("sampling_rate", 16_000)),
            n_fft=int(section.get("n_fft", 400)),
            win_length=int(section.get("win_length", 400)),
            hop_length=int(section.get("hop_length", 160)),
            n_mels=int(section.get("n_mels", 40)),
            n_mfcc=int(section.get("n_mfcc", 40)),
        )
    if component_type == "LogMelSpectrogram":
        return LogMelSpectrogram(
            sampling_rate=int(section.get("sampling_rate", 16_000)),
            n_fft=int(section.get("n_fft", 400)),
            win_length=int(section.get("win_length", 400)),
            hop_length=int(section.get("hop_length", 160)),
            n_mels=int(section.get("n_mels", 40)),
            log_offset=float(section.get("log_offset", 1e-6)),
        )
    if component_type == "FeatureWiseStandardization":
        return FeatureWiseStandardization()
    if component_type == "FeatureWiseNormalization":
        return FeatureWiseNormalization()
    raise ValueError(
        f"unsupported transformer type: {component_type!r}",
    )


def _build_transformers(
    value:object,
) -> object|list[object]|None:
    if value is None:
        return None
    if isinstance(value, list):
        return [_build_transformer(item) for item in value]
    if isinstance(value, dict):
        return _build_transformer(value)
    raise ValueError("transformer must be a mapping or list of mappings")


def _build_example_constructor(section:dict[str, Any]) -> ExampleConstructor:
    component_type = _get_component_type(section, "example_constructor")
    if component_type != "ExampleConstructor":
        raise ValueError(
            f"unsupported example_constructor type: {component_type!r}",
        )

    return ExampleConstructor(
        sampling_rate=int(section.get("sampling_rate", 16_000)),
        train_segmenter=_build_segmenter(
            _require_section(section, "train_segmenter"),
            "train_segmenter",
        ),
        validation_segmenter=_build_segmenter(
            _require_section(section, "validation_segmenter"),
            "validation_segmenter",
        ),
        test_segmenter=_build_segmenter(
            _require_section(section, "test_segmenter"),
            "test_segmenter",
        ),
        segment_labeler=_build_segment_labeler(
            _require_section(section, "segment_labeler"),
        ),
        transformer=_build_transformers(section.get("transformer")),
        cache_directory=Path(
            section.get(
                "cache_directory",
                "outputs/cache/elderly_cough_audio_waveform_16khz_v3",
            ),
        ),
        verbose=bool(section.get("verbose", False)),
    )


def _build_oversampler(section:dict[str, Any]|None) -> UniformOversampler|None:
    if section is None:
        return None
    component_type = _get_component_type(section, "oversampler")
    if component_type != "UniformOversampler":
        raise ValueError(
            f"unsupported oversampler type: {component_type!r}",
        )
    return UniformOversampler(
        random_seed=section.get("random_seed", 42),
    )


def _required_value(
    section:dict[str, Any],
    section_name:str,
    name:str,
) -> object:
    if name not in section:
        raise ValueError(f"missing required field '{section_name}.{name}'")
    return section[name]


def _build_model_config(
    section:dict[str, Any],
    label_mapping:BinaryInfectionLabelMapping|None=None,
) -> ModelConfig:
    component_type = _get_component_type(section, "model")
    if component_type == "MLP":
        architecture = _build_mlp(section)
    elif component_type in {"LeNet1D", "LeNet"}:
        architecture = _build_lenet_1d(section)
    elif component_type == "LeNet2D":
        architecture = _build_lenet_2d(section)
    elif component_type == "ResNet":
        architecture = _build_resnet(section)
    elif component_type == "TemporalStatisticsMLP":
        architecture = _build_temporal_statistics_mlp(section)
    elif component_type == "PatchTST":
        architecture = _build_patchtst(section)
    else:
        raise ValueError(f"unsupported model type: {component_type!r}")
    if label_mapping is not None:
        _output_dim(_model_value(section, "output_dim"), label_mapping)

    return ModelConfig(
        architecture=architecture,
        behavior=ClassificationBehavior(),
    )


def _reject_unknown_model_fields(
    section:dict[str, Any],
    allowed_fields:set[str],
) -> None:
    unknown_fields = sorted(set(section) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"unknown model field: {unknown_fields[0]!r}")
    return


def _model_value(section:dict[str, Any], name:str) -> object:
    return _required_value(section, "model", name)


def _positive_int(value:object, name:str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"model.{name} must be a positive integer")
    return value


def _dimension_list(value:object, name:str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise ValueError(f"model.{name} must be a non-empty list of positive integers")
    return value


def _dropout(value:object, name:str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"model.{name} must be numeric")
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"model.{name} must be between 0 and 1")
    return value


def _output_dim(value:object, mapping:BinaryInfectionLabelMapping|None=None) -> int:
    output_dim = _positive_int(value, "output_dim")
    if output_dim != 3:
        raise ValueError("model.output_dim must be exactly 3")
    mapping = mapping or BinaryInfectionLabelMapping()
    raw_classes = {
        raw_class
        for group in mapping.raw_class_groups
        for raw_class in group
    }
    if raw_classes != {0, 1, 2}:
        raise ValueError("model.output_dim requires raw label mapping coverage {0, 1, 2}")
    return output_dim


def _build_mlp(section:dict[str, Any]) -> MLP:
    _reject_unknown_model_fields(section, {"type", "linear_dims", "dropout", "output_dim"})
    return MLP(
        linear_dims=_dimension_list(_model_value(section, "linear_dims"), "linear_dims"),
        dropout=_dropout(_model_value(section, "dropout"), "dropout"),
        output_dim=_output_dim(_model_value(section, "output_dim")),
    )


def _build_lenet_1d(section:dict[str, Any]) -> LeNet1D:
    _reject_unknown_model_fields(
        section,
        {"type", "conv_channels", "linear_dims", "dropout", "output_dim"},
    )
    return LeNet1D(
        conv_channels=_dimension_list(_model_value(section, "conv_channels"), "conv_channels"),
        linear_dims=_dimension_list(_model_value(section, "linear_dims"), "linear_dims"),
        dropout=_dropout(_model_value(section, "dropout"), "dropout"),
        output_dim=_output_dim(_model_value(section, "output_dim")),
    )


def _build_lenet_2d(section:dict[str, Any]) -> LeNet2D:
    _reject_unknown_model_fields(
        section,
        {"type", "conv_channels", "linear_dims", "dropout", "output_dim"},
    )
    return LeNet2D(
        conv_channels=_dimension_list(_model_value(section, "conv_channels"), "conv_channels"),
        linear_dims=_dimension_list(_model_value(section, "linear_dims"), "linear_dims"),
        dropout=_dropout(_model_value(section, "dropout"), "dropout"),
        output_dim=_output_dim(_model_value(section, "output_dim")),
    )


def _build_resnet(section:dict[str, Any]) -> ResNet:
    _reject_unknown_model_fields(
        section,
        {"type", "block_channels", "blocks_per_stage", "output_dim"},
    )
    return ResNet(
        block_channels=_dimension_list(_model_value(section, "block_channels"), "block_channels"),
        blocks_per_stage=_positive_int(
            _model_value(section, "blocks_per_stage"),
            "blocks_per_stage",
        ),
        output_dim=_output_dim(_model_value(section, "output_dim")),
    )


def _build_temporal_statistics_mlp(section:dict[str, Any]) -> TemporalStatisticsMLP:
    _reject_unknown_model_fields(section, {"type", "linear_dims", "dropout", "output_dim"})
    return TemporalStatisticsMLP(
        linear_dims=_dimension_list(_model_value(section, "linear_dims"), "linear_dims"),
        dropout=_dropout(_model_value(section, "dropout"), "dropout"),
        output_dim=_output_dim(_model_value(section, "output_dim")),
    )


def _build_patchtst(section:dict[str, Any]) -> PatchTST:
    allowed_fields = {
        "type", "seq_len", "patch_len", "stride", "enc_in_feature",
        "e_layers_num", "n_heads_num", "d_model", "d_ff", "dropout",
        "fc_dropout", "head_dropout", "attn_dropout", "use_pre_norm",
        "max_seq_len", "norm", "act", "res_attention", "store_attn",
        "pe", "learn_pe", "head_type", "individual", "padding_patch",
        "use_revin", "use_affine", "use_subtract_last",
        "use_positional_encoding", "decomposition", "verbose",
        "output_dim",
    }
    _reject_unknown_model_fields(section, allowed_fields)
    seq_len = _positive_int(_model_value(section, "seq_len"), "seq_len")
    if seq_len != 51:
        raise ValueError("model.seq_len must be 51 for PatchTST")
    patch_len = _positive_int(_model_value(section, "patch_len"), "patch_len")
    if patch_len > seq_len:
        raise ValueError("model.patch_len must not exceed model.seq_len")
    stride = _positive_int(_model_value(section, "stride"), "stride")
    enc_in_feature = _positive_int(_model_value(section, "enc_in_feature"), "enc_in_feature")
    if enc_in_feature != 40:
        raise ValueError("model.enc_in_feature must be 40 for PatchTST")
    e_layers_num = _positive_int(_model_value(section, "e_layers_num"), "e_layers_num")
    n_heads_num = _positive_int(_model_value(section, "n_heads_num"), "n_heads_num")
    d_model = _positive_int(_model_value(section, "d_model"), "d_model")
    if d_model % n_heads_num != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads_num")
    d_ff = _positive_int(_model_value(section, "d_ff"), "d_ff")
    for name in ("dropout", "fc_dropout", "head_dropout", "attn_dropout"):
        _dropout(_model_value(section, name), name)
    for name in ("use_pre_norm", "res_attention", "store_attn", "learn_pe", "individual",
                 "use_revin", "use_affine", "use_subtract_last", "use_positional_encoding",
                 "decomposition", "verbose"):
        if type(_model_value(section, name)) is not bool:
            raise ValueError(f"model.{name} must be a boolean")
    for name in ("norm", "act", "pe", "head_type"):
        if not isinstance(_model_value(section, name), str):
            raise ValueError(f"model.{name} must be a string")
    max_seq_len = _positive_int(_model_value(section, "max_seq_len"), "max_seq_len")
    padding_patch = _model_value(section, "padding_patch")
    if padding_patch is not None and not isinstance(padding_patch, str):
        raise ValueError("model.padding_patch must be a string or null")
    return PatchTST(
        seq_len=seq_len,
        pred_len=_output_dim(_model_value(section, "output_dim")),
        patch_len=patch_len,
        stride=stride,
        enc_in_feature=enc_in_feature,
        e_layers_num=e_layers_num,
        n_heads_num=n_heads_num,
        d_model=d_model,
        d_ff=d_ff,
        dropout=_dropout(_model_value(section, "dropout"), "dropout"),
        fc_dropout=_dropout(_model_value(section, "fc_dropout"), "fc_dropout"),
        head_dropout=_dropout(_model_value(section, "head_dropout"), "head_dropout"),
        attn_dropout=_dropout(_model_value(section, "attn_dropout"), "attn_dropout"),
        use_pre_norm=_model_value(section, "use_pre_norm"),
        max_seq_len=max_seq_len,
        norm=_model_value(section, "norm"),
        act=_model_value(section, "act"),
        res_attention=_model_value(section, "res_attention"),
        store_attn=_model_value(section, "store_attn"),
        pe=_model_value(section, "pe"),
        learn_pe=_model_value(section, "learn_pe"),
        head_type=_model_value(section, "head_type"),
        individual=_model_value(section, "individual"),
        padding_patch=padding_patch,
        use_revin=_model_value(section, "use_revin"),
        use_affine=_model_value(section, "use_affine"),
        use_subtract_last=_model_value(section, "use_subtract_last"),
        use_positional_encoding=_model_value(section, "use_positional_encoding"),
        decomposition=_model_value(section, "decomposition"),
        verbose=_model_value(section, "verbose"),
    )


def _build_training_config(section:dict[str, Any]) -> TrainingConfig:
    required_fields = (
        "random_seed",
        "num_epochs",
        "criterion_name",
        "optimizer_name",
        "learning_rate",
        "weight_decay",
    )
    for name in required_fields:
        _required_value(section, "training", name)

    criterion_name = section["criterion_name"]
    if not isinstance(criterion_name, str) or criterion_name != "cross_entropy":
        raise ValueError(
            f"unsupported training criterion: {criterion_name!r}",
        )
    optimizer_name = section["optimizer_name"]
    if (
        not isinstance(optimizer_name, str)
        or optimizer_name not in {"adam", "adamw"}
    ):
        raise ValueError(
            f"unsupported training optimizer: {optimizer_name!r}",
        )
    class_weighting = section.get("class_weighting", "none")
    if (
        not isinstance(class_weighting, str)
        or class_weighting not in {"none", "balanced"}
    ):
        raise ValueError(
            f"unsupported training class weighting: {class_weighting!r}",
        )
    if class_weighting == "balanced":
        raise ValueError("training.class_weighting 'balanced' is not supported for X")

    return TrainingConfig(
        random_seed=int(section["random_seed"]),
        num_epochs=int(section["num_epochs"]),
        criterion_name=criterion_name,
        optimizer_name=optimizer_name,
        learning_rate=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
        batch_size=int(section.get("batch_size", 32)),
        num_workers=int(section.get("num_workers", 0)),
        drop_last=bool(section.get("drop_last", False)),
        class_weighting=class_weighting,
        early_stopping_patience=section.get("early_stopping_patience"),
        load_best_model=bool(section.get("load_best_model", True)),
    )


def _build_evaluation_config(
    section:dict[str, Any],
) -> tuple[MetricsConfig, BinaryInfectionLabelMapping]:
    metric_names = section.get("metrics")
    if not isinstance(metric_names, list) or not metric_names:
        raise ValueError("evaluation.metrics must be a non-empty list")
    if any(not isinstance(name, str) for name in metric_names):
        raise ValueError("evaluation.metrics must contain metric names")
    unknown_names = [name for name in metric_names if name not in _METRIC_TYPES]
    if unknown_names:
        raise ValueError(f"unknown evaluation metric: {unknown_names[0]!r}")
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("evaluation.metrics must not contain duplicate names")

    label_mapping_section = section.get("label_mapping")
    if not isinstance(label_mapping_section, dict):
        raise ValueError("missing required section 'evaluation.label_mapping'")
    raw_class_groups = label_mapping_section.get("raw_class_groups")
    if (
        not isinstance(raw_class_groups, list)
        or len(raw_class_groups) != 2
        or any(
            not isinstance(group, list)
            or not group
            or any(
                not isinstance(raw_class, int)
                or isinstance(raw_class, bool)
                or raw_class < 0
                for raw_class in group
            )
            for group in raw_class_groups
        )
    ):
        raise ValueError(
            "evaluation.label_mapping.raw_class_groups must contain two nonempty integer lists",
        )
    raw_classes = [raw_class for group in raw_class_groups for raw_class in group]
    if len(raw_classes) != len(set(raw_classes)):
        raise ValueError(
            "evaluation.label_mapping.raw_class_groups must contain unique raw IDs",
        )

    class_names = label_mapping_section.get("class_names")
    if (
        not isinstance(class_names, list)
        or len(class_names) != 2
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
    ):
        raise ValueError(
            "evaluation.label_mapping.class_names must contain two nonempty strings",
        )

    return (
        MetricsConfig(metrics=tuple(_METRIC_TYPES[name]() for name in metric_names)),
        BinaryInfectionLabelMapping(
            raw_class_groups=tuple(tuple(group) for group in raw_class_groups),
            class_names=tuple(class_names),
        ),
    )


def build_orchestrator_config(
    config:dict[str, Any],
) -> ExperimentOrchestratorConfig:
    experiment_id = config.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string")
    training_config = _build_training_config(
        _require_section(config, "training"),
    )
    metrics_config, label_mapping = _build_evaluation_config(
        _require_section(config, "evaluation"),
    )
    model_config = _build_model_config(
        _require_section(config, "model"),
        label_mapping,
    )
    return ExperimentOrchestratorConfig(
        experiment_id=experiment_id,
        model_config=model_config,
        training_config=training_config,
        metrics_config=metrics_config,
        label_mapping=label_mapping,
    )


def build_pipeline(config:dict[str, Any]) -> StandardDataPipeline:
    data_pipeline_config = config.get("data_pipeline")
    if not isinstance(data_pipeline_config, dict):
        raise ValueError("missing required section 'data_pipeline'")

    pipeline_section = data_pipeline_config.get("pipeline")
    if not isinstance(pipeline_section, dict):
        raise ValueError("missing required data_pipeline section 'pipeline'")
    pipeline_type = pipeline_section.get("type")
    if not isinstance(pipeline_type, str):
        raise ValueError("missing pipeline type")
    if pipeline_type != "StandardDataPipeline":
        raise ValueError(f"unsupported pipeline type: {pipeline_type!r}")

    source_reader = _build_source_reader(
        _require_section(data_pipeline_config, "source_reader"),
    )
    partitioner = _build_partitioner(
        _require_section(data_pipeline_config, "partitioner"),
    )
    example_constructor = _build_example_constructor(
        _require_section(data_pipeline_config, "example_constructor"),
    )
    oversampler_value = data_pipeline_config.get("oversampler")
    if oversampler_value is not None and not isinstance(oversampler_value, dict):
        raise ValueError("oversampler must be a mapping when provided")
    return StandardDataPipeline(
        source_reader=source_reader,
        partitioner=partitioner,
        example_constructor=example_constructor,
        oversampler=_build_oversampler(oversampler_value),
    )


def build_pipeline_from_yaml(path:Path) -> StandardDataPipeline:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")
    return build_pipeline(config)


DEFAULT_YAML_DIRECTORY = Path("yaml/run/x")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAML_SUFFIXES = {".yaml", ".yml"}



def _display_path(path:Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def expand_yaml_paths(paths:list[Path]) -> list[Path]:
    expanded_paths = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"YAML input path does not exist: {path}")
        
        if path.is_dir():
            directory_paths = sorted(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in YAML_SUFFIXES
            )
            
            if not directory_paths:
                raise ValueError(f"directory contains no YAML files: {path}")
            
            directory_paths.sort(
                key=lambda child: child.relative_to(path).as_posix(),
            )
            expanded_paths.extend(directory_paths)
            
            continue
        
        if not path.is_file() or path.suffix.lower() not in YAML_SUFFIXES:
            raise ValueError(
                "unsupported YAML input path (expected a .yaml/.yml file "
                f"or directory): {path}",
            )
            
        expanded_paths.append(path)
        
    return expanded_paths


def parse_arguments(arguments:list[str]|None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the experiment from one or more YAML files or directories"
        ),
        epilog=(
            "Example: python scripts/run_experiment_yaml.py yaml/run"
        ),
    )
    parser.add_argument(
        "yaml_paths",
        nargs="*",
        type=Path,
        metavar="YAML_PATH",
        help="YAML file(s) or directory path(s) to run sequentially; "
        "defaults to yaml/run/x",
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help="send one final batch summary email",
    )
    parsed_arguments = parser.parse_args(arguments)
    if not parsed_arguments.yaml_paths:
        parsed_arguments.yaml_paths = [DEFAULT_YAML_DIRECTORY]
    if len(parsed_arguments.yaml_paths) == 1:
        parsed_arguments.yaml_path = parsed_arguments.yaml_paths[0]
    return parsed_arguments


def run_yaml(path:Path) -> None:
    display_path = _display_path(path)
    print(f"> Running {display_path}")
    print(f"loading config: {display_path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")

    data_pipeline = build_pipeline(config)
    orchestrator = ExperimentOrchestrator(
        config=build_orchestrator_config(config),
        data_pipeline=data_pipeline,
    )
    orchestrator.run()
    
    print()
    return


def _summary_text(
    paths:list[Path],
    successful_paths:list[Path],
    failures:list[tuple[Path, str]],
) -> str:
    successful_set = set(successful_paths)
    lines = [
        "",
        "Experiment batch summary",
        f"Total: {len(paths)}",
        f"Successes: {len(successful_paths)}",
        f"Failures: {len(failures)}",
        "Successful paths:",
    ]
    lines.extend(f"- {_display_path(path)}" for path in paths if path in successful_set)
    lines.append("Failed paths:")
    lines.extend(
        f"- {_display_path(path)}: {error_text}"
        for path, error_text in failures
    )
    if not failures:
        lines.append("DONE!")
    lines.extend(("",))
    return "\n".join(lines)


def _send_summary_email(summary:str, has_failures:bool) -> None:
    subject = (
        "Experiment batch failed"
        if has_failures
        else "Experiment batch succeeded"
    )
    send_email(subject=subject, body=summary)
    return


def main(arguments:list[str]|None=None) -> None:
    parsed_arguments = parse_arguments(arguments)
    paths = expand_yaml_paths(parsed_arguments.yaml_paths)
    successful_paths = []
    failures = []
    for path in paths:
        try:
            run_yaml(path)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            error_text = str(error)
            failures.append((path, error_text))
            print(f"FAILED {_display_path(path)}: {error_text}")
        else:
            successful_paths.append(path)

    summary = _summary_text(paths, successful_paths, failures)
    print(summary)
    if parsed_arguments.email:
        _send_summary_email(summary, bool(failures))
    return



if __name__ == "__main__":
    main()
