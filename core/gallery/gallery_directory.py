import hashlib
import json
from pathlib import Path
from typing import Any

from ..data_pipeline.data_pipeline_config import DataPipelineConfig



GALLERY_DIRECTORY = Path("outputs/gallery")



def compute_config_hash(
    config:DataPipelineConfig,
    random_seed:int|None,
) -> str:
    representation = {
        "name": config.name,
        "source_reader": _extract_init_params(config.source_reader),
        "segmenter": _extract_init_params(config.segmenter),
        "transformer": _extract_transformer_params(config.transformer),
        "padder": _extract_init_params(config.padder),
        "splitter": _extract_init_params(config.splitter),
        "seed": random_seed,
    }
    data = json.dumps(representation, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()[:12]


def resolve_gallery_directory(
    config:DataPipelineConfig,
    gallery_directory:Path,
) -> Path:
    config_hash = compute_config_hash(config, None)
    name = config.name

    if name is not None:
        base = f"{name}_{config_hash}"
    else:
        base = config_hash

    candidate = gallery_directory / base
    if not candidate.exists():
        return candidate

    existing_hash = candidate.name.removeprefix(f"{name}_") if name is not None else candidate.name
    if existing_hash == config_hash:
        return candidate

    suffix = 1
    while True:
        candidate = gallery_directory / f"{base}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _extract_init_params(component:Any) -> dict[str, Any]|None:
    if component is None:
        return None
    parameters = {
        name: value
        for name, value in vars(component).items()
        if _is_hashable_primitive(value)
    }
    return {
        "type": f"{component.__class__.__module__}.{component.__class__.__qualname__}",
        "parameters": parameters,
    }


def _extract_transformer_params(
    transformer:Any|list[Any]|None,
) -> Any:
    if transformer is None:
        return None
    if isinstance(transformer, list):
        return [_extract_init_params(t) for t in transformer]
    return _extract_init_params(transformer)


def _is_hashable_primitive(value:Any) -> bool:
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, Path):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_hashable_primitive(item) for item in value)
    return False
