

from dataclasses import fields, is_dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from ..data_pipeline.data_pipeline_config import DataPipelineConfig
from ..data_pipeline.intermediary import Example
from .gallery_directory import (
    GALLERY_DIRECTORY,
    compute_config_hash as _compute_cache_hash,
    resolve_gallery_directory,
)


FEATURE_COLORMAP = "plasma"
PATIENT_ID_METADATA_KEY = "patient_id"


class ExampleGalleryGenerator:

    def __init__(
        self,
        data_pipeline_config:DataPipelineConfig,
        gallery_directory:Path=GALLERY_DIRECTORY,
        num_examples:int=10,
        random_seed:int|None=None,
        class_names:dict[int, str]|None=None,
        x_axis_label:str="Frame",
        y_axis_label:str="Feature bin",
        colorbar_label:str="Value",
        regenerate:bool=False,
    ) -> None:
        self.data_pipeline_config = data_pipeline_config
        self.gallery_directory = gallery_directory
        self.num_examples = num_examples
        self.random_seed = random_seed
        self.class_names = class_names
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.colorbar_label = colorbar_label
        self.regenerate = regenerate
        return

    def generate(self, examples:list[Example]) -> Path:
        gallery_dir = self._gallery_directory()
        gallery_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = gallery_dir / "examples.pdf"

        if not self.regenerate and pdf_path.exists():
            return pdf_path

        save_data_pipeline_config(
            self.data_pipeline_config,
            gallery_dir / "data_pipeline_config.txt",
        )

        save_examples_pdf(
            examples=examples,
            path=pdf_path,
            num_examples=self.num_examples,
            seed=self.random_seed,
            class_names=self.class_names,
            x_axis_label=self.x_axis_label,
            y_axis_label=self.y_axis_label,
            colorbar_label=self.colorbar_label,
        )
        return pdf_path

    def save_config_text(self, path:Path) -> None:
        save_data_pipeline_config(self.data_pipeline_config, path)
        return

    def _gallery_directory(self) -> Path:
        return resolve_gallery_directory(
            self.data_pipeline_config,
            self.gallery_directory,
        )


def save_data_pipeline_config(
    config:DataPipelineConfig,
    path:Path,
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(_format_data_pipeline_config(config))
        f.write("\n")
    return


def save_examples_pdf(
    examples:list[Example],
    path:Path,
    num_examples:int=10,
    seed:int|None=None,
    class_names:dict[int, str]|None=None,
    x_axis_label:str="Frame",
    y_axis_label:str="Feature bin",
    colorbar_label:str="Value",
) -> list[int]:
    if not examples:
        raise ValueError("examples list must not be empty")
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")

    rng = np.random.default_rng(seed)
    sample_count = min(len(examples), num_examples)
    indices = _select_example_indices(examples, sample_count, rng)
    selected = [examples[i] for i in indices]

    color_limit = _compute_color_limit(selected)

    with PdfPages(path) as pdf:
        for index, example in zip(indices, selected):
            _save_example_page(
                pdf,
                example,
                int(index),
                color_limit,
                class_names,
                x_axis_label,
                y_axis_label,
                colorbar_label,
            )
    return indices.tolist()


def _select_example_indices(
    examples:list[Example],
    sample_count:int,
    rng:np.random.Generator,
) -> np.ndarray:
    labels = sorted({example.label for example in examples})
    selected_indices = []

    for position, label in enumerate(labels):
        target_count = sample_count // len(labels)
        if position < sample_count % len(labels):
            target_count += 1

        label_indices = [
            index
            for index, example in enumerate(examples)
            if example.label == label
        ]
        target_count = min(target_count, len(label_indices))
        selected_indices.extend(
            _select_label_indices(label_indices, examples, target_count, rng)
        )

    remaining_count = sample_count - len(selected_indices)
    if remaining_count > 0:
        remaining_indices = np.setdiff1d(
            np.arange(len(examples)),
            selected_indices,
        )
        selected_indices.extend(
            rng.choice(remaining_indices, size=remaining_count, replace=False)
        )

    return np.array(selected_indices, dtype=int)


def _select_label_indices(
    label_indices:list[int],
    examples:list[Example],
    target_count:int,
    rng:np.random.Generator,
) -> list[int]:
    indices_by_patient:dict[str, list[int]] = {}
    for index in label_indices:
        patient_id = str(
            examples[index].metadata.get(PATIENT_ID_METADATA_KEY, index)
        )
        indices_by_patient.setdefault(patient_id, []).append(index)

    patient_ids = list(indices_by_patient)
    selected_indices = []
    for patient_id in rng.permutation(patient_ids):
        patient_indices = indices_by_patient[patient_id]
        selected_indices.append(int(rng.choice(patient_indices)))
        if len(selected_indices) == target_count:
            return selected_indices

    remaining_indices = [
        index
        for index in label_indices
        if index not in selected_indices
    ]
    remaining_count = target_count - len(selected_indices)
    if remaining_count > 0:
        selected_indices.extend(
            rng.choice(remaining_indices, size=remaining_count, replace=False)
        )

    return selected_indices


def _compute_color_limit(examples:list[Example]) -> float:
    values = np.concatenate([ex.value.ravel() for ex in examples])
    color_limit = float(np.percentile(np.abs(values), 99))
    return color_limit if color_limit > 0 else 1.0


def _save_example_page(
    pdf:PdfPages,
    example:Example,
    sample_index:int,
    color_limit:float,
    class_names:dict[int, str]|None,
    x_axis_label:str,
    y_axis_label:str,
    colorbar_label:str,
) -> None:
    color_normalization = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0,
        vmax=color_limit,
    )
    figure, axis = plt.subplots(figsize=(11.69, 8.27), layout="constrained")

    label_text = (
        class_names.get(example.label, str(example.label))
        if class_names
        else str(example.label)
    )
    _plot_feature_values(
        axis,
        example.value,
        color_normalization,
        x_axis_label,
        y_axis_label,
        colorbar_label,
        figure,
    )
    figure.suptitle(
        f"Example #{sample_index} | Label: {label_text}",
        fontsize=20,
    )

    pdf.savefig(figure)
    plt.close(figure)
    return


def _plot_feature_values(
    axis,
    values:np.ndarray,
    color_normalization:TwoSlopeNorm,
    x_axis_label:str,
    y_axis_label:str,
    colorbar_label:str,
    figure,
) -> None:
    if values.ndim == 1:
        frames = np.arange(values.size)
        axis.plot(frames, values, color="midnightblue", linewidth=1.25)
        axis.set(xlabel=x_axis_label, ylabel=y_axis_label)
        axis.grid(axis="y", alpha=0.25)
        return

    if values.ndim != 2:
        raise ValueError("example features must be a one- or two-dimensional array")

    image = axis.imshow(
        values.T,
        aspect="auto",
        origin="lower",
        cmap=FEATURE_COLORMAP,
        norm=color_normalization,
    )
    axis.set(xlabel=x_axis_label, ylabel=y_axis_label)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    return


def _format_data_pipeline_config(config:DataPipelineConfig) -> str:
    return "\n\n".join(
        f"{name} = {_format_value(getattr(config, name), indentation=0)}"
        for name in vars(config)
        if not name.startswith("_")
    )


def _format_value(value:Any, indentation:int) -> str:
    if isinstance(value, Path):
        return pformat(str(value))

    if isinstance(value, (str, int, float, bool, type(None))):
        return pformat(value)

    if isinstance(value, dict):
        return _format_dictionary(value, indentation)

    if isinstance(value, (list, tuple)):
        return _format_sequence(value, indentation)

    if isinstance(value, (list,)):
        return _format_sequence(value, indentation)

    parameters = _get_display_parameters(value)
    return _format_object(value.__class__.__name__, parameters, indentation)


def _format_dictionary(value:dict[Any, Any], indentation:int) -> str:
    if not value:
        return "{}"

    nested_indentation = indentation + 1
    prefix = "    " * nested_indentation
    entries = [
        f"{prefix}{key!r}: {_format_value(item, nested_indentation)}"
        for key, item in value.items()
    ]
    closing_prefix = "    " * indentation
    return "{\n" + ",\n".join(entries) + f"\n{closing_prefix}}}"


def _format_sequence(value:list[Any]|tuple[Any, ...], indentation:int) -> str:
    if not value:
        return "[]"

    if all(_is_scalar(item) for item in value):
        return pformat(value, sort_dicts=False)

    nested_indentation = indentation + 1
    prefix = "    " * nested_indentation
    entries = [
        f"{prefix}{_format_value(item, nested_indentation)}"
        for item in value
    ]
    closing_prefix = "    " * indentation
    return "[\n" + ",\n".join(entries) + f"\n{closing_prefix}]"


def _format_object(
    class_name:str,
    parameters:dict[str, Any],
    indentation:int,
) -> str:
    if not parameters:
        return f"{class_name}()"

    nested_indentation = indentation + 1
    prefix = "    " * nested_indentation
    entries = [
        f"{prefix}{name}={_format_value(parameter, nested_indentation)}"
        for name, parameter in parameters.items()
    ]
    closing_prefix = "    " * indentation
    return f"{class_name}(\n" + ",\n".join(entries) + f"\n{closing_prefix})"


def _is_scalar(value:Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), Path))


def _get_display_parameters(value:Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: getattr(value, field.name)
            for field in fields(value)
        }
    return _get_public_parameters(value)


def _get_public_parameters(value:Any) -> dict[str, Any]:
    return {
        name: parameter
        for name, parameter in vars(value).items()
        if _is_display_parameter(name, parameter)
    }


def _is_display_parameter(name:str, parameter:Any) -> bool:
    if name.startswith("_") or name == "training":
        return False
    if isinstance(parameter, (np.ndarray,)):
        return False
    return True
