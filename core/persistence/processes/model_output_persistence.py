from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from ...data_pipeline.dataset import ExampleDataset
from ...gallery import ModelOutput, create_model_output
from ...model import FullModel
from ..persistence_config import PersistenceConfig


FEATURE_COLORMAP = "plasma"


def save_model_output_pdfs(
    figures_directory:Path,
    fold_index:int,
    model:FullModel,
    train_dataset:ExampleDataset,
    validation_dataset:ExampleDataset,
    class_names:dict[int, str],
    config:PersistenceConfig,
) -> None:
    color_limit = _get_color_limit(
        train_dataset,
        validation_dataset,
        config.feature_color_percentile,
    )
    _save_dataset_output_pdf(
        figures_directory / "output_train" / f"fold_{fold_index}-train.pdf",
        model,
        train_dataset,
        class_names,
        color_limit,
        config.number_of_train_model_outputs,
        config.include_grad_cam,
        config.include_legrad,
        config.x_axis_label,
        config.y_axis_label,
        config.colorbar_label,
    )
    _save_dataset_output_pdf(
        figures_directory / "output_validation" / f"fold_{fold_index}-validation.pdf",
        model,
        validation_dataset,
        class_names,
        color_limit,
        config.number_of_validation_model_outputs,
        config.include_grad_cam,
        config.include_legrad,
        config.x_axis_label,
        config.y_axis_label,
        config.colorbar_label,
    )
    return


def _get_color_limit(
    train_dataset:ExampleDataset,
    validation_dataset:ExampleDataset,
    percentile:float,
) -> float:
    values = np.concatenate([
        example.value.ravel()
        for example in train_dataset.examples + validation_dataset.examples
    ])
    color_limit = float(np.percentile(np.abs(values), percentile))
    return color_limit if color_limit > 0 else 1.0


def _save_dataset_output_pdf(
    path:Path,
    model:FullModel,
    dataset:ExampleDataset,
    class_names:dict[int, str],
    color_limit:float,
    number_of_outputs:int,
    include_grad_cam:bool,
    include_legrad:bool,
    x_axis_label:str,
    y_axis_label:str,
    colorbar_label:str,
) -> None:
    sample_indices = _select_sample_indices(len(dataset), number_of_outputs)
    outputs = [
        create_model_output(
            model,
            dataset.examples[sample_index],
            include_grad_cam=include_grad_cam,
            include_legrad=include_legrad,
        )
        for sample_index in sample_indices
    ]
    color_normalization = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0,
        vmax=color_limit,
    )

    with PdfPages(path) as pdf:
        for sample_index, output in zip(sample_indices, outputs):
            _save_output_page(
                pdf,
                dataset,
                sample_index,
                output,
                class_names,
                color_normalization,
                x_axis_label,
                y_axis_label,
                colorbar_label,
            )
    return


def _select_sample_indices(dataset_size:int, number_of_outputs:int) -> list[int]:
    if dataset_size == 0:
        raise ValueError("model output requires a dataset with at least one example")
    if number_of_outputs < 1:
        raise ValueError("number of model outputs must be at least 1")

    sample_count = min(dataset_size, number_of_outputs)
    return np.linspace(0, dataset_size - 1, sample_count, dtype=int).tolist()


def _save_output_page(
    pdf:PdfPages,
    dataset:ExampleDataset,
    sample_index:int,
    output:ModelOutput,
    class_names:dict[int, str],
    color_normalization:TwoSlopeNorm,
    x_axis_label:str,
    y_axis_label:str,
    colorbar_label:str,
) -> None:
    example = dataset.examples[sample_index]
    attribution_count = sum(
        attribution is not None
        for attribution in (output.grad_cam, output.legrad)
    )
    figure, axes = plt.subplots(
        nrows=1 + attribution_count,
        figsize=(11.69, 8.27),
        layout="constrained",
        squeeze=False,
        sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * attribution_count} if attribution_count else None,
    )
    feature_axis = axes[0, 0]
    _plot_feature_values(
        feature_axis,
        example.value,
        color_normalization,
        x_axis_label,
        y_axis_label,
        colorbar_label,
        figure,
    )
    true_label = class_names.get(example.label, str(example.label))
    predicted_label = class_names.get(output.prediction, str(output.prediction))
    title_color = "forestgreen" if output.prediction == example.label else "crimson"
    feature_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    if attribution_count == 0:
        feature_axis.set(xlabel=x_axis_label)
        feature_axis.tick_params(axis="x", bottom=True, labelbottom=True)
    figure.suptitle(
        f"True: {true_label} | Predicted: {predicted_label} ({output.confidence:.3f})",
        color=title_color,
        fontsize=24,
    )
    attribution_index = 1
    for attribution, label, color in (
        (output.grad_cam, "Grad-Cam", "crimson"),
        (output.legrad, "LeGrad", "darkorange"),
    ):
        if attribution is None:
            continue
        attribution_axis = axes[attribution_index, 0]
        frames = np.arange(attribution.size)
        attribution_axis.plot(frames, attribution, color=color, linewidth=1.5)
        attribution_axis.fill_between(frames, attribution, color=color, alpha=0.25)
        attribution_axis.set(xlabel=x_axis_label, ylabel=label, ylim=(0, 1))
        attribution_axis.grid(axis="y", alpha=0.25)
        attribution_index += 1
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
        axis.set(ylabel=y_axis_label)
        axis.grid(axis="y", alpha=0.25)
        return

    if values.ndim != 2:
        raise ValueError("model output features must be a one- or two-dimensional array")

    image = axis.imshow(
        values.T,
        aspect="auto",
        origin="lower",
        cmap=FEATURE_COLORMAP,
        norm=color_normalization,
    )
    axis.set(ylabel=y_axis_label)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    return
