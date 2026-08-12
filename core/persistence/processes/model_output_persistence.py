from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from ...data_pipeline.dataset import ExampleDataset
from ...gallery import ModelOutput, create_model_output
from ...model import FullModel
from ..persistence_config import PersistenceConfig


MFCC_COLORMAP = "magma"


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
        config.mfcc_color_percentile,
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
    )
    return


def _get_color_limit(
    train_dataset:ExampleDataset,
    validation_dataset:ExampleDataset,
    percentile:float,
) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("mfcc_color_percentile must be in the range (0, 100]")

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
    mfcc_axis = axes[0, 0]
    image = mfcc_axis.imshow(
        example.value.T,
        aspect="auto",
        origin="lower",
        cmap=MFCC_COLORMAP,
        norm=color_normalization,
    )
    true_label = class_names.get(example.label, str(example.label))
    predicted_label = class_names.get(output.prediction, str(output.prediction))
    title_color = "forestgreen" if output.prediction == example.label else "crimson"
    mfcc_axis.set(ylabel="MFCC coefficient")
    mfcc_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    if attribution_count == 0:
        mfcc_axis.set(xlabel="Frame")
        mfcc_axis.tick_params(axis="x", bottom=True, labelbottom=True)
    figure.suptitle(
        f"True: {true_label} | Predicted: {predicted_label} ({output.confidence:.3f})",
        color=title_color,
        fontsize=24,
    )
    figure.colorbar(image, ax=mfcc_axis, label="MFCC value")

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
        attribution_axis.set(xlabel="Frame", ylabel=label, ylim=(0, 1))
        attribution_axis.grid(axis="y", alpha=0.25)
        attribution_index += 1
    pdf.savefig(figure)
    plt.close(figure)
    return
