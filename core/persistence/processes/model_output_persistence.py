from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from modules.resolve_pytorch_device import get_model_device

from ...data_pipeline.dataset import ExampleDataset
from ...model import FullModel
from ..persistence_config import PersistenceConfig


MFCC_COLORMAP = "RdBu_r"


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
    )
    _save_dataset_output_pdf(
        figures_directory / "output_validation" / f"fold_{fold_index}-validation.pdf",
        model,
        validation_dataset,
        class_names,
        color_limit,
        config.number_of_validation_model_outputs,
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
) -> None:
    sample_indices = _select_sample_indices(len(dataset), number_of_outputs)
    outputs = _predict_samples(model, dataset, sample_indices)
    color_normalization = TwoSlopeNorm(
        vmin=-color_limit,
        vcenter=0,
        vmax=color_limit,
    )

    with PdfPages(path) as pdf:
        for output in outputs:
            _save_output_page(
                pdf,
                dataset,
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


def _predict_samples(
    model:FullModel,
    dataset:ExampleDataset,
    sample_indices:list[int],
) -> list[tuple[int, int, float]]:
    device = get_model_device(model)
    was_training = model.training
    model.eval()
    outputs:list[tuple[int, int, float]] = []

    try:
        with torch.inference_mode():
            for sample_index in sample_indices:
                example = dataset.examples[sample_index]
                values = torch.as_tensor(example.value, dtype=torch.float32).unsqueeze(0)
                probabilities = model.predict_probabilities(values.to(device))[0].cpu()
                prediction = int(probabilities.argmax().item())
                confidence = float(probabilities[prediction].item())
                outputs.append((sample_index, prediction, confidence))
    finally:
        model.train(was_training)

    return outputs


def _save_output_page(
    pdf:PdfPages,
    dataset:ExampleDataset,
    output:tuple[int, int, float],
    class_names:dict[int, str],
    color_normalization:TwoSlopeNorm,
) -> None:
    sample_index, prediction, confidence = output
    example = dataset.examples[sample_index]
    figure, axis = plt.subplots(figsize=(11.69, 8.27), layout="constrained")
    image = axis.imshow(
        example.value.T,
        aspect="auto",
        origin="lower",
        cmap=MFCC_COLORMAP,
        norm=color_normalization,
    )
    true_label = class_names.get(example.label, str(example.label))
    predicted_label = class_names.get(prediction, str(prediction))
    axis.set(xlabel="Frame", ylabel="MFCC coefficient")
    axis.text(
        0.5,
        -0.12,
        f"True: {true_label} | Predicted: {predicted_label} ({confidence:.3f})",
        ha="center",
        transform=axis.transAxes,
    )
    figure.colorbar(image, ax=axis, label="MFCC value")
    pdf.savefig(figure)
    plt.close(figure)
    return
