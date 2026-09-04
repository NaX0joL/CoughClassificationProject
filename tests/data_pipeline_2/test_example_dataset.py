import numpy as np
import torch

from core.data_pipeline_2 import Example, ExampleDataset
from core.data_pipeline_2.intermediary import ORIGINAL_LABEL_METADATA_KEY


def test_example_dataset_converts_example_to_pytorch_item() -> None:
    dataset = ExampleDataset([
        Example(
            value=np.array([1.0, 2.0], dtype=np.float64),
            label=2,
            metadata={ORIGINAL_LABEL_METADATA_KEY: "infectious"},
        ),
    ])

    item = dataset[0]

    assert len(dataset) == 1
    assert item["value"].dtype == torch.float32
    assert item["label"].dtype == torch.long
    assert item["label"].item() == 2
    assert item["metadata"] == {
        ORIGINAL_LABEL_METADATA_KEY: "infectious",
    }


def test_example_dataset_falls_back_to_numeric_label_metadata() -> None:
    dataset = ExampleDataset([
        Example(
            value=np.array([1.0]),
            label=1,
            metadata={},
        ),
    ])

    item = dataset[0]

    assert item["metadata"] == {ORIGINAL_LABEL_METADATA_KEY: "1"}
