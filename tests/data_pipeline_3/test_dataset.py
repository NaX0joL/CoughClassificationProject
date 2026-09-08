import numpy as np
import torch

from core.data_pipeline_3.dataset import ExampleDataset
from core.data_pipeline_3.intermediary import Example



def test_example_dataset_returns_tensors() -> None:
    dataset = ExampleDataset([
        Example(
            value=np.asarray([1.0, 2.0]),
            label=2,
            metadata={},
        ),
    ])

    item = dataset[0]

    assert torch.equal(item["value"], torch.tensor([1.0, 2.0]))
    assert torch.equal(item["label"], torch.tensor(2))
    assert set(item) == {"value", "label"}
