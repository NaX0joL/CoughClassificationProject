import numpy as np

from core.data_pipeline.dataset import ExampleDataset
from core.data_pipeline.intermediary import ORIGINAL_LABEL_METADATA_KEY, Example


def test_example_dataset_includes_original_label_as_metadata() -> None:
    dataset = ExampleDataset([
        Example(
            value=np.array([1.0, 2.0]),
            label=1,
            metadata={ORIGINAL_LABEL_METADATA_KEY: "POSITIVE"},
        ),
    ])

    item = dataset[0]

    assert item["label"].item() == 1
    assert item["metadata"] == {ORIGINAL_LABEL_METADATA_KEY: "POSITIVE"}
