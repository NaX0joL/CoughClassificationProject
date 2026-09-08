import numpy as np
from torch.utils.data import DataLoader

from core.data_pipeline_3.data_module import DataModule
from core.data_pipeline_3.dataset import ExampleDataset
from core.data_pipeline_3.intermediary import Example



def test_data_module_holds_three_data_loaders() -> None:
    dataset = ExampleDataset([
        Example(
            value=np.asarray([1.0]),
            label=0,
            metadata={},
        ),
    ])
    train_loader = DataLoader(dataset)
    validation_loader = DataLoader(dataset)
    test_loader = DataLoader(dataset)

    data_module = DataModule(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
    )

    assert data_module.train_loader is train_loader
    assert data_module.validation_loader is validation_loader
    assert data_module.test_loader is test_loader
