import torch
from torch.utils.data import DataLoader

from ..intermediary import Example
from .data_module import DataModule
from .dataset import ExampleDataset



def create_data_module(
    train_examples:list[Example],
    validation_examples:list[Example],
    test_examples:list[Example],
    batch_size:int,
    *,
    num_workers:int=0,
    drop_last_batch:bool=False,
    pin_memory:bool=False,
    random_seed:int|None=None,
) -> DataModule:
    generator = None
    if random_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(random_seed)

    return DataModule(
        train_loader=DataLoader(
            dataset=ExampleDataset(train_examples),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last_batch,
            pin_memory=pin_memory,
            generator=generator,
        ),
        validation_loader=DataLoader(
            dataset=ExampleDataset(validation_examples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        test_loader=DataLoader(
            dataset=ExampleDataset(test_examples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )
