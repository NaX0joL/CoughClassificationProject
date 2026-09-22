import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ..intermediary import Example



def split_development_and_test(
    examples:list[Example],
    group_metadata_key:str,
    test_ratio:float=0.2,
    random_seed:int=42,
) -> tuple[list[Example], list[Example]]:
    n_splits = max(2, round(1 / test_ratio))

    sample_indices = np.arange(len(examples))
    labels = np.asarray([
        example.label
        for example in examples
    ])
    groups = np.asarray([
        example.metadata[group_metadata_key]
        for example in examples
    ])

    stratifier = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    train_indices, test_indices = next(iter(stratifier.split(
        sample_indices,
        labels,
        groups,
    )))

    development_examples = [
        examples[index]
        for index in train_indices
    ]
    test_examples = [
        examples[index]
        for index in test_indices
    ]
    return development_examples, test_examples
