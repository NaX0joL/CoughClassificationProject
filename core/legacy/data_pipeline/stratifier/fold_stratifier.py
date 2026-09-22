import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ..intermediary import Example



def create_training_validation_folds(
    development_examples:list[Example],
    group_metadata_key:str,
    number_of_folds:int=5,
    random_seed:int=42,
) -> list[dict[str, list[Example]]]:
    sample_indices = np.arange(len(development_examples))
    labels = np.asarray([
        example.label
        for example in development_examples
    ])
    groups = np.asarray([
        example.metadata[group_metadata_key]
        for example in development_examples
    ])

    stratifier = StratifiedGroupKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=random_seed,
    )
    folds = []

    for train_indices, validation_indices in stratifier.split(
        sample_indices,
        labels,
        groups,
    ):
        train_examples = [
            development_examples[index]
            for index in train_indices
        ]
        validation_examples = [
            development_examples[index]
            for index in validation_indices
        ]
        
        folds.append({
            "train": train_examples,
            "validation": validation_examples
        })

    return folds
