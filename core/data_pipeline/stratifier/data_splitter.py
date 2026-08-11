from dataclasses import dataclass

from torch.utils.data import Dataset

from ..dataset import ExampleDataset
from ..intermediary import DataSplit, Example
from .fold_stratifier import create_training_validation_folds
from .test_splitter import split_development_and_test



@dataclass
class DataSplitter:
    group_metadata_key:str
    test_ratio:float=0.2
    number_of_folds:int=5
    random_seed:int=42

    def split(self, examples:list[Example]) -> DataSplit:
        development_examples, test_examples = split_development_and_test(
            examples=examples,
            group_metadata_key=self.group_metadata_key,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
        )
        development_folds = create_training_validation_folds(
            development_examples=development_examples,
            group_metadata_key=self.group_metadata_key,
            number_of_folds=self.number_of_folds,
            random_seed=self.random_seed,
        )
        data_split = DataSplit(
            test_set=ExampleDataset(test_examples),
            development_folds=self._to_dataset_folds(development_folds),
        )
        return data_split

    def _to_dataset_folds(
        self,
        development_folds:list[dict[str, list[Example]]],
    ) -> list[dict[str, Dataset]]:
        dataset_folds = []

        for fold in development_folds:
            dataset_fold = {
                "train": ExampleDataset(fold["train"]),
                "validation": ExampleDataset(fold["validation"]),
            }
            dataset_folds.append(dataset_fold)
            
        return dataset_folds
