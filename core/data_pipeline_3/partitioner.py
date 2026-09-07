from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

from .intermediary import FoldPartition, SourceRecord
from .abstract import AbstractPartitioner


class Partitioner(AbstractPartitioner):

    def __init__(
        self,
        label_key:str="isInfectious",
        group_key:str="PatientID",
        number_of_outer_folds:int=5,
        number_of_inner_folds:int=4,
        random_seed:int|None=None,
    ) -> None:
        self.label_key = label_key
        self.group_key = group_key
        self.number_of_outer_folds = number_of_outer_folds
        self.number_of_inner_folds = number_of_inner_folds
        self.random_seed = random_seed
        return

    def partition(self, source_records:list[SourceRecord]) -> list[FoldPartition]:
        indices = np.arange(len(source_records))
        labels = np.asarray([
            record.metadata[self.label_key]
            for record in source_records
        ])
        groups = np.asarray([
            record.metadata[self.group_key]
            for record in source_records
        ])
        
        outer_splitter = StratifiedGroupKFold(
            n_splits=self.number_of_outer_folds, 
            shuffle=True, 
            random_state=self.random_seed,
        )
        inner_splitter = StratifiedGroupKFold(
            n_splits=self.number_of_inner_folds,
            shuffle=True,
            random_state=self.random_seed,
        )
        
        fold_partitions = []
                
        for development_index, test_index in outer_splitter.split(
            indices, 
            labels, 
            groups,
        ):
            dev_indices = indices[development_index]
            dev_labels = labels[development_index]
            dev_groups = groups[development_index]
            test_indices = indices[test_index]
            
            for train_index, validation_index in inner_splitter.split(
                dev_indices, 
                dev_labels, 
                dev_groups
            ):
                train_indices = dev_indices[train_index]
                val_indices = dev_indices[validation_index]
                
                fold_partition = FoldPartition(
                    train=self._select(source_records, train_indices),
                    validation=self._select(source_records, val_indices),
                    test=self._select(source_records, test_indices),
                )
                fold_partitions.append(fold_partition)
                
                break

        return fold_partitions

    @staticmethod
    def _select(
        source_records:list[SourceRecord],
        selected_indices:np.ndarray,
    ) -> list[SourceRecord]:
        selected = [
            source_record
            for index, source_record in enumerate(source_records)
            if index in selected_indices
        ]
        return selected
