import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ..abstract import Partitioner
from ..intermediary import FoldPartition, SourceSeries



class FoldPartitioner(Partitioner):

    def __init__(
        self,
        group_metadata_key:str,
        label_metadata_key:str,
        number_of_folds:int=5,
        validation_ratio:float=0.2,
        random_seed:int=42,
    ) -> None:
        self.group_metadata_key = group_metadata_key
        self.label_metadata_key = label_metadata_key

        self._outer_fold_splitter = OuterFoldSplitter(
            number_of_folds=number_of_folds,
            random_seed=random_seed,
        )
        self._development_splitter = DevelopmentSplitter(
            validation_ratio=validation_ratio,
            random_seed=random_seed,
        )

        return

    def partition(self, source_series:list[SourceSeries]) -> list[FoldPartition]:
        self._validate_source_series(source_series)

        labels, groups = self._collect_partition_values(
            source_series=source_series,
        )
        outer_fold_indices = self._outer_fold_splitter.split(
            labels=labels,
            groups=groups,
        )

        fold_partitions:list[FoldPartition] = []
        for fold_index, (development_indices, test_indices) in enumerate(
            outer_fold_indices
        ):
            train_indices, validation_indices = self._development_splitter.split(
                development_indices=development_indices,
                labels=labels,
                groups=groups,
                fold_index=fold_index,
            )
            fold_partition = self._create_fold_partition(
                source_series=source_series,
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
            )
            fold_partitions.append(fold_partition)

        return fold_partitions

    @staticmethod
    def _validate_source_series(source_series:list[SourceSeries]) -> None:
        if not source_series:
            raise ValueError("source_series must contain at least one item")
        return

    def _collect_partition_values(
        self,
        source_series:list[SourceSeries],
    ) -> tuple[np.ndarray, np.ndarray]:
        labels:list[object] = []
        groups:list[object] = []

        for source_index, source in enumerate(source_series):
            labels.append(self._get_metadata_value(
                source,
                self.label_metadata_key,
                source_index,
            ))
            groups.append(self._get_metadata_value(
                source,
                self.group_metadata_key,
                source_index,
            ))

        return np.asarray(labels), np.asarray(groups)

    @staticmethod
    def _get_metadata_value(
        source:SourceSeries,
        metadata_key:str,
        source_index:int,
    ) -> object:
        if metadata_key not in source.metadata:
            raise ValueError(
                f"source_series[{source_index}].metadata is missing required "
                f"key: {metadata_key}"
            )

        metadata_value = source.metadata[metadata_key]
        if metadata_value is None:
            raise ValueError(
                f"source_series[{source_index}].metadata[{metadata_key!r}] "
                "cannot be None"
            )

        return metadata_value

    @staticmethod
    def _create_fold_partition(
        source_series:list[SourceSeries],
        train_indices:np.ndarray,
        validation_indices:np.ndarray,
        test_indices:np.ndarray,
    ) -> FoldPartition:
        fold_partition = FoldPartition(
            train=[source_series[index] for index in train_indices],
            validation=[source_series[index] for index in validation_indices],
            test=[source_series[index] for index in test_indices],
        )
        return fold_partition



class OuterFoldSplitter:

    def __init__(self, number_of_folds:int, random_seed:int) -> None:
        self.number_of_folds = number_of_folds
        self.random_seed = random_seed
        return

    def split(
        self,
        labels:np.ndarray,
        groups:np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        self._validate_configuration()
        self._validate_number_of_groups(groups)

        fold_indices = self._create_fold_indices(
            labels,
            groups,
        )
        return fold_indices

    def _validate_configuration(self) -> None:
        if self.number_of_folds < 2:
            raise ValueError("number_of_folds must be at least 2")
        return

    def _validate_number_of_groups(self, groups:np.ndarray) -> None:
        number_of_groups = len(np.unique(groups))
        if number_of_groups < self.number_of_folds:
            raise ValueError("number of patient groups must be at least number_of_folds")
        return

    def _create_fold_indices(
        self,
        labels:np.ndarray,
        groups:np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        sample_indices = np.arange(len(labels))
        splitter = StratifiedGroupKFold(
            n_splits=self.number_of_folds,
            shuffle=True,
            random_state=self.random_seed,
        )
        fold_indices = list(splitter.split(
            sample_indices,
            labels,
            groups,
        ))
        return fold_indices



class DevelopmentSplitter:

    def __init__(self, validation_ratio:float, random_seed:int) -> None:
        self.validation_ratio = validation_ratio
        self.random_seed = random_seed
        return

    def split(
        self,
        development_indices:np.ndarray,
        labels:np.ndarray,
        groups:np.ndarray,
        fold_index:int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._validate_configuration()

        validation_number_of_folds = round(1 / self.validation_ratio)
        fold_indices = self._create_fold_indices(
            development_indices,
            labels,
            groups,
            validation_number_of_folds,
            fold_index,
        )
        return fold_indices

    def _validate_configuration(self) -> None:
        if not 0 < self.validation_ratio <= 0.5:
            raise ValueError(
                "validation_ratio must be greater than 0 and at most 0.5"
            )
        return

    def _create_fold_indices(
        self,
        development_indices:np.ndarray,
        labels:np.ndarray,
        groups:np.ndarray,
        validation_number_of_folds:int,
        fold_index:int,
    ) -> tuple[np.ndarray, np.ndarray]:
        development_labels = labels[development_indices]
        development_groups = groups[development_indices]
        number_of_groups = len(np.unique(development_groups))

        if number_of_groups < validation_number_of_folds:
            raise ValueError(
                "development patient groups are insufficient for the "
                "requested validation_ratio"
            )

        splitter = StratifiedGroupKFold(
            n_splits=validation_number_of_folds,
            shuffle=True,
            random_state=self.random_seed + fold_index,
        )
        relative_train_indices, relative_validation_indices = next(iter(
            splitter.split(
                development_indices,
                development_labels,
                development_groups,
            )
        ))

        train_indices = development_indices[relative_train_indices]
        validation_indices = development_indices[relative_validation_indices]
        return train_indices, validation_indices
