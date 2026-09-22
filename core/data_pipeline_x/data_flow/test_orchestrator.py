import numpy as np

from core.data_pipeline_x.data_flow.orchestrator import DataFlowOrchestrator
from core.data_pipeline_x.intermediary import Example, FoldPartition
from core.data_pipeline_x.loading.data_module import DataModule



class FakeSourceReader:

    def __init__(self, source_records:list[object]) -> None:
        self.source_records = source_records
        self.calls = 0

    def get_source_records(self) -> list[object]:
        self.calls += 1
        return self.source_records


class FakePartitioner:

    def __init__(self, fold_partitions:list[FoldPartition]) -> None:
        self.fold_partitions = fold_partitions
        self.calls = 0

    def partition(self, source_records:list[object]) -> list[FoldPartition]:
        self.calls += 1
        return self.fold_partitions


class FakeExampleConstructor:

    def __init__(self) -> None:
        self.calls = []

    def construct(
        self,
        source_records:list[object],
        *,
        type:str,
    ) -> list[Example]:
        self.calls.append((source_records, type))
        return [
            Example(
                value=np.asarray([0.0]),
                label=0,
                metadata={"type": type},
            ),
        ]


def test_get_data_module_constructs_only_the_requested_fold() -> None:
    fold_partitions = [
        FoldPartition(
            train=["fold-0-train"],
            validation=["fold-0-validation"],
            test=["fold-0-test"],
        ),
        FoldPartition(
            train=["fold-1-train"],
            validation=["fold-1-validation"],
            test=["fold-1-test"],
        ),
    ]
    example_constructor = FakeExampleConstructor()
    data_flow = DataFlowOrchestrator(
        source_reader=FakeSourceReader(["source-0", "source-1"]),
        partitioner=FakePartitioner(fold_partitions),
        example_constructor=example_constructor,
        batch_size=1,
    )

    assert len(data_flow) == 2
    assert example_constructor.calls == []

    data_module = data_flow.get_data_module(1)

    assert isinstance(data_module, DataModule)
    assert example_constructor.calls == [
        (["fold-1-train"], "train"),
        (["fold-1-validation"], "validation"),
        (["fold-1-test"], "test"),
    ]
