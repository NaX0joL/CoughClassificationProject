import numpy as np
import pytest

import core.data_pipeline_2 as data_pipeline_2
from core.data_pipeline_2 import (
    Example,
    Segment,
    SourceSeries,
    TrainExampleGenerator,
    ValidationExampleGenerator,
)
from core.data_pipeline_2.abstract import (
    Balancer,
    Labeler,
    Padder,
    Segmenter,
    Transformer,
)



class FakeSegmenter(Segmenter):

    def __init__(self, calls:list[str]) -> None:
        self.calls = calls
        self.received_source_series:list[SourceSeries]|None = None
        return

    def segment(self, source_series:list[SourceSeries]) -> list[Segment]:
        self.calls.append("segment")
        self.received_source_series = source_series
        return [Segment(
            value=np.array([1.0]),
            source_series=source_series[0],
            original_index=(0, 1),
        )]



class FakeLabeler(Labeler):

    def __init__(self, calls:list[str]) -> None:
        self.calls = calls
        return

    def label(self, segments:list[Segment]) -> list[Example]:
        self.calls.append("label")
        return [Example(value=segments[0].value, label=0, metadata={})]



class FakeTransformer(Transformer):

    def __init__(self, name:str, calls:list[str]) -> None:
        self.name = name
        self.calls = calls
        return

    def transform(self, examples:list[Example]) -> list[Example]:
        self.calls.append(f"transform-{self.name}")
        return examples



class FakePadder(Padder):

    def __init__(self, calls:list[str]) -> None:
        self.calls = calls
        return

    def pad(self, examples:list[Example]) -> list[Example]:
        self.calls.append("pad")
        return examples



class FakeBalancer(Balancer):

    def __init__(self, calls:list[str]) -> None:
        self.calls = calls
        return

    def balance(self, examples:list[Example]) -> list[Example]:
        self.calls.append("balance")
        return examples



@pytest.mark.parametrize(
    "generator_type",
    [
        TrainExampleGenerator,
        ValidationExampleGenerator,
        data_pipeline_2.TestExampleGenerator,
    ],
)
def test_split_example_generator_runs_each_stage_in_order(generator_type:type) -> None:
    source_series = [
        SourceSeries(value=np.array([1.0]), metadata={}),
    ]
    calls:list[str] = []
    segmenter = FakeSegmenter(calls)
    generator = generator_type(
        segmenter=segmenter,
        labeler=FakeLabeler(calls),
        transformer=[
            FakeTransformer("1", calls),
            FakeTransformer("2", calls),
        ],
        padder=FakePadder(calls),
        balancer=FakeBalancer(calls),
    )

    examples = generator.generate(source_series)

    assert segmenter.received_source_series is source_series
    assert calls == [
        "segment",
        "label",
        "transform-1",
        "transform-2",
        "pad",
        "balance",
    ]
    assert len(examples) == 1
