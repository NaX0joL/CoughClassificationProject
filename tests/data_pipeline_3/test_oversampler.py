from collections import Counter

import numpy as np

from core.data_pipeline_3.intermediary import Example
from core.data_pipeline_3.oversampler import UniformOversampler



def _make_example(label:int, index:int) -> Example:
    return Example(
        value=np.asarray([index]),
        label=label,
        metadata={"index": index},
    )


def test_oversampler_balances_every_present_label() -> None:
    examples = [
        _make_example(0, 0),
        _make_example(0, 1),
        _make_example(0, 2),
        _make_example(1, 3),
        _make_example(2, 4),
        _make_example(2, 5),
    ]

    oversampled = UniformOversampler(random_seed=42).oversample(examples)

    label_counts = Counter(example.label for example in oversampled)
    assert label_counts == {0: 3, 1: 3, 2: 3}


def test_oversampler_does_not_modify_input_list() -> None:
    examples = [
        _make_example(0, 0),
        _make_example(0, 1),
        _make_example(1, 2),
    ]

    UniformOversampler(random_seed=42).oversample(examples)

    assert len(examples) == 3


def test_oversampler_leaves_single_label_unchanged() -> None:
    examples = [
        _make_example(1, 0),
        _make_example(1, 1),
    ]

    oversampled = UniformOversampler(random_seed=42).oversample(examples)

    assert oversampled == examples
    assert oversampled is not examples
