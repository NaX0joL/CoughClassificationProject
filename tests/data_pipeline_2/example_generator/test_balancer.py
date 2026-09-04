from collections import Counter

import numpy as np

from core.data_pipeline_2 import Example, UniformOversamplingBalancer


def make_example(label:int, identifier:str) -> Example:
    return Example(
        value=np.array([0.0]),
        label=label,
        metadata={"identifier": identifier},
    )


def test_balancer_oversamples_each_label_to_largest_class_count() -> None:
    examples = [
        make_example(0, "invalid-1"),
        make_example(0, "invalid-2"),
        make_example(0, "invalid-3"),
        make_example(1, "non-infectious-1"),
        make_example(2, "infectious-1"),
        make_example(2, "infectious-2"),
    ]

    balanced_examples = UniformOversamplingBalancer().balance(examples)
    label_counts = Counter(example.label for example in balanced_examples)

    assert label_counts == {0: 3, 1: 3, 2: 3}


def test_balancer_distributes_duplicates_across_examples_round_robin() -> None:
    majority_examples = [
        make_example(0, f"majority-{index}")
        for index in range(5)
    ]
    minority_examples = [
        make_example(1, "minority-1"),
        make_example(1, "minority-2"),
    ]

    balanced_examples = UniformOversamplingBalancer().balance(
        majority_examples + minority_examples,
    )
    minority_identifiers = [
        example.metadata["identifier"]
        for example in balanced_examples
        if example.label == 1
    ]

    assert Counter(minority_identifiers) == {
        "minority-1": 3,
        "minority-2": 2,
    }


def test_balancer_does_not_mutate_input_list() -> None:
    examples = [
        make_example(0, "majority-1"),
        make_example(0, "majority-2"),
        make_example(1, "minority-1"),
    ]
    original_examples = list(examples)

    UniformOversamplingBalancer().balance(examples)

    assert examples == original_examples


def test_balancer_leaves_equal_classes_unchanged() -> None:
    examples = [
        make_example(0, "invalid"),
        make_example(1, "non-infectious"),
        make_example(2, "infectious"),
    ]

    balanced_examples = UniformOversamplingBalancer().balance(examples)

    assert balanced_examples == examples
    assert balanced_examples is not examples


def test_balancer_accepts_empty_examples() -> None:
    assert UniformOversamplingBalancer().balance([]) == []
