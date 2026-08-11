from sklearn.model_selection import train_test_split

from ..intermediary import Example



def split_development_and_test(
    examples:list[Example],
    group_metadata_key:str,
    test_ratio:float=0.2,
    random_seed:int=42,
) -> tuple[list[Example], list[Example]]:
    group_labels = _get_group_labels(examples, group_metadata_key)
    group_ids = list(group_labels)
    labels = list(group_labels.values())

    development_group_ids, test_group_ids = train_test_split(
        group_ids,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_seed,
    )
    development_group_ids = set(development_group_ids)
    test_group_ids = set(test_group_ids)

    development_examples = [
        example
        for example in examples
        if example.metadata[group_metadata_key] in development_group_ids
    ]
    test_examples = [
        example
        for example in examples
        if example.metadata[group_metadata_key] in test_group_ids
    ]
    return development_examples, test_examples



def _get_group_labels(
    examples:list[Example],
    group_metadata_key:str,
) -> dict[object, int]:
    group_labels = {}

    for example in examples:
        group_id = example.metadata[group_metadata_key]

        if group_id in group_labels and group_labels[group_id] != example.label:
            raise ValueError("each group must have one consistent label")

        group_labels[group_id] = example.label

    return group_labels
