from ..abstract import Balancer
from ..intermediary import Example



class UniformOversamplingBalancer(Balancer):

    def balance(self, examples:list[Example]) -> list[Example]:
        if not examples:
            return []

        examples_by_label = self._group_examples_by_label(examples)
        target_count = max(
            len(label_examples)
            for label_examples in examples_by_label.values()
        )
        balanced_examples = list(examples)

        for label_examples in examples_by_label.values():
            number_of_duplicates = target_count - len(label_examples)
            for duplicate_index in range(number_of_duplicates):
                source_index = duplicate_index % len(label_examples)
                balanced_examples.append(label_examples[source_index])

        return balanced_examples

    @staticmethod
    def _group_examples_by_label(
        examples:list[Example],
    ) -> dict[int, list[Example]]:
        examples_by_label:dict[int, list[Example]] = {}

        for example in examples:
            examples_by_label.setdefault(example.label, []).append(example)

        return examples_by_label
