
from .test_example_generator import TestExampleGenerator
from .train_example_generator import TrainExampleGenerator
from .validation_example_generator import ValidationExampleGenerator
from ..intermediary import ExampleBundle, FoldPartition



class ExampleGenerator:

    def __init__(
        self,
        train_generator:TrainExampleGenerator,
        validation_generator:ValidationExampleGenerator,
        test_generator:TestExampleGenerator,
    ) -> None:
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.test_generator = test_generator
        return

    def generate(self, fold_partition:FoldPartition) -> ExampleBundle:
        train_examples = self.train_generator.generate(fold_partition.train)
        validation_examples = self.validation_generator.generate(
            fold_partition.validation,
        )
        test_examples = self.test_generator.generate(fold_partition.test)

        return ExampleBundle(
            train=train_examples,
            validation=validation_examples,
            test=test_examples,
        )
