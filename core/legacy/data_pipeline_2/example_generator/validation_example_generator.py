from ..abstract import Balancer, Labeler, Padder, Segmenter, Transformer
from ..intermediary import Example, SourceSeries



class ValidationExampleGenerator:

    def __init__(
        self,
        segmenter:Segmenter,
        labeler:Labeler,
        transformer:Transformer|list[Transformer]|None=None,
        padder:Padder|None=None,
        balancer:Balancer|None=None,
    ) -> None:
        self.segmenter = segmenter
        self.labeler = labeler

        if transformer is None:
            self.transformers:list[Transformer] = []
        elif isinstance(transformer, list):
            self.transformers = transformer
        else:
            self.transformers = [transformer]

        self.padder = padder
        self.balancer = balancer
        return

    def generate(self, source_series:list[SourceSeries]) -> list[Example]:
        segments = self.segmenter.segment(source_series)
        examples = self.labeler.label(segments)

        for transformer in self.transformers:
            examples = transformer.transform(examples)

        if self.padder is not None:
            examples = self.padder.pad(examples)

        if self.balancer is not None:
            examples = self.balancer.balance(examples)

        return examples
