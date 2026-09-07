from core.data_pipeline_2 import (
    DataPipeline,
    Example,
    ExampleGenerator,
    FoldPartitioner,
    MFCC,
    OverlapLabeler,
    SlidingWindowSegmenter,
    TestExampleGenerator,
    TrainExampleGenerator,
    UniformOversamplingBalancer,
    ValidationExampleGenerator,
)
from core.data_pipeline_2.source_reader import ElderlyCoughAudioSourceReader



def print_data_pipeline_examples(data_pipeline:DataPipeline) -> None:
    data_pipeline.initialize()
    fold_partitions = data_pipeline.fold_partitions

    if fold_partitions is None:
        raise RuntimeError("DataPipeline did not create fold partitions")

    for fold_index in range(len(fold_partitions)):
        example_bundle = data_pipeline.build_fold(fold_index)
        _print_examples(fold_index, "train", example_bundle.train)
        _print_examples(fold_index, "validation", example_bundle.validation)
        _print_examples(fold_index, "test", example_bundle.test)

    return


def _print_examples(
    fold_index:int,
    split_name:str,
    examples:list[Example],
) -> None:
    for example_index, example in enumerate(examples):
        print(
            f"fold={fold_index} split={split_name} index={example_index}: "
            f"{example}"
        )

    return


WINDOW_SIZE = 8_200
WINDOW_STRIDE = 4_100
SAMPLE_RATE = 16_000



def main() -> None:
    segmenter = SlidingWindowSegmenter(
            window_size=WINDOW_SIZE,
            stride=WINDOW_STRIDE,
        )
    labeler = OverlapLabeler(
        overlap_threshold=0.7,
        kept_metadata_keys=["patient_id", "cough_audio"],
    )
    transformer = MFCC(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        win_length=400,
        hop_length=200,
        n_mels=64,
        n_mfcc=40,
    )
    example_generator = ExampleGenerator(
        train_generator=TrainExampleGenerator(
            segmenter=segmenter,
            labeler=labeler,
            transformer=transformer,
            balancer=UniformOversamplingBalancer(),
        ),
        validation_generator=ValidationExampleGenerator(
            segmenter=segmenter,
            labeler=labeler,
            transformer=transformer,
        ),
        test_generator=TestExampleGenerator(
            segmenter=segmenter,
            labeler=labeler,
            transformer=transformer,
        ),
    )
    data_pipeline = DataPipeline(
        source_reader=ElderlyCoughAudioSourceReader(),
        partitioner=FoldPartitioner(
            group_metadata_key="patient_id",
            label_metadata_key="is_infectious",
            number_of_folds=5,
            validation_ratio=0.2,
            random_seed=42,
        ),
        example_generator=example_generator,
    )

    print_data_pipeline_examples(data_pipeline)
    return


if __name__ == "__main__":
    main()
