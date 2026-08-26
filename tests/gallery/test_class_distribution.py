
import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from core.data_pipeline.data_pipeline_config import DataPipelineConfig
from core.data_pipeline.dataset import ExampleDataset
from core.data_pipeline.intermediary import DataSplit, DevelopmentFold, Example
from core.data_pipeline.preprocessing import CoughSegmenter, MFCC, ZeroPadder
from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.stratifier import DataSplitter
from core.gallery.gallery_directory import compute_config_hash
from core.gallery.class_distribution import (
    ClassDistributionGenerator,
    collect_class_counts,
    save_class_distribution_figure,
    save_class_distribution_json,
)


def _make_example(label:int=0) -> Example:
    return Example(
        value=np.random.rand(10, 4).astype(np.float32),
        label=label,
        metadata={"patient_id": "p1"},
    )


def _make_dataset(labels:list[int]) -> ExampleDataset:
    return ExampleDataset([_make_example(label) for label in labels])


def _make_data_split() -> DataSplit:
    return DataSplit(
        test_dataset=_make_dataset([0, 0, 1]),
        development_folds=[
            DevelopmentFold(
                train_dataset=_make_dataset([0, 0, 0, 1, 1]),
                validation_dataset=_make_dataset([0, 1]),
            ),
            DevelopmentFold(
                train_dataset=_make_dataset([0, 0, 1, 1]),
                validation_dataset=_make_dataset([0, 1, 1]),
            ),
        ],
    )


def _make_empty_data_split() -> DataSplit:
    return DataSplit(
        test_dataset=_make_dataset([]),
        development_folds=[
            DevelopmentFold(
                train_dataset=_make_dataset([]),
                validation_dataset=_make_dataset([]),
            ),
        ],
    )


def _make_config() -> DataPipelineConfig:
    return DataPipelineConfig(
        source_reader=ElderlyCoughAudioSourceReader(),
        segmenter=CoughSegmenter(kept_metadata_key=["patient_id", "cough_audio"]),
        transformer=MFCC(sample_rate=16_000, n_fft=400, win_length=400, hop_length=160, n_mels=40, n_mfcc=40),
        padder=ZeroPadder(target_length=820, padding_type="random", random_seed=42),
        splitter=DataSplitter(group_metadata_key="patient_id", test_ratio=0.1, number_of_folds=5, random_seed=42),
        name="test_config",
    )



class TestCollectClassCounts:

    def test_computes_overall_and_per_split_counts(self) -> None:
        data_split = _make_data_split()

        counts = collect_class_counts(data_split)

        assert counts == {
            "overall": {0: 9, 1: 8},
            "fold_1-train": {0: 3, 1: 2},
            "fold_1-val": {0: 1, 1: 1},
            "fold_2-train": {0: 2, 1: 2},
            "fold_2-val": {0: 1, 1: 2},
            "test": {0: 2, 1: 1},
        }
        split_names = list(counts)
        assert split_names[0] == "overall"
        assert split_names[-1] == "test"

    def test_rejects_empty_split(self) -> None:
        data_split = _make_empty_data_split()

        with pytest.raises(ValueError, match="at least one example"):
            collect_class_counts(data_split)


class TestSaveClassDistributionJson:

    def test_writes_counts_with_default_style_payload(self, tmp_path) -> None:
        counts = collect_class_counts(_make_data_split())
        path = tmp_path / "distribution.json"

        save_class_distribution_json(counts, class_names=None, path=path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["class_names"] == {"0": "0", "1": "1"}
        assert payload["splits"]["overall"] == {
            "total": 17,
            "counts": {"0": 9, "1": 8},
            "percentages": {"0": 52.94, "1": 47.06},
        }
        assert payload["splits"]["fold_1-train"]["total"] == 5

    def test_custom_class_names_override_labels(self, tmp_path) -> None:
        counts = collect_class_counts(_make_data_split())
        path = tmp_path / "distribution.json"
        class_names = {0: "healthy", 1: "sick"}

        save_class_distribution_json(counts, class_names=class_names, path=path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["class_names"] == {"0": "healthy", "1": "sick"}
        assert payload["splits"]["overall"]["counts"] == {"healthy": 9, "sick": 8}


class TestSaveClassDistributionFigure:

    def test_creates_png_file(self, tmp_path) -> None:
        counts = collect_class_counts(_make_data_split())
        path = tmp_path / "distribution.png"

        save_class_distribution_figure(counts, class_names=None, path=path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestClassDistributionGenerator:

    def test_creates_files_in_hash_named_folder(self, tmp_path) -> None:
        generator = ClassDistributionGenerator(
            data_pipeline_config=_make_config(),
            gallery_directory=tmp_path / "gallery",
        )

        result_path = generator.generate(_make_data_split())

        gallery_dirs = list((tmp_path / "gallery").iterdir())
        assert len(gallery_dirs) == 1
        gallery_dir = gallery_dirs[0]
        assert gallery_dir.name.startswith("test_config_")
        assert result_path == gallery_dir / "class_distribution.png"
        assert (gallery_dir / "class_distribution.png").exists()
        assert (gallery_dir / "class_distribution.json").exists()
        assert result_path.stat().st_size > 0

    def test_skips_regeneration_when_output_exists(self, tmp_path) -> None:
        generator = ClassDistributionGenerator(
            data_pipeline_config=_make_config(),
            gallery_directory=tmp_path / "gallery",
        )
        figure_path = generator.generate(_make_data_split())
        original_bytes = figure_path.read_bytes()

        figure_path.write_bytes(b"sentinel")
        returned_path = generator.generate(_make_data_split())

        assert returned_path == figure_path
        assert figure_path.read_bytes() == b"sentinel"
        assert original_bytes != b"sentinel"

    def test_regenerate_flag_rewrites_file(self, tmp_path) -> None:
        generator = ClassDistributionGenerator(
            data_pipeline_config=_make_config(),
            gallery_directory=tmp_path / "gallery",
        )
        figure_path = generator.generate(_make_data_split())
        figure_path.write_bytes(b"sentinel")

        regenerating_generator = ClassDistributionGenerator(
            data_pipeline_config=_make_config(),
            gallery_directory=tmp_path / "gallery",
            regenerate=True,
        )
        regenerating_generator.generate(_make_data_split())

        assert figure_path.read_bytes() != b"sentinel"

    def test_different_config_creates_new_folder(self, tmp_path) -> None:
        config1 = _make_config()
        config2 = _make_config()
        config2.padder = ZeroPadder(target_length=500, padding_type="left")
        generator1 = ClassDistributionGenerator(
            data_pipeline_config=config1,
            gallery_directory=tmp_path / "gallery",
        )
        generator2 = ClassDistributionGenerator(
            data_pipeline_config=config2,
            gallery_directory=tmp_path / "gallery",
        )

        generator1.generate(_make_data_split())
        generator2.generate(_make_data_split())

        gallery_dirs = list((tmp_path / "gallery").iterdir())
        assert len(gallery_dirs) == 2
        assert len({d.name for d in gallery_dirs}) == 2

    def test_no_name_uses_hash_only(self, tmp_path) -> None:
        config = _make_config()
        config.name = None
        generator = ClassDistributionGenerator(
            data_pipeline_config=config,
            gallery_directory=tmp_path / "gallery",
        )

        result_path = generator.generate(_make_data_split())

        config_hash = compute_config_hash(config, None)
        assert result_path.parent.name == config_hash

    def test_uses_binary_class_names_by_default(self, tmp_path) -> None:
        generator = ClassDistributionGenerator(
            data_pipeline_config=_make_config(),
            gallery_directory=tmp_path / "gallery",
        )

        result_path = generator.generate(_make_data_split())

        json_path = result_path.parent / "class_distribution.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["class_names"] == {"0": "non-infectious", "1": "infectious"}
        assert payload["splits"]["overall"]["counts"] == {
            "non-infectious": 9,
            "infectious": 8,
        }
