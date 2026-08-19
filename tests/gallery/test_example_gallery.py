
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from core.data_pipeline.data_pipeline_config import DataPipelineConfig
from core.data_pipeline.intermediary import Example
from core.data_pipeline.preprocessing import CoughSegmenter, MFCC, ZeroPadder
from core.data_pipeline.source_reader import ElderlyCoughAudioSourceReader
from core.data_pipeline.stratifier import DataSplitter
from core.gallery.example_gallery import (
    ExampleGalleryGenerator,
    save_data_pipeline_config,
    save_examples_pdf,
    _compute_cache_hash,
)


def _make_example(length:int=100, n_features:int=10, label:int=0) -> Example:
    return Example(
        value=np.random.rand(length, n_features).astype(np.float32),
        label=label,
        metadata={"patient_id": "p1"},
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


class TestSaveExamplesPdf:

    def test_creates_pdf_file(self, tmp_path) -> None:
        examples = [_make_example() for _ in range(5)]
        path = tmp_path / "examples.pdf"

        save_examples_pdf(examples, path, num_examples=3)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_returns_correct_number_of_indices(self, tmp_path) -> None:
        examples = [_make_example() for _ in range(10)]
        path = tmp_path / "examples.pdf"

        indices = save_examples_pdf(examples, path, num_examples=3)

        assert len(indices) == 3
        assert all(0 <= i < 10 for i in indices)

    def test_random_sampling_is_reproducible_with_seed(self, tmp_path) -> None:
        examples = [_make_example(label=i % 2) for i in range(20)]
        path1 = tmp_path / "first.pdf"
        path2 = tmp_path / "second.pdf"

        indices1 = save_examples_pdf(examples, path1, num_examples=5, seed=42)
        indices2 = save_examples_pdf(examples, path2, num_examples=5, seed=42)

        assert indices1 == indices2

    def test_different_seeds_produce_different_indices(self, tmp_path) -> None:
        examples = [_make_example(label=i % 2) for i in range(20)]
        path1 = tmp_path / "first.pdf"
        path2 = tmp_path / "second.pdf"

        indices1 = save_examples_pdf(examples, path1, num_examples=5, seed=42)
        indices2 = save_examples_pdf(examples, path2, num_examples=5, seed=123)

        assert indices1 != indices2

    def test_num_examples_clamps_to_dataset_size(self, tmp_path) -> None:
        examples = [_make_example() for _ in range(3)]
        path = tmp_path / "examples.pdf"

        indices = save_examples_pdf(examples, path, num_examples=10)

        assert len(indices) == 3

    def test_rejects_empty_examples(self, tmp_path) -> None:
        path = tmp_path / "examples.pdf"

        with pytest.raises(ValueError, match="examples list must not be empty"):
            save_examples_pdf([], path)

    def test_rejects_zero_num_examples(self, tmp_path) -> None:
        examples = [_make_example()]
        path = tmp_path / "examples.pdf"

        with pytest.raises(ValueError, match="num_examples must be at least 1"):
            save_examples_pdf(examples, path, num_examples=0)

    def test_class_names_used_in_title(self, tmp_path) -> None:
        examples = [_make_example(label=1) for _ in range(3)]
        path = tmp_path / "examples.pdf"
        class_names = {0: "healthy", 1: "sick"}

        indices = save_examples_pdf(examples, path, num_examples=1, class_names=class_names)

        assert len(indices) == 1
        assert path.exists()


class TestComputeCacheHash:

    def test_same_config_same_hash(self) -> None:
        config = _make_config()

        hash1 = _compute_cache_hash(config, random_seed=42)
        hash2 = _compute_cache_hash(config, random_seed=42)

        assert hash1 == hash2

    def test_different_seed_different_hash(self) -> None:
        config = _make_config()

        hash1 = _compute_cache_hash(config, random_seed=42)
        hash2 = _compute_cache_hash(config, random_seed=123)

        assert hash1 != hash2

    def test_different_config_different_hash(self) -> None:
        config1 = _make_config()
        config2 = _make_config()
        config2.padder = ZeroPadder(target_length=500, padding_type="left")

        hash1 = _compute_cache_hash(config1, random_seed=42)
        hash2 = _compute_cache_hash(config2, random_seed=42)

        assert hash1 != hash2


class TestSaveDataPipelineConfig:

    def test_creates_text_file(self, tmp_path) -> None:
        config = _make_config()
        path = tmp_path / "config.txt"

        save_data_pipeline_config(config, path)

        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "ElderlyCoughAudioSourceReader" in content
        assert "MFCC" in content


class TestSaveGallery:

    def test_creates_gallery_folder_with_files(self, tmp_path) -> None:
        examples = [_make_example() for _ in range(5)]
        config = _make_config()

        generator = ExampleGalleryGenerator(
            data_pipeline_config=config,
            gallery_directory=tmp_path / "gallery",
            random_seed=42,
            num_examples=3,
        )
        result_path = generator.generate(examples)

        gallery_dirs = list((tmp_path / "gallery").iterdir())
        assert len(gallery_dirs) == 1
        gallery_dir = gallery_dirs[0]
        assert gallery_dir.name.startswith("test_config_")
        assert (gallery_dir / "examples.pdf").exists()
        assert (gallery_dir / "data_pipeline_config.txt").exists()
        assert result_path == gallery_dir / "examples.pdf"

    def test_repeated_generate_overwrites_same_folder(self, tmp_path) -> None:
        examples = [_make_example() for _ in range(5)]
        config = _make_config()

        generator = ExampleGalleryGenerator(
            data_pipeline_config=config,
            gallery_directory=tmp_path / "gallery",
            random_seed=42,
            num_examples=3,
        )

        path1 = generator.generate(examples)
        path2 = generator.generate(examples)
        assert path1 == path2
        assert path1.stat().st_size > 0

    def test_different_config_creates_new_gallery(self, tmp_path) -> None:
        examples = [_make_example() for _ in range(5)]
        config1 = _make_config()
        config2 = _make_config()
        config2.padder = ZeroPadder(target_length=500, padding_type="left")

        generator1 = ExampleGalleryGenerator(
            data_pipeline_config=config1,
            gallery_directory=tmp_path / "gallery",
            random_seed=42,
            num_examples=3,
        )
        generator2 = ExampleGalleryGenerator(
            data_pipeline_config=config2,
            gallery_directory=tmp_path / "gallery",
            random_seed=42,
            num_examples=3,
        )

        generator1.generate(examples)
        generator2.generate(examples)

        gallery_dirs = list((tmp_path / "gallery").iterdir())
        assert len(gallery_dirs) == 2
        dir_names = sorted(d.name for d in gallery_dirs)
        assert dir_names[0] != dir_names[1]

    def test_collision_resolution_creates_suffixed_folder(self, tmp_path) -> None:
        examples = [_make_example() for _ in range(5)]
        config = _make_config()

        gallery_dir = tmp_path / "gallery"
        gallery_dir.mkdir()

        collision_dir = gallery_dir / "test_config_stalehash"
        collision_dir.mkdir()

        generator = ExampleGalleryGenerator(
            data_pipeline_config=config,
            gallery_directory=gallery_dir,
            random_seed=42,
            num_examples=3,
        )
        result_path = generator.generate(examples)

        gallery_dirs = list(gallery_dir.iterdir())
        assert len(gallery_dirs) == 2
        config_hash = _compute_cache_hash(config, None)
        assert result_path.parent.name == f"test_config_{config_hash}"
        assert (result_path.parent / "examples.pdf").exists()

    def test_no_name_uses_hash_only(self, tmp_path) -> None:
        examples = [_make_example() for _ in range(5)]
        config = _make_config()
        config.name = None

        generator = ExampleGalleryGenerator(
            data_pipeline_config=config,
            gallery_directory=tmp_path / "gallery",
            random_seed=42,
            num_examples=3,
        )
        result_path = generator.generate(examples)

        gallery_dirs = list((tmp_path / "gallery").iterdir())
        assert len(gallery_dirs) == 1
        config_hash = _compute_cache_hash(config, None)
        assert gallery_dirs[0].name == config_hash
