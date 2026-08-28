import sys
from pathlib import Path

import pytest

from scripts.run_yaml_experiments import (
    DEFAULT_YAML_DIRECTORY,
    YamlExperimentRunner,
    get_arguments,
)



def test_runner_recursively_runs_yaml_files_in_sorted_order(
    tmp_path:Path,
    monkeypatch:pytest.MonkeyPatch,
) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    (tmp_path / "z.yaml").write_text("", encoding="utf-8")
    (nested_directory / "b.yml").write_text("", encoding="utf-8")
    (nested_directory / "a.yaml").write_text("", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("", encoding="utf-8")
    run_order = []

    def run_yaml_file(yaml_path:Path) -> int:
        run_order.append(yaml_path.relative_to(tmp_path).as_posix())
        return 0

    monkeypatch.setattr(
        "scripts.run_yaml_experiments.YamlExperimentRunner._run_yaml_file",
        lambda self, yaml_path: run_yaml_file(yaml_path),
    )

    YamlExperimentRunner().run(tmp_path)

    assert run_order == [
        "nested/a.yaml",
        "nested/b.yml",
        "z.yaml",
    ]


def test_runner_continues_after_failure(
    tmp_path:Path,
    monkeypatch:pytest.MonkeyPatch,
    capsys:pytest.CaptureFixture[str],
) -> None:
    first_yaml_path = tmp_path / "first.yaml"
    second_yaml_path = tmp_path / "second.yaml"
    first_yaml_path.write_text("", encoding="utf-8")
    second_yaml_path.write_text("", encoding="utf-8")
    return_codes = iter([1, 0])
    run_paths = []

    def run_yaml_file(yaml_path:Path) -> int:
        run_paths.append(yaml_path)
        return next(return_codes)

    monkeypatch.setattr(
        "scripts.run_yaml_experiments.YamlExperimentRunner._run_yaml_file",
        lambda self, yaml_path: run_yaml_file(yaml_path),
    )

    YamlExperimentRunner().run(tmp_path)

    assert run_paths == [first_yaml_path, second_yaml_path]
    assert "Failed: 1" in capsys.readouterr().out


def test_get_arguments_uses_default_yaml_directory(
    monkeypatch:pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_yaml_experiments.py"])

    arguments = get_arguments()

    assert arguments.yaml_directory == DEFAULT_YAML_DIRECTORY
