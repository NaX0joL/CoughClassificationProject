from core.persistence.persistence import _create_run_directory


def test_create_run_directory_adds_incrementing_suffix_for_duplicate_names(
    tmp_path,
) -> None:
    first_directory = _create_run_directory(tmp_path, "experiment")
    second_directory = _create_run_directory(tmp_path, "experiment")
    third_directory = _create_run_directory(tmp_path, "experiment")

    assert first_directory.name == "experiment"
    assert second_directory.name == "experiment_2"
    assert third_directory.name == "experiment_3"
