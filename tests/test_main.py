from pathlib import Path
from unittest.mock import Mock

import main as main_module



def test_main_runs_yaml_experiment(monkeypatch) -> None:
    yaml_path = Path("yaml/run/test.yaml")
    experiment = object()
    converter = Mock()
    converter.convert.return_value = experiment
    run_experiment = Mock()
    monkeypatch.setattr(
        main_module,
        "YamlToExperimentConverter",
        Mock(return_value=converter),
    )
    monkeypatch.setattr(main_module, "do_experiment", run_experiment)

    main_module.main(yaml_path)

    converter.convert.assert_called_once_with(yaml_path)
    run_experiment.assert_called_once_with(experiment)
