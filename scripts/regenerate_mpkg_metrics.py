import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.artifacts.artifact_reader import ExperimentArtifactReader
from core.artifacts.processes.metrics_writing import write_metrics_workbook
from core.metrics.evaluation import ModelEvaluator

from modules.resolve_pytorch_device import get_optimal_device



def main() -> None:
    arguments = get_arguments()
    
    mpkg_paths = _list_all_mpkgs(arguments.paths)
    metrics_excel_filename = "metrics.xlsx"
    
    for mpkg_path in mpkg_paths:
        metrics_excel_file_path = mpkg_path / metrics_excel_filename
        
        metrics_excel_exist = metrics_excel_file_path.is_file()
        force_regenerate = arguments.force_regenerate_all
        
        if force_regenerate or not metrics_excel_exist:
            print(f"> regenerating metrics for: {mpkg_path}")
            regenerate_mpkg_metrics(mpkg_path)
    
    return


def get_arguments(arguments:list[str]|None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate metrics Excel files for mpkg paths or directories "
            "containing mpkgs."
        ),
        epilog=(
            "Example: python scripts/regenerate_mpkg_metrics.py outputs/mpkg/tmp"
        ),
    )
    
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="mpkg folder(s) or directories containing mpkgs",
    )
    parser.add_argument(
        "--force-regenerate-all",
        action="store_true",
        help="regenerate metrics even when metrics.xlsx already exists",
    )
    
    return parser.parse_args(arguments)


def _list_all_mpkgs(paths:list[Path]) -> list[Path]:
    marker_filename = "__mpkg__.py"
    mpkg_paths:set[Path] = set()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"input path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"input path must be a directory: {path}")

        if (path / marker_filename).is_file():
            mpkg_paths.add(path)

        mpkg_paths.update(
            marker_path.parent
            for marker_path in path.rglob(marker_filename)
            if marker_path.is_file()
        )

    return sorted(mpkg_paths)


def regenerate_mpkg_metrics(mpkg_path:Path) -> None:
    reader = ExperimentArtifactReader(directory=mpkg_path)
    
    config = reader.load_configuration()
    data_pipeline = reader.create_data_pipeline()
    fold_indices = reader.fold_indices
    
    data_pipeline.initialize()
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    device = get_optimal_device()
    
    evaluator = ModelEvaluator(
        config=evaluation_config["metrics"],
        label_mapping=evaluation_config["label_mapping"],
    )
    
    fold_metrics:list[tuple[int, dict[str, float|None]]] = []
    
    for fold_index in fold_indices:
        model = reader.create_fold_model(fold_index)
        model = model.to(device)
        
        data_module = data_pipeline.get_data_module(
            index=fold_index - 1,
            
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            drop_last_batch=training_config.drop_last,
            random_seed=training_config.random_seed,
            pin_memory=device.type == "cuda",
        )
        data_loader = data_module.test_loader

        evaluation = evaluator.evaluate(
            model,
            data_loader,
        )
        fold_metrics.append(
            (fold_index, evaluation.metrics.to_dict()),
        )
        
    write_metrics_workbook(
        path=reader.directory / "metrics.xlsx",
        fold_metrics=fold_metrics,
    )
    return



if __name__ == "__main__":
    main()
    print("DONE!")
