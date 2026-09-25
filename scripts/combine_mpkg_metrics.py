import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.artifacts import ExperimentArtifactReader



OUTPUT_FILENAME = "combined_mpkg_metrics.xlsx"



def main() -> None:
    arguments = get_arguments()
    output_path = combine_mpkg_metrics(arguments.directory)
    print(f"combined metrics written to {output_path}")
    return


def get_arguments(arguments:list[str]|None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the average fold metrics from every mpkg under a directory "
            "into one Excel workbook."
        ),
        epilog=(
            "Example: python scripts/combine_mpkg_metrics.py outputs/mpkg/tmp"
        ),
    )
    
    parser.add_argument(
        "directory",
        type=Path,
        help="directory containing mpkg folders",
    )
    
    return parser.parse_args(arguments)



def combine_mpkg_metrics(directory:Path) -> Path:
    mpkg_paths = _list_all_mpkgs(directory)
    
    rows = []
    for path in mpkg_paths:
        metrics = _load_average_metrics(path)
        rows.append(metrics)
        print(f"> adding metrics from: {path}")
    
    output_path = directory / OUTPUT_FILENAME
    pd.DataFrame(rows).to_excel(
        output_path,
        sheet_name="Average Metrics",
        index=False,
    )
    return output_path


def _list_all_mpkgs(directory:Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"input directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"input path must be a directory: {directory}")

    marker_filename = "__mpkg__.py"
    mpkg_paths = {
        marker_path.parent
        for marker_path in directory.rglob(marker_filename)
        if marker_path.is_file()
    }
    
    if (directory / marker_filename).is_file():
        mpkg_paths.add(directory)
    if not mpkg_paths:
        raise FileNotFoundError(f"no mpkg folders found in: {directory}")
    
    return sorted(mpkg_paths)


def _load_average_metrics(mpkg_path:Path) -> dict[str, object]:
    reader = ExperimentArtifactReader(mpkg_path)
    
    summary = reader.load_metrics_summary()
    if summary["metric"].duplicated().any():
        raise ValueError(f"mpkg Summary contains duplicate metrics: {mpkg_path}")

    row:dict[str, object] = {
        "mpkg": reader.experiment_id,
        "path": str(mpkg_path),
    }
    row.update(dict(zip(summary["metric"], summary["mean"], strict=True)))
    
    return row



if __name__ == "__main__":
    main()
    print("DONE!")
