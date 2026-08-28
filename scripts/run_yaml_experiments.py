import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

import argparse
import subprocess
import sys
from pathlib import Path

from modules.email_notification import send_email



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML_DIRECTORY = PROJECT_ROOT / "yaml/run"
YAML_SUFFIXES = {".yaml", ".yml"}

SUCCESS_EMAIL_SUBJECT = "YAML experiments completed successfully"
FAILURE_EMAIL_SUBJECT = "YAML experiments completed with failures"



class YamlExperimentRunner:

    def run(self, yaml_directory:Path=DEFAULT_YAML_DIRECTORY) -> None:
        yaml_directory = self._resolve_yaml_directory(yaml_directory)
        yaml_paths = self._find_yaml_files(yaml_directory)
        succeeded_paths, failed_paths = self._run_yaml_files(
            yaml_paths,
            yaml_directory,
        )
        run_summary = self._create_run_summary(
            succeeded_paths,
            failed_paths,
            yaml_directory,
        )
        self._print_run_summary(run_summary)
        self._send_run_notification(run_summary, failed_paths)
        return

    def _resolve_yaml_directory(self, yaml_directory:Path) -> Path:
        return yaml_directory.resolve()

    def _find_yaml_files(self, yaml_directory:Path) -> list[Path]:
        if not yaml_directory.is_dir():
            raise NotADirectoryError(
                f"YAML directory does not exist: {yaml_directory}",
            )

        yaml_paths = [
            path
            for path in yaml_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in YAML_SUFFIXES
        ]
        yaml_paths.sort(
            key=lambda path: path.relative_to(yaml_directory).as_posix(),
        )

        if not yaml_paths:
            raise FileNotFoundError(f"No YAML files found in: {yaml_directory}")
        return yaml_paths

    def _run_yaml_files(
        self,
        yaml_paths:list[Path],
        yaml_directory:Path,
    ) -> tuple[list[Path], list[Path]]:
        succeeded_paths = []
        failed_paths = []

        for yaml_path in yaml_paths:
            self._print_running_yaml(yaml_path, yaml_directory)
            return_code = self._run_yaml_file(yaml_path)
            self._record_run_result(
                yaml_path,
                return_code,
                succeeded_paths,
                failed_paths,
            )

        return succeeded_paths, failed_paths

    def _print_running_yaml(
        self,
        yaml_path:Path,
        yaml_directory:Path,
    ) -> None:
        relative_path = yaml_path.relative_to(yaml_directory)
        print(f"\n> Running {relative_path}")
        return

    def _run_yaml_file(self, yaml_path:Path) -> int:
        completed_process = subprocess.run(
            [
                sys.executable,
                "-m",
                "modules.yaml_experiment",
                str(yaml_path),
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        return completed_process.returncode

    def _record_run_result(
        self,
        yaml_path:Path,
        return_code:int,
        succeeded_paths:list[Path],
        failed_paths:list[Path],
    ) -> None:
        if return_code == 0:
            succeeded_paths.append(yaml_path)
        else:
            failed_paths.append(yaml_path)
        return

    def _create_run_summary(
        self,
        succeeded_paths:list[Path],
        failed_paths:list[Path],
        yaml_directory:Path,
    ) -> str:
        summary_lines = [
            "YAML experiment summary",
            f"Directory: {yaml_directory}",
            f"Succeeded: {len(succeeded_paths)}",
        ]
        for yaml_path in succeeded_paths:
            summary_lines.append(
                f"  - {yaml_path.relative_to(yaml_directory)}"
            )

        summary_lines.append(f"Failed: {len(failed_paths)}")
        for yaml_path in failed_paths:
            summary_lines.append(
                f"  - {yaml_path.relative_to(yaml_directory)}"
            )

        run_summary = "\n".join(summary_lines)
        return run_summary

    def _print_run_summary(self, run_summary:str) -> None:
        print(f"\n{run_summary}\n")
        return

    def _send_run_notification(
        self,
        run_summary:str,
        failed_paths:list[Path],
    ) -> None:
        subject = (
            FAILURE_EMAIL_SUBJECT
            if failed_paths
            else SUCCESS_EMAIL_SUBJECT
        )
        send_email(subject, run_summary)
        return



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "yaml_directory",
        type=Path,
        nargs="?",
        default=DEFAULT_YAML_DIRECTORY,
    )
    arguments = parser.parse_args()
    return arguments


def main() -> None:
    arguments = get_arguments()
    YamlExperimentRunner().run(arguments.yaml_directory)
    return



if __name__ == "__main__":
    main()
    print("DONE!")
