import sys

from scripts.run_experiment_yaml import main as run_experiment_yaml



def main(arguments:list[str]|None=None) -> None:
    run_experiment_yaml(arguments)
    return



if __name__ == "__main__":
    main(sys.argv[1:])
