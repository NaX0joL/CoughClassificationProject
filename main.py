from pathlib import Path

from modules.yaml_experiment import YamlToExperimentConverter, do_experiment



def main(yaml_path:Path) -> None:
    experiment = YamlToExperimentConverter().convert(yaml_path)
    do_experiment(experiment)
    return



if __name__ == "__main__":
    main(Path("yaml/run/mfcc_sliding_windows_mlp_v3.yaml"))
    print("DONE!")
