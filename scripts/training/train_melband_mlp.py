from core.experiment import ExperimentOrchestrator, ExperimentConfig

from config_plan import (
    default_metrics_config,
    normal_batch_training_config,
    melband_data_pipeline_config,
    default_persistence_config,
    mlp_config,
)


def main() -> None:
    exp_config = ExperimentConfig(
        data_pipeline_config=melband_data_pipeline_config,
        model_config=mlp_config,
        training_config=normal_batch_training_config,
        metrics_config=default_metrics_config,
        persistence_config=default_persistence_config,
    )
    exp = ExperimentOrchestrator(config=exp_config, experiment_id="melband_mlp")
    exp.train_model()


if __name__ == "__main__":
    main()
    print("DONE!")
