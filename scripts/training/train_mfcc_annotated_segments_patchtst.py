from core.experiment import ExperimentOrchestrator, ExperimentConfig

from config_plan import (
    default_metrics_config,
    normal_batch_training_config,
    mfcc_annotated_segments_data_pipeline_config,
    mfcc_legrad_persistence_config,
    patchtst_config,
)


def main() -> None:
    exp_config = ExperimentConfig(
        data_pipeline_config=mfcc_annotated_segments_data_pipeline_config,
        model_config=patchtst_config,
        training_config=normal_batch_training_config,
        metrics_config=default_metrics_config,
        persistence_config=mfcc_legrad_persistence_config,
    )
    exp = ExperimentOrchestrator(config=exp_config, experiment_id="mfcc_annotated_segments_patchtst")
    exp.train_model()


if __name__ == "__main__":
    main()
    print("DONE!")
