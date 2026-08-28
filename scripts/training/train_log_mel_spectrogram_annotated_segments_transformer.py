import os
import sys

sys.path.append(os.getcwd())

from core.experiment import ExperimentOrchestrator, ExperimentConfig

from config_plan import (
    default_metrics_config,
    small_batch_training_config,
    log_mel_spectrogram_annotated_segments_data_pipeline_config,
    log_mel_spectrogram_legrad_persistence_config,
    transformer_config,
)


def main() -> None:
    exp_config = ExperimentConfig(
        data_pipeline_config=log_mel_spectrogram_annotated_segments_data_pipeline_config,
        model_config=transformer_config,
        training_config=small_batch_training_config,
        metrics_config=default_metrics_config,
        persistence_config=log_mel_spectrogram_legrad_persistence_config,
    )
    exp = ExperimentOrchestrator(config=exp_config, experiment_id="log_mel_spectrogram_annotated_segments_transformer")
    exp.train_model()


if __name__ == "__main__":
    main()
    print("DONE!")
