from dataclasses import asdict
from pathlib import Path

from core.experiment import ExperimentOrchestrator
from core.experiment_config import ExperimentConfig
from config_plan import (
    mfcc_annotated_segments_data_pipeline_config,
    mfcc_sliding_windows_data_pipeline_config,
    log_mel_spectrogram_annotated_segments_data_pipeline_config,
    log_mel_spectrogram_sliding_windows_data_pipeline_config,
    downsampled_waveform_sliding_window_data_pipeline_config,
    
    mlp_config,
    lenet_config,
    patchtst_config,
    
    default_training_config,
    default_metrics_config,
    default_persistence_config,
)



def main():
    
    exp_config = ExperimentConfig(
        data_pipeline_config=mfcc_sliding_windows_data_pipeline_config,
        model_config=mlp_config,
        training_config=default_training_config,
        metrics_config=default_metrics_config,
        persistence_config=default_persistence_config,
    )
    
    exp = ExperimentOrchestrator(
        config=exp_config,
        experiment_id="test_run",
    )
    exp.train_model()
    
    path = exp.run_directory if exp.run_directory is not None else Path("")
    exp.load(mpkg_path=path)
    
    return



if __name__ == "__main__":
    main()
    print("DONE!")
