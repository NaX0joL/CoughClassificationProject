from core.experiment import ExperimentOrchestrator, ExperimentConfig

from config_plan import (
    mfcc_data_pipeline_config,
    mlp_config,
    normal_batch_training_config,
    default_metrics_config,
    default_persistence_config,
)



def main():
    
    exp_config = ExperimentConfig(
        data_pipeline_config=mfcc_data_pipeline_config,
        model_config=mlp_config,
        training_config=normal_batch_training_config,
        metrics_config=default_metrics_config,
        persistence_config=default_persistence_config,
    )
    
    exp = ExperimentOrchestrator(
        config=exp_config,
        experiment_id="mfcc_mlp",
    )
    exp.train_model()
    return



if __name__ == "__main__":
    main()
    print("DONE!")
