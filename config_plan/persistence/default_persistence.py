from core.persistence import PersistenceConfig


default_persistence_config = PersistenceConfig(
    number_of_train_model_outputs=20,
    number_of_validation_model_outputs=20,
    mfcc_color_percentile=99.0,
    include_grad_cam=False,
    include_legrad=False,
)
