from core.persistence import PersistenceConfig


mfcc_legrad_persistence_config = PersistenceConfig(
    number_of_train_model_outputs=20,
    number_of_validation_model_outputs=20,
    feature_color_percentile=99.0,
    x_axis_label="Frame",
    y_axis_label="MFCC coefficient",
    colorbar_label="MFCC value",
    include_grad_cam=False,
    include_legrad=True,
)
