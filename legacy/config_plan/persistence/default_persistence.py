from core.persistence import PersistenceConfig


default_persistence_config = PersistenceConfig(
    number_of_train_model_outputs=20,
    number_of_validation_model_outputs=20,
    feature_color_percentile=99.0,
    feature_colormap="inferno",
    x_axis_label="Frame",
    y_axis_label="Feature bin",
    colorbar_label="Feature value",
    include_grad_cam=False,
    include_legrad=False,
)
