from dataclasses import dataclass


@dataclass(frozen=True)
class PersistenceConfig:
    number_of_train_model_outputs:int=12
    number_of_validation_model_outputs:int=12
    feature_color_percentile:float=99.0
    feature_colormap:str="inferno"
    x_axis_label:str="Frame"
    y_axis_label:str="Feature bin"
    colorbar_label:str="Feature value"
    include_grad_cam:bool=True
    include_legrad:bool=True

    def __post_init__(self) -> None:
        if not 0 < self.feature_color_percentile <= 100:
            raise ValueError("feature_color_percentile must be in the range (0, 100]")
        for name, value in {
            "feature_colormap": self.feature_colormap,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "colorbar_label": self.colorbar_label,
        }.items():
            if not value.strip():
                raise ValueError(f"{name} must not be blank")

    @classmethod
    def default(cls) -> "PersistenceConfig":
        return cls()
