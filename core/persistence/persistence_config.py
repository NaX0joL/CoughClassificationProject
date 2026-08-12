from dataclasses import dataclass


@dataclass(frozen=True)
class PersistenceConfig:
    number_of_train_model_outputs:int=12
    number_of_validation_model_outputs:int=12
    mfcc_color_percentile:float=99.0
    include_grad_cam:bool=True
    include_legrad:bool=True

    @classmethod
    def default(cls) -> "PersistenceConfig":
        return cls()
