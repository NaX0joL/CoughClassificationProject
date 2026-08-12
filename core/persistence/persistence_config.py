from dataclasses import dataclass


@dataclass(frozen=True)
class PersistenceConfig:
    number_of_train_model_outputs:int=12
    number_of_validation_model_outputs:int=12
    mfcc_color_percentile:float=99.0

    @classmethod
    def default(cls) -> "PersistenceConfig":
        return cls()
