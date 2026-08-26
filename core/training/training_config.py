from dataclasses import dataclass
from typing import Literal, TypeAlias



CRITERION_CHOICES: TypeAlias = Literal["cross_entropy"]
OPTIMIZER_CHOICES: TypeAlias = Literal["adam", "adamw"]
CLASS_WEIGHTING_CHOICES: TypeAlias = Literal["none", "balanced"]



@dataclass
class TrainingConfig:
    random_seed: int
    num_epochs: int
    
    criterion_name: CRITERION_CHOICES
    optimizer_name: OPTIMIZER_CHOICES
    
    learning_rate: float
    weight_decay: float
    
    batch_size: int
    num_workers: int
    drop_last: bool
    
    class_weighting: CLASS_WEIGHTING_CHOICES = "none"
    early_stopping_patience: int|None = None
    
    load_best_model: bool = True
    
    @classmethod
    def default(cls) -> "TrainingConfig":
        training_config = cls(
            random_seed = 42,
            num_epochs = 100,
            
            criterion_name="cross_entropy",
            optimizer_name="adamw",
            
            learning_rate = 0.0001,
            weight_decay = 0.001,
            class_weighting = "none",
            
            batch_size = 32,
            num_workers = 1,
            drop_last = False,
            early_stopping_patience = None,
            
            load_best_model = True,
        )
        return training_config
