from dataclasses import dataclass
from typing import Literal, TypeAlias



CRITERION_CHOICES: TypeAlias = Literal["cross_entropy"]
OPTIMIZER_CHOICES: TypeAlias = Literal["adam", "adamw"]



@dataclass
class TrainingConfig:
    num_epochs: int
    
    criterion_name: CRITERION_CHOICES
    optimizer_name: OPTIMIZER_CHOICES
    
    learning_rate: float
    weight_decay: float
    
    batch_size: int
    num_workers: int
    drop_last: bool
    
    @classmethod
    def default(cls) -> "TrainingConfig":
        training_config = cls(
            num_epochs = 100,
            
            criterion_name="cross_entropy",
            optimizer_name="adamw",
            
            learning_rate = 0.0001,
            weight_decay = 0.001,
            
            batch_size = 32,
            num_workers = 1,
            drop_last = False,
        )
        return training_config