from core.training import TrainingConfig


default_training_config = TrainingConfig(
    random_seed=42,
    num_epochs=2,
    
    criterion_name="cross_entropy",
    optimizer_name="adamw",
    
    learning_rate=0.0001,
    weight_decay=0.001,
    class_weighting="balanced",
    
    batch_size=32,
    num_workers=0,
    drop_last=False,
    early_stopping_patience=2,
    
    load_best_model=True,
)
