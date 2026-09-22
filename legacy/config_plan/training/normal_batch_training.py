from core.training import TrainingConfig


normal_batch_training_config = TrainingConfig(
    random_seed=42,
    num_epochs=100,
    
    criterion_name="cross_entropy",
    optimizer_name="adamw",
    
    learning_rate=0.0001,
    weight_decay=0.001,
    class_weighting="balanced",
    
    batch_size=16,
    num_workers=0,
    drop_last=False,
    early_stopping_patience=None,
    
    load_best_model=True,
)
