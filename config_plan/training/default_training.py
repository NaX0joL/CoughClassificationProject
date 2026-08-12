from core.training import TrainingConfig


default_training_config = TrainingConfig(
    num_epochs=2,
    
    criterion_name="cross_entropy",
    optimizer_name="adamw",
    
    learning_rate=0.0001,
    weight_decay=0.001,
    
    batch_size=1,
    num_workers=1,
    drop_last=False,
    random_seed=42,
    
    load_best_model=True,
)
