class EarlyStopping:
    
    def __init__(self, patience:int) -> None:
        if patience < 1:
            raise ValueError("early stopping patience must be at least 1")

        self.patience = patience
        self.best_validation_loss = None
        self.epochs_without_improvement = 0
        return

    def update(self, validation_loss:float) -> bool:
        if (
            self.best_validation_loss is None
            or validation_loss < self.best_validation_loss
        ):
            self.best_validation_loss = validation_loss
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience
