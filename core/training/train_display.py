from tqdm.auto import tqdm



class TrainDisplay:

    def __init__(self, number_of_epochs:int) -> None:
        self.progress_bar = tqdm(
            total=number_of_epochs,
            desc="Training",
            unit="epoch",
        )
        return

    def update(
        self,
        training_loss:float,
        validation_loss:float,
    ) -> None:
        self.progress_bar.set_postfix(
            training_loss=f"{training_loss:.4f}",
            validation_loss=f"{validation_loss:.4f}",
        )
        self.progress_bar.update()
        return

    def close(self) -> None:
        self.progress_bar.close()
        return
