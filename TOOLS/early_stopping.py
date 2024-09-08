class EarlyStoppingCurveWithSmooth:
    def __init__(self, patience=7, min_delta=0.0, smoothing_factor=0.9):
        """
        Args:
            patience (int): Number of epochs to wait after the last time the error decreased.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            smoothing_factor (float): Factor for exponential moving average.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_factor = smoothing_factor
        self.counter = 0
        self.train_losses = []
        self.smooth_losses = []
        self.early_stop = False

    def __call__(self, train_loss):
        self.train_losses.append(train_loss)

        if len(self.smooth_losses) == 0:
            smooth_loss = train_loss
        else:
            smooth_loss = self.smoothing_factor * self.smooth_losses[-1] + (1 - self.smoothing_factor) * train_loss

        self.smooth_losses.append(smooth_loss)

        print(f"Epoch {len(self.train_losses)}: train_loss={train_loss:.6f}, smooth_loss={smooth_loss:.6f}")

        if len(self.smooth_losses) > 1:
            if abs(self.smooth_losses[-1] - self.smooth_losses[-2]) < self.min_delta:
                self.counter += 1
                print(f"No significant improvement. Counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0
                print("Significant improvement detected. Counter reset.")

        return self.early_stop
