import numpy as np


class Callback:
    def on_epoch_begin(self, epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass


class ModelCheckpoint:
    def __init__(
        self,
        dir_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=True,
    ):
        self.dir_path = dir_path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value = None

    def on_epoch_end(self, epoch, logs):
        value = logs.get(self.monitor)
        if value is None:
            return

        is_best = False
        if self.best_value is None:
            is_best = True
        elif self.mode == "min":
            is_best = value < self.best_value
        elif self.mode == "max":
            is_best = value > self.best_value

        if is_best:
            self.best_value = value
            if self.verbose:
                print(
                    f"Epoch {epoch}: {self.monitor} = {value:.4}, saving best model..."
                )


class EarlyStopping:
    def __init__(
        self, monitor="val_loss", patience=5, min_delta=0.0, restore_best_weights=True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_value = None

    def on_epoch_end(self, epoch, logs):
        value = logs.get(self.monitor)
        if value is None:
            return

        if self.best_value is None:
            self.best_value = value
            self.counter = 0
        elif abs(value - self.best_value) > self.min_delta:
            if value < self.best_value:
                self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}!")


class LearningRateScheduler:
    def __init__(self, schedule="cosine", **kwargs):
        self.schedule = schedule
        self.lr = kwargs.get("lr", 0.01)
        self.step_size = kwargs.get("step_size", 10)
        self.gamma = kwargs.get("gamma", 0.1)
        self.T_max = kwargs.get("T_max", 100)
        self.eta_min = kwargs.get("eta_min", 0.0001)

    def on_epoch_end(self, epoch, logs):
        if self.schedule == "step":
            new_lr = self.lr * (self.gamma ** (epoch // self.step_size))
        elif self.schedule == "cosine":
            new_lr = (
                self.eta_min
                + (self.lr - self.eta_min)
                * (1 + np.cos(np.pi * epoch / self.T_max))
                / 2
            )
        else:
            new_lr = self.lr

        logs["lr"] = new_lr


class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.epoch = 0

    def on_epoch_end(self, epoch, logs):
        self.epoch = epoch
