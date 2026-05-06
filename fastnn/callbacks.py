import os
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


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value = None
        self.should_save = False

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
        else:
            if self.verbose:
                print(f"Warning: Unknown mode '{self.mode}' for ModelCheckpoint")
            return

        self.should_save = False
        if is_best:
            self.best_value = value
            self.should_save = True
            if self.verbose:
                print(
                    f"Epoch {epoch}: {self.monitor} = {value:.4f}, saving best model..."
                )
        elif not self.save_best_only:
            self.should_save = True
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} = {value:.4f}, saving model...")

    def save_model(self, model):
        if self.should_save:
            import fastnn as fnn

            fnn.save_model(model, self.filepath)
            self.should_save = False


class EarlyStopping(Callback):
    def __init__(
        self, monitor="val_loss", patience=5, min_delta=0.0, restore_best_weights=True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_value = None
        self.best_weights = None
        self.should_stop = False

    def on_epoch_end(self, epoch, logs):
        value = logs.get(self.monitor)
        if value is None:
            return

        if self.best_value is None:
            self.best_value = value
            self.counter = 0
            return

        # Check for improvement beyond min_delta
        improved = value < self.best_value - self.min_delta
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            # No significant improvement; count patience
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                self.should_stop = True

    def on_train_end(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            print("Restoring best model weights...")
            # Model should handle restoring weights


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
            raise ValueError(f"Unknown schedule: {self.schedule}")

        logs["lr"] = new_lr


class CSVLogger(Callback):
    def __init__(self, filepath, fields=None):
        self.filepath = filepath
        self.fields = fields
        self._file = None
        self._initialized = False

    def on_train_begin(self, logs):
        if self._initialized:
            return
        mode = "a" if os.path.exists(self.filepath) else "w"
        self._file = open(self.filepath, mode)
        if mode == "w" and self.fields is not None:
            self._file.write(",".join(self.fields) + "\n")
        self._initialized = True

    def on_epoch_end(self, epoch, logs):
        if not self._initialized:
            return

        if self.fields is None:
            field_order = [
                "epoch",
                "loss",
                "val_loss",
                "accuracy",
                "val_accuracy",
                "lr",
            ]
            fields = list(field_order)
            extra_keys = sorted(set(logs.keys()) - set(field_order))
            fields.extend(extra_keys)
            self.fields = fields
            self._file.write(",".join(fields) + "\n")

        values = [str(logs.get(field, "")) for field in self.fields]
        self._file.write(",".join(values) + "\n")
        self._file.flush()

    def on_train_end(self, logs):
        if self._file:
            self._file.close()
            self._file = None
