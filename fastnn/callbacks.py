import os
import numpy as np
import fastnn as fnn

# Default CSV fields for CSVLogger
DEFAULT_CSV_FIELDS = [
    "epoch",
    "loss",
    "val_loss",
    "accuracy",
    "val_accuracy",
    "lr",
]


class Callback:
    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class MonitorCallback(Callback):
    """Base class for callbacks that monitor a metric."""

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def get_monitor_value(self, logs):
        """Get the monitor value from logs."""
        if logs is None:
            return None
        return logs.get(self.monitor)

    def _check_monitor(self, logs):
        """Check monitor value and return (value, should_continue).
        
        Returns:
            tuple: (value, should_continue) where should_continue is False if value is None
        """
        value = self.get_monitor_value(logs)
        if value is None:
            return None, False
        return value, True


class ModelCheckpoint(MonitorCallback):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=True,
    ):
        super().__init__(monitor)
        self.filepath = filepath
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value = None
        self.should_save = False

    def on_epoch_end(self, epoch, logs=None):
        value, should_continue = self._check_monitor(logs)
        if not should_continue:
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
            fnn.save_model(model, self.filepath)
            self.should_save = False


class EarlyStopping(MonitorCallback):
    def __init__(
        self, monitor="val_loss", patience=5, min_delta=0.0, restore_best_weights=True
    ):
        super().__init__(monitor)
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_value = None
        self.best_weights = None
        self.should_stop = False

    def on_epoch_end(self, epoch, logs=None):
        value, should_continue = self._check_monitor(logs)
        if not should_continue:
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

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            print("Restoring best model weights...")


class LearningRateScheduler(Callback):
    """Callback for learning rate scheduling.

    Wraps a scheduler from fastnn.schedulers and calls its step() method
    at the end of each epoch.

    Args:
        scheduler: A scheduler object from fastnn.schedulers (e.g., StepLR, CosineAnnealingLR).
                   Or a string specifying the schedule type ("step" or "cosine").
        **kwargs: Additional arguments for creating the scheduler from string.
                  Required if scheduler is a string: optimizer, and schedule-specific args.
    """

    def __init__(self, scheduler, **kwargs):
        super().__init__()
        if isinstance(scheduler, str):
            # Create scheduler from string
            from fastnn.schedulers import StepLR, CosineAnnealingLR

            if scheduler == "step":
                optimizer = kwargs.get("optimizer")
                if optimizer is None:
                    raise ValueError("optimizer is required for step schedule")
                step_size = kwargs.get("step_size", 10)
                gamma = kwargs.get("gamma", 0.1)
                self.scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif scheduler == "cosine":
                optimizer = kwargs.get("optimizer")
                if optimizer is None:
                    raise ValueError("optimizer is required for cosine schedule")
                T_max = kwargs.get("T_max", 100)
                eta_min = kwargs.get("eta_min", 0.0001)
                self.scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            else:
                raise ValueError(f"Unknown schedule: {scheduler}")
        else:
            # Assume scheduler is a scheduler object
            self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        """Update learning rate and log the new value."""
        new_lr = self.scheduler.step()
        if logs is not None:
            logs["lr"] = new_lr


class CSVLogger(Callback):
    def __init__(self, filepath, fields=None):
        self.filepath = filepath
        self.fields = fields if fields is not None else DEFAULT_CSV_FIELDS
        self._file = None
        self._initialized = False

    def on_train_begin(self, logs=None):
        if self._initialized:
            return
        mode = "a" if os.path.exists(self.filepath) else "w"
        self._file = open(self.filepath, mode)

        if mode == "w":
            self._file.write(",".join(self.fields) + "\n")

        self._initialized = True

    def on_epoch_end(self, epoch, logs=None):
        if not self._initialized:
            return

        values = [str(logs.get(field, "")) for field in self.fields]
        self._file.write(",".join(values) + "\n")
        self._file.flush()

    def on_train_end(self, logs=None):
        if self._file:
            self._file.close()
            self._file = None

    def __del__(self):
        self.on_train_end()
