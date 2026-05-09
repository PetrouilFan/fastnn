import os
import numpy as np
import fastnn as fnn
from typing import Optional, Dict, Any, List, Union, Tuple

# Default CSV fields for CSVLogger
DEFAULT_CSV_FIELDS = [
    "epoch",
    "loss",
    "val_loss",
    "accuracy",
    "val_accuracy",
    "lr",
]

__all__ = [
    "Callback",
    "MonitorCallback",
    "ModelCheckpoint",
    "EarlyStopping",
    "CSVLogger",
    "LearningRateScheduler",
    "register_scheduler",
    "DEFAULT_CSV_FIELDS",
]

_SCHEDULER_REGISTRY: Dict[str, type] = {}

def register_scheduler(name: str, scheduler_cls: Optional[type] = None) -> Union[type, callable]:
    """Register a scheduler class for string-based creation.
    
    Can be used as a decorator:
        @register_scheduler("my_scheduler")
        class MyScheduler:
            ...
    Or called directly:
        register_scheduler("my_scheduler", MyScheduler)
    """
    if scheduler_cls is None:
        # Decorator mode
        def decorator(cls: type) -> type:
            _SCHEDULER_REGISTRY[name] = cls
            return cls
        return decorator
    else:
        _SCHEDULER_REGISTRY[name] = scheduler_cls


class Callback:
    __slots__ = ('model',)
    
    def __init__(self) -> None:
        self.model: Any = None

    def set_model(self, model: Any) -> None:
        self.model = model

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


class MonitorCallback(Callback):
    """Base class for callbacks that monitor a metric."""
    __slots__ = ('monitor',)

    def __init__(self, monitor: str) -> None:
        super().__init__()
        self.monitor = monitor

    def get_monitor_value(self, logs: Optional[Dict[str, Any]]) -> Optional[float]:
        """Get the monitor value from logs."""
        if logs is None:
            return None
        return logs.get(self.monitor)

    def _check_monitor(self, logs: Optional[Dict[str, Any]]) -> Tuple[Optional[float], bool]:
        """Check monitor value and return (value, should_continue).
        
        Returns:
            tuple: (value, should_continue) where should_continue is False if value is None
        """
        value = self.get_monitor_value(logs)
        if value is None:
            return None, False
        return value, True

    def _is_improvement(
        self,
        new_value: float,
        old_value: Optional[float],
        mode: str = "min",
        min_delta: float = 0.0
    ) -> bool:
        """Check if new_value is an improvement over old_value.
        
        Args:
            new_value: Current metric value.
            old_value: Previous best metric value.
            mode: "min" for lower is better, "max" for higher is better.
            min_delta: Minimum change to qualify as improvement.
            
        Returns:
            True if new_value is an improvement, False otherwise.
        """
        if old_value is None:
            return True
        if mode == "min":
            return new_value < old_value - min_delta
        elif mode == "max":
            return new_value > old_value + min_delta
        else:
            return False


class ModelCheckpoint(MonitorCallback):
    __slots__ = ('filepath', 'mode', 'save_best_only', 'verbose', 'best_value', 'should_save')

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(monitor)
        self.filepath = filepath
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value: Optional[float] = None
        self.should_save = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        value, should_continue = self._check_monitor(logs)
        if not should_continue:
            return

        is_best = self._is_improvement(value, self.best_value, mode=self.mode, min_delta=0.0)

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

    def save_model(self, model: Any) -> None:
        if self.should_save:
            import fastnn as fnn
            fnn.io.save(model, self.filepath)
            self.should_save = False


class EarlyStopping(MonitorCallback):
    __slots__ = ('patience', 'min_delta', 'restore_best_weights', 'counter', 'best_value', 'best_weights', 'should_stop')

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> None:
        super().__init__(monitor)
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter: int = 0
        self.best_value: Optional[float] = None
        self.best_weights: Optional[Dict[str, Any]] = None
        self.should_stop: bool = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        value, should_continue = self._check_monitor(logs)
        if not should_continue:
            return

        if self.best_value is None:
            self.best_value = value
            self.counter = 0
            if self.restore_best_weights and self.model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            return

        improved = self._is_improvement(
            value, self.best_value, mode="min", min_delta=self.min_delta
        )
        if improved:
            self.best_value = value
            self.counter = 0
            if self.restore_best_weights and self.model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                self.should_stop = True

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.restore_best_weights and self.best_weights is not None and self.model is not None:
            print("Restoring best model weights...")
            self.model.load_state_dict(self.best_weights)
            self.best_weights = None


class LearningRateScheduler(Callback):
    """Callback for learning rate scheduling.
    
    Wraps a scheduler from fastnn.schedulers and calls its step() method
    at the end of each epoch.
    
    Args:
        scheduler: A scheduler object from fastnn.schedulers (e.g., StepLR, CosineAnnealingLR).
                   Or a string specifying the schedule type (registered in _SCHEDULER_REGISTRY).
        **kwargs: Additional arguments for creating the scheduler from string.
                   Required if scheduler is a string: optimizer, and schedule-specific args.
    """
    __slots__ = ('scheduler',)

    def __init__(self, scheduler: Union[str, object], **kwargs: Any) -> None:
        super().__init__()
        if isinstance(scheduler, str):
            if not _SCHEDULER_REGISTRY:
                from fastnn.schedulers import StepLR, CosineAnnealingLR
                register_scheduler("step")(StepLR)
                register_scheduler("cosine")(CosineAnnealingLR)
            if scheduler not in _SCHEDULER_REGISTRY:
                raise ValueError(f"Unknown schedule: {scheduler}. Registered: {list(_SCHEDULER_REGISTRY.keys())}")
            scheduler_cls = _SCHEDULER_REGISTRY[scheduler]
            self.scheduler = scheduler_cls(**kwargs)
        else:
            self.scheduler = scheduler

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update learning rate and log the new value."""
        new_lr = self.scheduler.step()
        if logs is not None:
            logs["lr"] = new_lr


class CSVLogger(Callback):
    __slots__ = ('filepath', 'fields', '_file', '_initialized')

    def __init__(self, filepath: str, fields: Optional[List[str]] = None) -> None:
        super().__init__()
        self.filepath = filepath
        self.fields = fields if fields is not None else DEFAULT_CSV_FIELDS
        self._file: Optional[Any] = None
        self._initialized: bool = False

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._initialized:
            return
        mode = "a" if os.path.exists(self.filepath) else "w"
        self._file = open(self.filepath, mode)
        if mode == "w":
            self._file.write(",".join(self.fields) + "\n")
        self._initialized = True

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if not self._initialized:
            return
        values = [str(logs.get(field, "")) for field in self.fields]
        self._file.write(",".join(values) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the log file explicitly."""
        if self._file:
            self._file.close()
            self._file = None
            self._initialized = False

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        self.close()
