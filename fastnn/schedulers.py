"""Learning rate schedulers for fastnn.

This module provides various learning rate scheduling strategies, including
step decay, cosine annealing, exponential decay, and reduce on plateau.

Examples:
    >>> import fastnn as fnn
    >>> optimizer = fnn.optimizers.SGD(model.parameters(), lr=0.1)
    >>> scheduler = fnn.schedulers.StepLR(optimizer, step_size=30, gamma=0.1)
    >>> for epoch in range(100):
    ...     train(...)
    ...     scheduler.step()
"""

import math
from typing import Any



class LRScheduler:
    """Base class for learning rate schedulers.
    
    Modifies the optimizer's learning rate in-place via state_dict.
    
    Args:
        optimizer: Optimizer to schedule.
    
    Examples:
        >>> scheduler = LRScheduler(optimizer)
        >>> lr = scheduler.get_lr()
    """
    
    def __init__(self, optimizer: Any):
        self.optimizer = optimizer
        self.base_lr = self._get_lr()
        self.last_epoch = -1
    
    def _get_lr(self) -> float:
        sd = self.optimizer.state_dict()
        return sd.get("lr", 0.01)
    
    def _set_lr(self, lr: float) -> None:
        sd = self.optimizer.state_dict()
        sd["lr"] = lr
        self.optimizer.load_state_dict(sd)
    
    def get_lr(self) -> float:
        raise NotImplementedError
    
    def step(self) -> float:
        """Update learning rate and return the new learning rate."""
        self.last_epoch += 1
        lr = self.get_lr()
        self._set_lr(lr)
        return lr


class StepLR(LRScheduler):
    """Decays LR by gamma every step_size epochs.
    
    Args:
        optimizer: Optimizer to schedule.
        step_size: Number of epochs between LR decay.
        gamma: Multiplicative factor for LR decay.
    
    Examples:
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    """
    
    def __init__(self, optimizer: Any, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing LR schedule.
    
    Args:
        optimizer: Optimizer to schedule.
        T_max: Maximum number of epochs (half cycle).
        eta_min: Minimum learning rate.
    
    Examples:
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    """
    
    def __init__(self, optimizer: Any, T_max: int, eta_min: float = 0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self) -> float:
        return (
            self.eta_min
            + (self.base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
        )


class ExponentialLR(LRScheduler):
    """Decays LR by gamma every epoch.
    
    Args:
        optimizer: Optimizer to schedule.
        gamma: Multiplicative factor for LR decay.
    
    Examples:
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
    """
    
    def __init__(self, optimizer: Any, gamma: float):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** self.last_epoch


class ReduceLROnPlateau(LRScheduler):
    """Reduce LR when metric has stopped improving.

    Args:
        optimizer: Optimizer to schedule.
        mode: 'min' or 'max'.
        factor: Factor to multiply LR by.
        patience: Number of epochs with no improvement.
        min_lr: Minimum learning rate.

    Examples:
        >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        >>> for epoch in range(100):
        ...     val_loss = validate(...)
        ...     scheduler.step(val_loss)
    """

    def __init__(self, optimizer: Any, mode: str = "min", factor: float = 0.1, patience: int = 10, min_lr: float = 1e-6):
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.current_lr = self.base_lr

    def get_lr(self) -> float:
        """Not used for ReduceLROnPlateau. Required by base class."""
        return self.current_lr

    def step(self, metric: float) -> float:
        """Update learning rate based on metric value."""
        improved = (metric < self.best) if self.mode == "min" else (metric > self.best)
        if improved:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                new_lr = self.current_lr * self.factor
                if new_lr < self.min_lr:
                    new_lr = self.min_lr
                self._set_lr(new_lr)
                self.current_lr = new_lr
                return new_lr
        return self.current_lr
