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
    
    Modifies the optimizer's learning rate in-place.
    
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
        # Access param_groups directly (PyTorch-style) for efficiency
        if hasattr(self.optimizer, 'param_groups') and self.optimizer.param_groups:
            return self.optimizer.param_groups[0].get("lr", 0.01)
        # Fall back to state_dict
        sd = self.optimizer.state_dict()
        return sd.get("lr", 0.01)
    
    def _set_lr(self, lr: float) -> None:
        # Access param_groups directly (PyTorch-style) for efficiency
        if hasattr(self.optimizer, 'param_groups'):
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        else:
            # Fall back to state_dict
            sd = self.optimizer.state_dict()
            sd["lr"] = lr
            self.optimizer.load_state_dict(sd)
    
    def get_lr(self) -> float:
        raise NotImplementedError
    
    def step(self, **kwargs) -> float:
        """Update learning rate and return the new learning rate."""
        self.last_epoch += 1
        lr = self.get_lr()
        self._set_lr(lr)
        return lr


class DecayLR(LRScheduler):
    """Base class for decay-based LR schedulers that use multiplicative decay.
    
    Args:
        optimizer: Optimizer to schedule.
        gamma: Multiplicative factor for LR decay.
    
    Examples:
        >>> scheduler = DecayLR(optimizer, gamma=0.1)
    """
    
    def __init__(self, optimizer: Any, gamma: float = 0.1):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def get_decay_factor(self) -> float:
        """Calculate the decay factor. Subclasses should override this."""
        raise NotImplementedError
    
    def get_lr(self) -> float:
        return self.base_lr * self.get_decay_factor()


class StepLR(DecayLR):
    """Decays LR by gamma every step_size epochs.
    
    Args:
        optimizer: Optimizer to schedule.
        step_size: Number of epochs between LR decay.
        gamma: Multiplicative factor for LR decay.
    
    Examples:
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    """
    
    def __init__(self, optimizer: Any, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer, gamma)
        self.step_size = step_size
    
    def get_decay_factor(self) -> float:
        return self.gamma ** (self.last_epoch // self.step_size)


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


class ExponentialLR(DecayLR):
    """Decays LR by gamma every epoch.
    
    Args:
        optimizer: Optimizer to schedule.
        gamma: Multiplicative factor for LR decay.
    
    Examples:
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
    """
    
    def __init__(self, optimizer: Any, gamma: float):
        super().__init__(optimizer, gamma)
    
    def get_decay_factor(self) -> float:
        return self.gamma ** self.last_epoch


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

    def step(self, metric: float, **kwargs) -> float:
        """Update learning rate based on metric value."""
        improved = (metric < self.best) if self.mode == "min" else (metric > self.best)
        if improved:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                current_lr = self._get_lr()
                new_lr = current_lr * self.factor
                if new_lr < self.min_lr:
                    new_lr = self.min_lr
                self._set_lr(new_lr)
                return new_lr
        return self._get_lr()
