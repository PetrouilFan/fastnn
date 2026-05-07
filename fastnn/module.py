"""Module protocol for fastnn layers and models."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class Module(ABC):
    """Abstract base class for all neural network modules.
    
    All layers and models should inherit from this class to provide
    a consistent interface for parameters, training/eval modes, etc.
    """
    
    def __init__(self):
        self._training = True
    
    @abstractmethod
    def parameters(self) -> List:
        """Return list of parameters."""
        pass
    
    def named_parameters(self) -> List[Tuple[str, any]]:
        """Return list of (name, parameter) tuples."""
        return []
    
    def train_mode(self) -> None:
        """Set module to training mode (aligned with Rust Module trait)."""
        self._training = True
    
    def eval_mode(self) -> None:
        """Set module to evaluation mode (aligned with Rust Module trait)."""
        self._training = False
    
    def is_training(self) -> bool:
        """Return True if module is in training mode."""
        return self._training
    
    # Backward compatibility aliases
    train = train_mode
    eval = eval_mode
    
    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        pass
    
    def to_gpu(self, device_id: int) -> None:
        """Move module to GPU."""
        pass
