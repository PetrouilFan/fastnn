"""Module protocol for fastnn layers and models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


__all__ = ['Module']


class Module(ABC):
    """Abstract base class for all neural network modules.
    
    All layers and models should inherit from this class to provide
    a consistent interface for parameters, training/eval modes, etc.
    """
    
    __slots__ = ('_training',)
    
    def __init__(self) -> None:
        self._training = True
    
    @abstractmethod
    def parameters(self) -> List[Any]:
        """Return list of parameters."""
        pass
    
    @abstractmethod
    def named_parameters(self) -> List[Tuple[str, Any]]:
        """Return list of (name, parameter) tuples."""
        pass
    
    def train_mode(self) -> None:
        """Set module to training mode (aligned with Rust Module trait)."""
        self._training = True
    
    def eval_mode(self) -> None:
        """Set module to evaluation mode (aligned with Rust Module trait)."""
        self._training = False
    
    def is_training(self) -> bool:
        """Return True if module is in training mode."""
        return self._training
    
    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        pass
    
    def to_gpu(self, device_id: int) -> None:
        """Move module to GPU."""
        pass
    
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary of module."""
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary into module."""
        pass
