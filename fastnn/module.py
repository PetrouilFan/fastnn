"""Module protocol for fastnn layers and models."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class Module(ABC):
    """Abstract base class for all neural network modules.
    
    All layers and models should inherit from this class to provide
    a consistent interface for parameters, training/eval modes, etc.
    """
    
    @abstractmethod
    def parameters(self) -> List:
        """Return list of parameters."""
        pass
    
    def named_parameters(self) -> List[Tuple[str, any]]:
        """Return list of (name, parameter) tuples."""
        return []
    
    def train(self) -> None:
        """Set module to training mode."""
        pass
    
    def eval(self) -> None:
        """Set module to evaluation mode."""
        pass
    
    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        pass
    
    def to_gpu(self, device_id: int) -> None:
        """Move module to GPU."""
        pass
