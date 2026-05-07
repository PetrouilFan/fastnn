"""Mixin for classes that wrap a fastnn model.

Provides delegation methods for common model operations.
"""


class ModuleWrapperMixin:
    """Mixin for classes that wrap a fastnn model.
    
    The wrapped model should be accessible via `self.model` or `self._model`.
    Subclasses can override `_get_wrapped_model()` for custom behavior.
    """
    
    def _get_wrapped_model(self):
        """Get the wrapped model. Override if using a different attribute name."""
        if hasattr(self, '_model') and self._model is not None:
            return self._model
        if hasattr(self, 'model') and self.model is not None:
            return self.model
        raise AttributeError("No wrapped model found. Set self._model or self.model")
    
    def parameters(self):
        """Return parameters of the wrapped model."""
        return self._get_wrapped_model().parameters()
    
    def train(self):
        """Set wrapped model to training mode."""
        return self._get_wrapped_model().train()
    
    def eval(self):
        """Set wrapped model to evaluation mode."""
        return self._get_wrapped_model().eval()
    
    def zero_grad(self):
        """Zero gradients of the wrapped model."""
        return self._get_wrapped_model().zero_grad()
    
    def __call__(self, x):
        """Forward pass through the wrapped model."""
        return self._get_wrapped_model()(x)
    
    def forward(self, x):
        """Forward pass through the wrapped model."""
        return self._get_wrapped_model()(x)
