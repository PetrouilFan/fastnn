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
    
    def train_mode(self):
        """Set wrapped model to training mode."""
        model = self._get_wrapped_model()
        if hasattr(model, 'train_mode'):
            return model.train_mode()
        # Backward compatibility
        return model.train()
    
    def eval_mode(self):
        """Set wrapped model to evaluation mode."""
        model = self._get_wrapped_model()
        if hasattr(model, 'eval_mode'):
            return model.eval_mode()
        # Backward compatibility
        return model.eval()
    
    # Backward compatibility
    train = train_mode
    eval = eval_mode
    
    def zero_grad(self):
        """Zero gradients of the wrapped model."""
        return self._get_wrapped_model().zero_grad()
    
    def __call__(self, x):
        """Forward pass through the wrapped model."""
        return self._get_wrapped_model()(x)
    
    def forward(self, x):
        """Forward pass through the wrapped model."""
        return self._get_wrapped_model()(x)
