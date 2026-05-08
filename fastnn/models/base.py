from fastnn._model_wrapper import ModuleWrapperMixin



__all__ = ["BaseModel"]


class BaseModel(ModuleWrapperMixin):
    def save(self, path: str):
        """Save the model using fastnn's serialization."""
        import fastnn as fnn
        fnn.save_model(self._model if hasattr(self, '_model') else self, path)

    @classmethod
    def load(cls, path: str):
        """Load a model from fastnn format."""
        import fastnn as fnn
        state = fnn.load_model(path)
        obj = cls.__new__(cls)
        obj.__init__()
        obj.load_state_dict(state)
        return obj
