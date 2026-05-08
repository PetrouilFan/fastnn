from fastnn._model_wrapper import ModuleWrapperMixin
import pickle


__all__ = ["BaseModel"]


class BaseModel(ModuleWrapperMixin):
    def save(self, path: str):
        """Save the model to a file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        """Load a model from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
