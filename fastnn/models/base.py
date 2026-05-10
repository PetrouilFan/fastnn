from fastnn._model_wrapper import ModuleWrapperMixin



__all__ = ["BaseModel"]


class BaseModel(ModuleWrapperMixin):
    def save(self, path: str):
        from fastnn.io import save
        model = self._model if hasattr(self, '_model') else self
        save(model, path)

    @staticmethod
    def load(path: str):
        from fastnn.io import load
        return load(path)
