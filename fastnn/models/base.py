import json
import struct
from pathlib import Path

from fastnn._model_wrapper import ModuleWrapperMixin
from fastnn.io import MODEL_MAGIC, MODEL_VERSION, write_tensor, read_tensor


__all__ = ["BaseModel"]


def _write_metadata(f, model_cls: str, model_config: dict) -> None:
    """Write model metadata (class name and config) to file."""
    metadata = {"model_cls": model_cls, "model_config": model_config}
    metadata_json = json.dumps(metadata)
    metadata_bytes = metadata_json.encode("utf-8")
    f.write(struct.pack("Q", len(metadata_bytes)))
    f.write(metadata_bytes)


def _read_metadata(f) -> tuple:
    """Read model metadata from file. Returns (model_cls, model_config)."""
    metadata_len = struct.unpack("Q", f.read(8))[0]
    metadata_bytes = f.read(metadata_len)
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    return metadata.get("model_cls"), metadata.get("model_config", {})


class BaseModel(ModuleWrapperMixin):
    def save(self, path: str):
        """Save model to file with metadata (class name and config)."""
        model = self._model if hasattr(self, '_model') else self

        params = model.named_parameters() if hasattr(model, "named_parameters") else []
        if not params:
            params = [(f"param_{i}", p) for i, p in enumerate(model.parameters())]

        model_cls = type(self).__name__
        model_config = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }

        with open(path, "wb") as f:
            f.write(MODEL_MAGIC)
            f.write(struct.pack("I", MODEL_VERSION))
            _write_metadata(f, model_cls, model_config)
            f.write(struct.pack("Q", len(params)))
            for name, tensor in params:
                write_tensor(f, name, tensor.inner.numpy())

    @staticmethod
    def load(path: str):
        """Load model from file, reconstructing from saved metadata."""
        from fastnn.io import serialization_error

        with serialization_error("load model"):
            with open(path, "rb") as f:
                magic = f.read(4)
                if magic != MODEL_MAGIC:
                    raise ValueError("Invalid file format: expected FNN magic bytes")

                version = struct.unpack("I", f.read(4))[0]
                if version != MODEL_VERSION:
                    raise ValueError(f"Unsupported file version: {version}")

                model_cls, model_config = _read_metadata(f)

                if not model_cls:
                    raise ValueError(
                        "Saved file does not contain model class information. "
                        "The file appears to contain only a state_dict without metadata. "
                        "Use fastnn.io.load() to load raw state_dicts, or ensure "
                        "the model was saved using model.save() to include metadata."
                    )

                num_params = struct.unpack("Q", f.read(8))[0]
                state_dict = {}
                for _ in range(num_params):
                    name, data = read_tensor(f)
                    from fastnn import tensor
                    state_dict[name] = tensor(data, list(data.shape))

        import fastnn.models
        model_class = getattr(fastnn.models, model_cls, None)
        if model_class is None:
            raise ValueError(f"Unknown model class: {model_cls}. Available models in fastnn.models.")

        model = model_class(**model_config)

        if hasattr(model, '_model'):
            target = model._model
        else:
            target = model

        for name, param in state_dict.items():
            parts = name.split('.')
            obj = target
            for part in parts[:-1]:
                obj = getattr(obj, part)
            if hasattr(obj, 'set_weight') and parts[-1] == 'weight':
                obj.set_weight(param)
            elif hasattr(obj, 'set_bias') and parts[-1] == 'bias':
                obj.set_bias(param)
            else:
                setattr(obj, parts[-1], param)

        return model
