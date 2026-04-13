import struct
import numpy as np
import mmap

HAS_SAFETENSORS = True


MAGIC = b"FNN\x00"
VERSION = 1


def load_safetensors(path: str):
    """Load weights from a HuggingFace safetensors file.

    Args:
        path: Path to .safetensors file or model directory

    Returns:
        Dictionary mapping parameter names to fastnn tensors
    """
    # Check if path is a directory (sharded model)
    import os

    if os.path.isdir(path):
        return load_hf_weights(path)
    return _load_safetensors_mmap(path)


def load_hf_weights(model_path: str):
    """Load HuggingFace weights, supporting sharded models.

    Handles:
    - Single model.safetensors file
    - Sharded models with model.safetensors.index.json

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary mapping parameter names to fastnn tensors
    """
    import os
    import json

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    state_dict = {}

    if os.path.exists(index_path):
        # Sharded model - read index
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]

        # Load each unique shard file
        loaded_files = set()
        for f_name in set(weight_map.values()):
            f_path = os.path.join(model_path, f_name)
            if os.path.exists(f_path) and f_name not in loaded_files:
                shard_dict = _load_safetensors_mmap(f_path)
                state_dict.update(shard_dict)
                loaded_files.add(f_name)
    else:
        # Single file - find any .safetensors
        import glob

        files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors found in {model_path}")
        for f_path in files:
            shard_dict = _load_safetensors_mmap(f_path)
            state_dict.update(shard_dict)

    return state_dict


def _load_safetensors_mmap(path: str):
    """Load safetensors using mmap for zero-copy disk access.

    Uses memory mapping to avoid loading entire file into RAM.
    Only copies data when converting to fastnn tensors.

    Args:
        path: Path to .safetensors file

    Returns:
        Dictionary mapping parameter names to fastnn tensors
    """
    import fastnn._core as _core

    with open(path, "rb") as f:
        # Memory map the file (read-only)
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            # Read header size
            header_size = struct.unpack("<Q", mapped[:8])[0]

            # Read header JSON
            header_json = mapped[8 : 8 + header_size].decode("utf-8")
            import json

            header = json.loads(header_json)

            result = {}
            for tensor_name, tensor_meta in header.items():
                if tensor_name == "__metadata__":
                    continue

                data_offsets = tensor_meta.get("data_offsets", [])
                if not data_offsets:
                    continue

                offset, size = data_offsets[0], data_offsets[1] - data_offsets[0]
                dtype_str = tensor_meta.get("dtype", "F32")
                shape = tensor_meta.get("shape", [])

                # Determine raw dtype for reading bytes
                if dtype_str == "BF16":
                    raw_dtype = np.uint16
                elif dtype_str == "F16":
                    raw_dtype = np.float16
                else:
                    raw_dtype = _dtype_to_numpy(dtype_str)

                tensor_start = 8 + header_size + offset
                tensor_end = tensor_start + size

                # Create numpy view directly from mmap (zero-copy)
                num_elements = size // np.dtype(raw_dtype).itemsize
                np_data = np.frombuffer(
                    mapped[tensor_start:tensor_end], dtype=raw_dtype
                ).copy()  # Need to copy because mmap can be invalid after close

                np_data = np_data.reshape(shape)

                # Convert bf16 to float32
                if dtype_str == "BF16":
                    np_data = _bf16_to_f32(np_data)

                ftensor = _core.tensor_from_array(np_data)
                result[tensor_name] = ftensor

    return result


def _bf16_to_f32(arr):
    """Convert bf16 (uint16) array to float32 array properly."""
    if arr.dtype != np.uint16:
        return arr.astype(np.float32)

    # Use numpy for vectorized conversion
    arr = np.ascontiguousarray(arr)
    bits = arr.astype(np.uint32)

    # Extract bf16 components
    sign = (bits >> 15) & 1
    exp = (bits >> 7) & 0xFF
    mant = bits & 0x7F

    # Initialize result as uint32
    result = np.zeros_like(bits, dtype=np.uint32)

    # Normal values: exp 1-254
    mask = (exp >= 1) & (exp <= 254)
    result[mask] = (
        (sign[mask] << 31) | ((exp[mask] + 127 - 127) << 23) | (mant[mask] << 16)
    )

    # Zeros: exp=0, mant=0
    mask = (exp == 0) & (mant == 0)
    result[mask] = sign[mask] << 31

    # Infs: exp=255, mant=0
    mask = (exp == 255) & (mant == 0)
    result[mask] = (sign[mask] << 31) | (255 << 23)

    # NaN: exp=255, mant>0
    mask = (exp == 255) & (mant > 0)
    result[mask] = (sign[mask] << 31) | (255 << 23) | (1 << 22)

    # View as float32
    return result.view(np.float32)


def _dtype_to_numpy(dtype_str: str):
    """Convert safetensors dtype string to numpy dtype."""
    dtype_map = {
        "F32": np.float32,
        "F64": np.float64,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U64": np.uint64,
        "U32": np.uint32,
        "U16": np.uint16,
        "U8": np.uint8,
        "BOOL": np.bool_,
        "F16": np.float16,
    }
    return dtype_map.get(dtype_str, np.float32)


def state_dict_to_fastnn(state_dict: dict):
    """Convert a state dict (dict of numpy arrays) to fastnn tensors.

    Args:
        state_dict: Dictionary mapping names to numpy arrays

    Returns:
        Dictionary mapping names to fastnn tensors
    """
    import fastnn._core as _core

    result = {}
    for key, np_data in state_dict.items():
        # Ensure float32
        if np_data.dtype == np.float16:
            np_data = np_data.astype(np.float32)
        elif np_data.dtype == np.bool_:
            np_data = np_data.astype(np.float32)

        shape = list(np_data.shape)
        ftensor = _core.tensor_from_data(np_data.flatten().tolist(), shape)
        result[key] = ftensor

    return result


def flatten_state_dict(state_dict: dict):
    """Flatten nested state dict into dot-separated keys.

    HuggingFace models often have nested dicts like:
    {'model.embed_tokens.weight': ...} stays as is
    {'model.layers.0.attn.q_proj.weight': ...}

    Args:
        state_dict: Dictionary to flatten

    Returns:
        Flattened dictionary
    """
    result = {}

    def _flatten(d, prefix=""):
        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _flatten(value, new_key)
            else:
                result[new_key] = value

    _flatten(state_dict)
    return result


def load_model(model, path: str):
    """Load a fastnn model from file.

    Args:
        model: fastnn model object
        path: Path to .fnn file

    Returns:
        Model with loaded weights
    """
    result = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("Invalid file format: expected FNN magic bytes")
        version = struct.unpack("<I", f.read(4))[0]
        if version > VERSION:
            raise ValueError(f"Unsupported format version: {version}")

        num_params = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_params):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            shape_len = struct.unpack("<Q", f.read(8))[0]
            shape = [struct.unpack("<q", f.read(8))[0] for _ in range(shape_len)]
            data_len = struct.unpack("<Q", f.read(8))[0]
            data = np.frombuffer(f.read(data_len * 4), dtype=np.float32)
            import fastnn._core as _core

            result[name] = _core.tensor_from_data(data.copy().tolist(), list(shape))

    # Load into model
    from fastnn import load_state_dict

    load_state_dict(model, result)
    return model


def save_model(model, path: str):
    """Save a fastnn model to file.

    Args:
        model: fastnn model object
        path: Path to output .fnn file
    """
    named = model.named_parameters() if hasattr(model, "named_parameters") else []
    params = model.parameters() if hasattr(model, "parameters") else []

    if named:
        param_list = named
    else:
        param_list = [(f"param_{i}", p) for i, p in enumerate(params)]

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(param_list)))

        for name, tensor in param_list:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)

            shape = tensor.shape
            f.write(struct.pack("<Q", len(shape)))
            for d in shape:
                f.write(struct.pack("<q", d))

            data = tensor.numpy().astype(np.float32).flatten()
            f.write(struct.pack("<Q", len(data)))
            f.write(data.tobytes())
