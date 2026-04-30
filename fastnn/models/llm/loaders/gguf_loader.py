"""GGUF loader for fastnn LLM models.

Loads GGUF quantized models, dequantizes weights to f32,
and creates fastnn tensors.
    
Supports: Q4_K, Q5_K, Q6_K, F32, F16, BF16
"""

import warnings
from typing import Dict, Optional, Tuple, Any
import numpy as np

import fastnn._core as _core
import fastnn
from fastnn.utils import tensor_from_array


class GGUFLoader:
    """Load GGUF models and convert weights to fastnn tensors."""

    # Tensor types we can dequantize
    SUPPORTED_QTYPES = {
        "Q4_K", "Q5_K", "Q6_K", "F32", "F16", "BF16", "Q5_0", "Q8_0", "Q4_0", "Q4_1"
    }

    def __init__(self, gguf_path: str):
        self.gguf_path = gguf_path
        self._reader = None
        self._metadata = None
        self._weights_cache = {}

    @property
    def reader(self):
        """Lazy-load GGUF reader."""
        if self._reader is None:
            try:
                from gguf import GGUFReader
            except ImportError:
                raise ImportError("gguf package required. Install with: uv pip install gguf")
            self._reader = GGUFReader(self.gguf_path, mode='r')
        return self._reader

    # ── Metadata ──────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        """Extract all GGUF metadata as a Python dict."""
        if self._metadata is not None:
            return self._metadata

        meta = {}
        for key, field in self.reader.fields.items():
            val = field.contents()
            meta[key] = val
        self._metadata = meta
        return meta

    def extract_config(self) -> Dict[str, Any]:
        """Extract model config dict from GGUF metadata.

        Maps GGUF keys to fastnn-friendly config keys.
        """
        meta = self.get_metadata()

        # Architecture name in metadata
        arch = str(meta.get("general.architecture", "gemma4"))

        def _get(keys):
            for k in keys:
                if k in meta:
                    return meta[k]
            return None

        config = {
            "model_type": arch,
            "vocab_size": _get([
                f"{arch}.vocab_size",
                "tokenizer.ggml.tokens"
            ]),
            "hidden_size": _get([f"{arch}.embedding_length"]),
            "num_hidden_layers": _get([f"{arch}.block_count"]),
            "num_attention_heads": _get([f"{arch}.attention.head_count"]),
            "num_key_value_heads": _get([f"{arch}.attention.head_count_kv"]),
            "head_dim": _get([f"{arch}.attention.key_length"]),
            "rms_norm_eps": _get([f"{arch}.attention.layer_norm_rms_epsilon"]),
            "max_position_embeddings": _get([f"{arch}.context_length"]),
            "rope_theta": _get([f"{arch}.rope.freq_base"]),
            "bos_token_id": _get(["tokenizer.ggml.bos_token_id"]),
            "eos_token_id": _get(["tokenizer.ggml.eos_token_id"]),
        }

        # Gemma 4 specific
        config["sliding_window"] = _get([f"{arch}.attention.sliding_window"])
        config["sliding_window_pattern"] = _get([f"{arch}.attention.sliding_window_pattern"])
        config["rope_theta_swa"] = _get([f"{arch}.rope.freq_base_swa"])
        config["head_dim_swa"] = _get([f"{arch}.attention.key_length_swa"])
        config["shared_kv_layers"] = _get([f"{arch}.attention.shared_kv_layers"])
        config["embedding_length_per_layer"] = _get([f"{arch}.embedding_length_per_layer_input"])
        config["final_logit_softcapping"] = _get([f"{arch}.final_logit_softcapping"])
        config["feed_forward_lengths"] = _get([f"{arch}.feed_forward_length"])

        # Per-layer MLP widths
        if config.get("feed_forward_lengths"):
            config["intermediate_sizes"] = config["feed_forward_lengths"]
        else:
            config["intermediate_size"] = _get([f"{arch}.feed_forward_length"])

        # Filter None values
        config = {k: v for k, v in config.items() if v is not None}

        # Convert lists to proper types
        if "vocab_size" in config and isinstance(config["vocab_size"], list):
            config["vocab_size"] = len(config["vocab_size"])
        config.setdefault("vocab_size", 262144)

        return config

    # ── Weight loading ──────────────────────────────────────

    def _np_to_tensor(self, np_arr: np.ndarray) -> Any:
        """Convert numpy array to fastnn tensor."""
        return tensor_from_array(np_arr)

    # ── Weight loading ──────────────────────────────────────

    def load_tensor(self, name: str) -> Any:
        """Load a single tensor by name, returning a fastnn Tensor."""
        if name in self._weights_cache:
            return self._weights_cache[name]

        for tensor in self.reader.tensors:
            if tensor.name == name:
                np_arr = self._dequantize_tensor(tensor)
                fnn_tensor = self._np_to_tensor(np_arr)
                self._weights_cache[name] = fnn_tensor
                return fnn_tensor

        raise KeyError(f"Tensor '{name}' not found in GGUF")

    def load_all_weights(self, skip_unsupported: bool = True) -> Dict[str, Any]:
        """Load all weights, skipping unsupported quantization types.

        Returns: {name: fastnn_tensor}
        """
        weights = {}
        for tensor in self.reader.tensors:
            qt_name = tensor.tensor_type.name
            if qt_name not in self.SUPPORTED_QTYPES:
                if not skip_unsupported:
                    raise ValueError(
                        f"Unsupported quant type '{qt_name}' for tensor '{tensor.name}'"
                    )
                print(f"  Skipping {tensor.name}: unsupported quant type {qt_name}")
                continue

            try:
                np_arr = self._dequantize_tensor(tensor)
                weights[tensor.name] = self._np_to_tensor(np_arr)
                print(f"  Loaded {tensor.name}: {list(tensor.shape)} ({qt_name})")
            except Exception as e:
                print(f"  Warning: Failed to load {tensor.name}: {e}")
                if not skip_unsupported:
                    raise

        return weights

    def _dequantize_tensor(self, tensor) -> np.ndarray:
        """Dequantize a GGUF tensor to f32 numpy array."""
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize

        t = tensor.tensor_type

        # Already f32
        if t == GGMLQuantizationType.F32:
            data = np.array(tensor.data, dtype=np.float32)
            return data.reshape(tensor.shape)

        # F16 / BF16
        if t == GGMLQuantizationType.F16:
            arr = np.frombuffer(tensor.data.tobytes(), dtype=np.float16)
            return arr.astype(np.float32).reshape(tensor.shape)

        if t == GGMLQuantizationType.BF16:
            arr = np.frombuffer(tensor.data.tobytes(), dtype=np.uint16)
            arr = arr.reshape(tensor.shape)
            # BF16 -> f32: shift left by 16
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
            return arr

        # Quantized: use gguf dequantize
        dequant = dequantize(tensor.data, t)
        return np.array(dequant, dtype=np.float32).reshape(tensor.shape)


    # ── Tensor name helpers ─────────────────────────────────

    def layer_exists(self, layer_idx: int) -> bool:
        """Check if a layer exists in the model."""
        prefix = f"blk.{layer_idx}."
        return any(t.name.startswith(prefix) for t in self.reader.tensors)

    def get_layer_weights(self, layer_idx: int) -> Dict[str, Any]:
        """Get all weights for a specific layer."""
        prefix = f"blk.{layer_idx}."
        weights = {}
        for tensor in self.reader.tensors:
            if tensor.name.startswith(prefix):
                qt = tensor.tensor_type.name
                if qt not in self.SUPPORTED_QTYPES:
                    continue
                np_arr = self._dequantize_tensor(tensor)
                shape = list(map(int, tensor.shape))
                weights[tensor.name] = tensor_from_array(np_arr, shape)
        return weights

    # ── Context manager ────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Reader uses mmap, will be GC'd
        pass


# ── Convenience function ──────────────────────────────────

def load_gguf(gguf_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a GGUF model and return (weights, metadata).

    Args:
        gguf_path: Path to .gguf file

    Returns:
        (weights_dict, config_dict)
    """
    with GGUFLoader(gguf_path) as loader:
        config = loader.extract_config()
        weights = loader.load_all_weights()
        return weights, config
