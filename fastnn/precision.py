"""Precision type system for fastnn.

Defines the Precision enum, Quantizer strategy classes, and PrecisionConfig
for per-layer precision assignment in the modular quantization system.
"""

from __future__ import annotations

import numpy as np
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union


class Precision(IntEnum):
    """Supported weight storage precisions.

    The integer value matches the dtype tag used in .fnn v3 serialization.
    """
    F32 = 0  # Full float32, no quantization
    F16 = 1  # PackedTensor<F16x2> (2 × f16 per u32 word)
    U8 = 2   # PackedTensor<U8x4>  (4 × i8 per u32 word)
    U4 = 3   # PackedTensor<U4x8>  (8 × i4 per u32 word)

    @property
    def bit_width(self) -> int:
        return {Precision.F32: 32, Precision.F16: 16,
                Precision.U8: 8, Precision.U4: 4}[self]

    @property
    def is_float(self) -> bool:
        return self in (Precision.F32, Precision.F16)

    @property
    def is_quantized(self) -> bool:
        return self in (Precision.U8, Precision.U4)

    @staticmethod
    def from_dtype_tag(tag: int) -> "Precision":
        for p in Precision:
            if p.value == tag:
                return p
        raise ValueError(f"Unknown dtype tag: {tag}")

    @staticmethod
    def from_string(s: str) -> "Precision":
        mapping = {
            "f32": Precision.F32, "float32": Precision.F32,
            "f16": Precision.F16, "float16": Precision.F16,
            "u8": Precision.U8, "uint8": Precision.U8,
            "u4": Precision.U4, "uint4": Precision.U4,
        }
        s = s.lower().replace("-", "").replace("_", "")
        if s not in mapping:
            raise ValueError(f"Unknown precision string: {s!r}. "
                             f"Options: {list(mapping.keys())}")
        return mapping[s]


class QuantizationScheme(str):
    """Supported quantization schemes."""
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_BLOCK = "per_block"


class Quantizer:
    """Strategy for quantizing f32 weights to a target precision.

    Args:
        precision: Target precision (U4, U8, F16, F32).
        scheme: Quantization scheme ("per_tensor", "per_channel", "per_block").
        block_size: Block size for per-block quantization (only used when
            scheme is "per_block").
    """

    def __init__(
        self,
        precision: Union[Precision, str],
        scheme: str = "per_channel",
        block_size: Optional[int] = None,
        asymmetric: bool = False,
    ):
        if isinstance(precision, str):
            precision = Precision.from_string(precision)
        self.precision = precision
        self.scheme = scheme
        self.block_size = block_size
        self.asymmetric = asymmetric

        if self.scheme not in (QuantizationScheme.PER_TENSOR,
                                QuantizationScheme.PER_CHANNEL,
                                QuantizationScheme.PER_BLOCK):
            raise ValueError(f"Unknown scheme: {self.scheme}")

    def should_quantize(self) -> bool:
        """Return True if this quantizer actually quantizes (not F32)."""
        return self.precision.is_quantized

    def quantize(
        self, f32_weights: np.ndarray
    ) -> Tuple[np.ndarray, List[float], List[float]]:
        """Quantize f32 weights to the target precision.

        Returns:
            Tuple of (scales: list[float], zeros: list[float]).
            The actual packed representation is handled by the Rust
            PackedTensor bindings (via `from_f32_per_channel` /
            `from_f32_auto`).
        """
        if self.precision == Precision.F32:
            return np.array([1.0]), np.array([0.0])

        if self.scheme == QuantizationScheme.PER_TENSOR:
            scale = self._compute_scale(f32_weights)
            return np.array([scale]), np.array([0.0])

        elif self.scheme == QuantizationScheme.PER_CHANNEL:
            if self.asymmetric:
                return self.quantize_asymmetric(f32_weights)
            assert f32_weights.ndim >= 2, (
                f"Per-channel requires 2D+ weights, got shape {f32_weights.shape}"
            )
            m = f32_weights.shape[0]
            scales = np.zeros(m, dtype=np.float64)
            for i in range(m):
                row = f32_weights[i].ravel()
                scales[i] = self._compute_scale(row)
            return scales, np.zeros(m, dtype=np.float64)

        elif self.scheme == QuantizationScheme.PER_BLOCK:
            return self._quantize_per_block(f32_weights)

        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")

    def _compute_scale(self, data: np.ndarray) -> float:
        """Compute quantization scale using max-abs."""
        max_abs = float(np.abs(data).max())
        if max_abs == 0.0:
            return 1.0
        max_val = float((1 << (self.precision.bit_width - 1)) - 1)
        return max_abs / max_val

    def _quantize_per_block(
        self, f32_weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Block-wise quantization with per-block scales."""
        block_size = self.block_size or 64
        flat = f32_weights.ravel()
        n = len(flat)
        n_blocks = (n + block_size - 1) // block_size
        scales = np.zeros(n_blocks, dtype=np.float64)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            scales[b] = self._compute_scale(flat[start:end])
        return scales, np.zeros(n_blocks, dtype=np.float64)

    def quantize_asymmetric(
        self, f32_weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Asymmetric min-max quantization per channel.

        Computes scale and zero_point from per-channel min/max
        instead of max-abs.

        Returns:
            Tuple of (scales, zeros) as ndarrays.
        """
        if self.precision == Precision.F32:
            return np.array([1.0]), np.array([0.0])

        assert f32_weights.ndim >= 2, "Asymmetric quantization requires 2D+ weights"
        m = f32_weights.shape[0]
        unsigned_max = float((1 << self.precision.bit_width) - 1)
        signed_bias = float(1 << (self.precision.bit_width - 1))

        scales = np.zeros(m, dtype=np.float64)
        zeros = np.zeros(m, dtype=np.float64)

        for i in range(m):
            row = f32_weights[i].ravel()
            row_min, row_max = row.min(), row.max()
            rng = row_max - row_min
            if rng == 0:
                scales[i] = 1.0
                zeros[i] = 0.0
            else:
                scales[i] = rng / unsigned_max
                zeros[i] = row_min + signed_bias * scales[i]

        return scales, zeros

    def dequantize(
        self,
        data: np.ndarray,
        scales: Union[List[float], np.ndarray],
        zeros: Union[List[float], np.ndarray],
        shape: List[int],
    ) -> np.ndarray:
        """Dequantize packed data back to f32.

        For quantized types, dequantizes as: f32 = q * scale + zero.
        For float types (F16, F32), this is a no-op.
        """
        if not self.should_quantize():
            return data.reshape(shape)

        scales = np.asarray(scales, dtype=np.float64)
        zeros = np.asarray(zeros, dtype=np.float64)
        result = data.astype(np.float64)

        if len(scales) == 1:
            result = result * scales[0] + zeros[0]
        else:
            m = shape[0]
            inner_stride = int(np.prod(shape[1:])) if len(shape) > 1 else result.size // m
            for i in range(m):
                start = i * inner_stride
                end = start + inner_stride
                s = scales[i] if i < len(scales) else scales[0]
                z = zeros[i] if i < len(zeros) else zeros[0]
                result[start:end] = result[start:end] * s + z

        return result.reshape(shape).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "precision": self.precision.name,
            "scheme": self.scheme,
            "block_size": self.block_size,
        }

    @staticmethod
    def from_dict(d: dict) -> "Quantizer":
        return Quantizer(
            precision=Precision[d["precision"]],
            scheme=d.get("scheme", "per_channel"),
            block_size=d.get("block_size"),
        )

    def __repr__(self) -> str:
        return (f"Quantizer({self.precision.name}, "
                f"scheme={self.scheme})")


class PrecisionConfig:
    """Per-layer precision assignment configuration.

    Controls which precision to use for each parameter in a model.
    Supports exact name matching and prefix matching.

    Args:
        default: Default quantizer for all parameters.
        overrides: Exact param name -> Quantizer mapping.
        prefixes: Param name prefix -> Quantizer mapping.
    """

    def __init__(
        self,
        default: Optional[Quantizer] = None,
        overrides: Optional[Dict[str, Quantizer]] = None,
        prefixes: Optional[Dict[str, Quantizer]] = None,
    ):
        self.default = default or Quantizer(Precision.F32)
        self.overrides = overrides or {}
        self.prefixes = prefixes or {}

    def get_quantizer(self, param_name: str) -> Quantizer:
        """Get the quantizer for a given parameter name."""
        if param_name in self.overrides:
            return self.overrides[param_name]
        for prefix, q in self.prefixes.items():
            if param_name.startswith(prefix):
                return q
        return self.default

    def should_quantize_param(self, param_name: str) -> bool:
        """Check if a parameter should be quantized."""
        return self.get_quantizer(param_name).should_quantize()

    def to_dict(self) -> dict:
        return {
            "default": self.default.to_dict(),
            "overrides": {k: v.to_dict() for k, v in self.overrides.items()},
            "prefixes": {k: v.to_dict() for k, v in self.prefixes.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "PrecisionConfig":
        return PrecisionConfig(
            default=Quantizer.from_dict(d["default"]),
            overrides={k: Quantizer.from_dict(v) for k, v in d.get("overrides", {}).items()},
            prefixes={k: Quantizer.from_dict(v) for k, v in d.get("prefixes", {}).items()},
        )

    @staticmethod
    def f32() -> "PrecisionConfig":
        """All parameters in full f32."""
        return PrecisionConfig(default=Quantizer(Precision.F32))

    @staticmethod
    def uniform(dtype: Union[Precision, str]) -> "PrecisionConfig":
        """Uniform precision for all parameters."""
        return PrecisionConfig(default=Quantizer(dtype))

    @staticmethod
    def from_dict_spec(spec: Dict[str, Union[str, Dict]]) -> "PrecisionConfig":
        """Build config from a user-friendly dict spec.

        Example:
            {
                "default": "u4",
                "overrides": {"layer1.weight": "f32"},
                "prefixes": {"head.": "u8"},
            }
        """
        default = Quantizer(spec.get("default", "f32"))
        overrides = {}
        for k, v in spec.get("overrides", {}).items():
            overrides[k] = Quantizer(v)
        prefixes = {}
        for k, v in spec.get("prefixes", {}).items():
            prefixes[k] = Quantizer(v)
        return PrecisionConfig(default=default, overrides=overrides, prefixes=prefixes)

    def __repr__(self) -> str:
        n_overrides = len(self.overrides)
        n_prefixes = len(self.prefixes)
        return (f"PrecisionConfig(default={self.default.precision.name}, "
                f"{n_overrides} overrides, {n_prefixes} prefixes)")


# ---- Built-in presets for common models ----

def linear_only_u4() -> PrecisionConfig:
    """Quantize only Linear/Gemm weights to U4, keep everything else f32."""
    return PrecisionConfig(
        default=Quantizer(Precision.F32),
        prefixes={
            "Gemm_": Quantizer(Precision.U4, scheme="per_channel"),
            "fc": Quantizer(Precision.U4, scheme="per_channel"),
            "classifier.": Quantizer(Precision.U4, scheme="per_channel"),
        },
    )


def conv_u8_linear_u4() -> PrecisionConfig:
    """U8 for Conv weights, U4 for Linear/Gemm weights."""
    return PrecisionConfig(
        default=Quantizer(Precision.U4, scheme="per_channel"),
        prefixes={
            "Conv_": Quantizer(Precision.U8, scheme="per_channel"),
        },
    )


def all_u4() -> PrecisionConfig:
    """All weight parameters in U4."""
    return PrecisionConfig.uniform(Precision.U4)


def all_u8() -> PrecisionConfig:
    """All weight parameters in U8."""
    return PrecisionConfig.uniform(Precision.U8)


def all_f16() -> PrecisionConfig:
    """All weight parameters in F16."""
    return PrecisionConfig.uniform(Precision.F16)
