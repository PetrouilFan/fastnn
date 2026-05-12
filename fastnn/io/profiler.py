"""Mixed-precision profiler that auto-selects precision per layer."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fastnn.precision import Precision, PrecisionConfig, Quantizer

logger = logging.getLogger(__name__)


class PrecisionProfiler:
    """Run sensitivity analysis to auto-select precision per layer.

    For each layer, measure the output perturbation when quantizing
    just that layer vs keeping all others in f32.

    Args:
        model: A runnable fastnn model with a ``forward`` or ``__call__`` method.
        calibration_data: numpy array of calibration inputs [N, ...].
    """

    def __init__(self, model, calibration_data: np.ndarray):
        self.model = model
        self.calibration_data = calibration_data

    def _get_f32_baseline(self) -> np.ndarray:
        """Run model in full f32 to get the reference output."""
        if isinstance(self.calibration_data, np.ndarray):
            sample = self.calibration_data[:1]
            if hasattr(self.model, "forward"):
                out = self.model.forward({"input": sample})
            elif hasattr(self.model, "__call__"):
                out = self.model(sample)
            else:
                out = self.model(sample)
            if isinstance(out, dict):
                key = next(iter(out.values()))
                if hasattr(key, "numpy"):
                    return key.numpy()
                return np.asarray(key)
            if hasattr(out, "numpy"):
                return out.numpy()
            return np.asarray(out)
        return np.array([])

    def _get_weight_params(self) -> Dict[str, np.ndarray]:
        """Get all weight parameters from the model."""
        weights: Dict[str, np.ndarray] = {}
        if hasattr(self.model, "params"):
            for name, tensor in self.model.params.items():
                if any(name.endswith(s) for s in (".weight", ".gamma", ".beta")):
                    if hasattr(tensor, "numpy"):
                        weights[name] = tensor.numpy().copy()
                    elif isinstance(tensor, np.ndarray):
                        weights[name] = tensor.copy()
        return weights

    def _quantize_weights(
        self, weights: np.ndarray, precision: Precision
    ) -> Tuple[np.ndarray, List[float], List[float]]:
        """Quantize weights to the target precision and return (quantized_data, scales, zeros)."""
        quantizer = Quantizer(precision, scheme="per_channel")
        scales, zeros = quantizer.quantize(weights)
        max_val = float((1 << (precision.bit_width - 1)) - 1)
        scales_arr = np.asarray(scales)
        zeros_arr = np.asarray(zeros)
        if len(scales_arr) == 1:
            quantized = np.round(weights / scales_arr[0] + zeros_arr[0]).clip(
                -max_val - 1, max_val
            )
            dequantized = (quantized - zeros_arr[0]) * scales_arr[0]
        else:
            m = weights.shape[0]
            dequantized = np.zeros_like(weights, dtype=np.float64)
            for i in range(m):
                row = weights[i].ravel()
                s = scales_arr[i] if i < len(scales_arr) else scales_arr[0]
                z = zeros_arr[i] if i < len(zeros_arr) else zeros_arr[0]
                q = np.round(row / s + z).clip(-max_val - 1, max_val)
                dequantized[i] = ((q - z) * s).reshape(row.shape)
        return dequantized.astype(np.float32), list(scales), list(zeros)

    def _set_param(self, name: str, arr: np.ndarray):
        """Set a parameter in the model."""
        if hasattr(self.model, "params") and name in self.model.params:
            import fastnn as fnn
            self.model.params[name] = fnn.tensor(arr, list(arr.shape))
        elif hasattr(self.model, "params") and name in self.model.params:
            self.model.params[name] = arr

    def profile_layer_sensitivity(self) -> Dict[str, float]:
        """Profile each layer's sensitivity to quantization.

        For each weight parameter, quantize just that param in isolation,
        run the calibration data, and measure the MSE perturbation vs f32.

        Returns:
            Dict of {param_name: sensitivity_score}.
            Lower score = more robust to quantization.
        """
        weights = self._get_weight_params()
        if not weights:
            logger.warning("No weight parameters found for sensitivity profiling")
            return {}

        f32_output = self._get_f32_baseline()
        original_params: Dict[str, np.ndarray] = {}
        for name, arr in weights.items():
            if hasattr(self.model, "params") and name in self.model.params:
                import fastnn as fnn
                original_params[name] = arr.copy()

        sensitivities: Dict[str, float] = {}
        for param_name, weight_arr in weights.items():
            import fastnn as fnn

            deq, _scales, _zeros = self._quantize_weights(weight_arr, Precision.U4)
            if hasattr(self.model, "params") and param_name in self.model.params:
                self.model.params[param_name] = fnn.tensor(deq, list(deq.shape))

            quantized_output = self._get_f32_baseline()

            if f32_output.size > 0 and quantized_output.size > 0:
                mse = float(np.mean((f32_output - quantized_output) ** 2))
                sensitivity = np.log10(max(mse, 1e-10))
            else:
                sensitivity = 0.0
            sensitivities[param_name] = sensitivity

            if hasattr(self.model, "params") and param_name in original_params:
                import fastnn as fnn
                orig = original_params[param_name]
                self.model.params[param_name] = fnn.tensor(orig, list(orig.shape))

        return sensitivities

    def _estimate_param_size(self, param_name: str, weight_arr: np.ndarray) -> int:
        """Estimate number of parameters."""
        return int(np.prod(weight_arr.shape))

    def auto_config(self, target_ratio: float = 0.5) -> PrecisionConfig:
        """Auto-generate PrecisionConfig.

        Quantize the least-sensitive layers up to target_ratio of total params.
        target_ratio=0.5 means 50% of params can be quantized.

        Args:
            target_ratio: Fraction of total parameters to quantize (0.0 to 1.0).

        Returns:
            A PrecisionConfig with appropriate overrides.
        """
        sensitivities = self.profile_layer_sensitivity()
        if not sensitivities:
            return PrecisionConfig.f32()

        weights = self._get_weight_params()
        total_params = sum(
            self._estimate_param_size(name, weights[name])
            for name in sensitivities
            if name in weights
        )
        if total_params == 0:
            return PrecisionConfig.f32()

        # Sort by sensitivity (ascending = least sensitive first)
        sorted_params = sorted(sensitivities.items(), key=lambda x: x[1])

        # Assign U4 to least-sensitive, U8 to mid, F32 to most-sensitive
        overrides: Dict[str, str] = {}
        quantized_count = 0
        quantized_params = 0

        for param_name, sens in sorted_params:
            weight_arr = weights.get(param_name)
            if weight_arr is None:
                continue
            psize = self._estimate_param_size(param_name, weight_arr)
            new_ratio = (quantized_params + psize) / total_params

            if new_ratio <= target_ratio:
                quantized_params += psize
                quantized_count += 1
                # U4 for the most robust (least sensitive) quarter
                # U8 for the next quarter
                quarter = total_params / 4
                if quantized_params <= quarter:
                    overrides[param_name] = "u4"
                else:
                    overrides[param_name] = "u8"
            else:
                overrides[param_name] = "f32"

        logger.info(
            "Auto config: %d params profiled, %.1f%% quantized (%d params to U4/U8, %d to F32)",
            len(sensitivities),
            100.0 * quantized_params / total_params,
            quantized_count,
            len(sorted_params) - quantized_count,
        )

        config = PrecisionConfig(
            default=Quantizer(Precision.F32),
            overrides={k: Quantizer(v) for k, v in overrides.items()},
        )
        return config
