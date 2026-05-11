"""Calibration module for quantization scale refinement.

Uses calibration data to improve quantization accuracy by observing
actual weight value distributions rather than assuming max-abs scaling.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


class Calibrator:
    """Run calibration data through a model to refine quantization scales.

    Args:
        model_path: Path to .onnx model for reference shapes.
        calibration_data: numpy array of calibration inputs [N, C, H, W] or [N, features].
    """

    def __init__(self, calibration_data: np.ndarray):
        self.calibration_data = calibration_data

    @staticmethod
    def percentile_scale(data: np.ndarray, percentile: float = 99.9) -> float:
        """Compute scale based on percentile instead of max-abs.

        Uses the given percentile of absolute values, which ignores outliers.
        """
        abs_data = np.abs(data)
        max_val = float((1 << (7)) - 1)  # int7 max for symmetric quantization
        p = float(np.percentile(abs_data, percentile))
        if p == 0.0:
            return 1.0
        return p / max_val

    @staticmethod
    def kl_divergence_scale(data: np.ndarray, num_bins: int = 2048) -> float:
        """Find optimal scale using KL-divergence (NVIDIA's method).

        Finds the threshold that minimizes information loss between
        the original distribution and the quantized distribution.
        """
        abs_data = np.abs(data)
        max_val = float(np.max(abs_data))
        if max_val == 0.0:
            return 1.0

        # Build histogram
        hist, bin_edges = np.histogram(abs_data, bins=num_bins, range=(0.0, max_val))

        best_kl = float('inf')
        best_threshold = max_val

        # Search over possible thresholds
        for threshold_idx in range(num_bins // 2, num_bins):
            threshold = bin_edges[threshold_idx + 1]

            # Truncated histogram
            truncated = hist[:threshold_idx + 1].copy()
            truncated[-1] += np.sum(hist[threshold_idx + 1:])

            if np.sum(truncated) == 0:
                continue

            # Quantize to 128 bins (int7)
            num_quant_bins = 128
            quantized = np.zeros(num_quant_bins)
            expanded = np.zeros(len(truncated))

            for i in range(num_quant_bins):
                start = i * len(truncated) // num_quant_bins
                end = (i + 1) * len(truncated) // num_quant_bins
                if end <= len(truncated):
                    quantized[i] = np.sum(truncated[start:end])
                    expanded[start:end] = quantized[i] / max(1, end - start)

            # Avoid zeros for KL divergence
            prob_p = truncated / np.sum(truncated) + 1e-10
            prob_q = expanded / np.sum(expanded) + 1e-10

            kl = np.sum(prob_p * np.log(prob_p / prob_q))
            if kl < best_kl:
                best_kl = kl
                best_threshold = threshold

        max_val_q = float((1 << 7) - 1)
        return best_threshold / max_val_q if best_threshold > 0 else 1.0

    def refine_scales(
        self,
        param_name: str,
        f32_weights: np.ndarray,
        current_scales: List[float],
        method: str = "percentile",
        percentile: float = 99.9,
    ) -> List[float]:
        """Refine quantization scales using calibration data.

        Args:
            param_name: Name of the parameter (for logging).
            f32_weights: The full-precision weights.
            current_scales: Current per-channel or per-tensor scales.
            method: "percentile" or "kl_divergence".
            percentile: Percentile to use when method="percentile".

        Returns:
            Refined scales (same length as current_scales).
        """
        method_fn = {
            "percentile": lambda d: self.percentile_scale(d, percentile),
            "kl_divergence": self.kl_divergence_scale,
        }
        fn = method_fn.get(method, method_fn["percentile"])

        if len(current_scales) == 1:
            # Per-tensor
            return [fn(f32_weights)]
        else:
            # Per-channel
            refined = []
            m = f32_weights.shape[0]
            inner = int(np.prod(f32_weights.shape[1:])) if f32_weights.ndim > 1 else f32_weights.size // m
            for i in range(m):
                row = f32_weights[i].ravel() if f32_weights.ndim > 1 else f32_weights[i * inner:(i + 1) * inner]
                refined.append(fn(row))
            return refined
