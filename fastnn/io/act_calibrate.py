"""Activation-aware calibration for quantization scale refinement.

Runs calibration data through a quantized model to observe activation
distributions and refine scales using NVIDIA's KL-divergence method.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from fastnn.io.calibrate import Calibrator

logger = logging.getLogger(__name__)


class ActivationCalibrator:
    """Calibrator that observes activations to refine quantization scales.

    Args:
        calibration_data: numpy array of calibration inputs [N, C, H, W]
        onnx_path: Path to .onnx model for computing reference activations
    """

    def __init__(self, calibration_data: np.ndarray, onnx_path: str):
        self.calibration_data = calibration_data
        self.onnx_path = onnx_path
        self.calibrator = Calibrator(calibration_data)

    def collect_activations(self, model) -> Dict[str, np.ndarray]:
        """Run calibration data through the model, collect per-layer activations.

        Hooks into the model's intermediate layers and captures activations
        for each Conv/Gemm layer.

        Args:
            model: A runnable fastnn model (DAGExecutor or Sequential).

        Returns:
            Dict mapping layer name to activation tensor.
        """
        activations: Dict[str, np.ndarray] = {}
        hooks = []
        activation_storage = {}

        if hasattr(model, "nodes"):
            for node in model.nodes:
                op_type = getattr(node, "op_type", None) or node.get("op_type", "")
                if op_type in ("Conv", "Gemm", "MatMul", "Linear"):
                    node_name = node.get("name", str(id(node)))
                    activation_storage[node_name] = []
                    original_forward = None
                    if hasattr(node, "forward"):
                        original_forward = node.forward
                        def make_hook(name, orig_fn):
                            def hooked_fn(*args, **kwargs):
                                result = orig_fn(*args, **kwargs)
                                if hasattr(result, "numpy"):
                                    activation_storage[name].append(result.numpy())
                                elif isinstance(result, np.ndarray):
                                    activation_storage[name].append(result)
                                return result
                            return hooked_fn
                        node.forward = make_hook(node_name, original_forward)
                        hooks.append((node, original_forward))
        elif hasattr(model, "named_children"):
            for name, module in model.named_children():
                if isinstance(module, (type(model),)):
                    continue
                activation_storage[name] = []
                original_forward = module.forward
                def make_hook(n, orig):
                    def hooked(*args, **kwargs):
                        out = orig(*args, **kwargs)
                        if hasattr(out, "numpy"):
                            activation_storage[n].append(out.numpy())
                        elif isinstance(out, np.ndarray):
                            activation_storage[n].append(out)
                        return out
                    return hooked
                module.forward = make_hook(name, original_forward)
                hooks.append((module, original_forward))

        try:
            if isinstance(self.calibration_data, np.ndarray):
                n_samples = min(self.calibration_data.shape[0], 8)
                for i in range(n_samples):
                    sample = self.calibration_data[i:i+1]
                    if hasattr(model, "forward"):
                        model.forward({"input": sample})
                    elif hasattr(model, "__call__"):
                        model(sample)
                    else:
                        model(sample)
            for name, storage in activation_storage.items():
                if storage:
                    activations[name] = np.concatenate(storage, axis=0)
                else:
                    activations[name] = np.array([])
        finally:
            for obj, orig_fn in hooks:
                obj.forward = orig_fn

        return activations

    def refine_conv_scales(
        self,
        activations: Dict[str, np.ndarray],
        weights: Dict[str, np.ndarray],
    ) -> Dict[str, List[float]]:
        """Refine conv weight scales using observed activation distributions.

        Uses weighted KL-divergence: weight_scale * activation_magnitude.

        Args:
            activations: Dict mapping layer name to activation tensor.
            weights: Dict mapping layer name to weight tensor.

        Returns:
            Dict mapping param_name to refined scales (list of float).
        """
        refined: Dict[str, List[float]] = {}

        for layer_name, act_data in activations.items():
            if act_data.size == 0:
                continue
            weight_key = f"{layer_name}.weight"
            weight_arr = weights.get(weight_key) or weights.get(layer_name)
            if weight_arr is None:
                continue

            act_magnitude = float(np.mean(np.abs(act_data)))
            if act_magnitude == 0.0:
                act_magnitude = 1.0

            weight_data = weight_arr if isinstance(weight_arr, np.ndarray) else np.array(weight_arr)
            weighted = weight_data * act_magnitude

            scale = self.calibrator.kl_divergence_scale(weighted)
            refined[weight_key] = [scale]

        return refined

    def calibrate(self, model) -> Dict[str, List[float]]:
        """Full calibration pipeline: collect activations + refine scales.

        Args:
            model: A runnable fastnn model.

        Returns:
            Dict mapping param_name to refined scales.
        """
        activations = self.collect_activations(model)

        weights: Dict[str, np.ndarray] = {}
        if hasattr(model, "params"):
            for name, tensor in model.params.items():
                if hasattr(tensor, "numpy"):
                    weights[name] = tensor.numpy()
                elif isinstance(tensor, np.ndarray):
                    weights[name] = tensor

        return self.refine_conv_scales(activations, weights)
