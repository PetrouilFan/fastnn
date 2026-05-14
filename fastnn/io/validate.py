"""ONNX Runtime validation module.

Compares FastNN model outputs against ONNX Runtime ground truth
for accuracy validation of quantized models.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class ValidationReport:
    """Report of validation results comparing FastNN vs ONNX Runtime."""

    def __init__(self):
        self.per_output: Dict[str, Dict[str, float]] = {}
        self.global_verdict: str = "UNKNOWN"

    def add_result(self, name: str, fastnn_output: np.ndarray, onnx_output: np.ndarray):
        """Compare a single output tensor."""
        fa = fastnn_output.ravel().astype(np.float64)
        on = onnx_output.ravel().astype(np.float64)

        diff = fa - on
        abs_diff = np.abs(diff)
        mae = float(np.mean(abs_diff))
        mse = float(np.mean(diff ** 2))
        max_abs = float(np.max(abs_diff))

        # Cosine similarity
        norm_fa = np.linalg.norm(fa)
        norm_on = np.linalg.norm(on)
        if norm_fa > 0 and norm_on > 0:
            cos_sim = float(np.dot(fa, on) / (norm_fa * norm_on))
        else:
            cos_sim = 1.0 if norm_fa == norm_on == 0 else 0.0

        self.per_output[name] = {
            "mae": mae,
            "mse": mse,
            "max_abs_error": max_abs,
            "cosine_similarity": cos_sim,
        }

    @property
    def verdict(self) -> str:
        """Overall verdict: PASS, APPROXIMATE, or DIFFERS."""
        if not self.per_output:
            return "NO_DATA"
        max_mae = max(v["mae"] for v in self.per_output.values())
        min_cos = min(v["cosine_similarity"] for v in self.per_output.values())

        if max_mae < 1e-3 and min_cos > 0.999:
            return "PASS"
        elif max_mae < 0.1 and min_cos > 0.95:
            return "APPROXIMATE"
        else:
            return "DIFFERS"

    def summary(self) -> str:
        lines = [f"Validation Verdict: {self.verdict}"]
        lines.append(f"{'Output':<30} {'MAE':<12} {'MSE':<12} {'MaxErr':<12} {'CosSim':<12}")
        lines.append("-" * 78)
        for name, metrics in self.per_output.items():
            lines.append(
                f"{name:<30} {metrics['mae']:<12.6f} {metrics['mse']:<12.6f} "
                f"{metrics['max_abs_error']:<12.6f} {metrics['cosine_similarity']:<12.6f}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def validate_against_onnx(
    fnn_model,
    onnx_path: str,
    input_data: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> ValidationReport:
    """Compare FastNN model output against ONNX Runtime.

    Args:
        fnn_model: A FastNN model (AotExecutor or callable).
        onnx_path: Path to the .onnx file.
        input_data: Input data as numpy array.
        output_names: Optional list of output tensor names to compare.

    Returns:
        ValidationReport with per-output metrics.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for validation. "
                         "Install with: pip install onnxruntime")

    # Run ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    ort_input_name = ort_session.get_inputs()[0].name
    ort_results = ort_session.run(output_names, {ort_input_name: input_data})

    # Get ONNX output names
    if output_names is None:
        output_names = [o.name for o in ort_session.get_outputs()]

    # Run FastNN model
    fnn_input = {ort_input_name: input_data}
    try:
        fnn_results_dict = fnn_model.forward(fnn_input)
    except (AttributeError, TypeError):
        # Fallback: call as function
        try:
            fnn_results_dict = fnn_model(input_data)
        except (TypeError, Exception):
            # Wrap in dict
            result = fnn_model(input_data)
            if isinstance(result, dict):
                fnn_results_dict = result
            else:
                fnn_results_dict = {output_names[0]: result}

    # Build report
    report = ValidationReport()
    for i, name in enumerate(output_names):
        onnx_out = ort_results[i] if isinstance(ort_results, (list, tuple)) else ort_results
        if isinstance(onnx_out, list):
            onnx_out = onnx_out[i]

        fnn_out = fnn_results_dict.get(name)
        if fnn_out is None:
            # Try positional
            if isinstance(fnn_results_dict, dict):
                values = list(fnn_results_dict.values())
                fnn_out = values[i] if i < len(values) else values[0]
            else:
                fnn_out = fnn_results_dict

        # Convert to numpy
        if hasattr(fnn_out, 'numpy'):
            fnn_out = fnn_out.numpy()
        elif hasattr(fnn_out, 'data_ptr'):
            fnn_out = np.array(fnn_out.to_f32_vec()).reshape(fnn_out.shape())
        else:
            fnn_out = np.asarray(fnn_out)

        onnx_out = np.asarray(onnx_out)

        report.add_result(name, fnn_out, onnx_out)

    return report


def validate_model_file(
    fnn_path: str,
    onnx_path: str,
    input_data: np.ndarray,
) -> ValidationReport:
    """Load a .fnn model and validate against ONNX Runtime.

    Args:
        fnn_path: Path to .fnn file.
        onnx_path: Path to .onnx file.
        input_data: Input data as numpy array.

    Returns:
        ValidationReport.
    """
    from fastnn.io.graph_builder import build_model_from_fnn
    model = build_model_from_fnn(fnn_path)
    return validate_against_onnx(model, onnx_path, input_data)
