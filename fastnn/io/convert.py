"""Conversion CLI for fastnn precision management.

Provides the `fastnn-convert` command-line tool and the
`convert_precision` function for converting between precisions.
"""

from __future__ import annotations

import argparse
import json
import struct


def _requantize_per_channel(
    f32_weights: np.ndarray,
    scales: list,
    bit_width: int,
    num_channels: int,
) -> bytes:
    """Re-quantize f32 weights with given per-channel scales.

    Manually quantizes and packs values into u32 words for U4/U8 formats.

    Returns:
        Packed bytes ready for PackedTensor.from_bytes().
    """
    flat = f32_weights.ravel()
    n = len(flat)
    inner = n // num_channels
    max_val = float((1 << (bit_width - 1)) - 1)

    if bit_width == 4:
        # U4: 8 values per u32 word
        items_per_word = 8
        quantized = np.zeros(n, dtype=np.int32)
        for ch in range(num_channels):
            start = ch * inner
            end = start + inner
            inv_s = 1.0 / float(scales[ch]) if scales[ch] != 0 else 1.0
            quantized[start:end] = np.clip(
                np.round(flat[start:end] * inv_s),
                -max_val - 1, max_val
            ).astype(np.int32)

        # Pack into u32 words (8 × 4-bit → 1 × u32)
        packed_words = []
        n_words = (n + items_per_word - 1) // items_per_word
        for w in range(n_words):
            word = 0
            for i in range(items_per_word):
                idx = w * items_per_word + i
                if idx < n:
                    nibble = quantized[idx] & 0xF
                    word |= (nibble << (i * 4))
            packed_words.append(word)
        return struct.pack(f"<{len(packed_words)}I", *packed_words)

    elif bit_width == 8:
        # U8: 4 values per u32 word
        items_per_word = 4
        quantized = np.zeros(n, dtype=np.int32)
        for ch in range(num_channels):
            start = ch * inner
            end = start + inner
            inv_s = 1.0 / float(scales[ch]) if scales[ch] != 0 else 1.0
            quantized[start:end] = np.clip(
                np.round(flat[start:end] * inv_s),
                -max_val - 1, max_val
            ).astype(np.int32)

        # Pack into u32 words (4 × 8-bit → 1 × u32)
        packed_words = []
        n_words = (n + items_per_word - 1) // items_per_word
        for w in range(n_words):
            word = 0
            for i in range(items_per_word):
                idx = w * items_per_word + i
                if idx < n:
                    byte_val = quantized[idx] & 0xFF
                    word |= (byte_val << (i * 8))
            packed_words.append(word)
        return struct.pack(f"<{len(packed_words)}I", *packed_words)

    else:
        raise ValueError(f"Unsupported bit_width for repacking: {bit_width}")
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def convert_precision(
    input_path: str,
    output_path: str,
    config: Any = None,
    calibration_data: Optional[np.ndarray] = None,
    validate_onnx: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a model between precisions.

    Args:
        input_path: Path to .onnx or .fnn file.
        output_path: Path to output .fnn file.
        config: PrecisionConfig for target precision.
        calibration_data: Optional calibration data for scale refinement.
        validate_onnx: Optional ONNX model path for validation after conversion.

    Returns:
        Dictionary with conversion results and optional validation report.
    """
    from fastnn.precision import PrecisionConfig

    if isinstance(config, dict):
        config = PrecisionConfig.from_dict_spec(config)
    elif config is None:
        config = PrecisionConfig.f32()

    result = {"input": input_path, "output": output_path}

    input_path = str(input_path)
    output_path = str(output_path)

    # Detect input type
    if input_path.endswith(".onnx"):
        result["source"] = "onnx"
        from fastnn.io.onnx import import_onnx
        info = import_onnx(input_path, output_path, config=config)
        result["import_info"] = info

        # Apply calibration if provided
        if calibration_data is not None and config.default.should_quantize():
            try:
                from fastnn.io.calibrate import Calibrator
                calibrator = Calibrator(calibration_data)
                # Reload and refine
                from fastnn.io import read_fnn_header, read_fnn_parameters_v3, write_fnn_file_v3, MODEL_MAGIC
                from fastnn.precision import Precision, Quantizer

                with open(output_path, "rb") as f:
                    magic, version, header, num_params = read_fnn_header(f)
                    raw_params = read_fnn_parameters_v3(f, num_params)

                # Refine scales where possible
                params_v3 = []
                for name, (data, dtype, scales, zeros) in raw_params.items():
                    if dtype in (2, 3) and config.should_quantize_param(name):
                        # Get original f32 weights by dequantizing
                        from fastnn import PackedTensor4, PackedTensor8
                        q = config.get_quantizer(name)
                        bit_width = q.precision.bit_width
                        cls_map = {4: PackedTensor4, 8: PackedTensor8}
                        cls = cls_map[bit_width]
                        shape = data.shape if hasattr(data, 'shape') else []
                        packed = cls.from_bytes(bytes(data), shape, scales, zeros)
                        f32_weights = np.array(packed.to_f32_vec()).reshape(shape)

                        refined_scales = calibrator.refine_scales(
                            name, f32_weights, scales, method="percentile"
                        )

                        # Re-quantize with refined scales
                        if len(refined_scales) == 1:
                            s = float(refined_scales[0])
                            p = cls(f32_weights.ravel().tolist(), list(f32_weights.shape), s, 0.0)
                            params_v3.append((
                                name, bytes(p.to_bytes()), dtype,
                                [s], [0.0]
                            ))
                        else:
                            # Manually quantize with refined per-channel scales
                            new_data = _requantize_per_channel(
                                f32_weights, refined_scales, bit_width, len(refined_scales)
                            )
                            params_v3.append((
                                name, new_data, dtype,
                                refined_scales, [0.0] * len(refined_scales)
                            ))
                    else:
                        params_v3.append((name, data, dtype, scales, zeros))

                header["calibrated"] = True
                with open(output_path, "wb") as f:
                    write_fnn_file_v3(f, header, params_v3)
                result["calibrated"] = True
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                result["calibrated"] = False

    elif input_path.endswith(".fnn"):
        result["source"] = "fnn"
        # Reload existing .fnn and re-quantize
        from fastnn.io import read_fnn_header, read_fnn_parameters, MODEL_MAGIC, MODEL_VERSION
        from fastnn.io import write_fnn_file_v3, write_fnn_file
        from fastnn.io import DTYPE_F32, DTYPE_U4, DTYPE_U8, DTYPE_F16
        from fastnn import PackedTensor4, PackedTensor8, PackedTensor16

        with open(input_path, "rb") as f:
            magic, version, header, num_params = read_fnn_header(f)
            params = read_fnn_parameters(f, num_params, version=version)

        params_v3 = []
        for name, value in params.items():
            if isinstance(value, tuple) and len(value) == 4:
                data, dtype, scales, zeros = value
                if dtype == DTYPE_F32:
                    params_v3.append((name, data, DTYPE_F32, [], []))
                else:
                    params_v3.append((name, data, dtype, scales, zeros))
            else:
                # v2 f32 tensor
                arr = value
                if config.should_quantize_param(name) and any(
                    name.endswith(s) for s in [".weight", ".gamma", ".beta"]
                ):
                    q = config.get_quantizer(name)
                    bit_width = q.precision.bit_width
                    cls_map = {4: PackedTensor4, 8: PackedTensor8, 16: PackedTensor16}
                    cls = cls_map[bit_width]
                    shape = list(arr.shape)
                    use_per_channel = q.scheme == "per_channel" and arr.ndim >= 2
                    if use_per_channel:
                        packed = cls.from_f32_per_channel(arr.ravel().tolist(), shape)
                    else:
                        s = 1.0
                        packed = cls(arr.ravel().tolist(), shape, s, 0.0)
                    params_v3.append((
                        name, bytes(packed.to_bytes()),
                        {4: DTYPE_U4, 8: DTYPE_U8, 16: DTYPE_F16}[bit_width],
                        packed.scales(), packed.zeros(),
                        shape,
                    ))
                else:
                    shape = list(arr.shape) if hasattr(arr, 'shape') else []
                    params_v3.append((name, arr, DTYPE_F32, [], [], shape))

        header["precision"] = config.to_dict()
        with open(output_path, "wb") as f:
            write_fnn_file_v3(f, header, params_v3)

        result["converted_params"] = len(params_v3)
    else:
        raise ValueError(f"Unknown input format: {input_path}")

    # Validate against ONNX Runtime if requested
    if validate_onnx:
        try:
            from fastnn.io.validate import validate_model_file
            val_report = validate_model_file(output_path, validate_onnx, calibration_data or np.random.randn(1, 3, 224, 224))
            result["validation"] = {
                "verdict": val_report.verdict,
                "details": {k: dict(v) for k, v in val_report.per_output.items()},
            }
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            result["validation"] = {"error": str(e)}

    return result


def main():
    """CLI entry point for fastnn-convert."""
    parser = argparse.ArgumentParser(
        description="Convert fastnn models between precisions"
    )
    parser.add_argument("input", help="Input model (.onnx or .fnn)")
    parser.add_argument("output", help="Output .fnn file")
    parser.add_argument(
        "--dtype", "-d",
        choices=["f32", "f16", "u8", "u4"],
        default="f32",
        help="Target precision dtype"
    )
    parser.add_argument(
        "--scheme", "-s",
        choices=["per_tensor", "per_channel", "per_block"],
        default="per_channel",
        help="Quantization scheme"
    )
    parser.add_argument(
        "--calibrate", "-c",
        type=str,
        default=None,
        help="Path to calibration data (.npy file)"
    )
    parser.add_argument(
        "--validate", "-v",
        type=str,
        default=None,
        help="Path to .onnx file for runtime validation"
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default=None,
        help="JSON string of per-layer precision overrides"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Build config
    spec = {"default": args.dtype, "scheme": args.scheme}
    if args.overrides:
        try:
            overrides = json.loads(args.overrides)
            spec["overrides"] = overrides
        except json.JSONDecodeError as e:
            print(f"Error parsing overrides JSON: {e}", file=sys.stderr)
            sys.exit(1)

    from fastnn.precision import PrecisionConfig
    config = PrecisionConfig.from_dict_spec(spec)

    # Load calibration data
    cal_data = None
    if args.calibrate:
        cal_data = np.load(args.calibrate)

    # Run conversion
    try:
        result = convert_precision(
            args.input, args.output,
            config=config,
            calibration_data=cal_data,
            validate_onnx=args.validate,
        )

        print(f"Conversion complete: {args.input} -> {args.output}")

        if "validation" in result:
            val = result["validation"]
            if "verdict" in val:
                print(f"Validation: {val['verdict']}")
                for name, metrics in val.get("details", {}).items():
                    print(f"  {name}: MAE={metrics['mae']:.6f}, CosSim={metrics['cosine_similarity']:.6f}")
            elif "error" in val:
                print(f"Validation error: {val['error']}")

        if "calibrated" in result:
            print(f"Calibration: {'applied' if result['calibrated'] else 'failed'}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
