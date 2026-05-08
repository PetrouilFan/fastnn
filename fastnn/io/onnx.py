"""ONNX model importer for fastnn.

Converts ONNX models (.onnx) to fastnn's native format.
Supports common operators: Conv, Gemm, Relu, BatchNormalization, MaxPool, etc.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    import onnx

from fastnn.io import MODEL_MAGIC, MODEL_VERSION, _pack_u32, _pack_u64, write_tensor, write_fnn_file

__all__ = ["import_onnx"]

logger = logging.getLogger(__name__)


def get_pooling_config(node, default_kernel=(2, 2)):
    """Get pooling config (kernel, stride, padding) from ONNX node attributes."""
    kernel = _get_attr(node, "kernel_shape", list(default_kernel))
    stride = _get_attr(node, "strides", kernel)
    padding = _get_attr(node, "pads", [0, 0, 0, 0])

    config = {
        "kernel_size": kernel[0] if isinstance(kernel, list) else kernel,
        "stride": stride[0] if isinstance(stride, list) else stride,
        "padding": padding[0] if isinstance(padding, list) else padding,
    }
    return config





def _get_attr(node: "onnx.NodeProto", name: str, default=None):
    """Get attribute value from node."""
    import onnx
    for attr in node.attribute:
        if attr.name == name:
            if attr.HasField("f"):
                return attr.f
            elif attr.HasField("i"):
                return attr.i
            elif attr.HasField("s"):
                return attr.s.decode("utf-8")
            elif attr.ints:
                return list(attr.ints)
            elif attr.floats:
                return list(attr.floats)
    return default


def import_onnx(onnx_path: str, fnn_path: str) -> Dict[str, Any]:
    """Import an ONNX model and save it in fastnn format.

    Args:
        onnx_path: Path to .onnx file
        fnn_path: Path to output .fnn file

    Returns:
        Dictionary with model info (layers, input_shape, output_shape)
    """
    import onnx
    import onnx.numpy_helper

    model = onnx.load(onnx_path)
    
    layers = []
    layer_index = 0
    
    # Build initializer dict for O(1) lookup
    initializer_map = {init.name: onnx.numpy_helper.to_array(init)
                       for init in model.graph.initializer}

    # Build a mapping of node outputs to their consumers
    output_to_node = {}
    for node in model.graph.node:
        for output in node.output:
            output_to_node[output] = node

    # Collect all parameters first
    params = []

    for node in model.graph.node:
        op_type = node.op_type
        layer_info = {"name": node.name or f"{op_type}_{layer_index}", "type": op_type}
        layer_index += 1

        if op_type == "Conv":
            weight = initializer_map.get(node.input[1])
            bias = initializer_map.get(node.input[2]) if len(node.input) > 2 else None

            out_channels, in_channels = weight.shape[:2]
            kernel_h, _kernel_w = weight.shape[2], weight.shape[3]
            stride = _get_attr(node, "strides", [1, 1])
            padding = _get_attr(node, "pads", [0, 0, 0, 0])
            dilation = _get_attr(node, "dilations", [1, 1])
            groups = _get_attr(node, "group", 1)

            layer_info["in_channels"] = in_channels
            layer_info["out_channels"] = out_channels
            layer_info["kernel_size"] = kernel_h
            layer_info["stride"] = stride[0] if isinstance(stride, list) else stride
            layer_info["padding"] = padding[0] if isinstance(padding, list) else padding
            layer_info["dilation"] = (
                dilation[0] if isinstance(dilation, list) else dilation
            )
            layer_info["groups"] = groups
            layer_info["bias"] = bias is not None

            params.append((f"{node.name}.weight", weight))
            if bias is not None:
                params.append((f"{node.name}.bias", bias))

        elif op_type == "Gemm":
            weight = initializer_map.get(node.input[1])
            bias = initializer_map.get(node.input[2]) if len(node.input) > 2 else None

            _get_attr(node, "alpha", 1.0)
            _get_attr(node, "beta", 1.0)
            _get_attr(node, "transA", 0)
            trans_b = _get_attr(node, "transB", 1)

            if trans_b:
                weight = weight.T

            in_features = weight.shape[0]
            out_features = weight.shape[1]

            layer_info["type"] = "Linear"
            layer_info["in_features"] = in_features
            layer_info["out_features"] = out_features
            layer_info["bias"] = bias is not None

            params.append((f"{node.name}.weight", weight))
            if bias is not None:
                params.append((f"{node.name}.bias", bias))

        elif op_type == "Relu":
            pass  # No parameters

        elif op_type == "BatchNormalization":
            scale = initializer_map.get(node.input[1])
            bias = initializer_map.get(node.input[2])
            mean = initializer_map.get(node.input[3])
            var = initializer_map.get(node.input[4])

            layer_info["type"] = "BatchNorm2d"
            layer_info["num_features"] = scale.shape[0]
            layer_info["eps"] = _get_attr(node, "epsilon", 1e-5)
            layer_info["momentum"] = 1.0 - _get_attr(node, "momentum", 0.9)

            params.append((f"{node.name}.weight", scale))
            params.append((f"{node.name}.bias", bias))
            params.append((f"{node.name}.running_mean", mean))
            params.append((f"{node.name}.running_var", var))

        elif op_type == "MaxPool":
            config = get_pooling_config(node)
            layer_info["kernel_size"] = config["kernel_size"]
            layer_info["stride"] = config["stride"]
            layer_info["padding"] = config["padding"]

        elif op_type == "AveragePool":
            config = get_pooling_config(node)
            layer_info["type"] = "AvgPool"
            layer_info["kernel_size"] = config["kernel_size"]
            layer_info["stride"] = config["stride"]

        elif op_type == "GlobalAveragePool":
            layer_info["type"] = "GlobalAvgPool"

        elif op_type == "Flatten":
            axis = _get_attr(node, "axis", 1)
            layer_info["axis"] = axis

        elif op_type == "Add":
            # Check if it's a bias add (one input is a constant)
            bias = initializer_map.get(node.input[1])
            if bias is not None:
                layer_info["type"] = "BiasAdd"
                params.append((f"{node.name}.bias", bias))
            else:
                layer_info["type"] = "ElementwiseAdd"

        elif op_type == "Mul":
            layer_info["type"] = "ElementwiseMul"

        elif op_type == "MatMul":
            layer_info["type"] = "MatMul"

        elif op_type == "Concat":
            axis = _get_attr(node, "axis", 1)
            layer_info["axis"] = axis

        elif op_type == "Reshape":
            shape = initializer_map.get(node.input[1])
            if shape is not None:
                layer_info["shape"] = shape.tolist()

        elif op_type == "Transpose":
            perm = _get_attr(node, "perm")
            if perm:
                layer_info["perm"] = perm

        elif op_type == "Sigmoid":
            pass

        elif op_type == "Tanh":
            pass

        elif op_type == "LeakyRelu":
            layer_info["alpha"] = _get_attr(node, "alpha", 0.01)

        elif op_type == "Elu":
            layer_info["alpha"] = _get_attr(node, "alpha", 1.0)

        elif op_type == "Clip":
            layer_info["min"] = _get_attr(node, "min", None)
            layer_info["max"] = _get_attr(node, "max", None)

        elif op_type == "Dropout":
            layer_info["ratio"] = _get_attr(node, "ratio", 0.5)

        elif op_type == "LRN":
            layer_info["size"] = _get_attr(node, "size", 5)
            layer_info["alpha"] = _get_attr(node, "alpha", 0.0001)
            layer_info["beta"] = _get_attr(node, "beta", 0.75)

        elif op_type == "InstanceNormalization":
            scale = initializer_map.get(node.input[1])
            bias = initializer_map.get(node.input[2])
            layer_info["type"] = "InstanceNorm"
            layer_info["num_features"] = scale.shape[0]
            layer_info["eps"] = _get_attr(node, "epsilon", 1e-5)
            params.append((f"{node.name}.weight", scale))
            params.append((f"{node.name}.bias", bias))

        else:
            logger.warning(f"Unsupported operator: {op_type}")
            layer_info["type"] = f"Unsupported_{op_type}"

        layers.append(layer_info)

    # Write to file with unified format
    header = {
        "layers": layers,
        "total_parameters": len(params),
    }
    with open(fnn_path, "wb") as f:
        write_fnn_file(f, header, params)

    # Get input/output shapes
    input_shape = None
    output_shape = None
    for inp in model.graph.input:
        if inp.type.tensor_type.HasField("shape"):
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            input_shape = dims
    for out in model.graph.output:
        if out.type.tensor_type.HasField("shape"):
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            output_shape = dims

    result = {
        "layers": layers,
        "parameters": len(params),
        "input_shape": input_shape,
        "output_shape": output_shape,
    }
    return result
