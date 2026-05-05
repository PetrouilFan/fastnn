"""ONNX model importer for fastnn.

Converts ONNX models (.onnx) to fastnn's native format.
Supports common operators: Conv, Gemm, Relu, BatchNormalization, MaxPool, etc.
"""

import onnx
import numpy as np
import struct
import logging
from typing import Dict, Optional, Any
from fastnn.serialization_utils import MODEL_MAGIC, MODEL_VERSION, write_tensor

logger = logging.getLogger(__name__)


def _get_initializer(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
    """Get initializer tensor by name."""
    for init in model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    return None


def _get_value_info(model: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
    """Get value info by name."""
    for vi in model.graph.value_info:
        if vi.name == name:
            return vi
    for inp in model.graph.input:
        if inp.name == name:
            return inp
    for out in model.graph.output:
        if out.name == name:
            return out
    return None


def _get_attr(node: onnx.NodeProto, name: str, default=None):
    """Get attribute value from node."""
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
    model = onnx.load(onnx_path)

    layers = []
    layer_index = 0
    param_count = 0

    # Build initializer dict for O(1) lookup
    initializer_map = {init.name: onnx.numpy_helper.to_array(init)
                       for init in model.graph.initializer}

    def _get_initializer(name: str) -> Optional[np.ndarray]:
        """Get initializer tensor by name (O(1) lookup)."""
        return initializer_map.get(name)

    # Build a mapping of node outputs to their consumers
    output_to_node = {}
    for node in model.graph.node:
        for output in node.output:
            output_to_node[output] = node

    # Open output file and write header
    f = open(fnn_path, "wb")
    try:
        f.write(MODEL_MAGIC)
        f.write(struct.pack("<I", MODEL_VERSION))
        # Placeholder for parameter count - will seek back to update
        param_count_offset = f.tell()
        f.write(struct.pack("<Q", 0))  # Placeholder

        # Track which tensors have been consumed (to avoid duplicate layer exports)

        for node in model.graph.node:
            op_type = node.op_type
            layer_info = {"name": node.name or f"{op_type}_{layer_index}", "type": op_type}
            layer_index += 1

            if op_type == "Conv":
                weight = _get_initializer(node.input[1])
                bias = (
                    _get_initializer(node.input[2]) if len(node.input) > 2 else None
                )

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

                write_tensor(f, f"{node.name}.weight", weight)
                param_count += 1
                if bias is not None:
                    write_tensor(f, f"{node.name}.bias", bias)
                    param_count += 1

            elif op_type == "Gemm":
                weight = _get_initializer(node.input[1])
                bias = (
                    _get_initializer(node.input[2]) if len(node.input) > 2 else None
                )

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

                write_tensor(f, f"{node.name}.weight", weight)
                param_count += 1
                if bias is not None:
                    write_tensor(f, f"{node.name}.bias", bias)
                    param_count += 1

            elif op_type == "Relu":
                pass  # No parameters

            elif op_type == "BatchNormalization":
                scale = _get_initializer(node.input[1])
                bias = _get_initializer(node.input[2])
                mean = _get_initializer(node.input[3])
                var = _get_initializer(node.input[4])

                layer_info["type"] = "BatchNorm2d"
                layer_info["num_features"] = scale.shape[0]
                layer_info["eps"] = _get_attr(node, "epsilon", 1e-5)
                layer_info["momentum"] = 1.0 - _get_attr(node, "momentum", 0.9)

                write_tensor(f, f"{node.name}.weight", scale)
                param_count += 1
                write_tensor(f, f"{node.name}.bias", bias)
                param_count += 1
                write_tensor(f, f"{node.name}.running_mean", mean)
                param_count += 1
                write_tensor(f, f"{node.name}.running_var", var)
                param_count += 1

            elif op_type == "MaxPool":
                kernel = _get_attr(node, "kernel_shape", [2, 2])
                stride = _get_attr(node, "strides", kernel)
                padding = _get_attr(node, "pads", [0, 0, 0, 0])

                layer_info["kernel_size"] = (
                    kernel[0] if isinstance(kernel, list) else kernel
                )
                layer_info["stride"] = stride[0] if isinstance(stride, list) else stride
                layer_info["padding"] = padding[0] if isinstance(padding, list) else padding

            elif op_type == "AveragePool":
                kernel = _get_attr(node, "kernel_shape", [2, 2])
                stride = _get_attr(node, "strides", kernel)
                layer_info["type"] = "AvgPool"
                layer_info["kernel_size"] = (
                    kernel[0] if isinstance(kernel, list) else kernel
                )
                layer_info["stride"] = stride[0] if isinstance(stride, list) else stride

            elif op_type == "GlobalAveragePool":
                layer_info["type"] = "GlobalAvgPool"

            elif op_type == "Flatten":
                axis = _get_attr(node, "axis", 1)
                layer_info["axis"] = axis

            elif op_type == "Add":
                # Check if it's a bias add (one input is a constant)
                bias = _get_initializer(node.input[1])
                if bias is not None:
                    layer_info["type"] = "BiasAdd"
                    write_tensor(f, f"{node.name}.bias", bias)
                    param_count += 1
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
                shape = _get_initializer(node.input[1])
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
                scale = _get_initializer(node.input[1])
                bias = _get_initializer(node.input[2])
                layer_info["type"] = "InstanceNorm"
                layer_info["num_features"] = scale.shape[0]
                layer_info["eps"] = _get_attr(node, "epsilon", 1e-5)
                write_tensor(f, f"{node.name}.weight", scale)
                param_count += 1
                write_tensor(f, f"{node.name}.bias", bias)
                param_count += 1

            else:
                logger.warning(f"Unsupported operator: {op_type}")
                layer_info["type"] = f"Unsupported_{op_type}"

            layers.append(layer_info)

        # Seek back to update parameter count
        f.seek(param_count_offset)
        f.write(struct.pack("<Q", param_count))

    except Exception:
        raise
    finally:
        f.close()

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
        "parameters": param_count,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }
    return result
