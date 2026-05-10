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

            alpha = _get_attr(node, "alpha", 1.0)
            beta = _get_attr(node, "beta", 1.0)
            _get_attr(node, "transA", 0)
            trans_b = _get_attr(node, "transB", 0)

            if trans_b:
                weight = weight.T

            weight = weight * alpha

            in_features = weight.shape[0]
            out_features = weight.shape[1]

            layer_info["type"] = "Linear"
            layer_info["in_features"] = in_features
            layer_info["out_features"] = out_features
            layer_info["bias"] = bias is not None

            params.append((f"{node.name}.weight", weight))
            if bias is not None:
                params.append((f"{node.name}.bias", bias * beta))

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
            layer_info["padding"] = config["padding"]

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

        elif op_type == "Split":
            split = _get_attr(node, "split", None)
            axis = _get_attr(node, "axis", 0)
            layer_info["axis"] = axis
            if split is not None:
                layer_info["split"] = list(split)

        elif op_type == "Shape":
            layer_info["type"] = "ShapeOp"

        elif op_type == "Cast":
            to = _get_attr(node, "to", 1)
            layer_info["to"] = to
            layer_info["type"] = "CastOp"

        elif op_type == "Gather":
            axis = _get_attr(node, "axis", 0)
            layer_info["axis"] = axis
            layer_info["type"] = "GatherOp"

        elif op_type == "Sub":
            bias = initializer_map.get(node.input[1])
            if bias is not None:
                layer_info["type"] = "BiasSub"
                params.append((f"{node.name}.bias", -bias))
            else:
                layer_info["type"] = "ElementwiseSub"

        elif op_type == "Div":
            layer_info["type"] = "ElementwiseDiv"

        elif op_type == "Pow":
            exponent = initializer_map.get(node.input[1])
            if exponent is not None:
                layer_info["exponent"] = float(exponent.flatten()[0])
            layer_info["type"] = "ElementwisePow"

        elif op_type == "Exp":
            layer_info["type"] = "ExpOp"

        elif op_type == "Sqrt":
            layer_info["type"] = "SqrtOp"

        elif op_type == "Neg":
            layer_info["type"] = "NegOp"

        elif op_type == "Resize":
            mode = _get_attr(node, "mode", "nearest")
            coord_mode = _get_attr(node, "coordinate_transformation_mode", "half_pixel")
            layer_info["type"] = "Resize"
            layer_info["mode"] = mode
            layer_info["coordinate_transformation_mode"] = coord_mode

        elif op_type == "ReduceMean":
            axes = _get_attr(node, "axes", None)
            keepdims = _get_attr(node, "keepdims", 1)
            layer_info["type"] = "ReduceMean"
            if axes is not None:
                layer_info["axes"] = list(axes)
            layer_info["keepdims"] = bool(keepdims)

        elif op_type == "ReduceSum":
            axes = _get_attr(node, "axes", None)
            keepdims = _get_attr(node, "keepdims", 1)
            layer_info["type"] = "ReduceSum"
            if axes is not None:
                layer_info["axes"] = list(axes)
            layer_info["keepdims"] = bool(keepdims)

        elif op_type == "Tile":
            layer_info["type"] = "TileOp"

        elif op_type == "Pad":
            mode = _get_attr(node, "mode", "constant")
            layer_info["type"] = "Pad"
            layer_info["mode"] = mode

        elif op_type == "Slice":
            layer_info["type"] = "SliceOp"

        elif op_type == "Where":
            layer_info["type"] = "WhereOp"

        elif op_type == "NonMaxSuppression":
            layer_info["type"] = "NonMaxSuppression"
            center_point_box = _get_attr(node, "center_point_box", 0)
            layer_info["center_point_box"] = center_point_box

        elif op_type == "TopK":
            axis = _get_attr(node, "axis", -1)
            layer_info["axis"] = axis
            layer_info["type"] = "TopKOp"

        elif op_type == "Softmax":
            axis = _get_attr(node, "axis", 1)
            layer_info["axis"] = axis

        elif op_type == "Log":
            layer_info["type"] = "LogOp"

        elif op_type == "Erf":
            layer_info["type"] = "ErfOp"

        elif op_type == "Constant":
            value = _get_attr(node, "value", None)
            if value is not None:
                layer_info["type"] = "ConstantOp"
                try:
                    import onnx.numpy_helper
                    const_tensor = onnx.numpy_helper.to_array(value)
                    layer_info["dims"] = list(const_tensor.shape)
                    params.append((f"{node.name}.value", const_tensor))
                except Exception:
                    pass

        elif op_type == "Unsqueeze":
            axes = _get_attr(node, "axes", None)
            if axes is not None:
                layer_info["axes"] = list(axes)
            layer_info["type"] = "UnsqueezeOp"

        elif op_type == "Squeeze":
            axes = _get_attr(node, "axes", None)
            if axes is not None:
                layer_info["axes"] = list(axes)
            layer_info["type"] = "SqueezeOp"

        elif op_type == "Identity":
            layer_info["type"] = "IdentityOp"

        elif op_type == "Loop":
            layer_info["type"] = "LoopOp"

        elif op_type == "If":
            layer_info["type"] = "IfOp"

        elif op_type == "And":
            layer_info["type"] = "AndOp"

        elif op_type == "Or":
            layer_info["type"] = "OrOp"

        elif op_type == "Xor":
            layer_info["type"] = "XorOp"

        elif op_type == "Not":
            layer_info["type"] = "NotOp"

        elif op_type == "Less":
            layer_info["type"] = "LessOp"

        elif op_type == "Greater":
            layer_info["type"] = "GreaterOp"

        elif op_type == "Equal":
            layer_info["type"] = "EqualOp"

        elif op_type == "Ceil":
            layer_info["type"] = "CeilOp"

        elif op_type == "Floor":
            layer_info["type"] = "FloorOp"

        elif op_type == "Round":
            layer_info["type"] = "RoundOp"

        elif op_type == "Sign":
            layer_info["type"] = "SignOp"

        elif op_type == "Reciprocal":
            layer_info["type"] = "ReciprocalOp"

        elif op_type == "IsNaN":
            layer_info["type"] = "IsNaNOp"

        elif op_type == "IsInf":
            layer_info["type"] = "IsInfOp"

        elif op_type == "LogSoftmax":
            axis = _get_attr(node, "axis", 1)
            layer_info["type"] = "LogSoftmax"
            layer_info["axis"] = axis

        elif op_type == "Selu":
            alpha = _get_attr(node, "alpha", 1.67326)
            gamma = _get_attr(node, "gamma", 1.0507)
            layer_info["type"] = "Selu"
            layer_info["alpha"] = alpha
            layer_info["gamma"] = gamma

        elif op_type == "HardSigmoid":
            alpha = _get_attr(node, "alpha", 0.2)
            beta = _get_attr(node, "beta", 0.5)
            layer_info["type"] = "HardSigmoid"
            layer_info["alpha"] = alpha
            layer_info["beta"] = beta

        elif op_type == "HardSwish":
            layer_info["type"] = "Hardswish"

        elif op_type == "LayerNormalization":
            layer_info["type"] = "LayerNorm"
            axis = _get_attr(node, "axis", -1)
            epsilon = _get_attr(node, "epsilon", 1e-5)
            layer_info["axis"] = axis
            layer_info["eps"] = epsilon
            scale = initializer_map.get(node.input[1])
            bias = initializer_map.get(node.input[2]) if len(node.input) > 2 else None
            if scale is not None:
                params.append((f"{node.name}.weight", scale))
            if bias is not None:
                params.append((f"{node.name}.bias", bias))

        elif op_type == "ConvTranspose":
            weight = initializer_map.get(node.input[1])
            bias = initializer_map.get(node.input[2]) if len(node.input) > 2 else None
            if weight is not None:
                out_channels, in_channels = weight.shape[:2]
                kernel_h, _ = weight.shape[2], weight.shape[3]
                stride = _get_attr(node, "strides", [1, 1])
                padding = _get_attr(node, "pads", [0, 0, 0, 0])
                output_padding = _get_attr(node, "output_padding", [0, 0])
                dilation = _get_attr(node, "dilations", [1, 1])
                groups = _get_attr(node, "group", 1)
                layer_info["type"] = "ConvTranspose"
                layer_info["in_channels"] = in_channels
                layer_info["out_channels"] = out_channels
                layer_info["kernel_size"] = kernel_h
                layer_info["stride"] = stride[0] if isinstance(stride, list) else stride
                layer_info["padding"] = padding[0] if isinstance(padding, list) else padding
                layer_info["output_padding"] = output_padding[0] if isinstance(output_padding, list) else output_padding
                layer_info["dilation"] = dilation[0] if isinstance(dilation, list) else dilation
                layer_info["groups"] = groups
                layer_info["bias"] = bias is not None
                params.append((f"{node.name}.weight", weight))
                if bias is not None:
                    params.append((f"{node.name}.bias", bias))

        elif op_type == "DepthToSpace":
            blocksize = _get_attr(node, "blocksize", 1)
            mode = _get_attr(node, "mode", "DCR")
            layer_info["type"] = "DepthToSpace"
            layer_info["blocksize"] = blocksize
            layer_info["mode"] = mode

        elif op_type == "SpaceToDepth":
            blocksize = _get_attr(node, "blocksize", 1)
            layer_info["type"] = "SpaceToDepth"
            layer_info["blocksize"] = blocksize

        else:
            logger.warning(f"Unsupported operator: {op_type}")
            layer_info["type"] = f"Unsupported_{op_type}"

        layers.append(layer_info)

    # Store full graph topology for DAG reconstruction
    graph = {
        "nodes": [{
            "name": node.name or f"{node.op_type}_{i}",
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        } for i, node in enumerate(model.graph.node)],
        "inputs": [{
            "name": inp.name,
            "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim] if inp.type.tensor_type.HasField("shape") else None
        } for inp in model.graph.input],
        "outputs": [{
            "name": out.name,
            "shape": [d.dim_value for d in out.type.tensor_type.shape.dim] if out.type.tensor_type.HasField("shape") else None
        } for out in model.graph.output],
    }

    # Write to file with unified format
    header = {
        "layers": layers,
        "graph": graph,
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
