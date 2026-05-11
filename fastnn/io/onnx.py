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

from fastnn.io import (
    MODEL_MAGIC, MODEL_VERSION, _pack_u32, _pack_u64, _pack_u8,
    write_tensor, write_tensor_v3, write_fnn_file, write_fnn_file_v3,
    DTYPE_F32, DTYPE_F16, DTYPE_U8, DTYPE_U4,
)

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
    from onnx import numpy_helper
    for attr in node.attribute:
        if attr.name == name:
            if attr.HasField("f"):
                return attr.f
            elif attr.HasField("i"):
                return attr.i
            elif attr.HasField("s"):
                return attr.s.decode("utf-8")
            elif attr.HasField("t"):
                # Tensor attribute: return as numpy array
                return numpy_helper.to_array(attr.t)
            elif attr.ints:
                return list(attr.ints)
            elif attr.floats:
                return list(attr.floats)
    return default


def import_onnx(
    onnx_path: str,
    fnn_path: str,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Import an ONNX model and save it in fastnn format.

    Args:
        onnx_path: Path to .onnx file
        fnn_path: Path to output .fnn file
        config: Optional PrecisionConfig for quantized import.
            When set, weight parameters matching the config are quantized.
            Non-weight params (bias, running_mean/var, constants) stay F32.

    Returns:
        Dictionary with model info (layers, input_shape, output_shape)
    """
    import onnx
    import onnx.numpy_helper

    model = onnx.load(onnx_path)
    
    # If config is given, resolve the precision module
    if config is not None:
        from fastnn.precision import Precision, Quantizer, PrecisionConfig as _PrecisionConfig
        if not isinstance(config, _PrecisionConfig):
            try:
                config = _PrecisionConfig.from_dict_spec(config) if isinstance(config, dict) else config
            except Exception:
                pass
    
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

    # Add all initializers as params (including scalar constants used by Gather, Range, etc.)
    for init_name, init_arr in initializer_map.items():
        params.append((init_name, init_arr))

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
            pool_config = get_pooling_config(node)
            layer_info["kernel_size"] = pool_config["kernel_size"]
            layer_info["stride"] = pool_config["stride"]
            layer_info["padding"] = pool_config["padding"]

        elif op_type == "AveragePool":
            pool_config = get_pooling_config(node)
            layer_info["type"] = "AvgPool"
            layer_info["kernel_size"] = pool_config["kernel_size"]
            layer_info["stride"] = pool_config["stride"]
            layer_info["padding"] = pool_config["padding"]

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
            pads = _get_attr(node, "pads", None)
            if pads is not None:
                layer_info["pads"] = list(pads)
            elif len(node.input) > 1 and node.input[1] in initializer_map:
                layer_info["pads"] = initializer_map[node.input[1]].flatten().tolist()

        elif op_type == "Slice":
            layer_info["type"] = "SliceOp"

        elif op_type == "Where":
            layer_info["type"] = "WhereOp"

        elif op_type == "NonZero":
            layer_info["type"] = "NonZeroOp"

        elif op_type == "Unique":
            layer_info["type"] = "UniqueOp"

        elif op_type == "Tril":
            layer_info["type"] = "TrilOp"

        elif op_type == "Triu":
            layer_info["type"] = "TriuOp"

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

        elif op_type == "Expand":
            layer_info["type"] = "Expand"

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

        elif op_type == "Compress":
            axis = _get_attr(node, "axis", None)
            layer_info["type"] = "Compress"
            if axis is not None:
                layer_info["axis"] = axis

        elif op_type == "CumSum":
            axis = _get_attr(node, "axis", None)
            exclusive = _get_attr(node, "exclusive", 0)
            reverse = _get_attr(node, "reverse", 0)
            layer_info["type"] = "CumSum"
            if axis is not None:
                layer_info["axis"] = axis if not isinstance(axis, list) else axis[0]
            layer_info["exclusive"] = bool(exclusive)
            layer_info["reverse"] = bool(reverse)

        elif op_type == "Einsum":
            equation = _get_attr(node, "equation", "")
            layer_info["type"] = "Einsum"
            layer_info["equation"] = equation

        elif op_type == "EyeLike":
            k = _get_attr(node, "k", 0)
            layer_info["type"] = "EyeLike"
            layer_info["k"] = k

        elif op_type == "OneHot":
            axis = _get_attr(node, "axis", -1)
            layer_info["type"] = "OneHot"
            layer_info["axis"] = axis

        elif op_type == "RandomNormal":
            mean = _get_attr(node, "mean", 0.0)
            scale = _get_attr(node, "scale", 1.0)
            shape = _get_attr(node, "shape", None)
            seed = _get_attr(node, "seed", None)
            layer_info["type"] = "RandomNormal"
            layer_info["mean"] = mean
            layer_info["scale"] = scale
            if shape is not None:
                layer_info["shape"] = list(shape)
            if seed is not None:
                layer_info["seed"] = seed

        elif op_type == "RandomUniform":
            low = _get_attr(node, "low", 0.0)
            high = _get_attr(node, "high", 1.0)
            shape = _get_attr(node, "shape", None)
            seed = _get_attr(node, "seed", None)
            layer_info["type"] = "RandomUniform"
            layer_info["low"] = low
            layer_info["high"] = high
            if shape is not None:
                layer_info["shape"] = list(shape)
            if seed is not None:
                layer_info["seed"] = seed

        elif op_type == "Range":
            layer_info["type"] = "RangeOp"

        elif op_type == "Attention":
            num_heads = _get_attr(node, "num_heads", 12)
            layer_info["type"] = "Attention"
            layer_info["num_heads"] = num_heads

        elif op_type == "BiasGelu":
            layer_info["type"] = "BiasGelu"

        elif op_type == "ConstantOfShape":
            value = _get_attr(node, "value", None)
            layer_info["type"] = "ConstantOfShape"
            if value is not None:
                const_tensor = value if isinstance(value, np.ndarray) else onnx.numpy_helper.to_array(value)
                layer_info["value"] = const_tensor.flatten()[0].item() if hasattr(const_tensor, 'flatten') else float(const_tensor)
                layer_info["dims"] = list(const_tensor.shape)

        elif op_type == "DequantizeLinear":
            axis = _get_attr(node, "axis", 1)
            layer_info["type"] = "DequantizeLinear"
            layer_info["axis"] = axis

        elif op_type == "EmbedLayerNormalization":
            layer_info["type"] = "EmbedLayerNormalization"

        elif op_type == "FastGelu":
            layer_info["type"] = "FastGelu"

        elif op_type == "GatherND":
            batch_dims = _get_attr(node, "batch_dims", 0)
            layer_info["type"] = "GatherND"
            layer_info["batch_dims"] = batch_dims

        elif op_type == "Gelu":
            layer_info["type"] = "Gelu"

        elif op_type == "GRU":
            hidden_size = _get_attr(node, "hidden_size", None)
            direction = _get_attr(node, "direction", "forward")
            layer_info["type"] = "GRU"
            if hidden_size is not None:
                layer_info["hidden_size"] = hidden_size
            layer_info["direction"] = direction

        elif op_type == "GroupQueryAttention":
            num_heads = _get_attr(node, "num_heads", 32)
            kv_num_heads = _get_attr(node, "kv_num_heads", 8)
            layer_info["type"] = "GroupQueryAttention"
            layer_info["num_heads"] = num_heads
            layer_info["kv_num_heads"] = kv_num_heads

        elif op_type == "LSTM":
            hidden_size = _get_attr(node, "hidden_size", None)
            direction = _get_attr(node, "direction", "forward")
            layer_info["type"] = "LSTM"
            if hidden_size is not None:
                layer_info["hidden_size"] = hidden_size
            layer_info["direction"] = direction

        elif op_type == "MultiHeadAttention":
            num_heads = _get_attr(node, "num_heads", 12)
            layer_info["type"] = "MultiHeadAttention"
            layer_info["num_heads"] = num_heads

        elif op_type == "QuantizeLinear":
            axis = _get_attr(node, "axis", 1)
            layer_info["type"] = "QuantizeLinear"
            layer_info["axis"] = axis

        elif op_type == "RMSNormalization":
            epsilon = _get_attr(node, "epsilon", 1e-5)
            layer_info["type"] = "RMSNorm"
            layer_info["eps"] = epsilon

        elif op_type == "RotaryEmbedding":
            layer_info["type"] = "RotaryEmbedding"

        elif op_type == "ScatterND":
            reduction = _get_attr(node, "reduction", "none")
            layer_info["type"] = "ScatterND"
            layer_info["reduction"] = reduction

        elif op_type == "SkipLayerNormalization":
            epsilon = _get_attr(node, "epsilon", 1e-5)
            layer_info["type"] = "SkipLayerNorm"
            layer_info["eps"] = epsilon

        elif op_type == "Swish":
            layer_info["type"] = "Swish"

        else:
            logger.warning(f"Unsupported operator: {op_type}")
            layer_info["type"] = f"Unsupported_{op_type}"

        layers.append(layer_info)

    # Store full graph topology for DAG reconstruction
    # Build a name->layer_info lookup so we can merge parsed attributes into graph nodes
    layer_by_name = {li["name"]: li for li in layers}
    graph_nodes = []
    for i, node in enumerate(model.graph.node):
        node_name = node.name or f"{node.op_type}_{i}"
        gn = {
            "name": node_name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        # Merge parsed attributes from layer_info (excluding name/type to avoid collision)
        layer_info = layer_by_name.get(node_name)
        if layer_info is not None:
            for k, v in layer_info.items():
                if k not in ("name", "type") and v is not None:
                    gn[k] = v
        graph_nodes.append(gn)

    # Fold QuantizeLinear/DequantizeLinear patterns
    graph_nodes, layers, params = _fold_qdq_patterns(
        graph_nodes, layers, params, initializer_map
    )

    graph = {
        "nodes": graph_nodes,
        "inputs": [{
            "name": inp.name,
            "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim] if inp.type.tensor_type.HasField("shape") else None
        } for inp in model.graph.input],
        "outputs": [{
            "name": out.name,
            "shape": [d.dim_value for d in out.type.tensor_type.shape.dim] if out.type.tensor_type.HasField("shape") else None
        } for out in model.graph.output],
    }

    # Write to file with unified format (v3 if quantized, v2 otherwise)
    header = {
        "layers": layers,
        "graph": graph,
        "total_parameters": len(params),
    }

    is_quantized = config is not None and config.default.should_quantize()

    if is_quantized:
        # v3 format: quantize weights per config, non-weights stay F32
        params_v3 = []
        for name, arr in params:
            quantizer = config.get_quantizer(name)
            is_weight = any(name.endswith(s) for s in [".weight", ".gamma", ".beta"])

            if is_weight and quantizer.should_quantize():
                # Quantize this weight
                scales, zeros = quantizer.quantize(arr)
                use_per_channel = quantizer.scheme == "per_channel" and len(scales) > 1

                if use_per_channel:
                    # Use per-channel quantization via Rust bindings
                    from fastnn import PackedTensor4, PackedTensor8, PackedTensor16
                    cls_map = {
                        4: PackedTensor4,
                        8: PackedTensor8,
                        16: PackedTensor16,
                    }
                    bit_width = quantizer.precision.bit_width
                    packed_cls = cls_map[bit_width]
                    shape = list(arr.shape)
                    packed = packed_cls.from_f32_per_channel(
                        arr.flatten().tolist(), shape
                    )
                    params_v3.append((
                        name,
                        bytes(packed.to_bytes()),
                        {
                            4: DTYPE_U4,
                            8: DTYPE_U8,
                            16: DTYPE_F16,
                        }[bit_width],
                        packed.scales(),
                        packed.zeros(),
                    ))
                else:
                    # Per-tensor quantization
                    from fastnn import PackedTensor4, PackedTensor8, PackedTensor16
                    cls_map = {
                        4: PackedTensor4,
                        8: PackedTensor8,
                        16: PackedTensor16,
                    }
                    bit_width = quantizer.precision.bit_width
                    packed_cls = cls_map[bit_width]
                    shape = list(arr.shape)
                    s = float(scales[0]) if len(scales) > 0 else 1.0
                    z = float(zeros[0]) if len(zeros) > 0 else 0.0
                    packed = packed_cls(arr.flatten().tolist(), shape, s, z)
                    params_v3.append((
                        name,
                        bytes(packed.to_bytes()),
                        {
                            4: DTYPE_U4,
                            8: DTYPE_U8,
                            16: DTYPE_F16,
                        }[bit_width],
                        packed.scales(),
                        packed.zeros(),
                    ))
            else:
                # Store as-is (F32)
                params_v3.append((name, arr, DTYPE_F32, [], []))

        header["precision"] = config.to_dict()
        with open(fnn_path, "wb") as f:
            write_fnn_file_v3(f, header, params_v3, version=3)
    else:
        # v2 format: all F32 (non-quantized ONNX import stays v2 for backward compat)
        with open(fnn_path, "wb") as f:
            write_fnn_file(f, header, params, version=2)

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
        "graph": graph,
        "parameters": len(params),
        "input_shape": input_shape,
        "output_shape": output_shape,
    }
    return result


def _fold_qdq_patterns(
    graph_nodes: list,
    layers: list,
    params: list,
    initializer_map: dict,
):
    """Fold QuantizeLinear/DequantizeLinear patterns in the graph.

    Detects Q → single_op → DQ patterns and folds them by removing
    the Q/DQ nodes and optionally quantizing the op's weights.
    Also handles standalone Q/DQ on weight initializers.

    Returns:
        Tuple of (modified_graph_nodes, modified_layers, modified_params).
    """
    if not graph_nodes:
        return graph_nodes, layers, params

    # Build consumer map: output_name -> list of consuming nodes
    output_to_consumers = {}
    for node in graph_nodes:
        for inp in node.get("inputs", []):
            if inp not in output_to_consumers:
                output_to_consumers[inp] = []
            output_to_consumers[inp].append(node)

    # Build output to producer map
    output_to_producer = {}
    for node in graph_nodes:
        for out in node.get("outputs", []):
            output_to_producer[out] = node

    nodes_to_remove = set()
    folded_count = 0

    # Pass 1: Detect Q → single_op → DQ patterns
    for node in list(graph_nodes):
        if node["op_type"] != "QuantizeLinear":
            continue
        q_name = node["name"]
        if q_name in nodes_to_remove:
            continue

        q_output = node["outputs"][0]
        consumers = output_to_consumers.get(q_output, [])

        if len(consumers) != 1:
            continue

        mid_node = consumers[0]
        if mid_node["op_type"] in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if mid_node["name"] in nodes_to_remove:
            continue

        mid_output = mid_node["outputs"][0]
        mid_consumers = output_to_consumers.get(mid_output, [])

        if len(mid_consumers) != 1:
            continue

        dq_node = mid_consumers[0]
        if dq_node["op_type"] != "DequantizeLinear":
            continue
        if dq_node["name"] in nodes_to_remove:
            continue

        # Found Q → mid_node → DQ pattern
        q_input = node["inputs"][0]

        # Extract scale/zero_point from initializers
        q_scale = 1.0
        q_zp = 0.0
        dq_scale = 1.0
        dq_zp = 0.0
        if len(node["inputs"]) > 1 and node["inputs"][1] in initializer_map:
            q_scale = float(initializer_map[node["inputs"][1]].flatten()[0])
        if len(node["inputs"]) > 2 and node["inputs"][2] in initializer_map:
            q_zp = float(initializer_map[node["inputs"][2]].flatten()[0])
        if len(dq_node["inputs"]) > 1 and dq_node["inputs"][1] in initializer_map:
            dq_scale = float(initializer_map[dq_node["inputs"][1]].flatten()[0])
        if len(dq_node["inputs"]) > 2 and dq_node["inputs"][2] in initializer_map:
            dq_zp = float(initializer_map[dq_node["inputs"][2]].flatten()[0])

        # Rewire: mid_node takes Q's input and its output goes to DQ's consumers
        mid_node["inputs"][0] = q_input
        mid_node["outputs"] = list(dq_node["outputs"])

        # Store Q/DQ metadata on the mid node for weight quantization
        mid_node["qdq_scale"] = q_scale
        mid_node["qdq_zp"] = q_zp
        mid_node["qdq_dq_scale"] = dq_scale
        mid_node["qdq_dq_zp"] = dq_zp

        nodes_to_remove.add(q_name)
        nodes_to_remove.add(dq_node["name"])
        folded_count += 1

    # Pass 2: Handle standalone QuantizeLinear on weight initializers
    for node in list(graph_nodes):
        if node["op_type"] != "QuantizeLinear":
            continue
        if node["name"] in nodes_to_remove:
            continue

        q_input = node["inputs"][0]
        if q_input not in initializer_map:
            continue

        # This Q applies directly to a weight initializer
        weight_arr = initializer_map[q_input]
        q_scale = 1.0
        q_zp = 0.0
        if len(node["inputs"]) > 1 and node["inputs"][1] in initializer_map:
            q_scale = float(initializer_map[node["inputs"][1]].flatten()[0])
        if len(node["inputs"]) > 2 and node["inputs"][2] in initializer_map:
            q_zp = float(initializer_map[node["inputs"][2]].flatten()[0])

        # Store as packed U8 with given scales
        quantized = np.round(weight_arr / q_scale + q_zp).astype(np.int8)
        params.append((q_input, quantized.astype(np.uint8)))
        nodes_to_remove.add(node["name"])

    # Pass 3: Handle standalone DequantizeLinear on weight initializers
    for node in list(graph_nodes):
        if node["op_type"] != "DequantizeLinear":
            continue
        if node["name"] in nodes_to_remove:
            continue

        dq_input = node["inputs"][0]
        if dq_input not in initializer_map:
            continue

        dq_scale = 1.0
        dq_zp = 0.0
        if len(node["inputs"]) > 1 and node["inputs"][1] in initializer_map:
            dq_scale = float(initializer_map[node["inputs"][1]].flatten()[0])
        if len(node["inputs"]) > 2 and node["inputs"][2] in initializer_map:
            dq_zp = float(initializer_map[node["inputs"][2]].flatten()[0])

        # Dequantize back to f32
        dq_data = initializer_map[dq_input]
        dequantized = (dq_data.astype(np.float32) - dq_zp) * dq_scale
        params.append((dq_input, dequantized))
        nodes_to_remove.add(node["name"])

    # Remove folded nodes from graph_nodes
    if nodes_to_remove:
        remaining_nodes = [n for n in graph_nodes if n["name"] not in nodes_to_remove]
        remaining_layers = [l for l in layers if l["name"] not in nodes_to_remove]

        if folded_count > 0:
            logger.info("Q/DQ folding: folded %d Q→Op→DQ pattern(s), removed %d total Q/DQ nodes", folded_count, len(nodes_to_remove))

        return remaining_nodes, remaining_layers, params

    return graph_nodes, layers, params
