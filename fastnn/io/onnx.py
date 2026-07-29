"""ONNX model importer for fastnn.

Converts ONNX models (.onnx) to ComputeGraph JSON (.fnn) format.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import onnx

__all__ = ["import_onnx", "import_onnx_to_compute_graph"]

logger = logging.getLogger(__name__)

ONNX_TO_IR_OP = {
    # Keep this adapter map limited to operations handled by the Rust converter.
    # Unmapped ONNX operations retain their original name and are rejected by the
    # authoritative Rust boundary instead of being silently approximated.
    "Relu": "Relu", "Sigmoid": "Sigmoid", "Tanh": "Tanh",
    "Add": "Add", "Sub": "Sub", "Mul": "Mul", "Div": "Div", "MatMul": "MatMul",
    "Conv": "Conv2d", "BatchNormalization": "BatchNormalization", "Reshape": "Reshape",
    "Transpose": "Transpose", "Gemm": "Gemm", "MaxPool": "MaxPool",
    "AveragePool": "AveragePool", "Softmax": "Softmax", "Concat": "Concat",
    "Flatten": "Flatten", "Slice": "Slice", "Pad": "Pad",
    "ReduceMean": "ReduceMean", "ReduceSum": "ReduceSum", "ReduceMax": "ReduceMax",
    "GlobalAveragePool": "GlobalAveragePool", "Constant": "Constant",
    "LeakyRelu": "LeakyRelu", "Elu": "Elu", "Clip": "Clip",
    "Dropout": "Identity", "InstanceNormalization": "InstanceNormalization",
    "Split": "Split", "Shape": "Shape", "Cast": "Cast",
    "Gather": "Gather", "GatherElements": "GatherElements",
    "Unsqueeze": "Unsqueeze", "Squeeze": "Squeeze",
    "Identity": "Identity", "Resize": "Resize", "Tile": "Tile",
    "Where": "Where", "CumSum": "CumSum",
    "LogSoftmax": "LogSoftmax", "Selu": "Selu",
    "HardSigmoid": "HardSigmoid", "HardSwish": "HardSwish",
    "LayerNormalization": "LayerNormalization", "ConvTranspose": "ConvTranspose",
    "TopK": "TopK", "ScatterND": "ScatterND",
    "Exp": "Exp", "Sqrt": "Sqrt", "Neg": "Neg", "Log": "Log", "Erf": "Erf",
    "Sign": "Sign", "Not": "Not",
    "Less": "Less", "Greater": "Greater", "Equal": "Equal",
    "Pow": "Pow", "Expand": "Expand",
    "GRU": "GRU", "LSTM": "LSTM", "Gelu": "Gelu", "Swish": "Swish",
    "QuantizeLinear": "QuantizeLinear", "DequantizeLinear": "DequantizeLinear",
    "QLinearMatMul": "QLinearMatMul", "QLinearConv": "QLinearConv",
    "RMSNormalization": "RMSNormalization",
    "ConstantOfShape": "ConstantOfShape", "Range": "Range",
}


def get_pooling_config(node, default_kernel=(2, 2)):
    kernel = _get_attr(node, "kernel_shape", list(default_kernel))
    stride = _get_attr(node, "strides", kernel)
    padding = _get_attr(node, "pads", [0, 0, 0, 0])
    return {
        "kernel_size": kernel[0] if isinstance(kernel, list) else kernel,
        "stride": stride[0] if isinstance(stride, list) else stride,
        "padding": padding[0] if isinstance(padding, list) else padding,
    }


def _get_attr(node: "onnx.NodeProto", name: str, default=None):
    import onnx
    from onnx import numpy_helper
    for attr in node.attribute:
        if attr.name == name:
            if attr.HasField("f"): return attr.f
            elif attr.HasField("i"): return attr.i
            elif attr.HasField("s"): return attr.s.decode("utf-8")
            elif attr.HasField("t"): return numpy_helper.to_array(attr.t)
            elif attr.ints: return list(attr.ints)
            elif attr.floats: return list(attr.floats)
    return default


def _extract_shape(value_info):
    shape, dtype = [], "F32"
    if value_info.type.tensor_type.HasField("shape"):
        for d in value_info.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(f"Known({d.dim_value})")
            elif d.HasField("dim_param") and d.dim_param:
                shape.append(f"Symbol({d.dim_param})")
            else:
                shape.append("Unknown")
    dtype = {1: "F32", 9: "BOOL", 7: "I64", 10: "F16", 11: "F64", 6: "I32"}.get(
        value_info.type.tensor_type.elem_type, "F32")
    return shape, dtype


def _build_vinfo(model) -> Dict[str, Tuple[List[str], str]]:
    vinfo = {}
    for src in (model.graph.value_info, model.graph.input, model.graph.output):
        for vi in src:
            vinfo[vi.name] = _extract_shape(vi)
    return vinfo


def _plain_dtype_name(array: np.ndarray, *, tensor_name: str) -> str:
    """Return the lossless plain-tensor dtype name supported by .fnn v3."""
    dtype = np.dtype(array.dtype)
    names = {
        np.dtype("float32"): "F32",
        np.dtype("int64"): "I64",
        np.dtype("int32"): "I32",
        np.dtype("bool"): "BOOL",
    }
    if dtype not in names:
        raise ValueError(
            f"ONNX initializer '{tensor_name}' uses unsupported dtype {dtype}; "
            "supported plain initializer dtypes are float32, int64, int32, and bool"
        )
    return names[dtype]


def _constant_param(array: np.ndarray, *, tensor_name: str) -> Dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "data": contiguous.tobytes(),
        "shape": list(contiguous.shape),
        "dtype": _plain_dtype_name(contiguous, tensor_name=tensor_name),
        "is_constant": True,
    }


_MAX_EXTERNAL_DATA_BYTES = 64 * 1024 * 1024 * 1024


def _validate_external_data(model, onnx_path: str) -> None:
    """Validate ONNX external tensor locations and their aggregate file budget."""
    model_dir = Path(onnx_path).resolve().parent
    external_files = set()
    for initializer in model.graph.initializer:
        if not initializer.external_data:
            continue
        metadata = {entry.key: entry.value for entry in initializer.external_data}
        location = metadata.get("location")
        if not location:
            raise ValueError(
                f"ONNX initializer '{initializer.name}' has external data without a location"
            )
        candidate = (model_dir / location).resolve()
        try:
            candidate.relative_to(model_dir)
        except ValueError as error:
            raise ValueError(
                f"ONNX initializer '{initializer.name}' external data escapes the model directory: {location!r}"
            ) from error
        if not candidate.is_file():
            raise ValueError(
                f"ONNX initializer '{initializer.name}' external data file is missing: {location!r}"
            )
        external_files.add(candidate)

    aggregate_size = sum(path.stat().st_size for path in external_files)
    if aggregate_size > _MAX_EXTERNAL_DATA_BYTES:
        raise ValueError(
            f"ONNX external data totals {aggregate_size} bytes, exceeding the "
            f"{_MAX_EXTERNAL_DATA_BYTES}-byte import budget"
        )


_OP_ATTRS = {
    "Conv": ["strides", "pads", "dilations", "group", "kernel_shape"],
    "Gemm": ["alpha", "transB"],
    "BatchNormalization": ["epsilon", "momentum"],
    "MaxPool": ["kernel_shape", "strides", "pads"],
    "AveragePool": ["kernel_shape", "strides", "pads"],
    "Softmax": ["axis"],
    "Concat": ["axis"],
    "Flatten": ["axis"],
    "Transpose": ["perm"],
    "LeakyRelu": ["alpha"],
    "Elu": ["alpha"],
    "Split": ["split", "axis"],
    "ReduceMean": ["axes", "keepdims"],
    "ReduceSum": ["axes", "keepdims"],
    "Pad": ["mode", "pads"],
    "Slice": ["starts", "ends", "axes", "steps"],
    "Cast": ["to"],
    "Gather": ["axis"],
    "Unsqueeze": ["axes"],
    "Squeeze": ["axes"],
    "Resize": ["mode", "coordinate_transformation_mode"],
    "DepthToSpace": ["blocksize", "mode"],
    "SpaceToDepth": ["blocksize"],
    "LogSoftmax": ["axis"],
    "Selu": ["alpha", "gamma"],
    "HardSigmoid": ["alpha", "beta"],
    "LayerNormalization": ["axis", "epsilon"],
    "ConvTranspose": ["strides", "pads", "output_padding", "dilations", "group", "kernel_shape"],
    "TopK": ["axis"],
    "GatherND": ["batch_dims"],
    "ScatterND": ["reduction"],
    "Compress": ["axis"],
    "CumSum": ["axis", "exclusive", "reverse"],
    "Einsum": ["equation"],
    "EyeLike": ["k"],
    "OneHot": ["axis"],
    "RandomNormal": ["mean", "scale", "shape", "seed"],
    "RandomUniform": ["low", "high", "shape", "seed"],
    "ConstantOfShape": ["value"],
    "Constant": ["value"],
    "GRU": ["hidden_size", "direction"],
    "LSTM": ["hidden_size", "direction"],
    "Attention": ["num_heads"],
    "MultiHeadAttention": ["num_heads"],
    "GroupQueryAttention": ["num_heads", "kv_num_heads"],
    "RMSNormalization": ["epsilon"],
    "SkipLayerNormalization": ["epsilon"],
    "LRN": ["size", "alpha", "beta"],
    "InstanceNormalization": ["epsilon"],
    "QuantizeLinear": ["axis"],
    "DequantizeLinear": ["axis"],
    "QLinearConv": ["strides", "pads", "dilations", "group", "kernel_shape"],
    "NonMaxSuppression": ["center_point_box"],
    "Clip": ["min", "max"],
}

_OP_PARAM_SLOTS = {
    "Conv": [("weight", 1), ("bias", 2)],
    "BatchNormalization": [("weight", 1), ("bias", 2), ("running_mean", 3), ("running_var", 4)],
    "InstanceNormalization": [("weight", 1), ("bias", 2)],
    "LayerNormalization": [("weight", 1), ("bias", 2)],
    "ConvTranspose": [("weight", 1), ("bias", 2)],
    "GRU": [("weight", 1), ("recurrent_weight", 2), ("bias", 3)],
    "LSTM": [("weight", 1), ("recurrent_weight", 2), ("bias", 3)],
    "QLinearMatMul": [("weight", 3)],
    "QLinearConv": [("weight", 3)],
}


def _extract_attrs(onnx_node) -> Dict[str, Any]:
    attrs = {}
    op = onnx_node.op_type
    for key in _OP_ATTRS.get(op, []):
        val = _get_attr(onnx_node, key, None)
        if val is not None:
            if isinstance(val, np.ndarray):
                attrs[key] = val.tolist()
            elif isinstance(val, list):
                attrs[key] = list(val)
            elif isinstance(val, (int, float, str, bool)):
                attrs[key] = val
    if op in ("Gemm",):
        if "alpha" not in attrs:
            attrs["alpha"] = 1.0
        if "transB" not in attrs:
            attrs["transB"] = 0
    if op in ("ReduceMean", "ReduceSum") and "keepdims" in attrs:
        attrs["keepdims"] = bool(attrs["keepdims"])
    if op in ("CumSum",):
        if "exclusive" in attrs:
            attrs["exclusive"] = bool(attrs["exclusive"])
        if "reverse" in attrs:
            attrs["reverse"] = bool(attrs["reverse"])
    if op == "ConstantOfShape" and "value" in attrs:
        v = attrs["value"]
        if isinstance(v, np.ndarray):
            attrs["value"] = float(v.flatten()[0])
        elif isinstance(v, (list, tuple)):
            attrs["value"] = float(np.asarray(v).flatten()[0])
        else:
            attrs["value"] = float(v)
    if op in ("MaxPool", "AveragePool"):
        cfg = get_pooling_config(onnx_node)
        attrs["kernel_size"] = cfg["kernel_size"]
        attrs["stride"] = cfg["stride"]
        attrs["padding"] = cfg["padding"]
        for k in ("kernel_shape", "strides", "pads"):
            attrs.pop(k, None)
    if op == "Conv":
        for rename in (("strides", "stride"), ("pads", "padding"), ("dilations", "dilation")):
            if rename[0] in attrs:
                v = attrs.pop(rename[0])
                attrs[rename[1]] = v[0] if isinstance(v, list) else v
    if op == "ConvTranspose":
        for rename in (("strides", "stride"), ("pads", "padding"), ("dilations", "dilation")):
            if rename[0] in attrs:
                v = attrs.pop(rename[0])
                attrs[rename[1]] = v[0] if isinstance(v, list) else v
        if "output_padding" in attrs:
            v = attrs["output_padding"]
            attrs["output_padding"] = v[0] if isinstance(v, list) else v
    if op == "Constant" and "value" in attrs:
        v = attrs.pop("value")
        attrs["dims"] = list(v.shape) if hasattr(v, "shape") else []
    return attrs


def _resolve_input_names(
    names: List[str], out_to_nid: Dict[str, int], gin_to_nid: Dict[str, int], init_names: set,
) -> List[str]:
    """Resolve input names to tensor names for the Rust converter.

    Returns the original ONNX tensor names (e.g., 'conv_out', 'X', 'W', 'B').
    """
    result = []
    for n in names:
        if n and (n in out_to_nid or n in gin_to_nid or n in init_names):
            result.append(n)
    return result


def _resolve_input_ids(
    names: List[str], out_to_nid: Dict[str, int], gin_to_nid: Dict[str, int], init_names: set,
) -> List[int]:
    """Resolve input names to node ids. Returns encoded ids.

    Encoded format: id | (slot << 20) for multi-output ops (e.g. Split).
    """
    result = []
    for n in names:
        if n and (n in out_to_nid or n in gin_to_nid):
            nid = out_to_nid[n] if n in out_to_nid else gin_to_nid[n]
            slot = 0
            # Extract slot from output name like Split_output_2
            if '_output_' in n:
                try:
                    slot = int(n.rsplit('_output_', 1)[1])
                except (ValueError, IndexError):
                    slot = 0
            result.append((slot << 20) | nid if slot else nid)
    return result


def import_onnx_to_compute_graph(onnx_path: str, config: Optional[Any] = None) -> Dict[str, Any]:
    import onnx
    import onnx.numpy_helper
    import onnx.shape_inference

    metadata_model = onnx.load(onnx_path, load_external_data=False)
    _validate_external_data(metadata_model, onnx_path)
    model = onnx.load(onnx_path, load_external_data=True)
    # Run ONNX shape inference to populate intermediate tensor shapes (value_info)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass  # Fall back to original model if inference fails
    init_map = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

    if config is not None:
        from fastnn.precision import PrecisionConfig as _PrecisionConfig
        if not isinstance(config, _PrecisionConfig):
            try:
                config = _PrecisionConfig.from_dict_spec(config) if isinstance(config, dict) else config
            except Exception:
                pass

    vinfo = _build_vinfo(model)
    nodes: List[Dict] = []
    params: Dict[str, Any] = {}
    nid = 0
    gin_to_nid: Dict[str, int] = {}
    gin_ids: List[int] = []

    for inp in model.graph.input:
        if inp.name in init_map:
            continue
        shape, dtype = vinfo.get(inp.name, ([], "F32"))
        nodes.append({"id": nid, "opcode": "Input", "inputs": [], "output_shape": {"shape": shape, "dtype": dtype}, "attrs": {}, "name": inp.name})
        gin_to_nid[inp.name] = nid
        gin_ids.append(nid)
        nid += 1

    for name, arr in init_map.items():
        params[name] = _constant_param(arr, tensor_name=name)

    out_to_nid: Dict[str, int] = {}
    onnx_nodes: List[Tuple[int, Any]] = []
    for onn in model.graph.node:
        oid = nid
        nid += 1
        onnx_nodes.append((oid, onn))
        for o in onn.output:
            out_to_nid[o] = oid

    init_names = set(init_map)

    for oid, onn in onnx_nodes:
        oname = onn.name or f"{onn.op_type}_{oid}"

        ir_op = ONNX_TO_IR_OP.get(onn.op_type, onn.op_type)
        attrs = _extract_attrs(onn)
        ins_ids = _resolve_input_ids(list(onn.input), out_to_nid, gin_to_nid, init_names)
        ins_names = _resolve_input_names(list(onn.input), out_to_nid, gin_to_nid, init_names)
        out_shape = vinfo.get(onn.output[0], ([], "F32")) if onn.output else ([], "F32")
        osd = {"shape": out_shape[0], "dtype": out_shape[1]}

        for suffix, idx in _OP_PARAM_SLOTS.get(onn.op_type, []):
            if idx < len(onn.input) and onn.input[idx] in init_map:
                arr = init_map[onn.input[idx]]
                param_name = f"{oname}.{suffix}"
                params[param_name] = _constant_param(arr, tensor_name=param_name)

        if onn.op_type == "Gemm":
            # Preserve Gemm as an ONNX operation. The Rust converter owns its
            # MatMul/alpha/bias decomposition and validates the complete input contract.
            nodes.append({"id": oid, "opcode": ir_op, "inputs": ins_names, "output_shape": osd, "attrs": attrs, "name": oname})

        elif onn.op_type in ("Conv", "ConvTranspose"):
            data_inputs = [n for n in ins_names if n not in init_names]
            conv_ins = list(data_inputs)
            for suffix, idx in _OP_PARAM_SLOTS.get(onn.op_type, []):
                if idx < len(onn.input) and onn.input[idx] in init_map:
                    param_name = f"{oname}.{suffix}"
                    conv_ins.append(param_name)
            nodes.append({"id": oid, "opcode": ir_op, "inputs": conv_ins, "output_shape": osd, "attrs": attrs, "name": oname})

        elif onn.op_type in ("Add", "Sub"):
            # Preserve elementwise semantics and the original initializer input.
            nodes.append({"id": oid, "opcode": ir_op, "inputs": ins_names, "output_shape": osd, "attrs": attrs, "name": oname})

        elif onn.op_type == "Constant":
            value = _get_attr(onn, "value", None)
            if value is not None:
                if isinstance(value, np.ndarray):
                    const_arr = value
                else:
                    try:
                        const_arr = onnx.numpy_helper.to_array(value)
                    except Exception:
                        const_arr = None
                if const_arr is not None:
                    param_name = f"{oname}.value"
                    params[param_name] = _constant_param(const_arr, tensor_name=param_name)
            nodes.append({"id": oid, "opcode": ir_op, "inputs": ins_names, "output_shape": osd, "attrs": attrs, "name": oname})

        elif onn.op_type == "Reshape":
            shape_arr = init_map.get(onn.input[1])
            if shape_arr is not None:
                attrs["shape"] = shape_arr.tolist()
            nodes.append({"id": oid, "opcode": ir_op, "inputs": ins_names, "output_shape": osd, "attrs": attrs, "name": oname})

        else:
            nodes.append({"id": oid, "opcode": ir_op, "inputs": ins_names, "output_shape": osd, "attrs": attrs, "name": oname})

    nid_map = {n["id"]: n for n in nodes}
    # Map from node ID to its output tensor names (from ONNX)
    node_output_names: Dict[int, List[str]] = {}
    for onn in model.graph.node:
        if onn.output:
            # Find the node with this output
            for o in onn.output:
                if o in out_to_nid:
                    # Store output names for the producer node
                    prod_id = out_to_nid[o]
                    if prod_id not in node_output_names:
                        node_output_names[prod_id] = []
                    node_output_names[prod_id].append(o)
    
    # Also include intermediate tensor names that were explicitly set on nodes
    # (e.g., MatMul outputs for Gemm decomposition)
    for node in nodes:
        if "outputs" in node and node["outputs"]:
            nid = node["id"]
            if nid not in node_output_names:
                node_output_names[nid] = []
            for out_name in node["outputs"]:
                if out_name not in node_output_names[nid]:
                    node_output_names[nid].append(out_name)
    
    # Set initial outputs based on ONNX tensor names
    for node in nodes:
        nid = node["id"]
        if nid in node_output_names:
            node["outputs"] = node_output_names[nid]
        else:
            node["outputs"] = []

    # For graph outputs, ensure the output node has the graph output tensor name
    out_name_to_nid = {}
    out_ids = []
    for out in model.graph.output:
        if out.name in out_to_nid:
            nid = out_to_nid[out.name]
            out_ids.append(nid)
            out_name_to_nid[nid] = out.name
        elif out.name in gin_to_nid:
            nid = gin_to_nid[out.name]
            out_ids.append(nid)
            out_name_to_nid[nid] = out.name

    # Add graph output names to the corresponding nodes' outputs
    for nid, name in out_name_to_nid.items():
        if nid in nid_map:
            # The output node's outputs should include the graph output tensor name
            # (overwrite any previously inferred names for final output)
            nid_map[nid]["outputs"] = [name]

    return {"nodes": nodes, "inputs": gin_ids, "outputs": out_ids, "params": params, "out_to_nid": out_to_nid}


def _convert_bytes(obj):
    if isinstance(obj, dict):
        return {k: _convert_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_bytes(v) for v in obj]
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('ascii')
    return obj


def _compute_graph_to_layers(cg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert ComputeGraph nodes to high-level layer format expected by tests."""
    nodes = cg.get("nodes", [])
    params = cg.get("params", {})
    nid_map = {n["id"]: n for n in nodes}

    layers = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        opcode = node.get("opcode", "")
        if opcode == "Input":
            i += 1
            continue  # Skip input nodes

        # Build layer dict
        attrs = node.get("attrs", {})
        layer = {"type": opcode, "name": node.get("name", "")}

        # Add attributes as layer properties
        for key, value in attrs.items():
            layer[key] = value

        # Add shape info if available
        output_shape = node.get("output_shape", {})
        if output_shape.get("shape"):
            layer["output_shape"] = output_shape["shape"]

        # Handle special cases
        if opcode == "Conv2d":
            layer["type"] = "Conv"
            # Extract kernel_size, stride, padding from attrs if not present
            if "kernel_size" not in layer and "kernel_shape" in attrs:
                ks = attrs["kernel_shape"]
                layer["kernel_size"] = ks[0] if isinstance(ks, list) else ks
            if "stride" not in layer and "strides" in attrs:
                s = attrs["strides"]
                layer["stride"] = s[0] if isinstance(s, list) else s
            if "padding" not in layer and "pads" in attrs:
                p = attrs["pads"]
                layer["padding"] = p[0] if isinstance(p, list) else p
            # Add in_channels/out_channels for test compatibility
            if "in_channels" not in layer and "group" in attrs:
                layer["in_channels"] = attrs.get("group", 1)
            out_shape = node.get("output_shape", {}).get("shape", [])
            if out_shape and len(out_shape) >= 2:
                layer["out_channels"] = int(out_shape[-1].replace("Known(", "").replace(")", ""))
        elif opcode == "ConvTranspose":
            layer["type"] = "ConvTranspose"
            # Add in_channels/out_channels for test compatibility
            if "kernel_size" not in layer and "kernel_shape" in attrs:
                ks = attrs["kernel_shape"]
                layer["kernel_size"] = ks[0] if isinstance(ks, list) else ks
            # Get in_channels/out_channels from weight tensor shape
            # Weight is stored as param with name pattern: {node_name}.weight
            weight_name = f"{node.get('name', '')}.weight"
            if weight_name in params:
                weight_shape = params[weight_name].get("shape", [])
                if len(weight_shape) >= 2:
                    # ConvTranspose weight shape: [out_channels, in_channels, kH, kW]
                    layer["out_channels"] = weight_shape[0]
                    layer["in_channels"] = weight_shape[1]
            # Check for bias
            bias_name = f"{node.get('name', '')}.bias"
            if bias_name in params:
                layer["bias"] = True
            # Fallback: get out_channels from output shape
            out_shape = node.get("output_shape", {}).get("shape", [])
            if "out_channels" not in layer and out_shape and len(out_shape) >= 2:
                layer["out_channels"] = int(out_shape[-1].replace("Known(", "").replace(")", ""))
        elif opcode == "Gemm":
            # Gemm remains a single executable ONNX op; expose it as a Linear
            # layer only in this compatibility summary.
            layer["type"] = "Linear"
            inputs = node.get("inputs", [])
            if len(inputs) >= 2 and inputs[1] in params:
                weight_shape = params[inputs[1]].get("shape", [])
                if len(weight_shape) == 2:
                    trans_b = bool(attrs.get("transB", 0))
                    if trans_b:
                        layer["in_features"] = weight_shape[1]
                        layer["out_features"] = weight_shape[0]
                    else:
                        layer["in_features"] = weight_shape[0]
                        layer["out_features"] = weight_shape[1]
        elif opcode == "MatMul":
            # Check if next node is BiasAdd for the same op
            if i + 1 < len(nodes):
                next_node = nodes[i + 1]
                # The BiasAdd should have this MatMul's output tensor as input
                matmul_outputs = node.get("outputs", [])
                if matmul_outputs and next_node.get("opcode") == "BiasAdd":
                    bias_inputs = next_node.get("inputs", [])
                    # Check if any of the MatMul's outputs is an input to BiasAdd
                    if any(out in bias_inputs for out in matmul_outputs):
                        # Merge MatMul + BiasAdd into Linear layer
                        layer["type"] = "Linear"
                    # Get out_features from output shape
                    out_shape = node.get("output_shape", {}).get("shape", [])
                    if out_shape and len(out_shape) >= 2:
                        layer["out_features"] = int(out_shape[-1].replace("Known(", "").replace(")", ""))
                    # Get in_features from input shape - map tensor names to producer node IDs or params
                    inputs = node.get("inputs", [])
                    if inputs and "out_to_nid" in cg:
                        out_to_nid = cg["out_to_nid"]
                        for inp_name in inputs:
                            producer_nid = out_to_nid.get(inp_name)
                            if producer_nid is not None:
                                producer = nid_map.get(producer_nid)
                                if producer:
                                    prod_out_shape = producer.get("output_shape", {}).get("shape", [])
                                    if prod_out_shape and len(prod_out_shape) >= 2:
                                        layer["in_features"] = int(prod_out_shape[-1].replace("Known(", "").replace(")", ""))
                                        break
                            # Also check if it's an initializer (weight parameter)
                            elif inp_name in params:
                                weight_shape = params[inp_name].get("shape", [])
                                if len(weight_shape) >= 2:
                                    # Weight shape is [out_features, in_features] for transB=1
                                    # or [in_features, out_features] for transB=0
                                    # Default assumption: second dim is in_features
                                    layer["in_features"] = weight_shape[1] if weight_shape[1] else weight_shape[0]
                                    break
                    # Transfer bias info
                    bias_attrs = next_node.get("attrs", {})
                    for key, value in bias_attrs.items():
                        if key not in layer:
                            layer[key] = value
                    i += 1  # Skip the BiasAdd node
                else:
                    layer["type"] = "Linear"
            else:
                layer["type"] = "Linear"
        elif opcode == "BatchNormalization":
            layer["type"] = "BatchNorm2d"
        elif opcode == "InstanceNormalization":
            layer["type"] = "InstanceNormalization"
            # Add num_features from scale/bias weight shape
            scale_name = f"{node.get('name', '')}.weight"
            if scale_name in params:
                scale_shape = params[scale_name].get("shape", [])
                if scale_shape:
                    layer["num_features"] = scale_shape[0]
            # Add eps from attrs (Rust expects 'eps' but ONNX uses 'epsilon')
            if "epsilon" in attrs:
                layer["eps"] = attrs["epsilon"]
        elif opcode == "RMSNormalization":
            layer["type"] = "RMSNormalization"
            # Add eps from attrs
            if "epsilon" in attrs:
                layer["eps"] = attrs["epsilon"]
        elif opcode == "Cast":
            layer["type"] = "CastOp"
        elif opcode == "Sub":
            layer["type"] = "ElementwiseSub"
        elif opcode in ("Add", "BiasAdd"):
            # Check if it's elementwise or bias add
            pass

        layers.append(layer)
        i += 1

    return layers


def import_onnx(onnx_path: str, fnn_path: str, config: Optional[Any] = None) -> Dict[str, Any]:
    cg = import_onnx_to_compute_graph(onnx_path, config=config)
    cg = _convert_bytes(cg)

    # Convert to layers format for return value and .fnn header
    layers = _compute_graph_to_layers(cg)
    num_params = sum(1 for p in cg.get("params", {}).values() if p.get("is_constant", False))

    # Build result dict with layers format
    result = {
        "layers": layers,
        "parameters": num_params,
        "graph": {
            "nodes": cg.get("nodes", []),
            "inputs": cg.get("inputs", []),
            "outputs": cg.get("outputs", []),
        },
        "params": cg.get("params", {}),
    }

    # Write .fnn file in proper binary format using write_fnn_file_v3
    from fastnn.io import (
        write_fnn_file_v3,
        MODEL_MAGIC,
        DTYPE_F32,
        DTYPE_I64,
        DTYPE_I32,
        DTYPE_BOOL,
    )

    # Build header with layers format for test compatibility
    fnn_header = {
        "layers": [{"type": l["type"], "name": l.get("name", "")} for l in layers],
        "graph": {
            "nodes": cg.get("nodes", []),
            "inputs": cg.get("inputs", []),
            "outputs": cg.get("outputs", []),
        },
        "parameter_count": num_params,
    }

    # Prepare parameters for v3 format (name, data, dtype, scales, zeros)
    params_v3 = []
    for name, p in cg.get("params", {}).items():
        if p.get("is_constant", False):
            data = p.get("data")
            shape = p.get("shape", [])
            if isinstance(data, str):
                # base64 encoded bytes
                import base64
                data = base64.b64decode(data)
            dtype_name = p.get("dtype", "F32")
            dtype_tag = {
                "F32": DTYPE_F32,
                "I64": DTYPE_I64,
                "I32": DTYPE_I32,
                "BOOL": DTYPE_BOOL,
            }.get(dtype_name)
            if dtype_tag is None:
                raise ValueError(
                    f"Parameter '{name}' has unsupported .fnn scalar dtype {dtype_name!r}"
                )
            params_v3.append((name, data, dtype_tag, [], [], shape))

    # Build graph nodes with op_type alias for test compatibility
    # Use the original cg nodes with numeric IDs for fnn header
    graph_nodes = cg.get("nodes", [])
    graph_nodes_with_op_type = []
    
    # Create mapping from node ID to output tensor name
    nid_to_name = {}
    for node in graph_nodes:
        nid = node.get("id")
        name = node.get("name", "")
        if nid is not None and name:
            nid_to_name[nid] = name
    
    for node in graph_nodes:
        n = dict(node)
        if "opcode" in n and "op_type" not in n:
            n["op_type"] = n["opcode"]
        # Convert numeric inputs/outputs to tensor names
        for key in ("inputs", "outputs"):
            val = n.get(key)
            if isinstance(val, list):
                # Convert list of numeric IDs to tensor names
                new_val = []
                for inp_id in val:
                    # Handle both int and encoded int (with slot)
                    if isinstance(inp_id, int):
                        producer_nid = inp_id & ((1 << 20) - 1)
                        # Add slot prefix if present (for multi-output nodes like Split)
                        slot = inp_id >> 20
                        name = nid_to_name.get(producer_nid, f"node_{producer_nid}")
                        if slot > 0:
                            name = f"{name}_output_{slot}"
                        new_val.append(name)
                    else:
                        new_val.append(str(inp_id))
                n[key] = ",".join(new_val)
            elif isinstance(val, int):
                producer_nid = val & ((1 << 20) - 1)
                slot = val >> 20
                name = nid_to_name.get(producer_nid, f"node_{producer_nid}")
                if slot > 0:
                    name = f"{name}_output_{slot}"
                n[key] = name
        graph_nodes_with_op_type.append(n)

    # Update fnn_header with the fixed graph nodes
    fnn_header["graph"]["nodes"] = graph_nodes_with_op_type

    # Fix inputs/outputs to include name info for test compatibility
    # Map input node IDs to their names
    input_nodes_info = []
    for inp_id in cg.get("inputs", []):
        # Find the Input node with this ID
        for node in graph_nodes:
            if node.get("id") == inp_id:
                input_nodes_info.append({"name": node.get("name", ""), "id": inp_id})
                break
        else:
            input_nodes_info.append({"name": f"input_{inp_id}", "id": inp_id})
    
    # Map output node IDs to their names
    output_nodes_info = []
    for out_id in cg.get("outputs", []):
        for node in graph_nodes:
            if node.get("id") == out_id:
                # Use the output tensor name from the node's outputs field
                # (set by import_onnx_to_compute_graph to the graph output tensor name)
                outputs = node.get("outputs", [])
                if outputs and isinstance(outputs, list) and len(outputs) > 0:
                    tensor_name = outputs[0]
                else:
                    tensor_name = node.get("name", "")
                output_nodes_info.append({"name": tensor_name, "id": out_id})
                break
        else:
            output_nodes_info.append({"name": f"output_{out_id}", "id": out_id})

    fnn_header["graph"]["inputs"] = input_nodes_info
    fnn_header["graph"]["outputs"] = output_nodes_info

    with open(fnn_path, "wb") as f:
        write_fnn_file_v3(f, fnn_header, params_v3, magic=MODEL_MAGIC, version=3)

    logger.info("Exported %s: %d layers, %d params to %s", onnx_path, len(layers), num_params, fnn_path)
    return result
