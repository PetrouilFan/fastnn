"""ONNX model importer for fastnn.

Converts ONNX models (.onnx) to ComputeGraph JSON (.fnn) format.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import onnx

__all__ = ["import_onnx", "import_onnx_to_compute_graph"]

logger = logging.getLogger(__name__)

ONNX_TO_IR_OP = {
    "Relu": "Relu", "Sigmoid": "Sigmoid", "Tanh": "Tanh",
    "Add": "Add", "Sub": "Sub", "Mul": "Mul", "Div": "Div", "MatMul": "MatMul",
    "Conv": "Conv2d", "BatchNormalization": "BatchNorm", "Reshape": "Reshape",
    "Transpose": "Transpose", "Gemm": "MatMul", "MaxPool": "MaxPool",
    "AveragePool": "AvgPool", "Softmax": "Softmax", "Concat": "Concat",
    "Flatten": "Flatten", "Slice": "Slice", "Pad": "Pad",
    "ReduceMean": "ReduceMean", "ReduceSum": "ReduceSum",
    "GlobalAveragePool": "ReduceMean", "Constant": "Constant",
    "LeakyRelu": "LeakyRelu", "Elu": "Elu", "Clip": "Clip",
    "Dropout": "Identity", "InstanceNormalization": "InstanceNorm",
    "Split": "Split", "Shape": "Shape", "Cast": "Cast",
    "Gather": "Gather", "Unsqueeze": "Unsqueeze", "Squeeze": "Squeeze",
    "Identity": "Identity", "Resize": "Resize", "Tile": "Tile",
    "Where": "Where", "Compress": "Compress", "CumSum": "CumSum",
    "DepthToSpace": "DepthToSpace", "SpaceToDepth": "SpaceToDepth",
    "LogSoftmax": "LogSoftmax", "Selu": "Selu",
    "HardSigmoid": "HardSigmoid", "HardSwish": "HardSwish",
    "LayerNormalization": "LayerNorm", "ConvTranspose": "ConvTranspose",
    "TopK": "TopK", "GatherND": "GatherND", "ScatterND": "ScatterND",
    "Exp": "Exp", "Sqrt": "Sqrt", "Neg": "Neg", "Log": "Log", "Erf": "Erf",
    "Ceil": "Ceil", "Floor": "Floor", "Round": "Round", "Sign": "Sign",
    "Reciprocal": "Reciprocal", "IsNaN": "IsNan", "IsInf": "IsInf",
    "And": "And", "Or": "Or", "Xor": "Xor", "Not": "Not",
    "Less": "Less", "Greater": "Greater", "Equal": "Equal",
    "NonMaxSuppression": "NonMaxSuppression", "Pow": "Pow", "Expand": "Expand",
    "GRU": "GRU", "LSTM": "LSTM", "Gelu": "Gelu", "Swish": "Swish",
    "BiasGelu": "BiasGelu", "FastGelu": "FastGelu",
    "Attention": "Attention", "MultiHeadAttention": "MultiHeadAttention",
    "GroupQueryAttention": "GroupQueryAttention",
    "EmbedLayerNormalization": "EmbedLayerNormalization",
    "QuantizeLinear": "QuantizeLinear", "DequantizeLinear": "DequantizeLinear",
    "RMSNormalization": "RmsNorm", "RotaryEmbedding": "RotaryEmbedding",
    "SkipLayerNormalization": "SkipLayerNorm",
    "ConstantOfShape": "ConstantOfShape", "LRN": "LRN",
    "Tril": "Tril", "Triu": "Triu", "Loop": "Loop", "If": "If",
    "Einsum": "Einsum", "EyeLike": "EyeLike", "OneHot": "OneHot",
    "RandomNormal": "RandomNormal", "RandomUniform": "RandomUniform",
    "Range": "Range", "NonZero": "NonZero", "Unique": "Unique",
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
            shape.append(f"Known({d.dim_value})" if d.HasField("dim_value") and d.dim_value > 0 else "Unknown")
    dtype = {1: "F32", 9: "I32", 7: "I64", 10: "I32", 11: "I64", 6: "I32"}.get(
        value_info.type.tensor_type.elem_type, "F32")
    return shape, dtype


def _build_vinfo(model) -> Dict[str, Tuple[List[str], str]]:
    vinfo = {}
    for src in (model.graph.value_info, model.graph.input, model.graph.output):
        for vi in src:
            vinfo[vi.name] = _extract_shape(vi)
    return vinfo


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
}


def _extract_attrs(onnx_node) -> Dict[str, Any]:
    attrs = {}
    op = onnx_node.op_type
    for key in _OP_ATTRS.get(op, []):
        val = _get_attr(onnx_node, key, None)
        if val is not None:
            if isinstance(val, np.ndarray):
                attrs[key] = val.flatten().tolist()
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
        attrs["value"] = float(v.flatten()[0]) if isinstance(v, np.ndarray) else float(v)
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


def _resolve_input_ids(
    names: List[str], out_to_nid: Dict[str, int], gin_to_nid: Dict[str, int], init_names: set,
) -> List[int]:
    return [
        out_to_nid[n] if n in out_to_nid else gin_to_nid[n]
        for n in names if n and n not in init_names and (n in out_to_nid or n in gin_to_nid)
    ]


def import_onnx_to_compute_graph(onnx_path: str, config: Optional[Any] = None) -> Dict[str, Any]:
    import onnx
    import onnx.numpy_helper

    model = onnx.load(onnx_path)
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
        params[name] = {"data": arr.flatten().tolist(), "shape": list(arr.shape), "dtype": "F32", "is_constant": True}

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
        ins = _resolve_input_ids(list(onn.input), out_to_nid, gin_to_nid, init_names)
        out_shape = vinfo.get(onn.output[0], ([], "F32")) if onn.output else ([], "F32")
        osd = {"shape": out_shape[0], "dtype": out_shape[1]}

        for suffix, idx in _OP_PARAM_SLOTS.get(onn.op_type, []):
            if idx < len(onn.input) and onn.input[idx] in init_map:
                arr = init_map[onn.input[idx]]
                params[f"{oname}.{suffix}"] = {"data": arr.flatten().tolist(), "shape": list(arr.shape), "dtype": "F32", "is_constant": True}

        if onn.op_type == "Gemm":
            nodes.append({"id": oid, "opcode": "MatMul", "inputs": list(ins), "output_shape": osd, "attrs": {"alpha": attrs.get("alpha", 1.0), "transB": attrs.get("transB", 0)}, "name": f"{oname}_matmul"})
            weight = init_map.get(onn.input[1])
            if weight is not None:
                if _get_attr(onn, "transB", 0):
                    weight = weight.T
                alpha = _get_attr(onn, "alpha", 1.0)
                if alpha != 1.0:
                    weight = (weight * alpha).astype(weight.dtype)
                params[f"{oname}.weight"] = {"data": weight.flatten().tolist(), "shape": list(weight.shape), "dtype": "F32", "is_constant": True}
            if len(onn.input) > 2 and onn.input[2] in init_map:
                bias = init_map[onn.input[2]]
                beta = _get_attr(onn, "beta", 1.0)
                if beta != 1.0:
                    bias = (bias * beta).astype(bias.dtype)
                bid = nid
                nid += 1
                nodes.append({"id": bid, "opcode": "BiasAdd", "inputs": [oid], "output_shape": osd, "attrs": {}, "name": f"{oname}_bias"})
                out_to_nid[onn.output[0]] = bid
                params[f"{oname}.bias"] = {"data": bias.flatten().tolist(), "shape": list(bias.shape), "dtype": "F32", "is_constant": True}

        elif onn.op_type == "Add":
            bias = init_map.get(onn.input[1])
            if bias is not None:
                params[f"{oname}.bias"] = {"data": bias.flatten().tolist(), "shape": list(bias.shape), "dtype": "F32", "is_constant": True}
            nodes.append({"id": oid, "opcode": "BiasAdd" if bias is not None else "Add", "inputs": list(ins), "output_shape": osd, "attrs": attrs, "name": oname})

        elif onn.op_type == "Sub":
            bias = init_map.get(onn.input[1])
            if bias is not None:
                params[f"{oname}.bias"] = {"data": (-bias).flatten().tolist(), "shape": list(bias.shape), "dtype": "F32", "is_constant": True}
            nodes.append({"id": oid, "opcode": "BiasSub" if bias is not None else "Sub", "inputs": list(ins), "output_shape": osd, "attrs": attrs, "name": oname})

        elif onn.op_type == "Constant":
            value = _get_attr(onn, "value", None)
            if value is not None:
                try:
                    const_arr = onnx.numpy_helper.to_array(value)
                    params[f"{oname}.value"] = {"data": const_arr.flatten().tolist(), "shape": list(const_arr.shape), "dtype": "F32", "is_constant": True}
                except Exception:
                    pass
            nodes.append({"id": oid, "opcode": ir_op, "inputs": list(ins), "output_shape": osd, "attrs": attrs, "name": oname})

        elif onn.op_type == "Reshape":
            shape_arr = init_map.get(onn.input[1])
            if shape_arr is not None:
                attrs["shape"] = shape_arr.flatten().tolist()
            nodes.append({"id": oid, "opcode": ir_op, "inputs": list(ins), "output_shape": osd, "attrs": attrs, "name": oname})

        else:
            nodes.append({"id": oid, "opcode": ir_op, "inputs": list(ins), "output_shape": osd, "attrs": attrs, "name": oname})

    out_ids = []
    for out in model.graph.output:
        if out.name in out_to_nid:
            out_ids.append(out_to_nid[out.name])
        elif out.name in gin_to_nid:
            out_ids.append(gin_to_nid[out.name])

    return {"nodes": nodes, "inputs": gin_ids, "outputs": out_ids, "params": params}


def import_onnx(onnx_path: str, fnn_path: str, config: Optional[Any] = None) -> Dict[str, Any]:
    cg = import_onnx_to_compute_graph(onnx_path, config=config)
    with open(fnn_path, "w") as f:
        json.dump(cg, f, indent=2)
    num_params = sum(1 for p in cg.get("params", {}).values() if p.get("is_constant", False))
    logger.info("Exported %s: %d nodes, %d params to %s", onnx_path, len(cg.get("nodes", [])), num_params, fnn_path)
    return cg
