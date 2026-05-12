"""ComputeGraph executor for fastnn v2.0.0."""
import logging
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import fastnn as fnn
from fastnn import functional as F
import fastnn._core as _core

logger = logging.getLogger(__name__)


def _apply_perm(tensor, perm: Sequence[int]):
    ndim = len(tensor.shape)
    if not perm or perm == list(range(ndim)):
        return tensor
    perm = list(perm)
    # ONNX/numpy convention: output[i] = input[perm[i]] (source lookup).
    # The swap-based algorithm below expects the INVERSE convention:
    #   "input position j should end up at output position inv[j]".
    # Compute the inverse permutation.
    inv = [0] * ndim
    for i in range(ndim):
        inv[perm[i]] = i
    result = tensor
    for i in range(ndim):
        for j in range(i, ndim):
            if inv[j] == i:
                if j != i:
                    result = result.transpose(i, j)
                    inv[i], inv[j] = inv[j], inv[i]
                break
    return result


def _dispatch_op(op: str, input_tensors, node: dict, params: dict):
    attrs = node.get("attrs", {})
    name = node.get("name", "")

    if op == "Relu":
        return F.relu(input_tensors[0])
    elif op == "Add":
        return fnn.add(input_tensors[0], input_tensors[1])
    elif op == "Sub":
        return fnn.sub(input_tensors[0], input_tensors[1])
    elif op == "Mul":
        return fnn.mul(input_tensors[0], input_tensors[1])
    elif op == "Div":
        return fnn.div(input_tensors[0], input_tensors[1])
    elif op == "MatMul":
        return fnn.matmul(input_tensors[0], input_tensors[1])
    elif op == "Conv2d":
        w_key, b_key = f"{name}.weight", f"{name}.bias"
        weight, bias = params.get(w_key), params.get(b_key)
        if weight is None:
            raise ValueError(f"Conv2d weight '{w_key}' not found")
        s = attrs.get("stride", 1)
        p = attrs.get("padding", 0)
        d = attrs.get("dilation", 1)
        g = attrs.get("group", 1)
        if isinstance(s, list): s = s[0]
        if isinstance(p, list): p = p[0]
        if isinstance(d, list): d = d[0]
        result = F.conv2d(input_tensors[0], weight, stride=s, padding=p, dilation=d, groups=g)
        if bias is not None:
            bs = list(bias.shape)
            rs = list(result.shape)
            if bs != rs and len(bs) == 1:
                bias = bias.reshape([1, bs[0]] + [1] * (len(rs) - 2))
            result = fnn.add(result, bias)
        return result
    elif op == "Reshape":
        shape = attrs.get("shape", [])
        if isinstance(shape, str):
            shape = eval(shape)
        if len(input_tensors) > 1:
            shape = input_tensors[1].numpy().flatten().tolist()
        resolved = [int(s) for s in shape]
        t = input_tensors[0]
        total_in = t.numel if hasattr(t, 'numel') else np.prod(list(t.shape))
        total_out = 1
        neg_idx = -1
        for i, s in enumerate(resolved):
            if s == -1:
                neg_idx = i
            elif s == 0 and i < len(t.shape):
                resolved[i] = list(t.shape)[i]
                total_out *= resolved[i]
            else:
                total_out *= s
        if neg_idx >= 0:
            resolved[neg_idx] = total_in // total_out if total_out > 0 else 1
            total_out = total_in
        if total_out != total_in and total_out > 0:
            return t  # shape mismatch, return as-is
        return t.reshape(resolved)
    elif op == "Transpose":
        perm = attrs.get("perm")
        return _apply_perm(input_tensors[0], perm) if perm else input_tensors[0].transpose(0, 1)
    elif op == "Flatten":
        return input_tensors[0].reshape([-1])
    elif op == "Sigmoid":
        return F.sigmoid(input_tensors[0])
    elif op == "Tanh":
        return F.tanh(input_tensors[0])
    elif op == "Exp":
        return fnn.exp(input_tensors[0])
    elif op == "Log":
        return fnn.log(input_tensors[0])
    elif op == "Softmax":
        # ONNX Softmax: axis defaults to 1 (not -1 like PyTorch)
        axis = int(attrs.get("axis", "1"))
        # Convert negative axis to positive
        ndim = len(input_tensors[0].shape)
        if axis < 0:
            axis += ndim
        return F.softmax(input_tensors[0], dim=axis)
    elif op == "ReduceMean":
        return input_tensors[0].mean(int(attrs.get("axis", "0")))
    elif op == "ReduceSum":
        return input_tensors[0].sum(int(attrs.get("axis", "0")))
    elif op == "Concat":
        axis = int(attrs.get("axis", "0"))
        max_ndim = max(len(t.shape) for t in input_tensors)
        normed = []
        for t in input_tensors:
            ndim = len(t.shape)
            if ndim < max_ndim:
                for _ in range(max_ndim - ndim):
                    t = t.unsqueeze(axis)
            normed.append(t)
        return fnn.cat(normed, axis)
    elif op == "BatchNorm":
        eps = float(attrs.get("epsilon", 1e-5))
        return F.batch_norm(input_tensors[0], input_tensors[1], input_tensors[2],
                             input_tensors[3], input_tensors[4], momentum=0.9, eps=eps)
    elif op == "MaxPool":
        return F.max_pool2d(input_tensors[0],
                             kernel_size=int(attrs.get("kernel_size", 2)),
                             stride=int(attrs.get("stride", 2)),
                             padding=int(attrs.get("padding", 0)))
    elif op == "AvgPool":
        return F.avg_pool2d(input_tensors[0],
                             kernel_size=int(attrs.get("kernel_size", 2)),
                             stride=int(attrs.get("stride", 2)))
    elif op == "Constant":
        result = params.get(name)
        if result is None:
            val, dims = attrs.get("value"), attrs.get("dims", [])
            if val is not None and dims:
                result = fnn.tensor_from_numpy(np.full(dims, val, dtype=np.float32))
            else:
                raise ValueError(f"Constant '{name}' not found")
        return result
    elif op == "ConstantOfShape":
        val = float(attrs.get("value", 0.0))
        shape = [int(x) for x in input_tensors[0].numpy().flatten().tolist()]
        return fnn.tensor_from_numpy(np.full(shape, val, dtype=np.float32))
    elif op == "BiasAdd":
        t, bias = input_tensors[0], input_tensors[1]
        bs = list(bias.shape)
        ts = list(t.shape)
        if bs != ts and len(bs) == 1:
            bias = bias.reshape([1, bs[0]] + [1] * (len(ts) - 2))
        return fnn.add(t, bias)
    elif op == "Gelu":
        return F.gelu(input_tensors[0])
    elif op == "Silu":
        return F.silu(input_tensors[0])
    elif op == "Sqrt":
        return fnn.sqrt(input_tensors[0])
    elif op == "Neg":
        return fnn.neg(input_tensors[0])
    elif op == "Abs":
        return fnn.abs(input_tensors[0])
    elif op == "Cast":
        return input_tensors[0]
    elif op == "Unsqueeze":
        axes = attrs.get("axes", [0])
        if isinstance(axes, int): axes = [axes]
        t = input_tensors[0]
        for ax in sorted(axes):
            t = t.unsqueeze(ax)
        return t
    elif op == "Slice":
        result = input_tensors[0]
        data_shape = list(result.shape)
        starts = attrs.get("starts")
        ends = attrs.get("ends")
        if starts is None and len(input_tensors) > 1:
            # ONNX opset 11+: starts/ends/axes/steps are input tensors (not attrs)
            try:
                starts = input_tensors[1].numpy().flatten().tolist()
            except Exception:
                starts = [0]
            try:
                ends = input_tensors[2].numpy().flatten().tolist() if len(input_tensors) > 2 else [2**31 - 1]
            except Exception:
                ends = [2**31 - 1]
            try:
                axes = input_tensors[3].numpy().flatten().tolist() if len(input_tensors) > 3 else None
            except Exception:
                axes = None
            try:
                steps = input_tensors[4].numpy().flatten().tolist() if len(input_tensors) > 4 else [1]
            except Exception:
                steps = [1]
        else:
            starts = starts or [0]
            ends = ends or [2**31 - 1]
            axes = attrs.get("axes")
            steps = attrs.get("steps", [1])
        if axes is None:
            axes = list(range(len(starts)))
        for ax, st, en, sp in zip(axes, starts, ends, steps):
            try:
                result = fnn.fnn_slice(result, int(ax), int(st), int(en), int(sp))
            except Exception as e:
                raise RuntimeError(
                    f"Slice '{name}' failed on dim {int(ax)} with data_shape={data_shape}, "
                    f"starts={starts}, ends={ends}, axes={axes}, steps={steps}: {e}")
        return result
    elif op == "Split":
        axis, splits = int(attrs.get("axis", 0)), attrs.get("split")
        t = input_tensors[0]
        if not splits:
            return [t]
        chunks, offset = [], 0
        for s in splits:
            s = int(s)
            chunks.append(fnn.fnn_slice(t, axis, offset, offset + s, 1))
            offset += s
        return chunks
    elif op == "Shape":
        return fnn.tensor_from_numpy(np.array(list(input_tensors[0].shape), dtype=np.float32))
    elif op == "Gather":
        axis = int(attrs.get("axis", 0))
        indices = attrs.get("indices")
        if indices is not None:
            idx_t = fnn.tensor_from_numpy(np.array(indices, dtype=np.float32))
        elif len(input_tensors) > 1 and input_tensors[1] is not None:
            idx_t = input_tensors[1]
        else:
            raise ValueError(f"Gather '{name}': no indices provided (attrs={attrs}, inputs={len(input_tensors)})")
        return fnn.gather(input_tensors[0], axis, idx_t)
    elif op == "Expand":
        target = [int(x) for x in input_tensors[1].numpy().flatten().tolist()]
        t = input_tensors[0]
        # Materialize: expand returns a view with broadcast strides, but downstream
        # ops (e.g. Concat) need contiguous data. Create a proper tensor via numpy.
        src_np = np.array(t.numpy(), dtype=np.float32).reshape(t.shape)
        full = np.broadcast_to(src_np, target).copy()
        return fnn.tensor_from_numpy(full)
    elif op == "Range":
        vals = [t.item() for t in input_tensors[:3]]
        arr = np.arange(vals[0], vals[1], vals[2], dtype=np.float32)
        return fnn.tensor_from_numpy(arr)
    elif op == "Resize":
        t = input_tensors[0]
        ih, iw = t.shape[-2], t.shape[-1]
        sh = float(attrs.get("scale_h", 1.0))
        sw = float(attrs.get("scale_w", 1.0))
        # Check if sizes/scales come from input tensor (skip ROI at [1], check [2])
        for idx in range(len(input_tensors) - 1, 0, -1):
            try:
                arr = input_tensors[idx].numpy()
                if arr.size >= 2:
                    vals = arr.flatten().tolist()
                    if len(vals) == 4: sh, sw = float(vals[2]), float(vals[3])
                    elif len(vals) == 2: sh, sw = float(vals[0]), float(vals[1])
                    break
            except Exception:
                continue
        th, tw = max(1, round(ih * sh)), max(1, round(iw * sw))
        return t if th == ih and tw == iw else F.interpolate(t, size=[th, tw], mode='nearest')
    else:
        raise ValueError(f"Unsupported opcode: {op}")


class ComputeGraphExecutor:
    def __init__(self, graph_data: Dict[str, Any]):
        self.nodes = graph_data["nodes"]
        self.input_ids = graph_data["inputs"]
        self.output_ids = graph_data["outputs"]
        self.params = {}
        for name, param in graph_data.get("params", {}).items():
            arr = np.array(param["data"], dtype=np.float32).reshape(param["shape"])
            self.params[name] = fnn.tensor_from_numpy(arr)
        self.node_map = {n["id"]: n for n in self.nodes}

    def _resolve_nid(self, encoded: int):
        """Decode encoded node id: returns (node_id, output_slot)."""
        return (encoded & 0xFFFFF, (encoded >> 20) & 0xF) if encoded > 0xFFFFF else (encoded, 0)

    def _resolve_input(self, encoded: int, tensors: dict):
        nid, slot = self._resolve_nid(encoded)
        t = tensors.get(nid)
        if t is None: return None
        if isinstance(t, list): return t[slot] if slot < len(t) else t[-1]
        return t

    def forward(self, *inputs):
        tensors: Dict[int, Any] = {}
        for i, (nid, t) in enumerate(zip(self.input_ids, inputs)):
            tensors[nid] = t

        for node in self.nodes:
            nid = node["id"]
            if nid in tensors:
                continue
            inp_tensors = []
            for inp_id in node["inputs"]:
                inp_tensors.append(self._resolve_input(inp_id, tensors))
            result = _dispatch_op(node["opcode"], inp_tensors, node, self.params)
            tensors[nid] = result

        outputs = []
        for out_id in self.output_ids:
            t = tensors.get(out_id)
            if isinstance(t, list): outputs.append(t[0])
            else: outputs.append(t)
        return outputs


class DAGModel:
    def __init__(self, nodes, params, input_names, output_names):
        self._executor = ComputeGraphExecutor(
            _convert_legacy_to_compute_graph(nodes, params, input_names, output_names))

    @classmethod
    def from_header(cls, header, params):
        g = header.get("graph", {})
        return cls(g.get("nodes", []), params,
                   [inp["name"] for inp in g.get("inputs", [])],
                   [out["name"] for out in g.get("outputs", [])])

    def forward(self, *inputs):
        return self._executor.forward(*inputs)


def _convert_legacy_to_compute_graph(nodes, params, input_names, output_names):
    compute_nodes = []
    for i, n in enumerate(nodes):
        nid = i + 1
        compute_nodes.append({
            "id": nid, "opcode": n.get("op_type", n.get("op_code", "Unknown")),
            "inputs": [], "output_shape": {"shape": [], "dtype": "F32"},
            "attrs": n.get("attrs", {}), "name": n.get("name", f"node_{i}")})
    return {"nodes": compute_nodes, "inputs": [], "outputs": [],
            "params": {k: {"data": v.tolist() if hasattr(v, 'tolist') else v,
                           "shape": list(v.shape) if hasattr(v, 'shape') else [],
                           "dtype": "F32"} for k, v in params.items()}}
