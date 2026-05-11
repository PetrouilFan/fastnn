"""Python DAG executor prototype for ONNX models.

This is a temporary implementation that will be replaced by the
Rust DAGExecutor once it's ready. It provides the same interface
and allows testing the ONNX import pipeline immediately.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import fastnn as fnn
import fastnn._core as _core

logger = logging.getLogger(__name__)


class DAGModel:
    """A DAG-based model that executes ONNX graph nodes.

    This is a Python prototype that dispatches to fastnn ops.
    """

    def __init__(
        self,
        nodes: List[Dict],
        params: Dict[str, Any],
        input_names: List[str],
        output_names: List[str],
    ):
        self.nodes = nodes
        self.params = {}
        for name, arr in params.items():
            self.params[name] = fnn.tensor(arr, list(arr.shape))
        self.input_names = input_names
        self.output_names = output_names
        self.initializer_to_param: Dict[str, str] = {}
        # Build parameter name resolution mapping
        self._init_param_mapping()

    @classmethod
    def from_header(cls, header: dict, params: dict) -> "DAGModel":
        """Create a DAGModel from an ONNX-imported .fnn header."""
        graph = header.get("graph", {})
        nodes = graph.get("nodes", [])
        input_names = [inp["name"] for inp in graph.get("inputs", [])]
        output_names = [out["name"] for out in graph.get("outputs", [])]
        return cls(nodes, params, input_names, output_names)

    def _init_param_mapping(self):
        """Build a mapping from initializer (short) names to parameter names.

        ONNX nodes reference initializers by their original names (e.g., "W", "B")
        but params are stored as {node_name}.{param_type} (e.g., "conv1.weight").
        This method builds the bridge between them based on op-specific positional
        conventions (first input is data, subsequent inputs are params).
        """
        OP_PARAM_SUFFIXES = [".weight", ".bias", ".running_mean", ".running_var", ".value", ".scale", ".beta", ".gamma"]
        OP_PARAM_MAP = {
            "Conv": [".weight", ".bias"],
            "Gemm": [".weight", ".bias"],
            "MatMul": [".weight", ".bias"],
            "BatchNormalization": [".weight", ".bias", ".running_mean", ".running_var"],
            "batchnormalization": [".weight", ".bias", ".running_mean", ".running_var"],
            "InstanceNormalization": [".weight", ".bias"],
            "instancenormalization": [".weight", ".bias"],
            "Constant": [".value"],
        }

        graph_input_names = set(self.input_names)

        # Pass 1: Use OP_PARAM_MAP for known ops
        for node in self.nodes:
            node_name = node.get("name", "")
            if not node_name:
                continue
            inputs = node.get("inputs", [])
            op_type = node.get("op_type", "")

            suffixes = OP_PARAM_MAP.get(op_type, [])
            if suffixes and len(inputs) >= 2:
                for i, input_name in enumerate(inputs[1:], 1):
                    if input_name in self.params or input_name in graph_input_names:
                        continue
                    if i - 1 < len(suffixes):
                        param_name = node_name + suffixes[i - 1]
                        if param_name in self.params:
                            self.initializer_to_param[input_name] = param_name

        # Pass 2: Fallback for any remaining unresolved inputs
        for node in self.nodes:
            node_name = node.get("name", "")
            if not node_name:
                continue
            inputs = node.get("inputs", [])
            prefix = node_name + "."
            for input_name in inputs:
                if input_name in self.params or input_name in graph_input_names or input_name in self.initializer_to_param:
                    continue
                for param_name in self.params:
                    if param_name.startswith(prefix):
                        self.initializer_to_param[input_name] = param_name
                        break

        # Pass 3: Map Constant node output names to their .value params
        for node in self.nodes:
            op_type = node.get("op_type", "")
            if op_type not in ("Constant", "constantop"):
                continue
            node_name = node.get("name", "")
            value_key = f"{node_name}.value"
            if value_key not in self.params:
                continue
            for output_name in node.get("outputs", []):
                if output_name not in self.initializer_to_param:
                    self.initializer_to_param[output_name] = value_key

    def _resolve_input(self, in_name: str, buffer: Dict) -> Optional[Any]:
        """Resolve an input tensor by name.

        Tries buffer first, then params by direct name, then params by
        mapped initializer name.

        Args:
            in_name: Input name from node.
            buffer: Dict of currently available tensors.

        Returns:
            Tensor if found, None otherwise.
        """
        if in_name in buffer:
            return buffer[in_name]
        if in_name in self.params:
            return self.params[in_name]
        mapped_name = self.initializer_to_param.get(in_name)
        if mapped_name is not None:
            if mapped_name in buffer:
                return buffer[mapped_name]
            if mapped_name in self.params:
                return self.params[mapped_name]
        return None

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the DAG forward pass.

        Args:
            inputs: Dict mapping input name to fastnn tensor or numpy array.

        Returns:
            Dict mapping output name to fastnn tensor.
        """
        buffer = {}
        for name, val in inputs.items():
            if isinstance(val, np.ndarray):
                buffer[name] = fnn.tensor(val, list(val.shape))
            else:
                buffer[name] = val

        for name, tensor in self.params.items():
            buffer[name] = tensor

        for node in self.nodes:
            op_type = node.get("op_type", "")
            node_name = node.get("name", "")
            input_names = node.get("inputs", [])
            output_names = node.get("outputs", [])

            tensors = []
            for in_name in input_names:
                tensor = self._resolve_input(in_name, buffer)
                if tensor is not None:
                    tensors.append(tensor)
                else:
                    logger.warning("Node %s: input '%s' not found, skipping", node_name, in_name)
                    tensors = None
                    break

            if tensors is None:
                continue

            outputs = self._dispatch_op(op_type, tensors, node)

            if outputs is not None:
                for i, out_name in enumerate(output_names):
                    if i < len(outputs):
                        buffer[out_name] = outputs[i]

        result = {}
        for name in self.output_names:
            if name in buffer:
                result[name] = buffer[name]
        return result

    def _dispatch_op(self, op_type: str, tensors: List, node: Dict) -> Optional[List]:
        """Dispatch a single operation based on op_type."""
        try:
            if op_type in ("Relu", "relu"):
                return [fnn.relu(tensors[0])]
            elif op_type in ("Sigmoid", "sigmoid"):
                return [fnn.sigmoid(tensors[0])]
            elif op_type in ("Tanh", "tanh"):
                return [fnn.tanh(tensors[0])]
            elif op_type in ("Silu", "silu"):
                return [fnn.silu(tensors[0])]
            elif op_type in ("Gelu", "gelu"):
                return [fnn.gelu(tensors[0])]
            elif op_type in ("LeakyRelu", "leaky_relu", "leakyrelu"):
                alpha = node.get("alpha", 0.01)
                return [_core.leaky_relu(tensors[0], alpha)]
            elif op_type in ("Softmax", "softmax"):
                axis = node.get("axis", 1)
                return [fnn.softmax(tensors[0], axis)]
            elif op_type == "Conv":
                return self._dispatch_conv(tensors, node)
            elif op_type in ("Gemm", "MatMul"):
                return self._dispatch_gemm(tensors, node)
            elif op_type in ("BatchNormalization", "batchnormalization"):
                return self._dispatch_batch_norm(tensors, node)
            elif op_type in ("MaxPool",):
                return self._dispatch_max_pool(tensors, node)
            elif op_type in ("AveragePool", "averagepool"):
                return self._dispatch_avg_pool(tensors, node)
            elif op_type in ("GlobalAveragePool", "globalaveragepool"):
                return self._dispatch_global_avg_pool(tensors, node)
            elif op_type in ("Add", "elementwiseadd"):
                return [fnn.add(tensors[0], tensors[1])]
            elif op_type in ("Sub", "elementwisesub"):
                return [fnn.sub(tensors[0], tensors[1])]
            elif op_type in ("Mul", "elementwisemul"):
                return [fnn.mul(tensors[0], tensors[1])]
            elif op_type in ("Div", "elementwisediv"):
                return [fnn.div(tensors[0], tensors[1])]
            elif op_type == "matmul":
                return [fnn.matmul(tensors[0], tensors[1])]
            elif op_type in ("Exp", "expop"):
                return [fnn.exp(tensors[0])]
            elif op_type in ("Sqrt", "sqrtop"):
                return [fnn.sqrt(tensors[0])]
            elif op_type in ("Neg", "negop"):
                return [fnn.neg(tensors[0])]
            elif op_type in ("Log", "logop"):
                return [fnn.log(tensors[0])]
            elif op_type == "Concat":
                axis = node.get("axis", 1)
                return [fnn.cat(tensors, axis)]
            elif op_type == "Reshape":
                return [tensors[0].reshape(tensors[1].numpy().astype(int).tolist())]
            elif op_type == "Flatten":
                axis = node.get("axis", 1)
                shape = tensors[0].shape
                outer = int(np.prod(shape[:axis]))
                inner = int(np.prod(shape[axis:]))
                return [tensors[0].reshape([outer, inner])]
            elif op_type == "Transpose":
                perm = node.get("perm", None)
                if perm:
                    result = tensors[0]
                    perm_list = list(perm)
                    current = list(range(len(perm_list)))
                    for target in range(len(perm_list)):
                        if current[target] != perm_list[target]:
                            swap_idx = current.index(perm_list[target])
                            result = result.transpose(target, swap_idx)
                            current[target], current[swap_idx] = current[swap_idx], current[target]
                    return [result]
                return [tensors[0].transpose(0, 1)]
            elif op_type == "Shape" or op_type == "shapeop":
                shape_arr = np.array(tensors[0].shape, dtype=np.int64)
                return [fnn.tensor(shape_arr, list(shape_arr.shape))]
            elif op_type == "Cast" or op_type == "castop":
                # ONNX Cast: to attribute is an int mapping to dtype
                to_dtype = node.get("to", 1)
                dtype_map = {1: np.float32, 7: np.int64, 9: np.bool_, 10: np.int32, 11: np.int64}
                target_dtype = dtype_map.get(to_dtype, np.float32)
                x_np = tensors[0].numpy().astype(target_dtype)
                return [fnn.tensor(x_np, list(x_np.shape))]
            elif op_type == "Gather" or op_type == "gatherop":
                # ONNX Gather: data[axes] with axis attribute
                if len(tensors) < 2:
                    return [tensors[0]]
                data_np = tensors[0].numpy()
                indices_np = tensors[1].numpy().astype(int)
                axis = node.get("axis", 0)
                result_np = np.take(data_np, indices_np, axis=axis)
                return [fnn.tensor(result_np, list(result_np.shape))]
            elif op_type in ("Identity", "identityop", "Dropout", "dropout"):
                return [tensors[0]]
            elif op_type == "Resize":
                x = tensors[0]
                mode = node.get("mode", "nearest").lower()
                scales = node.get("scales", None)
                if scales is not None and hasattr(x, "numpy"):
                    x_np = x.numpy()
                    if len(scales) >= 4:
                        scale_h, scale_w = scales[2], scales[3]
                        import math
                        new_h = int(x_np.shape[2] * scale_h)
                        new_w = int(x_np.shape[3] * scale_w)
                        result_np = np.zeros((x_np.shape[0], x_np.shape[1], new_h, new_w), dtype=x_np.dtype)
                        for b in range(x_np.shape[0]):
                            for c in range(x_np.shape[1]):
                                for h in range(new_h):
                                    for w in range(new_w):
                                        src_h = min(int(h / scale_h), x_np.shape[2] - 1)
                                        src_w = min(int(w / scale_w), x_np.shape[3] - 1)
                                        result_np[b, c, h, w] = x_np[b, c, src_h, src_w]
                        return [fnn.tensor(result_np, list(result_np.shape))]
                return [tensors[0]]
            elif op_type in ("ReduceMean",):
                axes = node.get("axes", None)
                result = tensors[0]
                if axes is not None:
                    for ax in sorted(axes, reverse=True):
                        result = fnn.mean(result, ax)
                else:
                    result = fnn.mean(result)
                return [result]
            elif op_type in ("ReduceSum",):
                x = tensors[0]
                axes = node.get("axes", None)
                if axes is not None:
                    result = x
                    for ax in sorted(axes, reverse=True):
                        result = fnn.sum(result, ax)
                    return [result]
                else:
                    return [fnn.sum(x)]
            elif op_type in ("Slice", "sliceop"):
                x = tensors[0]
                if hasattr(x, "numpy"):
                    x_np = x.numpy()
                else:
                    x_np = np.array(x)
                starts = node.get("starts", None)
                ends = node.get("ends", None)
                axes = node.get("axes", None)
                steps = node.get("steps", None)
                if starts is not None and ends is not None:
                    if axes is None:
                        axes = list(range(len(starts)))
                    if steps is None:
                        steps = [1] * len(starts)
                    result_np = x_np
                    for i, ax in enumerate(axes):
                        st = int(starts[i])
                        en = int(ends[i])
                        step = int(steps[i]) if i < len(steps) else 1
                        slices = [slice(None)] * len(result_np.shape)
                        slices[int(ax)] = slice(st, en, step)
                        result_np = result_np[tuple(slices)]
                    return [fnn.tensor(result_np, list(result_np.shape))]
                else:
                    return [tensors[0]]
            elif op_type in ("Pad",):
                x = tensors[0]
                pads = node.get("pads", None)
                if pads is not None and hasattr(x, "numpy"):
                    x_np = x.numpy()
                    rank = len(x_np.shape)
                    pad_width = [(pads[i], pads[i + rank]) for i in range(rank)]
                    padded = np.pad(x_np, pad_width, mode='constant', constant_values=0)
                    return [fnn.tensor(padded, list(padded.shape))]
                return [tensors[0]]
            elif op_type in ("Tile", "tileop"):
                x = tensors[0]
                if len(tensors) >= 2:
                    repeats_tensor = tensors[1]
                    repeats = repeats_tensor.numpy().astype(int).tolist() if hasattr(repeats_tensor, "numpy") else repeats_tensor
                    return [fnn.repeat(x, repeats)]
                return [tensors[0]]
            elif op_type in ("Where", "whereop"):
                if len(tensors) >= 3:
                    return [fnn.where(tensors[0], tensors[1], tensors[2])]
                return [tensors[0]]
            elif op_type in ("NonMaxSuppression",):
                return [tensors[0]]
            elif op_type in ("TopK", "topkop"):
                if hasattr(fnn, "topk"):
                    k = node.get("k", 1)
                    axis = node.get("axis", -1)
                    if isinstance(k, list) and len(k) > 0:
                        k = k[0]
                    return list(fnn.topk(tensors[0], k, axis))
                return [tensors[0]]
            elif op_type in ("Split",):
                axis = int(node.get("axis", 0))
                split_sizes = node.get("split", None)
                x = tensors[0]
                if hasattr(x, "numpy"):
                    x_np = x.numpy()
                else:
                    x_np = np.array(x)
                dim_size = x_np.shape[axis] if axis < len(x_np.shape) else x_np.shape[0]
                num_outputs = len(node.get("outputs", [1]))
                if split_sizes is not None:
                    sizes = split_sizes
                else:
                    size = dim_size // num_outputs
                    sizes = [size] * num_outputs
                results = []
                start = 0
                for sz in sizes:
                    slices = [slice(None)] * len(x_np.shape)
                    slices[axis] = slice(start, start + sz)
                    result_np = x_np[tuple(slices)]
                    results.append(fnn.tensor(result_np, list(result_np.shape)))
                    start += sz
                return results
            elif op_type in ("Squeeze", "squeezeop"):
                x = tensors[0]
                axes = node.get("axes", None)
                if axes is not None:
                    shape = list(x.shape)
                    new_shape = [s for i, s in enumerate(shape) if i not in axes]
                    if new_shape:
                        return [x.reshape(new_shape)]
                    return [x.reshape([1])]
                return [x.reshape([s for s in x.shape if s != 1])]
            elif op_type in ("Unsqueeze", "unsqueezeop"):
                x = tensors[0]
                axes = node.get("axes", None)
                if axes is not None:
                    shape = list(x.shape)
                    for ax in sorted(axes):
                        shape.insert(ax, 1)
                    return [x.reshape(shape)]
                return [tensors[0]]
            elif op_type in ("Constant", "constantop"):
                if tensors:
                    return [tensors[0]]
                node_name = node.get("name", "")
                value_key = f"{node_name}.value"
                if value_key in self.params:
                    return [self.params[value_key]]
                return None
            elif op_type in ("Loop", "loopop", "If", "ifop"):
                logger.warning("Control flow op %s not yet supported, passing through", op_type)
                return [tensors[0]] if tensors else None
            elif op_type in ("Erf", "erfop"):
                x = tensors[0]
                if hasattr(x, "numpy"):
                    x_np = x.numpy()
                    signs = np.where(x_np >= 0, 1.0, -1.0)
                    x_abs = np.abs(x_np)
                    t = 1.0 / (1.0 + 0.3275911 * x_abs)
                    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x_abs * x_abs)
                    result_np = signs * y
                    result = fnn.tensor(result_np, list(x.shape))
                    return [result]
                return [tensors[0]]
            elif op_type in ("Pow", "elementwisepow"):
                if len(tensors) > 1:
                    # Extract scalar exponent from tensor
                    exponent = tensors[1].numpy().flatten()[0].item() if hasattr(tensors[1], "numpy") else float(tensors[1])
                    return [fnn.pow(tensors[0], exponent)]
                return [tensors[0]]
            elif op_type == "BiasAdd":
                return [fnn.add(tensors[0], self.params.get(node.get("name", "") + ".bias", tensors[0]))]
            elif op_type == "BiasSub":
                return [fnn.sub(tensors[0], self.params.get(node.get("name", "") + ".bias", tensors[0]))]
            elif op_type == "Clip":
                min_val = node.get("min", None)
                max_val = node.get("max", None)
                result = tensors[0]
                if min_val is not None and max_val is not None:
                    return [fnn.clamp(result, min_val, max_val)]
                elif min_val is not None:
                    return [fnn.clamp(result, min_val, float('inf'))]
                elif max_val is not None:
                    return [fnn.clamp(result, float('-inf'), max_val)]
                return [result]
            elif op_type in ("Abs",):
                return [fnn.abs(tensors[0])]
            elif op_type in ("Elu",):
                alpha = node.get("alpha", 1.0)
                return [fnn.elu(tensors[0], alpha)]
            elif op_type in ("Ceil",):
                return [fnn.tensor(np.ceil(tensors[0].numpy()).astype(tensors[0].numpy().dtype), list(tensors[0].shape))]
            elif op_type in ("Floor",):
                return [fnn.tensor(np.floor(tensors[0].numpy()).astype(tensors[0].numpy().dtype), list(tensors[0].shape))]
            elif op_type in ("Round",):
                return [fnn.tensor(np.round(tensors[0].numpy()).astype(tensors[0].numpy().dtype), list(tensors[0].shape))]
            elif op_type in ("Sign",):
                return [fnn.tensor(np.sign(tensors[0].numpy()), list(tensors[0].shape))]
            elif op_type in ("Reciprocal",):
                return [fnn.div(fnn.tensor(np.array(1.0, dtype=np.float32), []), tensors[0])]
            elif op_type in ("And",):
                return [fnn.tensor(np.logical_and(tensors[0].numpy(), tensors[1].numpy()).astype(np.bool_), list(tensors[0].shape))]
            elif op_type in ("Or",):
                return [fnn.tensor(np.logical_or(tensors[0].numpy(), tensors[1].numpy()).astype(np.bool_), list(tensors[0].shape))]
            elif op_type in ("Xor",):
                return [fnn.tensor(np.logical_xor(tensors[0].numpy(), tensors[1].numpy()).astype(np.bool_), list(tensors[0].shape))]
            elif op_type in ("Not",):
                return [fnn.tensor(np.logical_not(tensors[0].numpy()).astype(np.bool_), list(tensors[0].shape))]
            elif op_type in ("Equal",):
                return [fnn.tensor(np.equal(tensors[0].numpy(), tensors[1].numpy()), list(tensors[0].shape))]
            elif op_type in ("Greater",):
                return [fnn.tensor(np.greater(tensors[0].numpy(), tensors[1].numpy()), list(tensors[0].shape))]
            elif op_type in ("Less",):
                return [fnn.tensor(np.less(tensors[0].numpy(), tensors[1].numpy()), list(tensors[0].shape))]
            elif op_type in ("IsNaN",):
                return [fnn.tensor(np.isnan(tensors[0].numpy()), list(tensors[0].shape))]
            elif op_type in ("IsInf",):
                return [fnn.tensor(np.isinf(tensors[0].numpy()), list(tensors[0].shape))]
            elif op_type in ("Expand",):
                shape = tensors[1].numpy().astype(int).tolist() if len(tensors) > 1 else node.get("shape", list(tensors[0].shape))
                result_np = np.broadcast_to(tensors[0].numpy(), shape)
                return [fnn.tensor(result_np, list(result_np.shape))]
            elif op_type in ("CumSum",):
                axis = node.get("axis", 0)
                return [fnn.tensor(np.cumsum(tensors[0].numpy(), axis=axis), list(tensors[0].shape))]
            elif op_type in ("Compress",):
                axis = node.get("axis", None)
                condition = tensors[1].numpy().astype(bool) if len(tensors) > 1 else np.array([True])
                result = np.compress(condition, tensors[0].numpy(), axis=axis)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("DepthToSpace",):
                blocksize = node.get("blocksize", 1)
                mode = node.get("mode", "DCR")
                x = tensors[0].numpy()
                b, c, h, w = x.shape
                if mode == "DCR":
                    x = x.reshape(b, blocksize, blocksize, c // (blocksize**2), h, w).transpose(0, 3, 4, 1, 5, 2)
                else:
                    x = x.reshape(b, c // (blocksize**2), blocksize, blocksize, h, w).transpose(0, 1, 4, 2, 5, 3)
                result = x.reshape(b, c // (blocksize**2), h * blocksize, w * blocksize)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("SpaceToDepth",):
                blocksize = node.get("blocksize", 1)
                x = tensors[0].numpy()
                b, c, h, w = x.shape
                x = x.reshape(b, c, h // blocksize, blocksize, w // blocksize, blocksize)
                x = x.transpose(0, 3, 5, 1, 2, 4).reshape(b, c * blocksize * blocksize, h // blocksize, w // blocksize)
                return [fnn.tensor(x, list(x.shape))]
            elif op_type in ("EyeLike",):
                shape = tensors[0].shape
                k = node.get("k", 0)
                result = np.eye(shape[0], shape[1] if len(shape) > 1 else shape[0], k=k, dtype=tensors[0].numpy().dtype)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("OneHot",):
                axis = node.get("axis", -1)
                indices = tensors[0].numpy().astype(int)
                depth = int(tensors[1].numpy().flatten()[0]) if len(tensors) > 1 else 1
                values = tensors[2].numpy() if len(tensors) > 2 else np.array([0.0, 1.0])
                off_val, on_val = values[0], values[1] if len(values) > 1 else 1.0
                shape = list(indices.shape)
                if axis < 0:
                    axis = len(shape) + 1 + axis
                shape.insert(axis, depth)
                result = np.full(shape, off_val, dtype=np.float32)
                idx_tuple = [np.arange(s) for s in indices.shape]
                idx_tuple.insert(axis, indices)
                result[tuple(idx_tuple)] = on_val
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("Range",):
                start = float(tensors[0].numpy().flatten()[0])
                limit = float(tensors[1].numpy().flatten()[0])
                delta = float(tensors[2].numpy().flatten()[0]) if len(tensors) > 2 else 1.0
                result = np.arange(start, limit, delta, dtype=np.float32)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("Reverse",):
                axes = node.get("axes", None)
                x = tensors[0].numpy()
                if axes is not None:
                    slices = [slice(None)] * x.ndim
                    for ax in axes:
                        slices[ax] = slice(None, None, -1)
                    result = x[tuple(slices)]
                else:
                    result = x[::-1]
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("RandomNormal",):
                shape = node.get("shape", [1])
                mean = node.get("mean", 0.0)
                scale = node.get("scale", 1.0)
                seed = node.get("seed", None)
                if seed is not None:
                    np.random.seed(int(seed))
                result = np.random.normal(mean, scale, size=shape).astype(np.float32)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("RandomUniform",):
                shape = node.get("shape", [1])
                low = node.get("low", 0.0)
                high = node.get("high", 1.0)
                seed = node.get("seed", None)
                if seed is not None:
                    np.random.seed(int(seed))
                result = np.random.uniform(low, high, size=shape).astype(np.float32)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("ConstantOfShape",):
                shape = tensors[0].numpy().astype(int).tolist() if tensors else node.get("shape", [1])
                value = node.get("value", 0.0)
                result = np.full(shape, value, dtype=np.float32)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("HardSigmoid",):
                alpha = node.get("alpha", 0.2)
                beta = node.get("beta", 0.5)
                x = tensors[0].numpy()
                result = np.maximum(0, np.minimum(1, alpha * x + beta)).astype(x.dtype)
                return [fnn.tensor(result, list(tensors[0].shape))]
            elif op_type in ("HardSwish",):
                return [fnn.hardswish(tensors[0])]
            elif op_type in ("Selu",):
                alpha = node.get("alpha", 1.67326)
                gamma = node.get("gamma", 1.0507)
                x = tensors[0].numpy()
                result = gamma * np.where(x >= 0, x, alpha * (np.exp(x) - 1))
                return [fnn.tensor(result, list(tensors[0].shape))]
            elif op_type in ("SoftPlus",):
                return [fnn.softplus(tensors[0])]
            elif op_type in ("LogSoftmax",):
                axis = node.get("axis", 1)
                return [fnn.log_softmax(tensors[0], axis)]
            elif op_type in ("Swish",):
                return [fnn.silu(tensors[0])]
            elif op_type in ("ArgMax",):
                axis = node.get("axis", None)
                keepdims = node.get("keepdims", 1)
                x = tensors[0].numpy()
                if axis is None:
                    result = np.array([np.argmax(x)])
                else:
                    result = np.argmax(x, axis=axis)
                    if keepdims:
                        result = np.expand_dims(result, axis)
                return [fnn.tensor(result.astype(np.int64), list(result.shape))]
            elif op_type in ("ArgMin",):
                axis = node.get("axis", None)
                keepdims = node.get("keepdims", 1)
                x = tensors[0].numpy()
                if axis is None:
                    result = np.array([np.argmin(x)])
                else:
                    result = np.argmin(x, axis=axis)
                    if keepdims:
                        result = np.expand_dims(result, axis)
                return [fnn.tensor(result.astype(np.int64), list(result.shape))]
            elif op_type in ("Min", "min"):
                result = tensors[0].numpy()
                for t in tensors[1:]:
                    result = np.minimum(result, t.numpy())
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("Max", "max"):
                result = tensors[0].numpy()
                for t in tensors[1:]:
                    result = np.maximum(result, t.numpy())
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("PRelu",):
                slope = tensors[1].numpy() if len(tensors) > 1 else 0.01
                x = tensors[0].numpy()
                result = np.where(x > 0, x, slope * x)
                return [fnn.tensor(result, list(tensors[0].shape))]
            elif op_type in ("GatherND",):
                data = tensors[0].numpy()
                indices = tensors[1].numpy().astype(int) if len(tensors) > 1 else np.array([[0]])
                idx_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))
                result = data[idx_tuple]
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("ScatterND",):
                data = tensors[0].numpy().copy()
                indices = tensors[1].numpy().astype(int) if len(tensors) > 1 else np.array([[0]])
                updates = tensors[2].numpy() if len(tensors) > 2 else np.array([0.0])
                idx_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))
                data[idx_tuple] = updates
                return [fnn.tensor(data, list(tensors[0].shape))]
            elif op_type in ("ConvTranspose",):
                x = tensors[0]
                node_name = node.get("name", "")
                weight = self.params.get(f"{node_name}.weight")
                if weight is None:
                    logger.warning("ConvTranspose %s: weight not found", node_name)
                    return [x]
                x_np = x.numpy()
                w_np = weight.numpy()
                stride = node.get("stride", node.get("strides", 1))
                if isinstance(stride, list): stride = stride[0]
                padding = node.get("padding", node.get("pads", 0))
                if isinstance(padding, list): padding = padding[0]
                b, c_in, h_in, w_in = x_np.shape
                c_out, c_in_w, k_h, k_w = w_np.shape
                h_out = (h_in - 1) * stride + k_h - 2 * padding
                w_out = (w_in - 1) * stride + k_w - 2 * padding
                result = np.zeros((b, c_out, h_out, w_out), dtype=np.float32)
                for bi in range(b):
                    for co in range(c_out):
                        for ci in range(c_in_w):
                            for kh in range(k_h):
                                for kw in range(k_w):
                                    for hi in range(h_in):
                                        for wi in range(w_in):
                                            oh = hi * stride + kh - padding
                                            ow = wi * stride + kw - padding
                                            if 0 <= oh < h_out and 0 <= ow < w_out:
                                                result[bi, co, oh, ow] += x_np[bi, ci, hi, wi] * w_np[co, ci, kh, kw]
                bias = self.params.get(f"{node_name}.bias")
                if bias is not None:
                    result += bias.numpy().reshape(1, c_out, 1, 1)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("InstanceNormalization", "instancenormalization"):
                x = tensors[0].numpy()
                eps = node.get("epsilon", node.get("eps", 1e-5))
                b, c, h, w = x.shape
                x_r = x.reshape(b, c, -1)
                mean = x_r.mean(axis=2, keepdims=True)
                var = x_r.var(axis=2, keepdims=True)
                normalized = (x_r - mean) / np.sqrt(var + eps)
                normalized = normalized.reshape(b, c, h, w)
                node_name = node.get("name", "")
                scale = self.params.get(f"{node_name}.weight")
                bias = self.params.get(f"{node_name}.bias")
                if scale is not None:
                    normalized = normalized * scale.numpy().reshape(1, c, 1, 1)
                if bias is not None:
                    normalized = normalized + bias.numpy().reshape(1, c, 1, 1)
                return [fnn.tensor(normalized, list(x.shape))]
            elif op_type in ("LayerNormalization", "layernormalization"):
                x = tensors[0].numpy()
                axis = node.get("axis", -1)
                eps = node.get("epsilon", node.get("eps", 1e-5))
                mean = x.mean(axis=axis, keepdims=True)
                var = x.var(axis=axis, keepdims=True)
                normalized = (x - mean) / np.sqrt(var + eps)
                node_name = node.get("name", "")
                weight = self.params.get(f"{node_name}.weight")
                bias = self.params.get(f"{node_name}.bias")
                if weight is not None:
                    normalized = normalized * weight.numpy()
                if bias is not None:
                    normalized = normalized + bias.numpy()
                return [fnn.tensor(normalized, list(x.shape))]
            elif op_type in ("RMSNormalization", "rmsnormalization"):
                x = tensors[0].numpy()
                eps = node.get("epsilon", node.get("eps", 1e-5))
                rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
                normalized = x / rms
                node_name = node.get("name", "")
                weight = self.params.get(f"{node_name}.weight")
                if weight is not None:
                    normalized = normalized * weight.numpy()
                return [fnn.tensor(normalized, list(x.shape))]
            elif op_type in ("SkipLayerNormalization",):
                x = tensors[0].numpy()
                skip = tensors[1].numpy() if len(tensors) > 1 else 0
                eps = node.get("epsilon", node.get("eps", 1e-5))
                residual = x + skip
                mean = residual.mean(axis=-1, keepdims=True)
                var = residual.var(axis=-1, keepdims=True)
                normalized = (residual - mean) / np.sqrt(var + eps)
                node_name = node.get("name", "")
                weight = self.params.get(f"{node_name}.weight")
                bias = self.params.get(f"{node_name}.bias")
                if weight is not None:
                    normalized = normalized * weight.numpy()
                if bias is not None:
                    normalized = normalized + bias.numpy()
                return [fnn.tensor(normalized, list(x.shape))]
            elif op_type in ("Attention", "MultiHeadAttention", "GroupQueryAttention"):
                # Basic scaled dot-product attention
                if len(tensors) >= 3:
                    q, k, v = [t.numpy() for t in tensors[:3]]
                    scale = 1.0 / np.sqrt(k.shape[-1])
                    scores = q @ k.transpose(0, 1, 3, 2) * scale
                    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
                    weights /= weights.sum(axis=-1, keepdims=True)
                    result = weights @ v
                    return [fnn.tensor(result, list(result.shape))]
                logger.warning("Attention op '%s' needs Q, K, V inputs", op_type)
                return [tensors[0]]
            elif op_type in ("GRU",):
                # GRU: gated recurrent unit
                # Inputs: [X, W, R, B?, sequence_lens?, initial_h?]
                # X: [seq_len, batch_size, input_size]
                # W: [num_directions, 3*hidden_size, input_size]  (z, r, h)
                # R: [num_directions, 3*hidden_size, hidden_size]
                # B: [num_directions, 6*hidden_size] (optional)
                # sequence_lens: [batch_size] (optional)
                # initial_h: [num_directions, batch_size, hidden_size] (optional)
                if len(tensors) < 3:
                    logger.warning("GRU needs X, W, R")
                    return [tensors[0]]
                x_np = tensors[0].numpy()
                w_np = tensors[1].numpy()
                r_np = tensors[2].numpy()
                b_np = tensors[3].numpy() if len(tensors) > 3 else None
                seq_lens = tensors[4].numpy().flatten().astype(int) if len(tensors) > 4 else None
                init_h = tensors[5].numpy() if len(tensors) > 5 else None
                seq_len, batch_size, input_size = x_np.shape
                num_directions = w_np.shape[0]
                hidden_size = w_np.shape[1] // 3
                # Initialize hidden state
                h = np.zeros((num_directions, batch_size, hidden_size), dtype=np.float32)
                if init_h is not None:
                    h[:] = init_h
                # Process biases
                w_b = np.zeros((num_directions, 3 * hidden_size), dtype=np.float32)
                r_b = np.zeros((num_directions, 3 * hidden_size), dtype=np.float32)
                if b_np is not None:
                    w_b = b_np[:, :3 * hidden_size]
                    r_b = b_np[:, 3 * hidden_size:]
                outputs = []
                for t in range(seq_len):
                    x_t = x_np[t]  # [batch_size, input_size]
                    for d in range(num_directions):
                        # Gate weights for this direction
                        w = w_np[d]  # [3*hidden, input_size]
                        r = r_np[d]  # [3*hidden, hidden_size]
                        w_b_d = w_b[d]
                        r_b_d = r_b[d]
                        h_d = h[d]  # [batch_size, hidden_size]
                        # Split gates: z (update), r (reset), h (new)
                        w_z, w_r, w_h = np.split(w, 3, axis=0)
                        r_z, r_r, r_h = np.split(r, 3, axis=0)
                        b_wz, b_wr, b_wh = np.split(w_b_d, 3)
                        b_rz, b_rr, b_rh = np.split(r_b_d, 3)
                        # Gate computations
                        z = 1.0 / (1.0 + np.exp(-(x_t @ w_z.T + h_d @ r_z.T + b_wz + b_rz)))
                        r_gate = 1.0 / (1.0 + np.exp(-(x_t @ w_r.T + h_d @ r_r.T + b_wr + b_rr)))
                        n = np.tanh(x_t @ w_h.T + r_gate * (h_d @ r_h.T + b_rh) + b_wh)
                        h_new = (1.0 - z) * n + z * h_d
                        h[d] = h_new
                    outputs.append(h.copy())
                # Output Y: [seq_len, num_directions, batch_size, hidden_size]
                y = np.stack(outputs, axis=0)
                ret = [fnn.tensor(y, list(y.shape))]
                # Y_h: [num_directions, batch_size, hidden_size]
                ret.append(fnn.tensor(h, list(h.shape)))
                return ret
            elif op_type in ("LSTM",):
                # LSTM: long short-term memory
                # Inputs: [X, W, R, B?, sequence_lens?, initial_h?, initial_c?]
                # X: [seq_len, batch_size, input_size]
                # W: [num_directions, 4*hidden_size, input_size]  (i, o, f, c)
                # R: [num_directions, 4*hidden_size, hidden_size]
                # B: [num_directions, 8*hidden_size] (optional)
                if len(tensors) < 3:
                    logger.warning("LSTM needs X, W, R")
                    return [tensors[0]]
                x_np = tensors[0].numpy()
                w_np = tensors[1].numpy()
                r_np = tensors[2].numpy()
                b_np = tensors[3].numpy() if len(tensors) > 3 else None
                init_h = tensors[5].numpy() if len(tensors) > 5 else None
                init_c = tensors[6].numpy() if len(tensors) > 6 else None
                seq_len, batch_size, input_size = x_np.shape
                num_directions = w_np.shape[0]
                hidden_size = w_np.shape[1] // 4
                h = np.zeros((num_directions, batch_size, hidden_size), dtype=np.float32)
                c = np.zeros((num_directions, batch_size, hidden_size), dtype=np.float32)
                if init_h is not None:
                    h[:] = init_h
                if init_c is not None:
                    c[:] = init_c
                w_b = np.zeros((num_directions, 4 * hidden_size), dtype=np.float32)
                r_b = np.zeros((num_directions, 4 * hidden_size), dtype=np.float32)
                if b_np is not None:
                    w_b = b_np[:, :4 * hidden_size]
                    r_b = b_np[:, 4 * hidden_size:]
                def _sig(x):
                    return 1.0 / (1.0 + np.exp(-x))
                outputs = []
                for t in range(seq_len):
                    x_t = x_np[t]
                    for d in range(num_directions):
                        w = w_np[d]
                        r = r_np[d]
                        w_b_d = w_b[d]
                        r_b_d = r_b[d]
                        h_d = h[d]
                        c_d = c[d]
                        # Split gates: i (input), o (output), f (forget), c (cell)
                        w_i, w_o, w_f, w_c_g = np.split(w, 4, axis=0)
                        r_i, r_o, r_f, r_c_g = np.split(r, 4, axis=0)
                        b_wi, b_wo, b_wf, b_wc = np.split(w_b_d, 4)
                        b_ri, b_ro, b_rf, b_rc = np.split(r_b_d, 4)
                        # Gate computations
                        i = _sig(x_t @ w_i.T + h_d @ r_i.T + b_wi + b_ri)
                        f = _sig(x_t @ w_f.T + h_d @ r_f.T + b_wf + b_rf)
                        o = _sig(x_t @ w_o.T + h_d @ r_o.T + b_wo + b_ro)
                        c_tilde = np.tanh(x_t @ w_c_g.T + h_d @ r_c_g.T + b_wc + b_rc)
                        c_new = f * c_d + i * c_tilde
                        h_new = o * np.tanh(c_new)
                        h[d] = h_new
                        c[d] = c_new
                    outputs.append(h.copy())
                y = np.stack(outputs, axis=0)
                ret = [fnn.tensor(y, list(y.shape))]
                ret.append(fnn.tensor(h, list(h.shape)))
                ret.append(fnn.tensor(c, list(c.shape)))
                return ret
            elif op_type in ("BiasGelu",):
                x = tensors[0].numpy()
                bias = tensors[1].numpy() if len(tensors) > 1 else np.array(0.0)
                x_b = x + bias
                result = 0.5 * x_b * (1.0 + np.tanh(0.7978846 * (x_b + 0.044715 * x_b ** 3)))
                return [fnn.tensor(result, list(tensors[0].shape))]
            elif op_type in ("FastGelu",):
                x = tensors[0].numpy()
                result = 0.5 * x * (1.0 + np.tanh(0.7978846 * (x + 0.044715 * x ** 3)))
                return [fnn.tensor(result, list(tensors[0].shape))]
            elif op_type in ("RotaryEmbedding",):
                # Rotary position embedding (RoPE)
                # Inputs: [Q, K, cos_cache, sin_cache, position_ids?]
                # Q/K shape: [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
                # cos/sin cache shape: [max_seq_len, head_dim] or [seq_len, head_dim]
                # If fewer than 4 inputs, pass through
                if len(tensors) < 4:
                    logger.warning("RotaryEmbedding needs Q, K, cos, sin; got %d inputs", len(tensors))
                    return [tensors[0]]
                q = tensors[0].numpy()
                k = tensors[1].numpy()
                cos = tensors[2].numpy()
                sin = tensors[3].numpy()
                # Get position IDs if provided
                if len(tensors) >= 5:
                    position_ids = tensors[4].numpy().flatten().astype(int)
                else:
                    position_ids = np.arange(q.shape[1], dtype=int)
                # Apply rotary embedding to Q and K
                def _rope_single(x, cos_vals, sin_vals, pos_ids):
                    # x shape: [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
                    # Determine if seq_len is axis 1 or 2
                    if x.shape[1] == len(pos_ids) or (x.ndim >= 3 and x.shape[-2] == len(pos_ids)):
                        seq_axis = 1
                    else:
                        seq_axis = -2 if x.ndim >= 2 else 1
                    head_dim = x.shape[-1]
                    half = head_dim // 2
                    if seq_axis == 1:
                        x_flat = x.reshape(-1, x.shape[1], head_dim)
                    else:
                        x_flat = x.reshape(x.shape[0], x.shape[1], -1, head_dim)
                        x_flat = x_flat.transpose(0, 2, 1, 3).reshape(-1, x_flat.shape[2], head_dim)
                    # Split into pairs and rotate
                    x_even = x_flat[..., :half]
                    x_odd = x_flat[..., half:]
                    # Gather cos/sin for each position (use first half for pair-wise rotation)
                    cos_gathered = cos_vals[pos_ids][:, :half]  # [seq_len, half]
                    sin_gathered = sin_vals[pos_ids][:, :half]  # [seq_len, half]
                    # Reshape for broadcasting: [1, seq_len, half]
                    cos_gathered = cos_gathered.reshape(1, -1, half)
                    sin_gathered = sin_gathered.reshape(1, -1, half)
                    # Apply rotation
                    x_even_rot = x_even * cos_gathered - x_odd * sin_gathered
                    x_odd_rot = x_even * sin_gathered + x_odd * cos_gathered
                    x_rot = np.concatenate([x_even_rot, x_odd_rot], axis=-1)
                    # Restore original layout
                    if seq_axis != 1:
                        x_rot = x_rot.reshape(x.shape[0], -1, x.shape[-2], head_dim).transpose(0, 2, 1, 3)
                    return x_rot.reshape(x.shape)
                q_rot = _rope_single(q, cos, sin, position_ids)
                k_rot = _rope_single(k, cos, sin, position_ids)
                return [fnn.tensor(q_rot, list(q_rot.shape)), fnn.tensor(k_rot, list(k_rot.shape))]
            elif op_type in ("DequantizeLinear",):
                # y = (x - x_zero_point) * x_scale
                x = tensors[0].numpy()
                x_scale = tensors[1].numpy().flatten()[0] if len(tensors) > 1 else 1.0
                x_zero_point = tensors[2].numpy().flatten()[0] if len(tensors) > 2 else 0.0
                result = (x.astype(np.float32) - float(x_zero_point)) * float(x_scale)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("QuantizeLinear",):
                # y = round(x / y_scale) + y_zero_point
                x = tensors[0].numpy()
                y_scale = tensors[1].numpy().flatten()[0] if len(tensors) > 1 else 1.0
                y_zero_point = tensors[2].numpy().flatten()[0] if len(tensors) > 2 else 0.0
                result = np.round(x / float(y_scale)).astype(np.int8) + int(y_zero_point)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("EmbedLayerNormalization",):
                # Embedding + Layer Normalization (BERT-style)
                # Inputs: [input_ids, word_embedding, position_embedding,
                #          segment_embedding?, gamma, beta, mask?]
                if len(tensors) < 5:
                    logger.warning("EmbedLayerNorm needs input_ids, word_emb, pos_emb, gamma, beta")
                    return [tensors[0]]
                input_ids = tensors[0].numpy().astype(int)
                word_emb = tensors[1].numpy()
                pos_emb = tensors[2].numpy()
                has_segment = len(tensors) >= 7
                gamma = tensors[4 if has_segment else 3].numpy()
                beta = tensors[5 if has_segment else 4].numpy()
                batch_size, seq_len = input_ids.shape
                hidden_size = word_emb.shape[-1]
                # Word embeddings
                word_out = word_emb[input_ids]
                # Position embeddings
                pos_ids = np.arange(seq_len, dtype=int)
                pos_out = pos_emb[pos_ids]
                pos_out = np.broadcast_to(pos_out, (batch_size, seq_len, hidden_size))
                output = word_out + pos_out
                # Add segment embeddings if provided
                if has_segment:
                    seg_emb = tensors[3].numpy()
                    seg_ids = np.zeros((batch_size, seq_len), dtype=int)
                    seg_out = seg_emb[seg_ids]
                    output = output + seg_out
                # Layer normalization
                mean = output.mean(axis=-1, keepdims=True)
                var = output.var(axis=-1, keepdims=True)
                eps = node.get("epsilon", node.get("eps", 1e-5))
                output = (output - mean) / np.sqrt(var + eps)
                output = output * gamma.reshape(1, 1, -1) + beta.reshape(1, 1, -1)
                return [fnn.tensor(output.astype(np.float32), list(output.shape))]
            elif op_type in ("Einsum",):
                equation = node.get("equation", "")
                tensors_np = [t.numpy() for t in tensors]
                result = np.einsum(equation, *tensors_np)
                return [fnn.tensor(result, list(result.shape))]
            elif op_type in ("LRN",):
                x = tensors[0].numpy()
                alpha = node.get("alpha", 0.0001)
                beta = node.get("beta", 0.75)
                size = node.get("size", 5)
                bias_val = node.get("bias", 1.0)
                pad = size // 2
                sq = x ** 2
                sq_pad = np.pad(sq, ((0,0), (pad,pad), (0,0), (0,0)), mode='constant')
                scale = np.zeros_like(x)
                for i in range(size):
                    scale += sq_pad[:, i:i+x.shape[1], :, :]
                scale = (bias_val + alpha * scale / size) ** beta
                return [fnn.tensor(x / scale, list(x.shape))]
            else:
                logger.warning("DAGModel: unsupported op '%s', passing through", op_type)
                return [tensors[0]] if tensors else None
        except Exception as e:
            logger.error("DAGModel: error dispatching op '%s': %s", op_type, e)
            return [tensors[0]] if tensors else None

    def _dispatch_conv(self, tensors: List, node: Dict) -> Optional[List]:
        x = tensors[0]
        node_name = node.get("name", "")
        weight_name = f"{node_name}.weight"
        bias_name = f"{node_name}.bias"

        weight = self.params.get(weight_name)
        if weight is None:
            logger.warning("Conv %s: weight not found", node_name)
            return [x]

        bias = self.params.get(bias_name, None)

        stride = node.get("stride", node.get("strides", [1, 1]))
        if isinstance(stride, list):
            stride = stride[0]
        padding = node.get("padding", node.get("pads", [0, 0, 0, 0]))
        if isinstance(padding, list):
            padding = padding[0]
        dilation = node.get("dilation", node.get("dilations", [1, 1]))
        if isinstance(dilation, list):
            dilation = dilation[0]
        groups = node.get("groups", 1)

        conv = fnn.Conv2d(
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=weight.shape[2],
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias is not None,
        )
        conv.set_weight(weight)
        if bias is not None:
            conv.set_bias(bias)
        return [conv(x)]

    def _dispatch_gemm(self, tensors: List, node: Dict) -> Optional[List]:
        x = tensors[0]
        node_name = node.get("name", "")
        weight_name = f"{node_name}.weight"
        bias_name = f"{node_name}.bias"

        weight = self.params.get(weight_name)
        if weight is None:
            logger.warning("Gemm %s: weight not found", node_name)
            return [x]

        bias = self.params.get(bias_name, None)
        trans_b = node.get("transB", node.get("trans_b", 0))

        if trans_b:
            w = weight.transpose(0, 1)
        else:
            w = weight

        result = fnn.matmul(x, w)
        if bias is not None:
            result = fnn.add(result, bias)
        return [result]

    def _dispatch_batch_norm(self, tensors: List, node: Dict) -> Optional[List]:
        x = tensors[0]
        node_name = node.get("name", "")
        weight = self.params.get(f"{node_name}.weight")
        bias = self.params.get(f"{node_name}.bias")
        mean = self.params.get(f"{node_name}.running_mean")
        var = self.params.get(f"{node_name}.running_var")

        if any(t is None for t in [weight, bias, mean, var]):
            logger.warning("BatchNorm %s: missing parameters", node_name)
            return [x]

        eps = node.get("epsilon", node.get("eps", 1e-5))
        normalized = fnn.div(fnn.sub(x, mean.reshape([1, -1, 1, 1])), fnn.sqrt(fnn.add(var.reshape([1, -1, 1, 1]), eps)))
        return [fnn.add(fnn.mul(normalized, weight.reshape([1, -1, 1, 1])), bias.reshape([1, -1, 1, 1]))]

    def _dispatch_max_pool(self, tensors: List, node: Dict) -> Optional[List]:
        x = tensors[0]
        kernel = node.get("kernel_size", node.get("kernel_shape", 2))
        if isinstance(kernel, list):
            kernel = kernel[0]
        stride = node.get("stride", node.get("strides", kernel))
        if isinstance(stride, list):
            stride = stride[0]
        padding = node.get("padding", node.get("pads", 0))
        if isinstance(padding, list):
            padding = padding[0]
        dilation = node.get("dilation", node.get("dilations", 1))
        if isinstance(dilation, list):
            dilation = dilation[0]

        pool = fnn.MaxPool2d(kernel, stride=stride, padding=padding, dilation=dilation)
        return [pool(x)]

    def _dispatch_avg_pool(self, tensors: List, node: Dict) -> Optional[List]:
        x = tensors[0]
        kernel = node.get("kernel_size", node.get("kernel_shape", 2))
        if isinstance(kernel, list):
            kernel = kernel[0]
        stride = node.get("stride", node.get("strides", kernel))
        if isinstance(stride, list):
            stride = stride[0]
        padding = node.get("padding", node.get("pads", 0))
        if isinstance(padding, list):
            padding = padding[0]

        pool = fnn.AvgPool2d(kernel, stride=stride, padding=padding)
        return [pool(x)]

    def _dispatch_global_avg_pool(self, tensors: List, node: Dict) -> Optional[List]:
        x = tensors[0]
        shape = x.shape
        h, w = shape[2], shape[3]
        pool = fnn.AvgPool2d(h, stride=1, padding=0)
        return [pool(x)]

    def __call__(self, *args, **kwargs):
        if args:
            return self.forward({"input": args[0]})
        return self.forward(kwargs)
