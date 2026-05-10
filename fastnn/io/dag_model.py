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
                return [tensors[0]]
            elif op_type == "Gather" or op_type == "gatherop":
                return [tensors[0]]
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
                starts = node.get("starts", None)
                ends = node.get("ends", None)
                axes = node.get("axes", None)
                steps = node.get("steps", None)
                if starts is not None and ends is not None:
                    if axes is None:
                        axes = list(range(len(starts)))
                    if steps is None:
                        steps = [1] * len(starts)
                    result = x
                    for i, ax in enumerate(axes):
                        st = int(starts[i])
                        en = int(ends[i])
                        step = int(steps[i]) if i < len(steps) else 1
                        slices = [slice(None)] * len(result.shape)
                        slices[ax] = slice(st, en, step)
                        result = result[tuple(slices)]
                    return [result]
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
                axis = node.get("axis", 0)
                split_sizes = node.get("split", None)
                x = tensors[0]
                shape = x.shape
                dim_size = shape[axis] if axis < len(shape) else shape[0]
                num_outputs = len(node.get("outputs", [1]))
                if split_sizes is not None:
                    sizes = split_sizes
                else:
                    size = dim_size // num_outputs
                    sizes = [size] * num_outputs
                results = []
                start = 0
                for sz in sizes:
                    slices = [slice(None)] * len(shape)
                    slices[axis] = slice(start, start + sz)
                    result = x[tuple(slices)]
                    results.append(result)
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
                return [fnn.pow(tensors[0], tensors[1])] if len(tensors) > 1 else [tensors[0]]
            elif op_type == "BiasAdd":
                return [fnn.add(tensors[0], self.params.get(node.get("name", "") + ".bias", tensors[0]))]
            elif op_type == "BiasSub":
                return [fnn.sub(tensors[0], self.params.get(node.get("name", "") + ".bias", tensors[0]))]
            elif op_type == "Clip":
                min_val = node.get("min", None)
                max_val = node.get("max", None)
                result = tensors[0]
                if min_val is not None:
                    result = fnn.clamp(result, min=min_val)
                if max_val is not None:
                    result = fnn.clamp(result, max=max_val)
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
                logger.warning("GatherND not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("ScatterND",):
                logger.warning("ScatterND not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("ConvTranspose",):
                logger.warning("ConvTranspose not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("InstanceNormalization",):
                logger.warning("InstanceNormalization not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("LayerNormalization",):
                logger.warning("LayerNormalization not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("RMSNormalization",):
                logger.warning("RMSNormalization not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("SkipLayerNormalization",):
                logger.warning("SkipLayerNormalization not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("Attention", "MultiHeadAttention", "GroupQueryAttention"):
                logger.warning("Attention op %s not fully implemented, passing through", op_type)
                return [tensors[0]]
            elif op_type in ("GRU", "LSTM"):
                logger.warning("RNN op %s not fully implemented, passing through", op_type)
                return [tensors[0]]
            elif op_type in ("BiasGelu", "FastGelu"):
                logger.warning("Fused op %s not fully implemented, passing through", op_type)
                return [tensors[0]]
            elif op_type in ("DequantizeLinear", "QuantizeLinear"):
                logger.warning("Quantization op %s not fully implemented, passing through", op_type)
                return [tensors[0]]
            elif op_type in ("RotaryEmbedding",):
                logger.warning("RotaryEmbedding not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("EmbedLayerNormalization",):
                logger.warning("EmbedLayerNormalization not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("Einsum",):
                logger.warning("Einsum not fully implemented, passing through")
                return [tensors[0]]
            elif op_type in ("LRN",):
                logger.warning("LRN not fully implemented, passing through")
                return [tensors[0]]
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
