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

            if tensors is None or not tensors:
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
                return [tensors[0]] if tensors else None
            elif op_type in ("Loop", "loopop", "If", "ifop"):
                logger.warning("Control flow op %s not yet supported, passing through", op_type)
                return [tensors[0]] if tensors else None
            elif op_type in ("Erf", "erfop"):
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
