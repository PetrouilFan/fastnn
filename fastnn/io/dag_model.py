"""ComputeGraph executor for fastnn v2.0.0."""
import logging

import fastnn._core as _core

logger = logging.getLogger(__name__)


def _normalize_nodes(nodes):
    """Convert list-valued inputs/outputs in node dicts to comma-separated strings.

    The Rust AotExecutor expects all values in the node dicts to be strings.
    Also flattens 'attrs' dict into the parent node dict.
    """
    normalized = []
    for node in nodes:
        n = dict(node)
        for key in ("inputs", "outputs"):
            val = n.get(key)
            if isinstance(val, list):
                n[key] = ",".join(str(v) for v in val)
            elif isinstance(val, int):
                n[key] = str(val)
            elif val is None:
                n[key] = ""
        # Also convert attrs values to strings for Rust FFI
        if "attrs" in n and isinstance(n["attrs"], dict):
            attrs = n.pop("attrs")
            for k, v in attrs.items():
                n[k] = str(v)
        # Convert all other values to strings
        for k, v in list(n.items()):
            if not isinstance(v, str):
                n[k] = str(v)
        normalized.append(n)
    return normalized


class DAGModel:
    def __init__(self, nodes, params, input_names, output_names, quantize=None, input_shapes=None):
        """Create an AOT-compiled executor for the given compute graph.

        Args:
            nodes: List of node dicts with op_type, inputs, outputs, and attrs.
            params: Dict mapping parameter names to Tensors.
            input_names: List of input tensor names.
            output_names: List of output tensor names.
            quantize: Optional quantization bit width. Pass 4 for U4x8
                (4-bit packed) or 8 for U8x4 (8-bit packed) quantization.
                None means no quantization (default f32).
            input_shapes: Optional dict mapping input names to shape lists.
        """
        import fastnn._core as _core
        self._executor = _core.AotExecutor(
            _normalize_nodes(nodes), params, input_names, output_names,
            input_shapes=input_shapes,
            quantize=quantize,
        )

    @classmethod
    def from_header(cls, header, params, quantize=None):
        g = header.get("graph", {})
        # Convert params from v3 format (tuples) to PyTensor objects if needed
        import fastnn
        import numpy as np
        tensor_params = {}
        for name, p in params.items():
            if isinstance(p, tuple) and len(p) >= 5:
                # v3 format: (data, dtype, scales, zeros, shape)
                data, dtype, scales, zeros, shape = p[0], p[1], p[2], p[3], p[4]
                if isinstance(data, np.ndarray):
                    tensor_params[name] = fastnn.tensor(data.tolist(), shape)
                elif isinstance(data, bytes):
                    # For packed data, create empty tensor with correct shape
                    # and then we'd need to set data differently
                    arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
                    tensor_params[name] = fastnn.tensor(arr.tolist(), shape)
                else:
                    tensor_params[name] = p  # Already a tensor or unknown format
            else:
                tensor_params[name] = p

        # Extract input shapes from graph Input nodes
        input_shapes = {}
        for node in g.get("nodes", []):
            if node.get("op_type") == "Input" or node.get("opcode") == "Input":
                name = node.get("name", "")
                output_shape = node.get("output_shape", {})
                shape_list = output_shape.get("shape", [])
                if name and shape_list:
                    # Convert "Known(N)" format to int
                    dims = []
                    for dim in shape_list:
                        if isinstance(dim, str) and dim.startswith("Known("):
                            dims.append(int(dim[6:-1]))
                        elif isinstance(dim, (int, float)):
                            dims.append(int(dim))
                        else:
                            dims.append(-1)  # symbolic
                    if dims:
                        input_shapes[name] = dims

        return cls(g.get("nodes", []), tensor_params,
                   [inp["name"] for inp in g.get("inputs", [])],
                   [out["name"] for out in g.get("outputs", [])],
                   quantize=quantize,
                   input_shapes=input_shapes)

    def forward(self, *inputs):
        return self._executor.forward(*inputs)
