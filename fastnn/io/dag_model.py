"""ComputeGraph executor for fastnn v2.0.0."""
import logging

import fastnn._core as _core

logger = logging.getLogger(__name__)


def _normalize_nodes(nodes):
    """Convert list-valued inputs/outputs in node dicts to comma-separated strings.

    The Rust AotExecutor expects all values in the node dicts to be strings.
    """
    normalized = []
    for node in nodes:
        n = dict(node)
        for key in ("inputs", "outputs"):
            val = n.get(key)
            if isinstance(val, list):
                n[key] = ",".join(str(v) for v in val)
            elif val is None:
                n[key] = ""
        normalized.append(n)
    return normalized


class DAGModel:
    def __init__(self, nodes, params, input_names, output_names, quantize=None):
        """Create an AOT-compiled executor for the given compute graph.

        Args:
            nodes: List of node dicts with op_type, inputs, outputs, and attrs.
            params: Dict mapping parameter names to Tensors.
            input_names: List of input tensor names.
            output_names: List of output tensor names.
            quantize: Optional quantization bit width. Pass 4 for U4x8
                (4-bit packed) or 8 for U8x4 (8-bit packed) quantization.
                None means no quantization (default f32).
        """
        self._executor = _core.AotExecutor(
            _normalize_nodes(nodes), params, input_names, output_names,
            quantize=quantize,
        )

    @classmethod
    def from_header(cls, header, params, quantize=None):
        g = header.get("graph", {})
        return cls(g.get("nodes", []), params,
                   [inp["name"] for inp in g.get("inputs", [])],
                   [out["name"] for out in g.get("outputs", [])],
                   quantize=quantize)

    def forward(self, *inputs):
        return self._executor.forward(*inputs)
