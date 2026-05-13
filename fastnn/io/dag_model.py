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
    def __init__(self, nodes, params, input_names, output_names):
        self._executor = _core.AotExecutor(
            _normalize_nodes(nodes), params, input_names, output_names,
        )

    @classmethod
    def from_header(cls, header, params):
        g = header.get("graph", {})
        return cls(g.get("nodes", []), params,
                   [inp["name"] for inp in g.get("inputs", [])],
                   [out["name"] for out in g.get("outputs", [])])

    def forward(self, *inputs):
        return self._executor.forward(*inputs)
