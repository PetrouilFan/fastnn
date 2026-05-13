"""ComputeGraph executor for fastnn v2.0.0."""
import logging

import fastnn._core as _core

logger = logging.getLogger(__name__)


class DAGModel:
    def __init__(self, nodes, params, input_names, output_names):
        self._executor = _core.AotExecutor(nodes, params, input_names, output_names)

    @classmethod
    def from_header(cls, header, params):
        g = header.get("graph", {})
        return cls(g.get("nodes", []), params,
                   [inp["name"] for inp in g.get("inputs", [])],
                   [out["name"] for out in g.get("outputs", [])])

    def forward(self, *inputs):
        return self._executor.forward(*inputs)
