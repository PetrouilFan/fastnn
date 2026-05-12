"""ComputeGraph executor for fastnn v2.0.0.

Executes a ComputeGraph JSON by dispatching to fastnn ops in order.
This is the Python analog of the AOT compiled ExecutablePlan.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import fastnn as fnn
import fastnn._core as _core

logger = logging.getLogger(__name__)


class ComputeGraphExecutor:
    """Execute a ComputeGraph in node order."""
    
    def __init__(self, graph_data: Dict[str, Any]):
        """Initialize from a ComputeGraph JSON dict."""
        self.nodes = graph_data["nodes"]
        self.input_ids = graph_data["inputs"]
        self.output_ids = graph_data["outputs"]
        self.params = {}
        for name, param in graph_data.get("params", {}).items():
            arr = np.array(param["data"], dtype=np.float32).reshape(param["shape"])
            self.params[name] = fnn.tensor(arr, list(arr.shape))
        
        # Build node lookup
        self.node_map = {n["id"]: n for n in self.nodes}
    
    def forward(self, *inputs: fnn.Tensor) -> List[fnn.Tensor]:
        """Execute the compute graph with given input tensors."""
        tensors: Dict[int, fnn.Tensor] = {}
        
        # Bind inputs
        for i, (node_id, tensor) in enumerate(zip(self.input_ids, inputs)):
            tensors[node_id] = tensor
        
        # Execute nodes in order
        for node in self.nodes:
            nid = node["id"]
            if nid in tensors:
                continue  # Already have this tensor (input or constant)
            
            op = node["opcode"]
            input_tensors = [tensors[i] for i in node["inputs"]]
            
            # Dispatch to fastnn ops
            if op == "Relu":
                result = fnn.relu(input_tensors[0])
            elif op == "Add":
                result = fnn.add(input_tensors[0], input_tensors[1])
            elif op == "Sub":
                result = fnn.sub(input_tensors[0], input_tensors[1])
            elif op == "Mul":
                result = fnn.mul(input_tensors[0], input_tensors[1])
            elif op == "Div":
                result = fnn.div(input_tensors[0], input_tensors[1])
            elif op == "MatMul":
                result = fnn.matmul(input_tensors[0], input_tensors[1])
            elif op == "Conv2d":
                result = fnn.conv2d(input_tensors[0], input_tensors[1], 
                                     stride=node["attrs"].get("stride", 1),
                                     padding=node["attrs"].get("padding", 0))
            elif op == "Reshape":
                shape = eval(node["attrs"].get("shape", "[]"))
                result = input_tensors[0].reshape(shape)
            elif op == "Transpose":
                result = input_tensors[0].transpose()
            elif op == "Flatten":
                result = input_tensors[0].flatten()
            elif op == "Sigmoid":
                result = fnn.sigmoid(input_tensors[0])
            elif op == "Tanh":
                result = fnn.tanh(input_tensors[0])
            elif op == "Exp":
                result = fnn.exp(input_tensors[0])
            elif op == "Log":
                result = fnn.log(input_tensors[0])
            elif op == "Softmax":
                result = fnn.softmax(input_tensors[0])
            elif op == "ReduceMean":
                axis = int(node["attrs"].get("axis", "0"))
                result = input_tensors[0].mean(axis)
            elif op == "ReduceSum":
                axis = int(node["attrs"].get("axis", "0"))
                result = input_tensors[0].sum(axis)
            elif op == "Concat":
                result = fnn.cat(input_tensors, axis=int(node["attrs"].get("axis", "0")))
            elif op == "BatchNorm":
                result = fnn.batch_norm(input_tensors[0], input_tensors[1], input_tensors[2],
                                         input_tensors[3], input_tensors[4],
                                         momentum=0.9, eps=1e-5)
            elif op == "MaxPool":
                result = fnn.max_pool2d(input_tensors[0],
                                         kernel_size=int(node["attrs"].get("kernel_size", 2)),
                                         stride=int(node["attrs"].get("stride", 2)))
            elif op == "AvgPool":
                result = fnn.avg_pool2d(input_tensors[0],
                                         kernel_size=int(node["attrs"].get("kernel_size", 2)),
                                         stride=int(node["attrs"].get("stride", 2)))
            elif op == "Constant":
                result = self.params.get(node["name"])
                if result is None:
                    raise ValueError(f"Constant {node['name']} not found in params")
            elif op == "BiasAdd":
                result = fnn.add(input_tensors[0], input_tensors[1])
            elif op == "Gelu":
                result = fnn.gelu(input_tensors[0])
            elif op == "Silu":
                result = fnn.silu(input_tensors[0])
            elif op == "Sqrt":
                result = fnn.sqrt(input_tensors[0])
            elif op == "Neg":
                result = fnn.neg(input_tensors[0])
            elif op == "Abs":
                result = fnn.abs(input_tensors[0])
            else:
                raise ValueError(f"Unsupported opcode: {op}")
            
            tensors[nid] = result
        
        return [tensors[out_id] for out_id in self.output_ids]


class DAGModel:
    """Legacy wrapper that delegates to ComputeGraphExecutor."""
    
    def __init__(self, nodes, params, input_names, output_names):
        # Convert legacy format to ComputeGraph format
        graph_data = _convert_legacy_to_compute_graph(nodes, params, input_names, output_names)
        self._executor = ComputeGraphExecutor(graph_data)
    
    @classmethod
    def from_header(cls, header: dict, params: dict) -> "DAGModel":
        """Create from legacy .fnn header."""
        graph = header.get("graph", {})
        nodes = graph.get("nodes", [])
        input_names = [inp["name"] for inp in graph.get("inputs", [])]
        output_names = [out["name"] for out in graph.get("outputs", [])]
        return cls(nodes, params, input_names, output_names)
    
    def forward(self, *inputs):
        return self._executor.forward(*inputs)


def _convert_legacy_to_compute_graph(nodes, params, input_names, output_names):
    """Convert legacy DAGModel format to ComputeGraph JSON."""
    node_map = {}
    compute_nodes = []
    next_id = 1
    
    for i, node in enumerate(nodes):
        node_id = next_id
        next_id += 1
        node_map[id(node)] = node_id
        
        compute_nodes.append({
            "id": node_id,
            "opcode": node.get("op_type", node.get("op_code", "Unknown")),
            "inputs": [],
            "output_shape": {"shape": [], "dtype": "F32"},
            "attrs": node.get("attrs", {}),
            "name": node.get("name", f"node_{i}"),
        })
    
    return {
        "nodes": compute_nodes,
        "inputs": [],
        "outputs": [],
        "params": {k: {"data": v.tolist() if hasattr(v, 'tolist') else v,
                       "shape": list(v.shape) if hasattr(v, 'shape') else [],
                       "dtype": "F32"}
                  for k, v in params.items()},
    }
