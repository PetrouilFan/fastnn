"""Build runnable models from .fnn headers.

Supports both Sequential models (from PyTorch export) and
DAG models (from ONNX import).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastnn.io import read_fnn_header, read_fnn_parameters, SerializationError, MODEL_MAGIC

logger = logging.getLogger(__name__)


def build_model_from_fnn(path: str) -> Any:
    """Build a runnable model from a .fnn file.

    Automatically detects whether the file contains a sequential
    layer list (PyTorch export) or a full graph topology (ONNX import).

    Args:
        path: Path to .fnn file.

    Returns:
        A fastnn model (Sequential for PyTorch-exported, DAGExecutor for ONNX-imported).
    """
    with open(path, "rb") as f:
        magic, file_version, header, num_params = read_fnn_header(f)
        if magic != MODEL_MAGIC:
            raise SerializationError("Invalid .fnn file: missing magic bytes")

        if "graph" in header:
            return build_dag_model(header, path)
        elif "layers" in header:
            return build_sequential_model(path)
        else:
            raise ValueError("Unknown .fnn format: header has neither 'graph' nor 'layers'")


def fuse_silu(graph: dict) -> dict:
    """Fuse Sigmoid + Mul into SiLU where possible.

    Detects pattern: input -> Sigmoid -> Mul(input, sigmoid_output) -> ...
    Replaces with:   input -> Silu -> ...
    """
    nodes = graph.get("nodes", [])
    if not nodes:
        return graph

    consumer_map = {}
    for node in nodes:
        for inp in node.get("inputs", []):
            consumer_map.setdefault(inp, []).append(node)

    fused_nodes = []
    skip_names = set()
    silu_count = 0

    for node in nodes:
        name = node.get("name", "")
        if name in skip_names:
            continue

        op_type = node.get("op_type", "")

        if op_type == "Sigmoid":
            sig_outputs = node.get("outputs", [])
            sig_inputs = node.get("inputs", [])
            if not sig_outputs or not sig_inputs:
                fused_nodes.append(node)
                continue

            sig_output = sig_outputs[0]
            sig_input = sig_inputs[0]

            consumers = consumer_map.get(sig_output, [])

            # Only fuse when sigmoid has exactly one consumer that is a matching Mul
            if len(consumers) == 1:
                consumer = consumers[0]
                if consumer.get("op_type") == "Mul":
                    mul_inputs = consumer.get("inputs", [])
                    mul_name = consumer.get("name", "")

                    if len(mul_inputs) >= 2:
                        # Check both orderings: x * sigmoid(x) or sigmoid(x) * x
                        if (mul_inputs[0] == sig_input and mul_inputs[1] == sig_output) or \
                           (mul_inputs[0] == sig_output and mul_inputs[1] == sig_input):
                            silu_count += 1
                            silu_node = {
                                "name": f"fused_silu_{silu_count}",
                                "op_type": "Silu",
                                "inputs": [sig_input],
                                "outputs": consumer.get("outputs", []),
                            }
                            fused_nodes.append(silu_node)
                            skip_names.add(mul_name)
                            continue

            fused_nodes.append(node)

        elif op_type == "Mul" and name in skip_names:
            continue
        else:
            fused_nodes.append(node)

    graph["nodes"] = fused_nodes
    if silu_count:
        logger.info("Fused %d Sigmoid+Mul -> SiLU node(s)", silu_count)
    return graph


def build_dag_model(header: dict, path: str) -> Any:
    """Build a Rust DAGExecutor from an ONNX-imported .fnn file.

    Uses the high-performance Rust DAGExecutor for graph execution.
    """
    import fastnn as fnn

    # Load parameters from file
    with open(path, "rb") as f:
        _, _, _, num_params = read_fnn_header(f)
        params = read_fnn_parameters(f, num_params)

    graph = header.get("graph", {})
    onnx_nodes = graph.get("nodes", [])
    input_names = [inp.get("name", "") if isinstance(inp, dict) else inp for inp in graph.get("inputs", [])]
    output_names = [out.get("name", "") if isinstance(out, dict) else out for out in graph.get("outputs", [])]

    # Build param name mapping: ONNX initializer names -> {node_name}.{param_type}
    # Bridges the gap between ONNX node input references and how import_onnx stores params.
    OP_PARAM_SUFFIXES = [".weight", ".bias", ".running_mean", ".running_var", ".value", ".scale", ".beta", ".gamma"]
    # Known op-type to suffix mapping (positional)
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
    initializer_to_param = {}
    # Collect graph input names for exclusion
    graph_input_names = set()
    for inp in graph.get("inputs", []):
        name = inp.get("name", "") if isinstance(inp, dict) else inp
        graph_input_names.add(name)

    # Pass 1: Use OP_PARAM_MAP for known ops
    for node in onnx_nodes:
        node_name = node.get("name", "")
        if not node_name:
            continue
        inputs = node.get("inputs", [])
        op_type = node.get("op_type", "")

        suffixes = OP_PARAM_MAP.get(op_type, [])
        if suffixes and len(inputs) >= 2:
            for i, input_name in enumerate(inputs[1:], 1):
                if input_name in params or input_name in graph_input_names:
                    continue
                if i - 1 < len(suffixes):
                    param_name = node_name + suffixes[i - 1]
                    if param_name in params:
                        initializer_to_param[input_name] = param_name

    # Pass 2: Fallback for any remaining unresolved inputs
    # Try matching input names to params via common suffixes
    for node in onnx_nodes:
        node_name = node.get("name", "")
        if not node_name:
            continue
        inputs = node.get("inputs", [])
        prefix = node_name + "."
        for input_name in inputs:
            if input_name in params or input_name in graph_input_names or input_name in initializer_to_param:
                continue
            # Try to find a param whose name starts with node_name + "."
            for param_name in params:
                if param_name.startswith(prefix):
                    initializer_to_param[input_name] = param_name
                    break

    # Pass 3: Map Constant node output names to their .value params
    for node in onnx_nodes:
        op_type = node.get("op_type", "")
        if op_type not in ("Constant",):
            continue
        node_name = node.get("name", "")
        if not node_name:
            continue
        value_key = f"{node_name}.value"
        if value_key not in params:
            continue
        for output_name in node.get("outputs", []):
            if output_name not in initializer_to_param:
                initializer_to_param[output_name] = value_key

    # Run graph optimization passes (Sigmoid+Mul -> SiLU fusion, constant folding, dead node elimination, Conv+BN fusion)
    from fastnn.io.graph_optimizer import optimize_graph
    graph = fuse_silu(graph)
    header = optimize_graph(header)

    # Re-read nodes after optimization passes
    graph = header.get("graph", {})
    onnx_nodes = graph.get("nodes", [])

    # Convert ONNX nodes to DAGExecutor's node format
    dag_nodes = []
    for node in onnx_nodes:
        dag_node = {
            "name": node.get("name", ""),
            "op_type": node.get("op_type", ""),
            "inputs": ", ".join(node.get("inputs", [])),
            "outputs": ", ".join(node.get("outputs", [])),
        }
        # Copy attributes from the node (skip name/op_type/inputs/outputs)
        for key, value in node.items():
            if key not in ("name", "op_type", "inputs", "outputs"):
                if isinstance(value, (list, tuple)):
                    dag_node[key] = str(list(value))
                elif isinstance(value, bool):
                    dag_node[key] = "true" if value else "false"
                elif isinstance(value, (int, float)):
                    dag_node[key] = str(value)
                elif isinstance(value, str):
                    dag_node[key] = value
        dag_nodes.append(dag_node)

    # Convert numpy params to fastnn tensors
    fnn_params = {}
    for name, arr in params.items():
        fnn_params[name] = fnn.tensor(arr, list(arr.shape))
    # Add aliases for ONNX initializer names that differ from storage names
    for init_name, param_name in initializer_to_param.items():
        if init_name not in fnn_params and param_name in fnn_params:
            fnn_params[init_name] = fnn_params[param_name]

    # Build the Rust DAGExecutor
    executor = fnn.DAGExecutor(dag_nodes, fnn_params, input_names, output_names)
    return executor


def build_sequential_model(path: str) -> Any:
    """Build a Sequential model from a PyTorch-exported .fnn file."""
    from fastnn.io.export import load_fnn_model
    return load_fnn_model(path)
