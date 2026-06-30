"""Build runnable models from .fnn headers.

Supports both Sequential models (from PyTorch export) and
DAG models (from ONNX import).
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fastnn.io import read_fnn_header, read_fnn_parameters, SerializationError, MODEL_MAGIC

logger = logging.getLogger(__name__)


def _attr_to_str(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def build_model_from_fnn(path: str) -> Any:
    """Build a runnable model from a .fnn file.

    Automatically detects whether the file contains a sequential
    layer list (PyTorch export) or a full graph topology (ONNX import).

    Args:
        path: Path to .fnn file.

    Returns:
        A fastnn model (Sequential for PyTorch-exported, AotExecutor for ONNX-imported).
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


def build_dag_model(header: dict, path: str, quantize: int | None = None) -> Any:
    """Build a Rust AotExecutor from an ONNX-imported .fnn file.

    Uses the high-performance Rust AotExecutor for graph execution.

    Args:
        header: The .fnn file header dict containing graph metadata.
        path: Path to the .fnn parameters file.
        quantize: Optional quantization bit width. Pass 4 for I4x8
            (4-bit packed) or 8 for I8x4 (8-bit packed) quantization.
            None means no quantization (default f32).
    """
    import fastnn as fnn

    # Load parameters from file (version-aware)
    with open(path, "rb") as f:
        _, file_version, _, num_params = read_fnn_header(f)
        raw_params = read_fnn_parameters(f, num_params, version=file_version)

    # Unpack v3 format if needed: convert (data, dtype, scales, zeros, shape) tuples -> tensors + packed_params
    from fastnn.io import DTYPE_F32, DTYPE_I4, DTYPE_I8, DTYPE_F16, DTYPE_F8, DTYPE_F8R, DTYPE_F4
    params = {}
    packed_params_dict = {}
    for name, value in raw_params.items():
        if isinstance(value, tuple) and len(value) >= 4:
            # Tuple formats: v3.0 = (data, dtype, scales, zeros)
            #               v3.1 = (data, dtype, scales, zeros, shape)
            data, dtype = value[0], value[1]
            scales = value[2] if len(value) > 2 else []
            zeros = value[3] if len(value) > 3 else []
            shape = value[4] if len(value) > 4 else (
                list(data.shape) if hasattr(data, 'shape') else []
            )
            if dtype == DTYPE_F32:
                # F32: data is numpy array
                params[name] = fnn.tensor(data, list(data.shape))
            else:
                # Packed types require Rust-side PackedTensor for dequantization,
                # which is not exposed to Python yet.
                dtype_map = {
                    DTYPE_I4: "i4",
                    DTYPE_I8: "i8",
                    DTYPE_F16: "f16",
                    DTYPE_F8: "f8",
                    DTYPE_F8R: "f8r",
                    DTYPE_F4: "f4",
                }
                dtype_str = dtype_map.get(dtype, "f32")
                raise NotImplementedError(
                    f"Loading packed dtype '{dtype_str}' from .fnn files is not yet "
                    f"supported from Python. Use the Rust-side AotExecutor instead."
                )

        else:
            # v2 format: value is already a numpy array
            params[name] = fnn.tensor(value, list(value.shape))

    graph = header.get("graph", {})
    onnx_nodes = graph.get("nodes", [])
    input_names = [inp.get("name", "") if isinstance(inp, dict) else inp for inp in graph.get("inputs", [])]
    output_names = [out.get("name", "") if isinstance(out, dict) else out for out in graph.get("outputs", [])]

    # Extract input shapes from Input nodes BEFORE optimization passes
    # (dead node elimination removes Input nodes since they have no outputs)
    input_shapes: Dict[str, List[int]] = {}
    for nd in onnx_nodes:
        if nd.get("op_type", "") == "Input" or nd.get("opcode", "") == "Input":
            node_name = nd.get("name", "")
            shape_info = nd.get("output_shape", {})
            shape_list = shape_info.get("shape", [])
            if node_name and shape_list:
                dims = []
                for dim in shape_list:
                    if isinstance(dim, str) and dim.startswith("Known("):
                        dims.append(int(dim[6:-1]))
                    elif isinstance(dim, (int, float)):
                        dims.append(int(dim))
                    else:
                        dims.append(-1)
                if dims and all(d > 0 for d in dims):
                    input_shapes[node_name] = dims

    # Build param name mapping: ONNX initializer names -> {node_name}.{param_type}
    # Bridges the gap between ONNX node input references and how import_onnx stores params.
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
        raw_outputs = node.get("outputs", [])
        if isinstance(raw_outputs, str):
            out_list = [o.strip() for o in raw_outputs.split(",") if o.strip()]
        elif isinstance(raw_outputs, (list, tuple)):
            out_list = list(raw_outputs)
        else:
            out_list = []
        for output_name in out_list:
            if output_name not in initializer_to_param:
                initializer_to_param[output_name] = value_key

    # Run graph optimization passes (Sigmoid+Mul -> SiLU fusion, constant folding, dead node elimination, Conv+BN fusion)
    numpy_params = {}
    for pname, pval in params.items():
        if hasattr(pval, 'numpy'):
            numpy_params[pname] = np.asarray(pval.numpy())
        elif isinstance(pval, np.ndarray):
            numpy_params[pname] = pval

    from fastnn.io.graph_optimizer import optimize_graph
    graph = fuse_silu(graph)
    if not packed_params_dict:
        header = optimize_graph(header, params=numpy_params if numpy_params else None)

    # Re-read nodes after optimization passes
    graph = header.get("graph", {})
    onnx_nodes = graph.get("nodes", [])

    # Convert ONNX nodes to AotExecutor's node format
    dag_nodes = []
    for node in onnx_nodes:
        if node.get("op_type", "") == "Input" or node.get("opcode", "") == "Input":
            continue
        raw_inputs = node.get("inputs", [])
        raw_outputs = node.get("outputs", [])

        if isinstance(raw_inputs, str):
            inputs_str = raw_inputs
        elif isinstance(raw_inputs, (list, tuple)):
            inputs_str = ",".join(str(v) for v in raw_inputs)
        else:
            inputs_str = ""

        if isinstance(raw_outputs, str):
            outputs_str = raw_outputs
        elif isinstance(raw_outputs, (list, tuple)):
            outputs_str = ",".join(str(v) for v in raw_outputs)
        else:
            outputs_str = ""

        dag_node = {
            "name": node.get("name", ""),
            "op_type": node.get("op_type", ""),
            "inputs": inputs_str,
            "outputs": outputs_str,
        }
        for key, value in node.items():
            if key in ("name", "op_type", "inputs", "outputs"):
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    dag_node[sub_key] = _attr_to_str(sub_value)
            elif isinstance(value, (list, tuple)):
                dag_node[key] = str(list(value))
            elif isinstance(value, bool):
                dag_node[key] = "true" if value else "false"
            elif isinstance(value, (int, float)):
                dag_node[key] = str(value)
            elif isinstance(value, str):
                dag_node[key] = value
        dag_nodes.append(dag_node)

    # Replicate _make_fastnn_executor's constant folding for Shape→Gather→Add/Sub/Mul/Div chains
    # so Slice/Resize get resolved to string attributes instead of unresolved input references.
    # Seeds from params (weights/biases/constants), then propagates through shape-dependent chains.
    const_values = {}
    for pname, pval in numpy_params.items():
        if hasattr(pval, 'numpy'):
            const_values[pname] = np.asarray(pval.numpy())
        elif isinstance(pval, np.ndarray):
            const_values[pname] = pval

    known_shapes: Dict[str, List[int]] = {}
    for nd in onnx_nodes:
        node_name = nd.get("name", "")
        shape_info = nd.get("output_shape", {})
        shape_list = shape_info.get("shape", [])
        dims: List[int] = []
        if node_name and shape_list:
            for dim in shape_list:
                if isinstance(dim, str) and dim.startswith("Known("):
                    dims.append(int(dim[6:-1]))
                elif isinstance(dim, (int, float)):
                    dims.append(int(dim))
                else:
                    dims.append(-1)
            if dims and all(d > 0 for d in dims):
                known_shapes[node_name] = dims
        raw_outputs = nd.get("outputs", [])
        if isinstance(raw_outputs, str):
            out_names = [o.strip() for o in raw_outputs.split(",") if o.strip()]
        elif isinstance(raw_outputs, (list, tuple)):
            out_names = list(raw_outputs)
        else:
            out_names = []
        for out_name in out_names:
            if out_name and dims and all(d > 0 for d in dims):
                known_shapes[out_name] = dims

    for nd in onnx_nodes:
        op_type = nd.get("op_type", "")
        node_name = nd.get("name", "")
        inputs = nd.get("inputs", [])
        if isinstance(inputs, str):
            inputs = [s.strip() for s in inputs.split(",") if s.strip()]
        outputs = nd.get("outputs", [])
        if isinstance(outputs, str):
            outputs = [s.strip() for s in outputs.split(",") if s.strip()]
        out_name = outputs[0] if outputs else ""

        attrs = nd.get("attrs", {})
        if isinstance(attrs, dict):
            pass
        else:
            attrs = {}

        if op_type == "Constant":
            value_key = f"{node_name}.value"
            if value_key in const_values and out_name:
                const_values[out_name] = const_values[value_key]
        elif op_type == "Shape" and inputs and inputs[0] in known_shapes:
            if out_name:
                const_values[out_name] = np.asarray(known_shapes[inputs[0]], dtype=np.int64)
        elif op_type == "Gather" and len(inputs) >= 2:
            if inputs[0] in const_values and inputs[1] in const_values:
                axis = int(attrs.get("axis", 0))
                if out_name:
                    const_values[out_name] = np.take(
                        const_values[inputs[0]],
                        const_values[inputs[1]].astype(np.int64),
                        axis=axis,
                    )
        elif op_type in {"Add", "Sub", "Mul", "Div"} and len(inputs) >= 2:
            if inputs[0] in const_values and inputs[1] in const_values:
                a, b = const_values[inputs[0]], const_values[inputs[1]]
                if out_name:
                    if op_type == "Add":
                        const_values[out_name] = a + b
                    elif op_type == "Sub":
                        const_values[out_name] = a - b
                    elif op_type == "Mul":
                        const_values[out_name] = a * b
                    elif op_type == "Div":
                        const_values[out_name] = np.floor_divide(a, b)

    for node in onnx_nodes:
        op_type = node.get("op_type", "")
        node_name = node.get("name", "")
        inputs = node.get("inputs", [])
        if isinstance(inputs, str):
            inputs = [s.strip() for s in inputs.split(",") if s.strip()]

        if op_type == "Slice" and len(inputs) >= 3:
            dag = next((d for d in dag_nodes if d.get("name") == node_name), None)
            if dag is None:
                continue
            starts_val = const_values.get(inputs[1])
            ends_val = const_values.get(inputs[2]) if len(inputs) > 2 else None
            axes_val = const_values.get(inputs[3]) if len(inputs) > 3 else None
            steps_val = const_values.get(inputs[4]) if len(inputs) > 4 else None
            if starts_val is not None:
                dag["starts"] = str(int(np.asarray(starts_val).reshape(-1)[0]))
            if ends_val is not None:
                dag["ends"] = str(int(np.asarray(ends_val).reshape(-1)[0]))
            if axes_val is not None:
                dag["axes"] = str(int(np.asarray(axes_val).reshape(-1)[0]))
            if steps_val is not None:
                dag["steps"] = str(int(np.asarray(steps_val).reshape(-1)[0]))

        elif op_type == "Resize" and len(inputs) >= 3:
            dag = next((d for d in dag_nodes if d.get("name") == node_name), None)
            if dag is None:
                continue
            scales_val = const_values.get(inputs[2])
            if scales_val is not None:
                scales = np.asarray(scales_val, dtype=np.float32).reshape(-1)
                if scales.size >= 4:
                    dag["scale_h"] = str(int(scales[2]))
                    dag["scale_w"] = str(int(scales[3]))

    # params already contains fastnn tensors from the unpacking step above
    fnn_params = dict(params)  # copy, since we'll add aliases
    # Add aliases for ONNX initializer names that differ from storage names
    for init_name, param_name in initializer_to_param.items():
        if init_name not in fnn_params and param_name in fnn_params:
            fnn_params[init_name] = fnn_params[param_name]

    # Extract input shapes from Input nodes so Rust can run shape inference
    # Note: input_shapes is extracted earlier (before optimization) and reused here

    executor = fnn.AotExecutor(
        dag_nodes, fnn_params, input_names, output_names,
        input_shapes=input_shapes if input_shapes else None,
        quantize=quantize,
    )
    return executor


def build_sequential_model(path: str) -> Any:
    """Build a Sequential model from a PyTorch-exported .fnn file."""
    from fastnn.io.export import load_fnn_model
    return load_fnn_model(path)
