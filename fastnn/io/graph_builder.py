"""Build runnable models from .fnn headers.

Supports both Sequential models (from PyTorch export) and
DAG models (from ONNX import).
"""

import ast
import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

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


def _expression_capacity(expression: str, bounds: Mapping[str, int]) -> int:
    def evaluate(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in bounds:
                raise ValueError(f"missing capacity for symbolic dimension '{node.id}'")
            return bounds[node.id]
        if isinstance(node, ast.BinOp):
            lhs, rhs = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return max(0, lhs - rhs)
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                if rhs == 0:
                    raise ValueError("symbolic dimension capacity divides by zero")
                return lhs // rhs
            if isinstance(node.op, ast.Pow):
                return lhs**rhs
        raise ValueError(f"unsupported symbolic dimension expression: {expression}")

    capacity = evaluate(ast.parse(expression.replace("^", "**"), mode="eval"))
    if capacity < 0 or capacity > (1 << 64) - 1:
        raise ValueError(f"symbolic dimension capacity is outside u64: {capacity}")
    return capacity


def _bound_dimension_descriptor(descriptor: str, bounds: Mapping[str, int]) -> str:
    if not (descriptor.startswith("Symbol(") and descriptor.endswith(")")):
        return descriptor
    expression = descriptor[7:-1].strip()
    try:
        capacity = _expression_capacity(expression, bounds)
    except ValueError as error:
        if "missing capacity" in str(error):
            return descriptor
        raise
    return f"Bounded({expression};{capacity})"


def _dimension_expression(descriptor: Any) -> str:
    """Return the semantic integer expression carried by a shape descriptor."""
    if isinstance(descriptor, (int, np.integer)):
        return str(int(descriptor))
    if not isinstance(descriptor, str):
        raise ValueError(f"unsupported shape descriptor {descriptor!r}")
    if descriptor.startswith("Known(") and descriptor.endswith(")"):
        return descriptor[6:-1]
    if descriptor.startswith("Symbol(") and descriptor.endswith(")"):
        return descriptor[7:-1]
    if descriptor.startswith("Bounded(") and descriptor.endswith(")"):
        return descriptor[8:-1].rsplit(";", 1)[0]
    raise ValueError(f"unsupported shape descriptor {descriptor!r}")


def _reshape_descriptors(input_shape: List[Any], target: List[Any], bounds: Mapping[str, int]) -> List[str]:
    """Resolve ONNX 0/-1 reshape semantics without freezing runtime dimensions."""
    resolved: List[Any] = []
    inferred_axis: Optional[int] = None
    for axis, value in enumerate(target):
        if isinstance(value, (int, np.integer)):
            value = int(value)
            if value == 0:
                if axis >= len(input_shape):
                    raise ValueError("Reshape target 0 refers past the input rank")
                resolved.append(input_shape[axis])
            elif value == -1:
                if inferred_axis is not None:
                    raise ValueError("Reshape target contains more than one -1")
                inferred_axis = axis
                resolved.append(None)
            elif value > 0:
                resolved.append(f"Known({value})")
            else:
                raise ValueError(f"unsupported Reshape target dimension {value}")
        else:
            resolved.append(value)
    if inferred_axis is not None:
        input_factors = [_dimension_expression(d) for d in input_shape]
        target_factors = [
            _dimension_expression(d) for d in resolved if d is not None
        ]
        remaining = list(input_factors)
        for factor in target_factors:
            if factor in remaining:
                remaining.remove(factor)
            else:
                break
        else:
            expression = "*".join(f"({factor})" for factor in remaining) or "1"
            resolved[inferred_axis] = f"Symbol({expression})"
            return [
                _bound_dimension_descriptor(d, bounds) if isinstance(d, str) else d
                for d in resolved
            ]
        input_product = "*".join(f"({factor})" for factor in input_factors) or "1"
        known_product = "*".join(f"({factor})" for factor in target_factors) or "1"
        resolved[inferred_axis] = f"Symbol(({input_product})/({known_product}))"
    return [_bound_dimension_descriptor(d, bounds) if isinstance(d, str) else d for d in resolved]


def build_model_from_fnn(
    path: str, *, symbolic_dim_bounds: Optional[Mapping[str, int]] = None
) -> Any:
    """Build a runnable model from a .fnn file.

    Automatically detects whether the file contains a sequential
    layer list (PyTorch export) or a full graph topology (ONNX import).

    Args:
        path: Path to .fnn file.
        symbolic_dim_bounds: Optional allocation capacities keyed by ONNX symbolic
            dimension name. Live dimensions remain dynamic and are validated against
            these bounds.

    Returns:
        A fastnn model (Sequential for PyTorch-exported, AotExecutor for ONNX-imported).
    """
    bounds = dict(symbolic_dim_bounds or {})
    for name, capacity in bounds.items():
        if not isinstance(name, str) or not name or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("symbolic_dim_bounds must map non-empty names to positive integers")

    with open(path, "rb") as f:
        magic, file_version, header, num_params = read_fnn_header(f)
        if magic != MODEL_MAGIC:
            raise SerializationError("Invalid .fnn file: missing magic bytes")

        if "graph" in header:
            return build_dag_model(header, path, symbolic_dim_bounds=bounds)
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


def build_dag_model(
    header: dict,
    path: str,
    quantize: int | None = None,
    symbolic_dim_bounds: Optional[Mapping[str, int]] = None,
) -> Any:
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

    dimension_bounds = dict(symbolic_dim_bounds or {})

    # Load parameters from file (version-aware)
    with open(path, "rb") as f:
        _, file_version, _, num_params = read_fnn_header(f)
        raw_params = read_fnn_parameters(f, num_params, version=file_version)

    # Unpack v3 format if needed: convert (data, dtype, scales, zeros, shape) tuples -> tensors + packed_params
    from fastnn.io import (
        DTYPE_F32,
        DTYPE_I4,
        DTYPE_I8,
        DTYPE_F16,
        DTYPE_F8,
        DTYPE_F8R,
        DTYPE_F4,
        DTYPE_I64,
        DTYPE_I32,
        DTYPE_BOOL,
    )
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
            if dtype in (DTYPE_F32, DTYPE_I64, DTYPE_I32, DTYPE_BOOL):
                # Plain scalar tensors preserve their numpy dtype through fnn.tensor.
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
    # Serialized DAG edges use comma-delimited strings. Normalize them before
    # graph optimization; optimizer passes operate on tensor-name lists and must
    # never iterate individual characters from a serialized edge name.
    for node in onnx_nodes:
        for edge_key in ("inputs", "outputs"):
            edges = node.get(edge_key, [])
            if isinstance(edges, str):
                node[edge_key] = [edge.strip() for edge in edges.split(",") if edge.strip()]
    input_names = [inp.get("name", "") if isinstance(inp, dict) else inp for inp in graph.get("inputs", [])]
    output_names = [out.get("name", "") if isinstance(out, dict) else out for out in graph.get("outputs", [])]

    # Extract input shapes from Input nodes BEFORE optimization passes
    # (dead node elimination removes Input nodes since they have no outputs)
    input_shapes: Dict[str, List[int]] = {}
    symbolic_input_shapes: Dict[str, List[str]] = {}
    symbolic_dimension_ids: Dict[str, int] = {}
    next_symbolic_id = 1
    for nd in onnx_nodes:
        if nd.get("op_type", "") == "Input" or nd.get("opcode", "") == "Input":
            node_name = nd.get("name", "")
            shape_info = nd.get("output_shape", {})
            shape_list = shape_info.get("shape", [])
            if node_name and shape_list:
                dims = []
                symbolic_dims = []
                for axis, dim in enumerate(shape_list):
                    if isinstance(dim, str) and dim.startswith("Known("):
                        dims.append(int(dim[6:-1]))
                        symbolic_dims.append(dim)
                    elif isinstance(dim, str) and dim.startswith("Symbol("):
                        symbol_name = dim[7:-1]
                        if symbol_name not in symbolic_dimension_ids:
                            symbolic_dimension_ids[symbol_name] = next_symbolic_id
                            next_symbolic_id += 1
                        dims.append(-symbolic_dimension_ids[symbol_name])
                        symbolic_dims.append(
                            _bound_dimension_descriptor(dim, dimension_bounds)
                        )
                    elif dim == "Unknown":
                        unknown_name = f"{node_name}:axis:{axis}"
                        symbolic_dimension_ids[unknown_name] = next_symbolic_id
                        dims.append(-next_symbolic_id)
                        symbolic_dims.append(f"Symbol({unknown_name})")
                        next_symbolic_id += 1
                    elif isinstance(dim, (int, float)):
                        dims.append(int(dim))
                        symbolic_dims.append(str(int(dim)))
                    else:
                        raise ValueError(
                            f"input {node_name!r} has unsupported dimension descriptor {dim!r}"
                        )
                input_shapes[node_name] = dims
                symbolic_input_shapes[node_name] = symbolic_dims

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
        if op_type == "Constant":
            outputs = node.get("outputs", [])
            value_name = node_name + ".value"
            if outputs and value_name in params:
                initializer_to_param[outputs[0]] = value_name
        elif suffixes and len(inputs) >= 2:
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
            # v3 ONNX Constant payloads are stored under the node name itself.
            # Keep accepting the older `.value` convention for legacy artifacts.
            value_key = node_name
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
    for pname, raw_value in raw_params.items():
        if isinstance(raw_value, tuple) and raw_value:
            numpy_params[pname] = np.asarray(raw_value[0])
        elif isinstance(raw_value, np.ndarray):
            numpy_params[pname] = raw_value
        elif pname in params and hasattr(params[pname], "numpy"):
            numpy_params[pname] = np.asarray(params[pname].numpy())

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
                    if sub_key == "shape" and isinstance(sub_value, (list, tuple)):
                        sub_value = [
                            _bound_dimension_descriptor(dimension, dimension_bounds)
                            if isinstance(dimension, str)
                            else dimension
                            for dimension in sub_value
                        ]
                    dag_node[sub_key] = _attr_to_str(sub_value)
            elif isinstance(value, (list, tuple)):
                dag_node[key] = str(list(value))
            elif isinstance(value, bool):
                dag_node[key] = "true" if value else "false"
            elif isinstance(value, (int, float)):
                dag_node[key] = str(value)
            elif isinstance(value, str):
                dag_node[key] = value
        output_shape = node.get("output_shape", {})
        if isinstance(output_shape, dict) and output_shape.get("shape"):
            dag_node["output_rank"] = str(len(output_shape["shape"]))
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
    # Optimization can eliminate Constant nodes while consumers retain their
    # tensor-edge names. Preserve those aliases so static shape inputs remain
    # promotable after the optimized graph is re-read.
    for tensor_name, param_name in initializer_to_param.items():
        if param_name in const_values:
            const_values[tensor_name] = const_values[param_name]

    known_shapes: Dict[str, List[int]] = {}
    tensor_shapes: Dict[str, List[Any]] = {
        name: list(shape) for name, shape in symbolic_input_shapes.items()
    }
    for name, value in numpy_params.items():
        tensor_shapes[name] = [f"Known({extent})" for extent in np.asarray(value).shape]
    for tensor_name, param_name in initializer_to_param.items():
        if param_name in tensor_shapes:
            tensor_shapes[tensor_name] = list(tensor_shapes[param_name])
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
            if out_name and shape_list:
                tensor_shapes[out_name] = [
                    _bound_dimension_descriptor(dimension, dimension_bounds)
                    if isinstance(dimension, str) else dimension
                    for dimension in shape_list
                ]
            if out_name and dims and all(d > 0 for d in dims):
                known_shapes[out_name] = dims

    shape_values: Dict[str, List[Any]] = {}
    for name, value in const_values.items():
        constant = np.asarray(value).reshape(-1)
        if np.issubdtype(constant.dtype, np.integer):
            shape_values[name] = [int(item) for item in constant]

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
            if value_key not in const_values:
                value_key = node_name
            if value_key in const_values and out_name:
                const_values[out_name] = const_values[value_key]
        elif op_type == "Shape" and inputs and inputs[0] in known_shapes:
            if out_name:
                const_values[out_name] = np.asarray(known_shapes[inputs[0]], dtype=np.int64)
        elif op_type == "Shape" and inputs and inputs[0] in tensor_shapes:
            if out_name:
                shape_values[out_name] = list(tensor_shapes[inputs[0]])
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

    # Resolve symbolic shape-value chains to a fixed point. Runtime Reshape may
    # feed Shape/Gather/Concat into a later runtime Reshape.
    changed = True
    while changed:
        changed = False
        for node in onnx_nodes:
            op_type = node.get("op_type", "")
            node_name = node.get("name", "")
            inputs = node.get("inputs", [])
            if isinstance(inputs, str):
                inputs = [value.strip() for value in inputs.split(",") if value.strip()]
            outputs = node.get("outputs", [])
            if isinstance(outputs, str):
                outputs = [value.strip() for value in outputs.split(",") if value.strip()]
            out_name = outputs[0] if outputs else ""
            attrs = node.get("attrs", {}) if isinstance(node.get("attrs", {}), dict) else {}
            candidate = None
            if op_type == "Shape" and inputs and inputs[0] in tensor_shapes:
                candidate = list(tensor_shapes[inputs[0]])
            elif op_type == "Gather" and len(inputs) >= 2 and inputs[0] in shape_values and inputs[1] in const_values and int(attrs.get("axis", 0)) == 0:
                source = shape_values[inputs[0]]
                candidate = []
                for raw_index in np.asarray(const_values[inputs[1]]).astype(np.int64).reshape(-1):
                    index = int(raw_index)
                    if index < 0:
                        index += len(source)
                    if index < 0 or index >= len(source):
                        raise ValueError(f"shape-value Gather {node_name!r} index is out of range")
                    candidate.append(source[index])
            elif op_type == "Slice" and inputs and inputs[0] in shape_values:
                starts = attrs.get("starts", const_values.get(inputs[1]) if len(inputs) > 1 else None)
                ends = attrs.get("ends", const_values.get(inputs[2]) if len(inputs) > 2 else None)
                axes = attrs.get("axes", const_values.get(inputs[3]) if len(inputs) > 3 else [0])
                if starts is not None and ends is not None:
                    def _shape_ints(value):
                        if isinstance(value, (list, tuple)):
                            return [int(item) for item in value]
                        return [int(item) for item in str(value).strip("[]").split(",") if item]
                    starts, ends, axes = _shape_ints(starts), _shape_ints(ends), _shape_ints(axes)
                    if len(starts) == len(ends) == len(axes) == 1 and axes[0] == 0:
                        candidate = list(shape_values[inputs[0]][starts[0]:ends[0]])
            elif op_type in {"Unsqueeze", "Squeeze", "Cast"} and inputs and inputs[0] in shape_values:
                candidate = list(shape_values[inputs[0]])
            elif op_type == "Concat" and int(attrs.get("axis", 0)) == 0 and inputs and all(name in shape_values for name in inputs):
                candidate = [value for name in inputs for value in shape_values[name]]
            if out_name and candidate is not None and shape_values.get(out_name) != candidate:
                shape_values[out_name] = candidate
                changed = True

            tensor_candidate = None
            if op_type == "Gather" and len(inputs) >= 2 and inputs[0] in tensor_shapes and inputs[1] in tensor_shapes:
                data_shape = tensor_shapes[inputs[0]]
                index_shape = tensor_shapes[inputs[1]]
                axis = int(attrs.get("axis", 0))
                if axis < 0:
                    axis += len(data_shape)
                if 0 <= axis < len(data_shape):
                    tensor_candidate = data_shape[:axis] + index_shape + data_shape[axis + 1:]
            elif op_type in {"Add", "Sub", "Mul", "Div", "Pow", "Max", "Min"}:
                known = [tensor_shapes[name] for name in inputs if name in tensor_shapes]
                if known:
                    tensor_candidate = list(max(known, key=len))
            elif op_type in {"Sqrt", "Tanh", "Cast", "Identity", "Dropout"} and inputs and inputs[0] in tensor_shapes:
                tensor_candidate = list(tensor_shapes[inputs[0]])
            elif op_type == "ReduceMean" and inputs and inputs[0] in tensor_shapes:
                tensor_candidate = list(tensor_shapes[inputs[0]])
                axes = attrs.get("axes", [])
                if not isinstance(axes, (list, tuple)):
                    axes = [int(value) for value in str(axes).split(",") if value]
                if int(attrs.get("keepdims", attrs.get("keepdim", 1))):
                    for axis in axes:
                        normalized = int(axis) % len(tensor_candidate)
                        tensor_candidate[normalized] = "Known(1)"
            if out_name and tensor_candidate is not None and tensor_shapes.get(out_name) != tensor_candidate:
                tensor_shapes[out_name] = tensor_candidate
                changed = True

            if op_type == "Reshape" and len(inputs) >= 2 and inputs[0] in tensor_shapes and inputs[1] in shape_values:
                inferred_shape = _reshape_descriptors(tensor_shapes[inputs[0]], shape_values[inputs[1]], dimension_bounds)
                dag = next((item for item in dag_nodes if item.get("name") == node_name), None)
                if dag is not None:
                    dag["shape"] = _attr_to_str(inferred_shape)
                for output in outputs:
                    if tensor_shapes.get(output) != inferred_shape:
                        tensor_shapes[output] = list(inferred_shape)
                        changed = True

    for node in onnx_nodes:
        op_type = node.get("op_type", "")
        node_name = node.get("name", "")
        inputs = node.get("inputs", [])
        if isinstance(inputs, str):
            inputs = [s.strip() for s in inputs.split(",") if s.strip()]

        if op_type == "Reshape" and len(inputs) >= 2:
            dag = next((d for d in dag_nodes if d.get("name") == node_name), None)
            if dag is None:
                continue
            target_val = const_values.get(inputs[1])
            if target_val is not None:
                target = np.asarray(target_val).reshape(-1)
                if not np.all(np.isfinite(target)) or not np.all(target == np.trunc(target)):
                    raise ValueError(f"Reshape node {node_name!r} has a non-integral constant target")
                dag["target_shape"] = _attr_to_str([int(value) for value in target])


        elif op_type == "Slice" and len(inputs) >= 3:
            dag = next((d for d in dag_nodes if d.get("name") == node_name), None)
            if dag is None:
                continue
            starts_val = const_values.get(inputs[1])
            ends_val = const_values.get(inputs[2]) if len(inputs) > 2 else None
            axes_val = const_values.get(inputs[3]) if len(inputs) > 3 else None
            steps_val = const_values.get(inputs[4]) if len(inputs) > 4 else None
            def _slice_ints(value, label):
                array = np.asarray(value).reshape(-1)
                if not np.all(np.isfinite(array)) or not np.all(array == np.trunc(array)):
                    raise ValueError(f"Slice node {node_name!r} has non-integral {label}")
                return _attr_to_str([int(item) for item in array])

            if starts_val is not None and "starts" not in dag:
                dag["starts"] = _slice_ints(starts_val, "starts")
            if ends_val is not None and "ends" not in dag:
                dag["ends"] = _slice_ints(ends_val, "ends")
            if axes_val is not None and "axes" not in dag:
                dag["axes"] = _slice_ints(axes_val, "axes")
            if steps_val is not None and "steps" not in dag:
                dag["steps"] = _slice_ints(steps_val, "steps")

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
        symbolic_input_shapes=symbolic_input_shapes if symbolic_input_shapes else None,
        quantize=quantize,
    )
    return executor


def build_sequential_model(path: str) -> Any:
    """Build a Sequential model from a PyTorch-exported .fnn file."""
    from fastnn.io.export import load_fnn_model
    return load_fnn_model(path)
