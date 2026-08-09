"""Build runnable models from .fnn headers.

Supports both Sequential models (from PyTorch export) and
DAG models (from ONNX import).
"""

import ast
import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


def _integer_list(value: Any) -> List[int]:
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.reshape(-1)]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in str(value).strip("[]").split(",") if item]


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
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
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
    path: str,
    *,
    symbolic_dim_bounds: Optional[Mapping[str, int]] = None,
    quantize: int | str | None = None,
    diagnostic_outputs: Optional[Sequence[str]] = None,
    w4a8_clip_ratios: Optional[Mapping[str, Sequence[float]]] = None,
) -> Any:
    """Build a runnable model from a .fnn file.

    Automatically detects whether the file contains a sequential
    layer list (PyTorch export) or a full graph topology (ONNX import).

    Args:
        path: Path to .fnn file.
        symbolic_dim_bounds: Optional allocation capacities keyed by ONNX symbolic
            dimension name. Live dimensions remain dynamic and are validated against
            these bounds.
        quantize: Optional AOT compile target. Grouped dynamic W4A8 accepts
            ``"w4a8-g32"``, ``"w4a8-g64"``, or ``"w4a8-g128"``. Sensitive
            MatMuls can remain in native precision by appending comma-separated
            node-name substrings, for example
            ``"w4a8-g128:exclude=attn/c_proj,lm_head"``.
        diagnostic_outputs: Optional additional graph values to retain and return.
            This is intended for deterministic intermediate-output audits.
        w4a8_clip_ratios: Optional calibrated clipping ratios keyed by exact
            MatMul/Gemm provenance name.

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
            return build_dag_model(
                header,
                path,
                quantize=quantize,
                symbolic_dim_bounds=bounds,
                diagnostic_outputs=diagnostic_outputs,
                w4a8_clip_ratios=w4a8_clip_ratios,
            )
        elif "layers" in header:
            if quantize is not None:
                raise ValueError("quantize is only supported for graph .fnn artifacts")
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
    quantize: int | str | None = None,
    symbolic_dim_bounds: Optional[Mapping[str, int]] = None,
    diagnostic_outputs: Optional[Sequence[str]] = None,
    w4a8_clip_ratios: Optional[Mapping[str, Sequence[float]]] = None,
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
    calibrated_ratios = {
        name: [float(value) for value in values]
        for name, values in (w4a8_clip_ratios or {}).items()
    }
    if any(not isinstance(name, str) or not name for name in calibrated_ratios):
        raise ValueError("w4a8_clip_ratios keys must be non-empty strings")

    requested_diagnostics = list(diagnostic_outputs or ())
    if any(not isinstance(name, str) or not name for name in requested_diagnostics):
        raise ValueError("diagnostic_outputs must contain non-empty strings")
    if len(set(requested_diagnostics)) != len(requested_diagnostics):
        raise ValueError("diagnostic_outputs must not contain duplicates")

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
    produced_values = {
        value
        for node in graph.get("nodes", [])
        for value in node.get("outputs", [])
    }
    unknown_diagnostics = sorted(set(requested_diagnostics) - produced_values)
    if unknown_diagnostics:
        raise ValueError(
            "diagnostic_outputs are not produced by the graph: "
            + ", ".join(unknown_diagnostics)
        )
    output_names.extend(name for name in requested_diagnostics if name not in output_names)
    if requested_diagnostics:
        graph["outputs"] = [*graph.get("outputs", []), *[
            {"name": name} for name in requested_diagnostics if name not in {
                output.get("name", "") if isinstance(output, dict) else output
                for output in graph.get("outputs", [])
            }
        ]]

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

    # Preserve serialized tensor contracts across Python graph optimization.
    # Folded Constant nodes retain the original output name but historically
    # lost scalar rank and dtype metadata, turning scalar shape arithmetic into
    # rank-1 tensors during Rust shape inference.
    output_contracts = {}
    for node in onnx_nodes:
        contract = node.get("output_shape")
        raw_outputs = node.get("outputs", [])
        if isinstance(raw_outputs, str):
            contract_outputs = [value.strip() for value in raw_outputs.split(",") if value.strip()]
        elif isinstance(raw_outputs, (list, tuple)):
            contract_outputs = list(raw_outputs)
        else:
            contract_outputs = []
        if isinstance(contract, dict):
            for output_name in contract_outputs:
                output_contracts[output_name] = dict(contract)

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

    # Re-read nodes after optimization passes and restore contracts on folded aliases.
    graph = header.get("graph", {})
    onnx_nodes = graph.get("nodes", [])
    for node in onnx_nodes:
        if isinstance(node.get("output_shape"), dict):
            continue
        raw_outputs = node.get("outputs", [])
        if isinstance(raw_outputs, str):
            contract_outputs = [value.strip() for value in raw_outputs.split(",") if value.strip()]
        elif isinstance(raw_outputs, (list, tuple)):
            contract_outputs = list(raw_outputs)
        else:
            contract_outputs = []
        if len(contract_outputs) == 1 and contract_outputs[0] in output_contracts:
            node["output_shape"] = dict(output_contracts[contract_outputs[0]])

    # Convert ONNX nodes to AotExecutor's node format
    output_producers = {}
    for producer in onnx_nodes:
        producer_outputs = producer.get("outputs", [])
        if isinstance(producer_outputs, str):
            producer_outputs = [value.strip() for value in producer_outputs.split(",") if value.strip()]
        for output in producer_outputs:
            output_producers[output] = producer

    dag_nodes = []
    legacy_unknown_shape_outputs: set[str] = set()
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
        elif node.get("op_type") == "Constant":
            dag_node["output_rank"] = "0"
        elif node.get("op_type") == "Gather":
            gather_input_names = [
                value.strip() for value in inputs_str.split(",") if value.strip()
            ]
            data_producer = (
                output_producers.get(gather_input_names[0]) if gather_input_names else None
            )
            # Scalar gathers in exporter shape programs consume a Shape result.
            # Do not apply this to embedding gathers whose output metadata may be
            # absent: their runtime index tensor preserves its own rank.
            if data_producer is not None and data_producer.get("op_type") == "Shape":
                dag_node["output_rank"] = "0"
        if (
            isinstance(output_shape, dict)
            and dag_node.get("output_rank") != "0"
            and (
                output_shape.get("shape") == []
                or (
                    node.get("op_type") == "Slice"
                    and any(
                        not (
                            isinstance(dimension, str)
                            and dimension.startswith("Known(")
                        )
                        and not isinstance(dimension, (int, float))
                        for dimension in output_shape.get("shape", [])
                    )
                )
            )
        ):
            legacy_unknown_shape_outputs.update(
                value.strip() for value in outputs_str.split(",") if value.strip()
            )
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
        elif (
            constant.size <= 64
            and np.issubdtype(constant.dtype, np.floating)
            and np.all(np.isfinite(constant))
            and np.all(constant == np.trunc(constant))
        ):
            # Serialized ONNX shape constants currently travel through the F32
            # parameter container even when their graph dtype is I64. Recover
            # only small, exactly integral payloads; large model weights must
            # never be treated as shape programs.
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
            if op_type == "Constant" and out_name in const_values:
                constant = np.asarray(const_values[out_name]).reshape(-1)
                if np.issubdtype(constant.dtype, np.integer):
                    candidate = [int(item) for item in constant]
                elif (
                    constant.size <= 64
                    and np.issubdtype(constant.dtype, np.floating)
                    and np.all(np.isfinite(constant))
                    and np.all(constant == np.trunc(constant))
                ):
                    candidate = [int(item) for item in constant]
            elif op_type == "Shape" and inputs and inputs[0] in tensor_shapes:
                candidate = list(tensor_shapes[inputs[0]])
            elif op_type == "Gather" and len(inputs) >= 2 and inputs[0] in shape_values and inputs[1] in const_values and int(attrs.get("axis", 0)) == 0:
                source = shape_values[inputs[0]]
                candidate = []
                for raw_index in np.asarray(const_values[inputs[1]]).astype(np.int64).reshape(-1):
                    index = int(raw_index)
                    if index < 0:
                        index += len(source)
                    if index < 0 or index >= len(source):
                        candidate = None
                        break
                    candidate.append(source[index])
            elif op_type in {"Add", "Sub", "Mul", "Div"} and len(inputs) >= 2 and inputs[0] in shape_values and inputs[1] in shape_values:
                left, right = shape_values[inputs[0]], shape_values[inputs[1]]
                if len(left) == 1 and len(right) > 1:
                    left = left * len(right)
                elif len(right) == 1 and len(left) > 1:
                    right = right * len(left)
                if len(left) == len(right):
                    operator = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}[op_type]
                    candidate = []
                    for a, b in zip(left, right):
                        lhs_expr = _dimension_expression(a)
                        rhs_expr = _dimension_expression(b)
                        try:
                            lhs_value, rhs_value = int(lhs_expr), int(rhs_expr)
                        except ValueError:
                            candidate.append(
                                _bound_dimension_descriptor(
                                    f"Symbol(({lhs_expr}){operator}({rhs_expr}))",
                                    dimension_bounds,
                                )
                            )
                            continue
                        if op_type == "Add":
                            value = lhs_value + rhs_value
                        elif op_type == "Sub":
                            value = lhs_value - rhs_value
                        elif op_type == "Mul":
                            value = lhs_value * rhs_value
                        else:
                            if rhs_value == 0:
                                raise ValueError(f"shape-value Div {node_name!r} divides by zero")
                            value = lhs_value // rhs_value
                        candidate.append(f"Known({value})")
            elif op_type == "Slice" and inputs and inputs[0] in shape_values:
                starts = attrs.get("starts", const_values.get(inputs[1]) if len(inputs) > 1 else None)
                ends = attrs.get("ends", const_values.get(inputs[2]) if len(inputs) > 2 else None)
                axes = attrs.get("axes", const_values.get(inputs[3]) if len(inputs) > 3 else [0])
                if starts is not None and ends is not None:
                    starts, ends, axes = _integer_list(starts), _integer_list(ends), _integer_list(axes)
                    if len(starts) == len(ends) == len(axes) == 1 and axes[0] == 0:
                        candidate = list(shape_values[inputs[0]][starts[0]:ends[0]])
            elif op_type in {"Unsqueeze", "Squeeze", "Cast"} and inputs and inputs[0] in shape_values:
                candidate = list(shape_values[inputs[0]])
            elif op_type == "Reshape" and inputs and inputs[0] in shape_values:
                # Reshape changes the container geometry, not the scalar values
                # carried by a shape tensor.
                candidate = list(shape_values[inputs[0]])
            elif op_type == "ConstantOfShape" and inputs and inputs[0] in shape_values:
                extents = [_dimension_expression(value) for value in shape_values[inputs[0]]]
                fill_value = attrs.get("value", 0)
                integral_fill = 0
                try:
                    integral_fill = int(fill_value)
                    fill_is_integral = float(fill_value) == integral_fill
                except (TypeError, ValueError):
                    fill_is_integral = False
                try:
                    concrete_extents = [_expression_capacity(extent, {}) for extent in extents]
                except ValueError:
                    concrete_extents = []
                if len(concrete_extents) == len(extents) and fill_is_integral:
                    element_count = int(np.prod(concrete_extents, dtype=np.int64))
                    candidate = [f"Known({integral_fill})"] * element_count
            elif op_type == "Equal" and len(inputs) >= 2 and all(name in shape_values for name in inputs[:2]):
                left, right = shape_values[inputs[0]], shape_values[inputs[1]]
                if len(left) == 1 and len(right) > 1:
                    left = left * len(right)
                elif len(right) == 1 and len(left) > 1:
                    right = right * len(left)
                if len(left) == len(right):
                    candidate = []
                    for lhs, rhs in zip(left, right):
                        lhs_expr = _dimension_expression(lhs)
                        rhs_expr = _dimension_expression(rhs)
                        try:
                            lhs_value, rhs_value = int(lhs_expr), int(rhs_expr)
                        except ValueError:
                            candidate.append(f"Symbol(({lhs_expr})==({rhs_expr}))")
                        else:
                            candidate.append(f"Known({int(lhs_value == rhs_value)})")
            elif op_type == "Where" and len(inputs) >= 3 and all(name in shape_values for name in inputs[:3]):
                condition, if_true, if_false = (shape_values[name] for name in inputs[:3])
                if len(condition) == len(if_true) == len(if_false):
                    candidate = []
                    for predicate, true_value, false_value in zip(condition, if_true, if_false):
                        predicate_expr = _dimension_expression(predicate)
                        if predicate_expr == "1":
                            candidate.append(true_value)
                        elif predicate_expr == "0":
                            candidate.append(false_value)
                        elif _dimension_expression(true_value) == "1":
                            # Exporters use Equal(shape, 0) + Where(..., 1,
                            # shape) to normalize Expand targets. Runtime still
                            # computes the exact zero-dimension behavior; for
                            # bounded allocation, the false symbolic extent has
                            # the same maximum and preserves its live identity.
                            candidate.append(false_value)
            elif op_type == "Concat" and int(attrs.get("axis", 0)) == 0 and inputs and all(name in shape_values for name in inputs):
                candidate = [value for name in inputs for value in shape_values[name]]
            if out_name and candidate is not None and shape_values.get(out_name) != candidate:
                shape_values[out_name] = candidate
                changed = True
            tensor_candidate = None
            if op_type == "Shape" and inputs and inputs[0] in tensor_shapes:
                tensor_candidate = [f"Known({len(tensor_shapes[inputs[0]])})"]
            elif op_type == "Split" and inputs and inputs[0] in tensor_shapes and outputs:
                source_shape = tensor_shapes[inputs[0]]
                axis = int(attrs.get("axis", 0))
                if axis < 0:
                    axis += len(source_shape)
                sizes = None
                if len(inputs) > 1 and inputs[1] in const_values:
                    sizes = [int(value) for value in np.asarray(const_values[inputs[1]]).reshape(-1)]
                if sizes is None and 0 <= axis < len(source_shape):
                    expression = _dimension_expression(source_shape[axis])
                    if expression.isdigit() and int(expression) % len(outputs) == 0:
                        sizes = [int(expression) // len(outputs)] * len(outputs)
                if sizes is not None and len(sizes) == len(outputs):
                    for output, size in zip(outputs, sizes):
                        split_shape = list(source_shape)
                        split_shape[axis] = f"Known({size})"
                        if tensor_shapes.get(output) != split_shape:
                            tensor_shapes[output] = split_shape
                            changed = True
            if op_type == "Range" and len(inputs) >= 3 and all(name in shape_values for name in inputs[:3]):
                start, limit, step = (shape_values[name][0] for name in inputs[:3])
                expression = (
                    f"Symbol((({_dimension_expression(limit)})-({_dimension_expression(start)}))"
                    f"/({_dimension_expression(step)}))"
                )
                tensor_candidate = [_bound_dimension_descriptor(expression, dimension_bounds)]
            elif op_type == "Slice" and inputs and inputs[0] in tensor_shapes:
                source_shape = tensor_shapes[inputs[0]]
                axes_value = attrs.get("axes", const_values.get(inputs[3]) if len(inputs) > 3 else [0])
                if axes_value is None:
                    axes = []
                elif isinstance(axes_value, np.ndarray):
                    axes = [int(value) for value in axes_value.reshape(-1)]
                elif isinstance(axes_value, (list, tuple)):
                    axes = [int(value) for value in axes_value]
                else:
                    axes = [int(value) for value in str(axes_value).strip("[]").split(",") if value]
                starts_value = shape_values.get(inputs[1]) if len(inputs) > 1 else None
                ends_value = shape_values.get(inputs[2]) if len(inputs) > 2 else None
                if starts_value is None and len(inputs) > 1 and inputs[1] in const_values:
                    starts_value = [int(value) for value in np.asarray(const_values[inputs[1]]).reshape(-1)]
                if ends_value is None and len(inputs) > 2 and inputs[2] in const_values:
                    ends_value = [int(value) for value in np.asarray(const_values[inputs[2]]).reshape(-1)]
                if starts_value is None and "starts" in attrs:
                    starts_value = [f"Known({value})" for value in _integer_list(attrs["starts"])]
                if ends_value is None and "ends" in attrs:
                    ends_value = [f"Known({value})" for value in _integer_list(attrs["ends"])]
                if starts_value is not None and ends_value is not None and len(axes) == len(starts_value) == len(ends_value):
                    tensor_candidate = list(source_shape)
                    for axis, start, end in zip(axes, starts_value, ends_value):
                        axis %= len(source_shape)
                        start_expression = _dimension_expression(start)
                        end_expression = _dimension_expression(end)
                        source_expression = _dimension_expression(source_shape[axis])
                        try:
                            source_extent = ast.literal_eval(source_expression)
                            raw_start = ast.literal_eval(start_expression)
                            raw_end = ast.literal_eval(end_expression)
                            if not all(
                                isinstance(value, int)
                                for value in (source_extent, raw_start, raw_end)
                            ):
                                raise ValueError("slice extent is not an integer literal")
                        except (ValueError, SyntaxError):
                            pass
                        else:
                            normalized = slice(raw_start, raw_end, 1).indices(source_extent)
                            tensor_candidate[axis] = f"Known({len(range(*normalized))})"
                            continue
                        try:
                            unbounded_end = int(end_expression) >= np.iinfo(np.int64).max
                        except ValueError:
                            unbounded_end = False
                        # ONNX uses INT64_MAX as the positive-step "to the end"
                        # sentinel. Preserve the live input extent instead of
                        # allocating an effectively unbounded output dimension.
                        if unbounded_end:
                            end_expression = _dimension_expression(source_shape[axis])
                        extent = f"Symbol(({end_expression})-({start_expression}))"
                        tensor_candidate[axis] = _bound_dimension_descriptor(extent, dimension_bounds)
            elif op_type == "Gather" and len(inputs) >= 2 and inputs[0] in tensor_shapes and inputs[1] in tensor_shapes:
                data_shape = tensor_shapes[inputs[0]]
                index_shape = tensor_shapes[inputs[1]]
                axis = int(attrs.get("axis", 0))
                if axis < 0:
                    axis += len(data_shape)
                if 0 <= axis < len(data_shape):
                    tensor_candidate = data_shape[:axis] + index_shape + data_shape[axis + 1:]
            elif op_type in {"Gemm", "MatMul"} and len(inputs) >= 2 and inputs[0] in tensor_shapes and inputs[1] in tensor_shapes:
                lhs, rhs = tensor_shapes[inputs[0]], tensor_shapes[inputs[1]]
                if len(lhs) >= 2 and len(rhs) >= 2:
                    output_width = rhs[-2] if op_type == "Gemm" and int(attrs.get("transB", 0)) else rhs[-1]
                    tensor_candidate = list(lhs[:-1]) + [output_width]
            elif op_type == "Transpose" and inputs and inputs[0] in tensor_shapes:
                source = tensor_shapes[inputs[0]]
                perm = attrs.get("perm", list(reversed(range(len(source)))))
                if not isinstance(perm, (list, tuple)):
                    perm = [int(value) for value in str(perm).strip("[]").split(",") if value]
                if len(perm) == len(source):
                    tensor_candidate = [source[int(axis)] for axis in perm]
            elif op_type == "ConstantOfShape" and inputs and inputs[0] in shape_values:
                tensor_candidate = list(shape_values[inputs[0]])
            elif op_type == "Expand" and len(inputs) >= 2 and inputs[1] in shape_values:
                target_shape = list(shape_values[inputs[1]])
                data_shape = tensor_shapes.get(inputs[0], [])
                rank = max(len(data_shape), len(target_shape))
                padded_data = ["Known(1)"] * (rank - len(data_shape)) + list(data_shape)
                padded_target = ["Known(1)"] * (rank - len(target_shape)) + target_shape
                tensor_candidate = []
                for data_dimension, target_dimension in zip(padded_data, padded_target):
                    data_expression = _dimension_expression(data_dimension)
                    target_expression = _dimension_expression(target_dimension)
                    if target_expression == "1":
                        tensor_candidate.append(data_dimension)
                    elif data_expression == "1" or data_expression == target_expression:
                        tensor_candidate.append(target_dimension)
                    else:
                        # Preserve the target's runtime identity; backend
                        # validation rejects an actually incompatible broadcast.
                        tensor_candidate.append(target_dimension)
            elif op_type in {
                "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min",
                "Greater", "Less", "Equal", "GreaterOrEqual", "LessOrEqual",
                "And", "Or", "Where",
            }:
                known = [tensor_shapes[name] for name in inputs if name in tensor_shapes]
                if known:
                    rank = max(map(len, known))
                    tensor_candidate = ["Known(1)"] * rank
                    for shape in known:
                        offset = rank - len(shape)
                        for axis, dimension in enumerate(shape):
                            target_axis = offset + axis
                            current = tensor_candidate[target_axis]
                            if _dimension_expression(current) == "1":
                                tensor_candidate[target_axis] = dimension
                            elif _dimension_expression(dimension) == "1":
                                continue
                            elif _dimension_expression(current) != _dimension_expression(dimension):
                                continue
            elif op_type == "Concat" and inputs and all(name in tensor_shapes for name in inputs):
                shapes = [tensor_shapes[name] for name in inputs]
                axis = int(attrs.get("axis", 0))
                if axis < 0:
                    axis += len(shapes[0])
                if 0 <= axis < len(shapes[0]) and all(len(shape) == len(shapes[0]) for shape in shapes):
                    tensor_candidate = list(shapes[0])
                    expressions = [_dimension_expression(shape[axis]) for shape in shapes]
                    combined = "+".join(f"({expression})" for expression in expressions)
                    try:
                        tensor_candidate[axis] = _bound_dimension_descriptor(
                            f"Symbol({combined})", dimension_bounds
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"Concat node {node_name!r} cannot bound axis {axis} from {expressions}: {error}"
                        ) from error
            elif op_type == "Unsqueeze" and inputs and inputs[0] in tensor_shapes:
                tensor_candidate = list(tensor_shapes[inputs[0]])
                axes_value = const_values.get(inputs[1]) if len(inputs) > 1 else attrs.get("axes", [0])
                for axis in sorted(_integer_list(axes_value)):
                    output_rank = len(tensor_candidate) + 1
                    tensor_candidate.insert(axis % output_rank, "Known(1)")
            elif op_type == "Squeeze" and inputs and inputs[0] in tensor_shapes:
                tensor_candidate = list(tensor_shapes[inputs[0]])
                axes_value = const_values.get(inputs[1]) if len(inputs) > 1 else attrs.get("axes")
                if axes_value is not None:
                    for axis in sorted((_integer_list(axes_value)), reverse=True):
                        tensor_candidate.pop(axis % len(tensor_candidate))
                else:
                    tensor_candidate = [dimension for dimension in tensor_candidate if _dimension_expression(dimension) != "1"]
            elif op_type in {
                "Sqrt", "Tanh", "Cast", "Identity", "Dropout", "Softmax",
                "Sigmoid", "Silu", "Gelu", "IsNaN", "Not",
            } and inputs and inputs[0] in tensor_shapes:
                tensor_candidate = list(tensor_shapes[inputs[0]])
            elif op_type == "ReduceMean" and inputs and inputs[0] in tensor_shapes:
                tensor_candidate = list(tensor_shapes[inputs[0]])
                axes = attrs.get("axes", [])
                if not isinstance(axes, (list, tuple)):
                    axes = [int(value) for value in str(axes).split(",") if value]
                if int(attrs.get("keepdims", attrs.get("keepdim", 1))) and tensor_candidate:
                    for axis in axes:
                        normalized = int(axis) % len(tensor_candidate)
                        tensor_candidate[normalized] = "Known(1)"
            dag = next((item for item in dag_nodes if item.get("name") == node_name), None)
            scalar_output = dag is not None and dag.get("output_rank") == "0"
            if scalar_output:
                tensor_candidate = []
            current_shape = tensor_shapes.get(out_name) if out_name else None
            needs_inferred_shape = (
                scalar_output
                or current_shape is None
                or out_name in legacy_unknown_shape_outputs
                or any(
                    isinstance(dimension, str) and dimension.startswith("Symbol(")
                    for dimension in (current_shape or [])
                )
            )
            if out_name and tensor_candidate is not None and needs_inferred_shape and current_shape != tensor_candidate:
                tensor_shapes[out_name] = tensor_candidate
                if dag is not None:
                    dag["shape"] = _attr_to_str(tensor_candidate)
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

            axes_for_static_check = _integer_list(axes_val) if axes_val is not None else [0]
            source_shape_for_static_check = tensor_shapes.get(inputs[0])
            dynamic_slice_extent = bool(source_shape_for_static_check) and any(
                not _dimension_expression(source_shape_for_static_check[axis % len(source_shape_for_static_check)]).isdigit()
                for axis in axes_for_static_check
            )
            if dynamic_slice_extent:
                for bound_attr in ("starts", "ends", "axes", "steps"):
                    dag.pop(bound_attr, None)
            if starts_val is not None and "starts" not in dag and not dynamic_slice_extent:
                dag["starts"] = _slice_ints(starts_val, "starts")
            if ends_val is not None and "ends" not in dag and not dynamic_slice_extent:
                dag["ends"] = _slice_ints(ends_val, "ends")
            if axes_val is not None and "axes" not in dag and not dynamic_slice_extent:
                dag["axes"] = _slice_ints(axes_val, "axes")
            if steps_val is not None and "steps" not in dag and not dynamic_slice_extent:
                dag["steps"] = _slice_ints(steps_val, "steps")
            if (
                starts_val is not None
                and ends_val is not None
                and axes_val is not None
                and inputs[0] in tensor_shapes
            ):
                starts = _integer_list(starts_val)
                ends = _integer_list(ends_val)
                axes = _integer_list(axes_val)
                steps = _integer_list(steps_val) if steps_val is not None else [1] * len(starts)
                if len(starts) == len(ends) == len(axes) == len(steps) and all(step == 1 for step in steps):
                    inferred_shape = list(tensor_shapes[inputs[0]])
                    for raw_axis, start, end in zip(axes, starts, ends):
                        axis = raw_axis % len(inferred_shape)
                        source_expression = _dimension_expression(inferred_shape[axis])
                        try:
                            source_extent = ast.literal_eval(source_expression)
                            if not isinstance(source_extent, int):
                                raise ValueError("slice extent is not an integer literal")
                        except (ValueError, SyntaxError):
                            end_expression = (
                                source_expression
                                if end >= np.iinfo(np.int64).max
                                else str(end)
                            )
                            inferred_shape[axis] = _bound_dimension_descriptor(
                                f"Symbol(({end_expression})-({start}))", dimension_bounds
                            )
                        else:
                            normalized = slice(start, end, 1).indices(source_extent)
                            inferred_shape[axis] = f"Known({len(range(*normalized))})"
                    dag["shape"] = _attr_to_str(inferred_shape)
                    node_outputs = node.get("outputs", [])
                    if isinstance(node_outputs, str):
                        node_outputs = [value.strip() for value in node_outputs.split(",") if value.strip()]
                    for output in node_outputs:
                        tensor_shapes[output] = list(inferred_shape)

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

    # Materialized constants created by graph optimization are added to
    # numpy_params after the original Rust tensors were constructed.
    for param_name, value in numpy_params.items():
        if param_name not in params:
            array = np.asarray(value)
            params[param_name] = fnn.tensor(array, list(array.shape))

    # params already contains fastnn tensors from the unpacking step above
    fnn_params = dict(params)  # copy, since we'll add aliases
    for node in onnx_nodes:
        if node.get("op_type") == "Constant":
            value_name = f"{node.get('name', '')}.value"
            outputs = node.get("outputs", [])
            if isinstance(outputs, str):
                outputs = [value.strip() for value in outputs.split(",") if value.strip()]
            if value_name in fnn_params:
                for output in outputs:
                    fnn_params.setdefault(output, fnn_params[value_name])
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
        w4a8_clip_ratios=calibrated_ratios or None,
    )
    return executor


def build_sequential_model(path: str) -> Any:
    """Build a Sequential model from a PyTorch-exported .fnn file."""
    from fastnn.io.export import load_fnn_model
    return load_fnn_model(path)
