"""Static-shape specialisation for ONNX graphs.

The AOT engine executes static shapes: every tensor extent is known when a graph
is planned, so an ONNX model whose intermediate shapes are computed at run time
(`Shape` -> `Gather`/`Mod`/`Concat` -> `Reshape`/`Resize`) cannot be expressed
until those computed shapes are resolved to numbers.

This module resolves them, once, at import:

* a tensor is **shape-derived** when it is produced by ``Shape``, or by integer
  arithmetic over shape-derived tensors and integer constants;
* every shape-derived tensor is evaluated with ONNX Runtime at the (static)
  input shapes and replaced with an initializer;
* a constant ``If`` condition then selects one branch, which is inlined;
* dead nodes are pruned and shape inference runs again on the specialised
  graph, so the rest of the import sees only concrete dimensions.

Only integer and boolean tensors that are functions of the input *shapes* are
frozen. Nothing that depends on input *data* is ever replaced, which is what
keeps the transformation sound: the exported graph computes the same values for
every input of the specialised shape.

Graphs without `If`/`Loop`/`Mod` are returned untouched, so models that already
import keep their exact previous path.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Ops that produce or transform shape information rather than feature data.
_SHAPE_OPS = {
    "Shape",
    "Gather",
    "GatherElements",
    "GatherND",
    "Unsqueeze",
    "Squeeze",
    "Concat",
    "Cast",
    "Mod",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Equal",
    "Less",
    "Greater",
    "And",
    "Or",
    "Not",
    "Where",
    "ReduceProd",
    "ReduceSum",
    "Slice",
    "Range",
    "ConstantOfShape",
    "Expand",
    "Reshape",
    "Flatten",
    "Transpose",
    "Identity",
}

# Ops whose presence means the graph needs specialisation before it can be
# imported at all.
_NEEDS_SPECIALISATION = {"If", "Mod"}

_INT_ELEM_TYPES = {6, 7, 9}  # int32, int64, bool


def _logical_ops(model) -> Set[str]:
    return {node.op_type for node in model.graph.node}


def needs_specialisation(model) -> bool:
    """True when the graph uses control flow or integer shape arithmetic."""
    ops = _logical_ops(model)
    if "Loop" in ops:
        raise ValueError(
            "ONNX graph uses Loop, which cannot be specialised statically; "
            "re-export the model without a loop, or run it through a runtime"
        )
    return bool(ops & _NEEDS_SPECIALISATION)


def _static_input_shapes(model) -> Optional[Dict[str, List[int]]]:
    """Return every graph input's shape, or None if any extent is symbolic."""
    shapes: Dict[str, List[int]] = {}
    initializer_names = {init.name for init in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        dims: List[int] = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                return None
            dims.append(dim.dim_value)
        shapes[value_info.name] = dims
    return shapes


def _int_initializers(model) -> Set[str]:
    names = set()
    for init in model.graph.initializer:
        if init.data_type in _INT_ELEM_TYPES:
            names.add(init.name)
    return names


def _shape_derived(model) -> Set[str]:
    """Tensors that are functions of the input shapes only."""
    derived = _int_initializers(model)
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if node.op_type == "Constant":
                if _constant_dtype(node) in _INT_ELEM_TYPES:
                    for out in node.output:
                        if out and out not in derived:
                            derived.add(out)
                            changed = True
                continue
            if node.op_type == "Shape":
                for out in node.output:
                    if out and out not in derived:
                        derived.add(out)
                        changed = True
                continue
            if node.op_type not in _SHAPE_OPS:
                continue
            inputs = [name for name in node.input if name]
            if not inputs:
                continue
            # A shape op is derived when it only consumes derived tensors, or a
            # mix of derived tensors and constants (constants are seeded above).
            if all(name in derived for name in inputs):
                for out in node.output:
                    if out and out not in derived:
                        derived.add(out)
                        changed = True
    return derived


def _value_info_types(model) -> Dict[str, int]:
    types: Dict[str, int] = {}
    for collection in (
        model.graph.input,
        model.graph.value_info,
        model.graph.output,
    ):
        for value_info in collection:
            tensor_type = value_info.type.tensor_type
            if tensor_type.elem_type:
                types[value_info.name] = tensor_type.elem_type
    return types


_BOOL_OPS = {"Equal", "Less", "Greater", "And", "Or", "Not"}
# Ops whose output element type is the same as their first input's.
_SAME_TYPE_OPS = {
    "Gather",
    "GatherElements",
    "GatherND",
    "Unsqueeze",
    "Squeeze",
    "Concat",
    "Split",
    "Slice",
    "Mod",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "ReduceProd",
    "ReduceSum",
    "Range",
    "Expand",
    "Reshape",
    "Flatten",
    "Transpose",
    "Identity",
    "Where",
    "ScatterND",
}


def _constant_dtype(node) -> Optional[int]:
    # `elem_type` is unset on an empty TensorProto, and this protobuf build
    # raises AttributeError for an unset scalar field instead of returning the
    # default, so the field is read defensively and 0 (undefined) becomes None.
    for attribute in node.attribute:
        if attribute.name == "value" and attribute.type == 4:  # TENSOR
            elem_type = getattr(attribute.t, "data_type", 0)
            return elem_type or None
    return None


def _infer_dtypes(model) -> Dict[str, int]:
    """Propagate element types through the graph, one pass per dependency.

    Element types are what decide whether a tensor may be frozen, so they are
    derived rather than guessed: a declared type wins, ``Shape``/comparisons/
    ``Cast``/``Constant`` fix their own output type, and the remaining shape ops
    copy their first input's type. A tensor whose type cannot be established is
    left out of the map and can never be frozen.
    """
    types: Dict[str, int] = dict(_value_info_types(model))
    for init in model.graph.initializer:
        types[init.name] = init.data_type

    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if node.op_type == "Shape":
                out_type = 7
            elif node.op_type in _BOOL_OPS:
                out_type = 9
            elif node.op_type == "Cast":
                out_type = next(
                    (attr.i for attr in node.attribute if attr.name == "to"), None
                )
            elif node.op_type == "Constant":
                out_type = _constant_dtype(node)
            elif node.op_type in _SAME_TYPE_OPS:
                out_type = next(
                    (
                        types[name]
                        for name in node.input
                        if name and name in types
                    ),
                    None,
                )
            else:
                out_type = None
            if out_type is None:
                continue
            for out in node.output:
                if out and types.get(out) != out_type:
                    types[out] = out_type
                    changed = True
    return types


def _probe_values(
    model,
    tensor_names: Iterable[str],
    input_shapes: Dict[str, List[int]],
) -> Dict[str, "object"]:
    """Evaluate `tensor_names` with ONNX Runtime at the given input shapes."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    types = _infer_dtypes(model)
    probe = onnx.ModelProto()
    probe.CopyFrom(model)

    added = 0
    existing = {out.name for out in probe.graph.output}
    for name in tensor_names:
        if name in existing:
            continue
        elem_type = types.get(name)
        if elem_type not in _INT_ELEM_TYPES:
            continue
        value_info = helper.ValueInfoProto()
        value_info.name = name
        value_info.type.tensor_type.elem_type = elem_type
        probe.graph.output.append(value_info)
        added += 1
    if added == 0:
        return {}

    session = _make_session(probe)
    feeds = {}
    for name, shape in input_shapes.items():
        elem_type = types.get(name, TensorProto.FLOAT)
        dtype = {
            TensorProto.FLOAT: np.float32,
            TensorProto.DOUBLE: np.float64,
            TensorProto.FLOAT16: np.float16,
            TensorProto.INT64: np.int64,
            TensorProto.INT32: np.int32,
            TensorProto.BOOL: np.bool_,
        }.get(elem_type, np.float32)
        feeds[name] = np.zeros(shape, dtype=dtype)

    output_names = [out.name for out in probe.graph.output]
    results = session.run(output_names, feeds)
    return {
        name: value
        for name, value in zip(output_names, results)
        if name in tensor_names or name not in existing
    }


def _make_session(model):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def _freeze(model, values: Dict[str, "object"]) -> Set[str]:
    """Append the evaluated shape tensors to the initializer list."""
    import numpy as np
    from onnx import numpy_helper

    frozen: Set[str] = set()
    for name, value in values.items():
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(value), name=name)
        )
        frozen.add(name)
    return frozen


def _remove_frozen_producers(model, frozen: Set[str]) -> None:
    graph_outputs = {out.name for out in model.graph.output}
    kept = []
    for node in model.graph.node:
        outputs = [out for out in node.output if out]
        if outputs and all(out in frozen for out in outputs):
            if any(out in graph_outputs for out in outputs):
                kept.append(node)
                continue
            continue
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)


def _prune_dead_nodes(model) -> None:
    """Drop nodes no longer reachable from the graph outputs."""
    consumers: Dict[str, int] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers[name] = consumers.get(name, 0) + 1

    changed = True
    while changed:
        changed = False
        keep = []
        for node in reversed(model.graph.node):
            required = any(
                out in consumers and consumers[out] > 0 for out in node.output if out
            ) or any(out in {o.name for o in model.graph.output} for out in node.output if out)
            if required:
                keep.append(node)
            else:
                changed = True
                for name in node.output:
                    if name:
                        consumers.pop(name, None)
        keep.reverse()
        del model.graph.node[:]
        model.graph.node.extend(keep)


def _inline_constant_ifs(model, frozen: Set[str]) -> bool:
    """Inline `If` nodes whose condition is a frozen scalar."""
    import numpy as np
    from onnx import numpy_helper

    initializers = {init.name: init for init in model.graph.initializer}
    changed = False
    kept = []
    for node in model.graph.node:
        if node.op_type != "If":
            kept.append(node)
            continue
        condition = node.input[0] if node.input else ""
        if condition not in frozen or condition not in initializers:
            kept.append(node)
            continue
        taken = np.asarray(numpy_helper.to_array(initializers[condition])).reshape(-1)
        # An `If` holds `then_branch` first and `else_branch` second, and the
        # then-branch is taken when the condition is true. Matching by name
        # rather than by position keeps that mapping explicit.
        wanted = "then_branch" if bool(taken[0]) else "else_branch"
        branch = next((attr for attr in node.attribute if attr.name == wanted), None)
        if branch is None:
            kept.append(node)
            continue
        subgraph = branch.g
        prefix = f"{node.name or 'If'}/inlined/"
        rename = {}
        for inner in subgraph.node:
            for out in inner.output:
                if out:
                    rename[out] = prefix + out
        # Branch outputs map onto the If outputs, in order.
        outputs = [out for out in node.output if out]
        for index, graph_output in enumerate(subgraph.output):
            if index < len(outputs):
                rename[graph_output.name] = outputs[index]

        for initializer in subgraph.initializer:
            copied = numpy_helper.from_array(
                numpy_helper.to_array(initializer), name=prefix + initializer.name
            )
            model.graph.initializer.append(copied)
            rename.setdefault(initializer.name, prefix + initializer.name)

        for inner in subgraph.node:
            clone = type(inner)()
            clone.CopyFrom(inner)
            clone.name = prefix + (inner.name or "node")
            for index, name in enumerate(clone.input):
                if name in rename:
                    clone.input[index] = rename[name]
                elif name:
                    # Captured from the enclosing scope; keep the outer name.
                    pass
            for index, name in enumerate(clone.output):
                if name in rename:
                    clone.output[index] = rename[name]
            kept.append(clone)
        changed = True
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(kept)
    return changed


def specialize(model, *, log=None):
    """Return a static-shape specialisation of `model`, or `model` unchanged."""
    import onnx

    if not needs_specialisation(model):
        return model

    input_shapes = _static_input_shapes(model)
    if input_shapes is None:
        raise ValueError(
            "ONNX graph needs shape specialisation but its inputs are dynamic; "
            "pass concrete input shapes"
        )

    specialised = onnx.ModelProto()
    specialised.CopyFrom(model)
    try:
        specialised = onnx.shape_inference.infer_shapes(specialised)
    except Exception:  # pragma: no cover - best effort
        pass

    for round_index in range(8):
        derived = _shape_derived(specialised)
        types = _infer_dtypes(specialised)
        candidates = [
            name
            for name in derived
            if name not in {init.name for init in specialised.graph.initializer}
            and name not in {out.name for out in specialised.graph.output}
            and types.get(name) in _INT_ELEM_TYPES
        ]
        values = _probe_values(specialised, candidates, input_shapes)
        # Bool conditions decide control flow first; freezing them lets the If
        # inline in the same round.
        frozen = _freeze(specialised, values) if values else set()
        _remove_frozen_producers(specialised, frozen)
        inlined = _inline_constant_ifs(specialised, frozen | {
            init.name for init in specialised.graph.initializer
        })
        _prune_dead_nodes(specialised)
        if log is not None:
            log(
                f"shape specialisation round {round_index}: "
                f"{len(frozen)} tensors frozen, if_inlined={inlined}, "
                f"nodes={len(specialised.graph.node)}"
            )
        if not inlined and not values:
            break
        if not any(node.op_type in _NEEDS_SPECIALISATION for node in specialised.graph.node):
            break

    try:
        specialised = onnx.shape_inference.infer_shapes(specialised)
    except Exception:  # pragma: no cover - inference is best effort
        pass
    try:
        from onnx import checker

        checker.check_model(specialised)
    except Exception:  # pragma: no cover - checker is best effort
        pass
    return specialised
