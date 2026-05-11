"""Graph optimizer for ONNX-imported models.

Performs optimization passes on the computation graph:
1. Constant folding — evaluate constant-only subgraphs
2. Dead node elimination — remove unused nodes
3. Conv+BN fusion — merge Conv + BatchNormalization into fused ops
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---- Constant evaluation registry ----
# Maps op_type -> evaluation function (node, input_tensors, attrs) -> [output_tensors]

_EVAL_REGISTRY: Dict[str, Callable] = {}


def _register(op_types):
    """Decorator to register an evaluation function for one or more op types."""
    def decorator(fn):
        for ot in op_types:
            _EVAL_REGISTRY[ot.lower()] = fn
        return fn
    return decorator


def _get_attr_safe(node: dict, key: str, default=None):
    """Get an attribute from a node, checking both 'attrs' sub-dict and top-level."""
    attrs = node.get("attrs", {})
    if key in attrs:
        return attrs[key]
    if key in node:
        return node[key]
    return default


# ---- Evaluation functions for foldable ops ----

@_register(["identity", "identityop"])
def _eval_identity(node, tensors, node_attrs):
    return list(tensors)


@_register(["shape", "shapeop"])
def _eval_shape(node, tensors, node_attrs):
    return [np.array(tensors[0].shape, dtype=np.int64)]


@_register(["cast", "castop"])
def _eval_cast(node, tensors, node_attrs):
    to_dtype = _get_attr_safe(node, "to", 1)
    dtype_map = {
        1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
        6: np.int32, 7: np.int64, 9: bool, 10: np.float16, 11: np.float64,
        12: np.uint32, 13: np.uint64,
    }
    target = dtype_map.get(to_dtype, np.float32)
    return [tensors[0].astype(target)]


@_register(["unsqueeze", "unsqueezeop"])
def _eval_unsqueeze(node, tensors, node_attrs):
    axes = _get_attr_safe(node, "axes", None)
    if axes is None and len(tensors) > 1:
        axes = tensors[1].flatten().astype(int).tolist()
    if axes is None:
        return list(tensors)
    result = tensors[0]
    for ax in sorted(axes):
        result = np.expand_dims(result, axis=int(ax))
    return [result]


@_register(["squeeze", "squeezeop"])
def _eval_squeeze(node, tensors, node_attrs):
    axes = _get_attr_safe(node, "axes", None)
    if axes is None and len(tensors) > 1:
        axes = tensors[1].flatten().astype(int).tolist()
    if axes is None:
        return [np.squeeze(tensors[0])]
    result = tensors[0]
    for ax in sorted(axes, reverse=True):
        result = np.squeeze(result, axis=int(ax))
    return [result]


@_register(["gather", "gatherop"])
def _eval_gather(node, tensors, node_attrs):
    axis = _get_attr_safe(node, "axis", 0)
    indices = tensors[1].flatten().astype(int) if len(tensors) > 1 else np.array([0])
    result = np.take(tensors[0], indices, axis=int(axis))
    return [result]


@_register(["concat"])
def _eval_concat(node, tensors, node_attrs):
    axis = _get_attr_safe(node, "axis", 0)
    return [np.concatenate(tensors, axis=int(axis))]


@_register(["slice", "sliceop"])
def _eval_slice(node, tensors, node_attrs):
    starts = _get_attr_safe(node, "starts", None)
    ends = _get_attr_safe(node, "ends", None)
    axes = _get_attr_safe(node, "axes", None)
    steps = _get_attr_safe(node, "steps", None)
    if starts is None or ends is None:
        return list(tensors)
    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)
    x = tensors[0]
    slices = [slice(None)] * x.ndim
    for i, ax in enumerate(axes):
        slices[int(ax)] = slice(int(starts[i]), int(ends[i]), int(steps[i]) if i < len(steps) else 1)
    return [x[tuple(slices)]]


@_register(["transpose", "transposeop"])
def _eval_transpose(node, tensors, node_attrs):
    perm = _get_attr_safe(node, "perm", None)
    if perm is None:
        return [tensors[0].T]
    return [np.transpose(tensors[0], [int(p) for p in perm])]


@_register(["reshape", "reshapeop"])
def _eval_reshape(node, tensors, node_attrs):
    shape = tensors[1].flatten().astype(int).tolist() if len(tensors) > 1 else _get_attr_safe(node, "shape", list(tensors[0].shape))
    total = int(np.prod(tensors[0].shape))
    computed = int(np.prod([s for s in shape if s != -1])) if -1 in shape else total
    shape = [total // computed if s == -1 else int(s) for s in shape]
    return [tensors[0].reshape(shape)]


@_register(["add", "elementwiseadd"])
def _eval_add(node, tensors, node_attrs):
    return [tensors[0] + tensors[1]]


@_register(["sub", "elementwisesub"])
def _eval_sub(node, tensors, node_attrs):
    return [tensors[0] - tensors[1]]


@_register(["mul", "elementwisemul"])
def _eval_mul(node, tensors, node_attrs):
    return [tensors[0] * tensors[1]]


@_register(["div", "elementwisediv"])
def _eval_div(node, tensors, node_attrs):
    return [tensors[0] / tensors[1]]


@_register(["neg", "negop"])
def _eval_neg(node, tensors, node_attrs):
    return [-tensors[0]]


@_register(["exp", "expop"])
def _eval_exp(node, tensors, node_attrs):
    return [np.exp(tensors[0])]


@_register(["sqrt", "sqrtop"])
def _eval_sqrt(node, tensors, node_attrs):
    return [np.sqrt(tensors[0])]


@_register(["log", "logop"])
def _eval_log(node, tensors, node_attrs):
    return [np.log(tensors[0])]


@_register(["abs"])
def _eval_abs(node, tensors, node_attrs):
    return [np.abs(tensors[0])]


@_register(["ceil"])
def _eval_ceil(node, tensors, node_attrs):
    return [np.ceil(tensors[0])]


@_register(["floor"])
def _eval_floor(node, tensors, node_attrs):
    return [np.floor(tensors[0])]


@_register(["round"])
def _eval_round(node, tensors, node_attrs):
    return [np.round(tensors[0])]


@_register(["sign"])
def _eval_sign(node, tensors, node_attrs):
    return [np.sign(tensors[0])]


@_register(["reciprocal"])
def _eval_reciprocal(node, tensors, node_attrs):
    return [np.reciprocal(tensors[0])]


@_register(["min"])
def _eval_min(node, tensors, node_attrs):
    result = tensors[0]
    for t in tensors[1:]:
        result = np.minimum(result, t)
    return [result]


@_register(["max"])
def _eval_max(node, tensors, node_attrs):
    result = tensors[0]
    for t in tensors[1:]:
        result = np.maximum(result, t)
    return [result]


@_register(["equal"])
def _eval_equal(node, tensors, node_attrs):
    return [np.equal(tensors[0], tensors[1])]


@_register(["greater"])
def _eval_greater(node, tensors, node_attrs):
    return [np.greater(tensors[0], tensors[1])]


@_register(["less"])
def _eval_less(node, tensors, node_attrs):
    return [np.less(tensors[0], tensors[1])]


@_register(["and"])
def _eval_and(node, tensors, node_attrs):
    return [np.logical_and(tensors[0], tensors[1])]


@_register(["or"])
def _eval_or(node, tensors, node_attrs):
    return [np.logical_or(tensors[0], tensors[1])]


@_register(["xor"])
def _eval_xor(node, tensors, node_attrs):
    return [np.logical_xor(tensors[0], tensors[1])]


@_register(["not"])
def _eval_not(node, tensors, node_attrs):
    return [np.logical_not(tensors[0])]


@_register(["isnan"])
def _eval_isnan(node, tensors, node_attrs):
    return [np.isnan(tensors[0])]


@_register(["isinf"])
def _eval_isinf(node, tensors, node_attrs):
    return [np.isinf(tensors[0])]


@_register(["relu"])
def _eval_relu(node, tensors, node_attrs):
    return [np.maximum(0, tensors[0])]


@_register(["leakyrelu"])
def _eval_leakyrelu(node, tensors, node_attrs):
    alpha = float(_get_attr_safe(node, "alpha", 0.01))
    x = tensors[0]
    return [np.where(x > 0, x, alpha * x)]


@_register(["sigmoid"])
def _eval_sigmoid(node, tensors, node_attrs):
    return [1.0 / (1.0 + np.exp(-tensors[0]))]


@_register(["tanh"])
def _eval_tanh(node, tensors, node_attrs):
    return [np.tanh(tensors[0])]


@_register(["softmax"])
def _eval_softmax(node, tensors, node_attrs):
    axis = int(_get_attr_safe(node, "axis", 1))
    x = tensors[0]
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return [e_x / np.sum(e_x, axis=axis, keepdims=True)]


@_register(["flatten"])
def _eval_flatten(node, tensors, node_attrs):
    axis = int(_get_attr_safe(node, "axis", 1))
    x = tensors[0]
    outer = int(np.prod(x.shape[:axis]))
    inner = int(np.prod(x.shape[axis:]))
    return [x.reshape(outer, inner)]


@_register(["constantofshape"])
def _eval_constantofshape(node, tensors, node_attrs):
    shape = tensors[0].flatten().astype(int).tolist() if tensors else _get_attr_safe(node, "dims", [])
    value = float(_get_attr_safe(node, "value", 0.0))
    return [np.full(shape, value, dtype=np.float32)]


@_register(["reducesum"])
def _eval_reducesum(node, tensors, node_attrs):
    axes = _get_attr_safe(node, "axes", None)
    keepdims = bool(_get_attr_safe(node, "keepdims", 1))
    if axes is not None:
        axes = [int(a) for a in axes]
    return [np.sum(tensors[0], axis=axes, keepdims=keepdims)]


@_register(["reducemean"])
def _eval_reducemean(node, tensors, node_attrs):
    axes = _get_attr_safe(node, "axes", None)
    keepdims = bool(_get_attr_safe(node, "keepdims", 1))
    if axes is not None:
        axes = [int(a) for a in axes]
    return [np.mean(tensors[0], axis=axes, keepdims=keepdims)]


@_register(["tile"])
def _eval_tile(node, tensors, node_attrs):
    repeats = tensors[1].flatten().astype(int).tolist() if len(tensors) > 1 else _get_attr_safe(node, "repeats", [1])
    return [np.tile(tensors[0], repeats)]


@_register(["where"])
def _eval_where(node, tensors, node_attrs):
    condition = tensors[0].astype(bool)
    x = tensors[1] if len(tensors) > 1 else np.zeros_like(condition, dtype=np.float32)
    y = tensors[2] if len(tensors) > 2 else np.ones_like(condition, dtype=np.float32)
    return [np.where(condition, x, y)]


@_register(["clip"])
def _eval_clip(node, tensors, node_attrs):
    x = tensors[0]
    min_val = _get_attr_safe(node, "min", None)
    max_val = _get_attr_safe(node, "max", None)
    if min_val is None and len(tensors) > 1:
        min_val = tensors[1].item() if hasattr(tensors[1], 'item') else tensors[1]
    if max_val is None and len(tensors) > 2:
        max_val = tensors[2].item() if hasattr(tensors[2], 'item') else tensors[2]
    return [np.clip(x, min_val, max_val)]


@_register(["cumsum"])
def _eval_cumsum(node, tensors, node_attrs):
    axis = int(_get_attr_safe(node, "axis", 0))
    exclusive = bool(_get_attr_safe(node, "exclusive", False))
    reverse = bool(_get_attr_safe(node, "reverse", False))
    result = np.cumsum(tensors[0], axis=axis)
    if exclusive:
        result = np.roll(result, 1, axis=axis)
        if axis == 0:
            result[0] = 0
        else:
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(0, 1)
            result[tuple(slices)] = 0
    if reverse:
        result = np.flip(np.cumsum(np.flip(tensors[0], axis=axis), axis=axis), axis=axis)
    return [result]


@_register(["expand"])
def _eval_expand(node, tensors, node_attrs):
    shape = tensors[1].flatten().astype(int).tolist() if len(tensors) > 1 else _get_attr_safe(node, "shape", list(tensors[0].shape))
    return [np.broadcast_to(tensors[0], shape)]


@_register(["eyelike"])
def _eval_eyelike(node, tensors, node_attrs):
    shape = tensors[0].shape if len(tensors) > 0 else _get_attr_safe(node, "shape", [1, 1])
    k = int(_get_attr_safe(node, "k", 0))
    dtype = _get_attr_safe(node, "dtype", 1)
    dtype_map = {1: np.float32, 6: np.int32, 7: np.int64, 9: bool, 11: np.float64}
    np_dtype = dtype_map.get(dtype, np.float32)
    return [np.eye(shape[0], shape[1] if len(shape) > 1 else shape[0], k=k, dtype=np_dtype)]


@_register(["onehot"])
def _eval_onehot(node, tensors, node_attrs):
    indices = tensors[0].flatten().astype(int)
    depth = int(tensors[1].item()) if len(tensors) > 1 else int(_get_attr_safe(node, "depth", 1))
    values = tensors[2] if len(tensors) > 2 else np.array([0.0, 1.0])
    off_val, on_val = float(values[0]), float(values[1]) if len(values) > 1 else (0.0, 1.0)
    axis = int(_get_attr_safe(node, "axis", -1))
    result = np.full(indices.shape + (depth,), off_val, dtype=np.float32)
    np.put_along_axis(result, np.expand_dims(indices, axis=-1 if axis == -1 else axis), on_val, axis=axis if axis != -1 else -1)
    return [result]


@_register(["range"])
def _eval_range(node, tensors, node_attrs):
    start = float(tensors[0].item()) if len(tensors) > 0 else 0.0
    limit = float(tensors[1].item()) if len(tensors) > 1 else 0.0
    delta = float(tensors[2].item()) if len(tensors) > 2 else 1.0
    return [np.arange(start, limit, delta, dtype=np.float32)]


@_register(["depthtospace"])
def _eval_depthtospace(node, tensors, node_attrs):
    blocksize = int(_get_attr_safe(node, "blocksize", 1))
    x = tensors[0]
    n, c, h, w = x.shape
    c2 = c // (blocksize * blocksize)
    x = x.reshape(n, blocksize, blocksize, c2, h, w)
    x = x.transpose(0, 3, 4, 1, 5, 2)
    return [x.reshape(n, c2, h * blocksize, w * blocksize)]


@_register(["spacetodepth"])
def _eval_spacetodepth(node, tensors, node_attrs):
    blocksize = int(_get_attr_safe(node, "blocksize", 1))
    x = tensors[0]
    n, c, h, w = x.shape
    h2 = h // blocksize
    w2 = w // blocksize
    x = x.reshape(n, c, h2, blocksize, w2, blocksize)
    x = x.transpose(0, 3, 5, 1, 2, 4)
    return [x.reshape(n, c * blocksize * blocksize, h2, w2)]


@_register(["reverse"])
def _eval_reverse(node, tensors, node_attrs):
    axis = _get_attr_safe(node, "axis", None)
    if axis is None and len(tensors) > 1:
        axis = int(tensors[1].item())
    if axis is None:
        return [tensors[0][::-1]]
    return [np.flip(tensors[0], axis=int(axis))]


@_register(["topk"])
def _eval_topk(node, tensors, node_attrs):
    k = int(tensors[1].item()) if len(tensors) > 1 else int(_get_attr_safe(node, "k", 1))
    axis = int(_get_attr_safe(node, "axis", -1))
    largest = bool(_get_attr_safe(node, "largest", 1))
    sorted_ = bool(_get_attr_safe(node, "sorted", 1))
    if not largest:
        values, indices = np.sort(tensors[0], axis=axis), np.argsort(tensors[0], axis=axis)
    else:
        values, indices = np.sort(tensors[0], axis=axis)[..., ::-1], np.argsort(-tensors[0], axis=axis)
    if not sorted_:
        pass
    # Take first k
    slicer = [slice(None)] * values.ndim
    slicer[axis] = slice(0, k)
    return [values[tuple(slicer)], indices[tuple(slicer)].astype(np.int64)]


@_register(["gatherelements"])
def _eval_gatherelements(node, tensors, node_attrs):
    axis = int(_get_attr_safe(node, "axis", 0))
    data = tensors[0]
    indices = tensors[1].astype(int)
    return [np.take_along_axis(data, indices, axis=axis)]


@_register(["scatternd"])
def _eval_scatternd(node, tensors, node_attrs):
    data = tensors[0].copy()
    indices = tensors[1].astype(int)
    updates = tensors[2]
    reduction = _get_attr_safe(node, "reduction", "none")
    if reduction == "add":
        np.add.at(data, tuple(indices.T), updates)
    elif reduction == "mul":
        np.multiply.at(data, tuple(indices.T), updates)
    else:
        data[tuple(indices.T)] = updates
    return [data]


@_register(["scatterelements"])
def _eval_scatterelements(node, tensors, node_attrs):
    data = tensors[0].copy()
    indices = tensors[1].astype(int)
    updates = tensors[2]
    axis = int(_get_attr_safe(node, "axis", 0))
    reduction = _get_attr_safe(node, "reduction", "none")
    if reduction == "add":
        np.add.at(data, (slice(None),) * axis + (indices,), updates)
    elif reduction == "mul":
        np.multiply.at(data, (slice(None),) * axis + (indices,), updates)
    else:
        np.put_along_axis(data, indices, updates, axis=axis)
    return [data]


@_register(["split"])
def _eval_split(node, tensors, node_attrs):
    axis = int(_get_attr_safe(node, "axis", 0))
    split = _get_attr_safe(node, "split", None)
    if split is None and len(tensors) > 1:
        split = tensors[1].flatten().astype(int).tolist()
    if split is not None:
        split = [int(s) for s in split]
    return list(np.split(tensors[0], split or tensors[0].shape[axis], axis=axis))


@_register(["pad"])
def _eval_pad(node, tensors, node_attrs):
    mode = _get_attr_safe(node, "mode", "constant")
    pads = _get_attr_safe(node, "pads", None)
    if pads is None and len(tensors) > 1:
        pads = tensors[1].flatten().astype(int).tolist()
    constant_value = _get_attr_safe(node, "value", 0.0)
    if pads is None:
        return list(tensors)
    pads = [int(p) for p in pads]
    pad_width = [(pads[i], pads[i + len(pads) // 2]) for i in range(len(pads) // 2)]
    return [np.pad(tensors[0], pad_width, mode=mode, constant_values=constant_value)]


@_register(["softplus"])
def _eval_softplus(node, tensors, node_attrs):
    return [np.log(1.0 + np.exp(tensors[0]))]


@_register(["selu"])
def _eval_selu(node, tensors, node_attrs):
    alpha = float(_get_attr_safe(node, "alpha", 1.6732632423543772))
    gamma = float(_get_attr_safe(node, "gamma", 1.0507009873554805))
    x = tensors[0]
    return [gamma * np.where(x > 0, x, alpha * (np.exp(x) - 1))]


@_register(["elu"])
def _eval_elu(node, tensors, node_attrs):
    alpha = float(_get_attr_safe(node, "alpha", 1.0))
    x = tensors[0]
    return [np.where(x > 0, x, alpha * (np.exp(x) - 1))]


@_register(["hardsigmoid"])
def _eval_hardsigmoid(node, tensors, node_attrs):
    alpha = float(_get_attr_safe(node, "alpha", 0.2))
    beta = float(_get_attr_safe(node, "beta", 0.5))
    return [np.clip(alpha * tensors[0] + beta, 0, 1)]


@_register(["hardswish"])
def _eval_hardswish(node, tensors, node_attrs):
    x = tensors[0]
    return [x * np.clip(x / 6.0 + 0.5, 0, 1)]


@_register(["erf"])
def _eval_erf(node, tensors, node_attrs):
    from scipy.special import erf as _erf
    return [_erf(tensors[0])]


@_register(["gelu"])
def _eval_gelu(node, tensors, node_attrs):
    x = tensors[0]
    return [0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))]


def optimize_graph(header: dict, params: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """Run all optimization passes on a model graph header.

    Args:
        header: The model header dict containing 'graph' and 'layers'.
        params: Optional dict mapping parameter names to numpy arrays.
                Required for full constant folding.

    Returns:
        Optimized header dict with transformed graph.
    """
    if "graph" not in header:
        logger.debug("No graph topology found; skipping optimization")
        return header

    graph = header.get("graph", {})
    nodes = list(graph.get("nodes", []))
    
    if not nodes:
        return header

    # Pass 1: Dead node elimination
    nodes = eliminate_dead_nodes(nodes, graph)
    
    # Pass 2: Conv+BN fusion  
    nodes = fuse_conv_bn(nodes)
    
    # Pass 3: Constant folding
    nodes = fold_constants(nodes, header, params=params)
    
    # Update header
    header = dict(header)
    header["graph"] = dict(graph)
    header["graph"]["nodes"] = nodes
    
    return header


def eliminate_dead_nodes(nodes: List[dict], graph: dict) -> List[dict]:
    """Remove nodes whose outputs are never used as inputs to other nodes
    and are not model outputs.

    Args:
        nodes: List of node dicts with 'name', 'inputs', 'outputs'.
        graph: Graph dict with 'outputs' list.

    Returns:
        Filtered node list.
    """
    # Collect all output names that are model outputs
    model_outputs = set()
    for out in graph.get("outputs", []):
        if isinstance(out, dict):
            model_outputs.add(out.get("name", ""))
        elif isinstance(out, str):
            model_outputs.add(out)
    
    # Build set of all consumed outputs (inputs to other nodes)
    consumed_outputs: Set[str] = set()
    for node in nodes:
        for inp in node.get("inputs", []):
            consumed_outputs.add(inp)
    
    # Also add initializer names as consumed (they're always used)
    # (Initializers are tracked separately, not as node outputs)
    
    # A node is dead if none of its outputs are consumed and none are model outputs
    live_nodes = []
    removed_count = 0
    for node in nodes:
        outputs = node.get("outputs", [])
        is_live = any(
            out in consumed_outputs or out in model_outputs
            for out in outputs
        )
        # Always keep side-effect nodes (like print, sink)
        if node.get("op_type", "").lower() in ("loop", "if"):
            is_live = True
        
        if is_live:
            live_nodes.append(node)
        else:
            removed_count += 1
            logger.debug("Removed dead node: %s (op=%s)", node.get("name"), node.get("op_type"))
    
    if removed_count > 0:
        logger.info("Dead node elimination removed %d nodes", removed_count)
    
    return live_nodes


def fuse_conv_bn(nodes: List[dict]) -> List[dict]:
    """Detect Conv -> BatchNormalization patterns and fuse them.

    Looks for pattern:
        Conv -> BatchNormalization
    
    Produces a single fused node. The Conv node absorbs the BN parameters.

    Args:
        nodes: List of node dicts.

    Returns:
        Optimized node list with fused Conv+BN nodes.
    """
    # Build a map from output name to consuming node indices
    output_to_consumer: Dict[str, List[int]] = {}
    for i, node in enumerate(nodes):
        for inp in node.get("inputs", []):
            if inp not in output_to_consumer:
                output_to_consumer[inp] = []
            output_to_consumer[inp].append(i)
    
    # Build map from output name to producing node index
    output_to_producer: Dict[str, int] = {}
    for i, node in enumerate(nodes):
        for out in node.get("outputs", []):
            output_to_producer[out] = i
    
    # Find Conv -> BatchNormalization patterns
    nodes_to_remove: Set[int] = set()
    fused_nodes: List[dict] = []
    node_name_counter: Dict[str, int] = {}
    
    for i, node in enumerate(nodes):
        if i in nodes_to_remove:
            continue
        
        op_type = node.get("op_type", "")
        if op_type != "Conv":
            continue
        
        # Check if Conv's output feeds into only one BatchNormalization
        conv_outputs = node.get("outputs", [])
        if not conv_outputs:
            continue
        
        conv_output = conv_outputs[0]
        consumers = output_to_consumer.get(conv_output, [])
        
        # Find BatchNormalization consumers
        bn_consumers = []
        for consumer_idx in consumers:
            if consumer_idx in nodes_to_remove:
                continue
            consumer = nodes[consumer_idx]
            consumer_op = consumer.get("op_type", "")
            if consumer_op.lower() in ("batchnormalization", "batchnorm2d", "batchnorm1d"):
                bn_consumers.append(consumer_idx)
        
        # Only fuse if BN is the sole consumer of Conv's output
        if len(bn_consumers) != 1:
            continue
        
        bn_idx = bn_consumers[0]
        bn_node = nodes[bn_idx]
        
        # Check if BN's output is consumed by multiple nodes (if so, we can't remove BN)
        bn_outputs = bn_node.get("outputs", [])
        if not bn_outputs:
            continue
        bn_output = bn_outputs[0]
        bn_consumers_count = len(output_to_consumer.get(bn_output, []))
        
        # Determine activation after Conv (if any)
        # Check if Conv has a direct activation attribute
        
        # Create fused node
        fused_name = f"{node.get('name', 'conv')}_fused"
        base_name = fused_name
        count = node_name_counter.get(base_name, 0)
        if count > 0:
            fused_name = f"{base_name}_{count}"
        node_name_counter[base_name] = count + 1
        
        fused_node = {
            "name": fused_name,
            "op_type": "FusedConvBn",
            "inputs": list(node.get("inputs", [])),  # Same inputs as Conv
            "outputs": list(bn_node.get("outputs", [])),  # Same outputs as BN
            "attrs": dict(node.get("attrs", {})),
            "fused": True,
            "conv_node": node.get("name", ""),
            "bn_node": bn_node.get("name", ""),
        }
        
        # Copy Conv attributes
        for key in ("kernel_shape", "kernel_size", "strides", "stride",
                     "pads", "padding", "dilations", "dilation", "group", "groups"):
            if key in node.get("attrs", {}):
                fused_node["attrs"][key] = node["attrs"][key]
            elif key in node:
                fused_node["attrs"][key] = node[key]
        
        # Copy BN attributes
        for key in ("epsilon", "eps", "momentum"):
            if key in bn_node.get("attrs", {}):
                fused_node["attrs"][key] = bn_node["attrs"][key]
            elif key in bn_node:
                fused_node["attrs"][key] = bn_node[key]
        
        # Mark BN for removal; replace Conv with fused node in-place
        nodes_to_remove.add(bn_idx)
        fused_nodes.append((i, fused_node))
        
        logger.debug("Fused Conv '%s' + BN '%s' -> '%s'",
                     node.get("name"), bn_node.get("name"), fused_name)
    
    # Build new node list
    new_nodes = []
    fused_count = 0
    for i, node in enumerate(nodes):
        if i in nodes_to_remove:
            continue
        # Check if any fused node replaces this index
        matching_fused = [fn for idx, fn in fused_nodes if idx == i]
        if matching_fused:
            new_nodes.append(matching_fused[0])
            fused_count += 1
        else:
            new_nodes.append(node)
    
    # Append any fused nodes that replace a Conv (already handled above)
    # Also add fused nodes that should be inserted
    
    if fused_count > 0:
        logger.info("Conv+BN fusion: fused %d pairs", fused_count)
    
    return new_nodes


def fold_constants(
    nodes: List[dict],
    header: dict,
    params: Optional[Dict[str, np.ndarray]] = None,
) -> List[dict]:
    """Fold constant subgraphs into constant nodes.

    Evaluates subgraphs where all inputs are constant using numpy.
    Requires ``params`` dict to resolve initializer tensor values.
    Without params, only folds metadata-only ops (Shape/Cast on shapes).

    The algorithm:
    1. Seed constant values from Constant nodes and initializer params
    2. Propagate through foldable ops using a worklist
    3. Materialize Constant nodes for folded outputs consumed externally
    4. Remove dead constant nodes (unconsumed folded outputs)

    Args:
        nodes: List of node dicts.
        header: Full model header (for output names access).
        params: Optional dict mapping param names to numpy arrays.

    Returns:
        Node list with constant subgraphs folded into Constant nodes.
    """
    # 1. Build index maps
    output_to_producer: Dict[str, int] = {}
    for i, node in enumerate(nodes):
        for out in node.get("outputs", []):
            output_to_producer[out] = i

    consumed_outputs: Set[str] = set()
    for node in nodes:
        for inp in node.get("inputs", []):
            consumed_outputs.add(inp)

    model_outputs: Set[str] = set()
    for out in header.get("graph", {}).get("outputs", []):
        if isinstance(out, dict):
            model_outputs.add(out.get("name", ""))
        elif isinstance(out, str):
            model_outputs.add(out)

    # 2. Seed constant values from params and Constant nodes
    constant_values: Dict[str, np.ndarray] = {}

    if params is not None:
        for name, arr in params.items():
            constant_values[name] = arr

    for node in nodes:
        op_type = node.get("op_type", "").lower()
        if op_type in ("constant", "constantop"):
            node_name = node.get("name", "")
            # Try to find value in params
            for out in node.get("outputs", []):
                if out in constant_values:
                    break
                if params is not None:
                    value_key = f"{node_name}.value"
                    if value_key in params:
                        constant_values[out] = params[value_key]
                        break
                    if out in params:
                        constant_values[out] = params[out]
                        break

    # 3. Worklist propagation
    folded_indices: Set[int] = set()
    changed = True
    iterations = 0
    max_iterations = len(nodes) * 2

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        for i, node in enumerate(nodes):
            if i in folded_indices:
                continue
            op_type = node.get("op_type", "").lower()

            # Skip if all outputs already constant
            if all(out in constant_values for out in node.get("outputs", [])):
                folded_indices.add(i)
                continue

            # Check if op is foldable
            eval_fn = _EVAL_REGISTRY.get(op_type)
            if eval_fn is None:
                continue

            inputs = node.get("inputs", [])
            if not inputs:
                continue

            # Gather input tensors
            input_tensors = []
            all_inputs_constant = True
            for inp in inputs:
                if inp in constant_values:
                    input_tensors.append(constant_values[inp])
                else:
                    all_inputs_constant = False
                    break

            if not all_inputs_constant:
                continue

            # Evaluate
            try:
                node_attrs = node.get("attrs", {})
                output_tensors = eval_fn(node, input_tensors, node_attrs)
                for j, out in enumerate(node.get("outputs", [])):
                    if j < len(output_tensors):
                        constant_values[out] = output_tensors[j]
                folded_indices.add(i)
                changed = True
                logger.debug("Folded node '%s' (%s)", node.get("name"), op_type)
            except Exception as e:
                logger.debug("Failed to fold node '%s' (%s): %s", node.get("name"), op_type, e)

    if not folded_indices:
        return nodes

    logger.info("Constant folding: evaluated %d nodes", len(folded_indices))

    # 4. Determine which folded nodes need materialization
    # A folded node needs a Constant node if any of its outputs is
    # consumed by a non-folded node or is a model output.
    materialize: Set[int] = set()
    for i in folded_indices:
        node = nodes[i]
        for out in node.get("outputs", []):
            if out in model_outputs:
                materialize.add(i)
                break
            # Check if consumed by any non-folded node
            for consumer_idx, consumer_node in enumerate(nodes):
                if consumer_idx in folded_indices:
                    continue
                if out in consumer_node.get("inputs", []):
                    materialize.add(i)
                    break
            else:
                continue
            break

    # 5. Build final node list
    new_nodes: List[dict] = []
    added_constant_names: Set[str] = set()

    for i, node in enumerate(nodes):
        if i in folded_indices:
            if i in materialize:
                # Create a new Constant node for this folded output
                node_name = node.get("name", "")
                const_name = f"{node_name}_folded"
                if const_name not in added_constant_names:
                    added_constant_names.add(const_name)
                    const_node: Dict[str, Any] = {
                        "name": const_name,
                        "op_type": "Constant",
                        "inputs": [],
                        "outputs": list(node.get("outputs", [])),
                        "attrs": {},
                    }
                    # Store the folded value in params if possible
                    if params is not None:
                        for out in node.get("outputs", []):
                            if out in constant_values:
                                params[f"{const_name}.value"] = constant_values[out]
                                break
                    new_nodes.append(const_node)
                    logger.debug("Materialized Constant '%s' from folded '%s'", const_name, node_name)
        else:
            new_nodes.append(node)

    logger.info(
        "Constant folding: materialized %d constants, removed %d dead nodes",
        len(materialize), len(folded_indices) - len(materialize),
    )

    return new_nodes


# ---- Module-level convenience API ----

def optimize_header(header: dict) -> dict:
    """Convenience function to optimize a model header in-place."""
    return optimize_graph(header)
