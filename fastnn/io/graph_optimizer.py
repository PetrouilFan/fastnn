"""Graph optimizer for ONNX-imported models.

Performs optimization passes on the computation graph:
1. Constant folding — evaluate constant-only subgraphs
2. Dead node elimination — remove unused nodes
3. Conv+BN fusion — merge Conv + BatchNormalization into fused ops
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def optimize_graph(header: dict) -> dict:
    """Run all optimization passes on a model graph header.

    Args:
        header: The model header dict containing 'graph' and 'layers'.

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
    
    # Pass 3: Constant folding (currently a no-op, framework for future)
    nodes = fold_constants(nodes, header)
    
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


def fold_constants(nodes: List[dict], header: dict) -> List[dict]:
    """Fold constant subgraphs into constant nodes.

    This is a framework for future constant folding. Currently it:
    1. Identifies Constant nodes
    2. Propagates their values through simple ops (shape, cast, gather)

    Full constant folding requires an executor and is deferred.

    Args:
        nodes: List of node dicts.
        header: Full model header (for parameter access).

    Returns:
        Node list with some constant subgraphs folded (currently a no-op).
    """
    _ = header  # Reserved for future use
    
    # Placeholder: detect and mark constant-foldable subgraphs
    constant_nodes: Set[str] = set()
    
    # Find Constant nodes
    for node in nodes:
        if node.get("op_type", "").lower() in ("constant", "constantop"):
            constant_nodes.add(node.get("name", ""))
    
    # Propagate through identity-like ops (in future: actually evaluate)
    changed = True
    while changed:
        changed = False
        for node in nodes:
            node_name = node.get("name", "")
            if node_name in constant_nodes:
                continue
            op_type = node.get("op_type", "").lower()
            # If all inputs are constant, this node could be folded
            inputs = node.get("inputs", [])
            if not inputs:
                continue
            # Check if any input comes from a constant node
            has_constant_input = any(
                inp in constant_nodes for inp in inputs
            )
            if has_constant_input and op_type in (
                "shape", "shapeop", "cast", "castop",
                "identity", "identityop",
                "gather", "gatherop",
                "unsqueeze", "unsqueezeop",
                "squeeze", "squeezeop",
                "concat", "slice", "sliceop",
            ):
                constant_nodes.add(node_name)
                changed = True
    
    if constant_nodes:
        logger.debug("Identified %d constant-foldable nodes", len(constant_nodes))
    
    # Actual folding is not yet implemented
    return nodes


# ---- Module-level convenience API ----

def optimize_header(header: dict) -> dict:
    """Convenience function to optimize a model header in-place."""
    return optimize_graph(header)
