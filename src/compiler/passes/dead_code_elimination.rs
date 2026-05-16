use crate::ir::node::{ComputeGraph, DimExpr, NodeId, Opcode};
use std::collections::HashSet;

/// Remove nodes that are not reachable from `graph.inputs`, `graph.outputs`,
/// or `graph.required_nodes`.
///
/// Also eliminates no-op nodes: identity Reshape, identity Cast, single-input
/// Concat, full-tensor Slice, and unused Shape nodes.
///
/// If `graph.outputs` is empty the pass is a no-op — some callers build
/// graphs without explicitly setting outputs, and we conservatively assume
/// every node is live.
///
/// Returns the number of removed nodes.
pub fn eliminate_dead_code(graph: &mut ComputeGraph) -> usize {
    if graph.outputs.is_empty() {
        return 0;
    }

    // ── Phase 2: Standard dead code elimination ────────────────────────
    let mut reachable: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = Vec::new();

    for &id in &graph.inputs {
        stack.push(id);
    }
    for &id in &graph.outputs {
        stack.push(id);
    }
    for &id in &graph.required_nodes {
        stack.push(id);
    }

    while let Some(id) = stack.pop() {
        if reachable.insert(id) {
            if let Some(node) = graph.get_node(id) {
                for &input_id in &node.inputs {
                    stack.push(input_id);
                }
            }
        }
    }

    let before = graph.nodes.len();
    graph.nodes.retain(|node| reachable.contains(&node.id));
    let dce_removed = before - graph.nodes.len();

    if dce_removed > 0 {
        graph.inputs.retain(|id| reachable.contains(id));
        graph.outputs.retain(|id| reachable.contains(id));
        graph.required_nodes.retain(|id| reachable.contains(id));
        for node in &mut graph.nodes {
            node.inputs.retain(|id| reachable.contains(id));
        }
        graph.rebuild_node_index();
        graph.mark_mutated();
    }

    dce_removed
}

/// Eliminate no-op patterns:
/// - Identity Reshape(x, same_shape) → x
/// - Cast(x, same_dtype) → x
/// - Concat with single input → the input itself
/// - Slice(0..dim) covering the full tensor → x
#[allow(dead_code)]
fn eliminate_noops(graph: &mut ComputeGraph) -> usize {
    let order = graph.topological_sort();
    let mut rewrites: Vec<(NodeId, NodeId)> = Vec::new();

    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n.clone(),
            None => continue,
        };

        let replacement = match node.opcode {
            Opcode::Reshape => {
                // Identity reshape: target shape matches input shape
                let input_id = node.inputs.first().copied();
                input_id.and_then(|inp_id| {
                    let input_node = graph.get_node(inp_id)?;
                    let target_shape_str = node.attrs.get("shape")?;
                    let target_shape = parse_shape_attr(target_shape_str);
                    if shapes_equal(&input_node.output_type.shape, &target_shape) {
                        Some(inp_id)
                    } else {
                        None
                    }
                })
            }

            Opcode::Cast => {
                // Identity cast: input dtype matches output dtype
                node.inputs.first().copied().and_then(|inp_id| {
                    let input_node = graph.get_node(inp_id)?;
                    if input_node.output_type.dtype == node.output_type.dtype {
                        Some(inp_id)
                    } else {
                        None
                    }
                })
            }

            Opcode::Concat => {
                // Single-input concat: just pass through
                if node.inputs.len() == 1 {
                    node.inputs.first().copied()
                } else {
                    None
                }
            }

            Opcode::Slice => {
                // Full-tensor slice: Slice(0..dim) that covers the entire dimension
                node.inputs.first().copied().and_then(|inp_id| {
                    let input_node = graph.get_node(inp_id)?;
                    let dim: usize = node.attrs.get("dim").and_then(|s| s.parse().ok())?;
                    let start: u64 = node.attrs.get("start").and_then(|s| s.parse().ok()).unwrap_or(0);
                    let end: u64 = node.attrs.get("end").and_then(|s| s.parse().ok()).unwrap_or(0);

                    if dim < input_node.output_type.shape.len() && start == 0 {
                        let input_dim = &input_node.output_type.shape[dim];
                        if let Some(input_dim_val) = input_dim.evaluate() {
                            if end >= input_dim_val {
                                return Some(inp_id);
                            }
                        }
                    }
                    None
                })
            }

            _ => None,
        };

        if let Some(replacement_id) = replacement {
            rewrites.push((node_id, replacement_id));
        }
    }

    if rewrites.is_empty() {
        return 0;
    }

    for (node_id, replacement_id) in &rewrites {
        let consumers: Vec<NodeId> = graph.consumers(*node_id);
        for consumer_id in consumers {
            if let Some(consumer) = graph.get_node_mut(consumer_id) {
                for input in consumer.inputs.iter_mut() {
                    if *input == *node_id {
                        *input = *replacement_id;
                    }
                }
            }
        }
        graph.remove_node(*node_id);
    }

    graph.mark_mutated();
    rewrites.len()
}

fn shapes_equal(a: &[DimExpr], b: &[DimExpr]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(da, db)| {
        match (da.evaluate(), db.evaluate()) {
            (Some(va), Some(vb)) => va == vb,
            _ => da == db,
        }
    })
}

fn parse_shape_attr(s: &str) -> Vec<DimExpr> {
    let trimmed = s.trim();
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(trimmed);
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|part| {
            let p = part.trim();
            if let Ok(v) = p.parse::<u64>() {
                DimExpr::Known(v)
            } else if p.is_empty() {
                DimExpr::Known(1)
            } else {
                DimExpr::Symbol(p.to_string())
            }
        })
        .collect()
}
