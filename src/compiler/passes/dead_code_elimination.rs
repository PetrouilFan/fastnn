use crate::ir::{ComputeGraph, DimExpr, NodeId, Opcode};
use crate::utils::parse_shape_attr;
use std::collections::{HashMap, HashSet};

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

    // ── Phase 1: Eliminate no-op patterns ──────────────────────────────
    eliminate_noops(graph);

    // ── Phase 2: Standard dead code elimination ────────────────────────
    let mut reachable: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = Vec::with_capacity(graph.nodes.len());

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
fn eliminate_noops(graph: &mut ComputeGraph) -> usize {
    let mut rewrites: Vec<(NodeId, NodeId)> = Vec::with_capacity(graph.nodes.len());

    let graph_ref = &*graph;
    let _ = crate::utils::traverse_graph(graph_ref, |node_id, node| {
        let replacement = match node.opcode {
            Opcode::Reshape => {
                // Identity reshape: target shape matches input shape
                let input_id = node.inputs.first().copied();
                input_id.and_then(|inp_id| {
                    let input_node = graph_ref.get_node(inp_id)?;
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
                // Identity cast: the complete input/output tensor contracts match.
                node.inputs.first().copied().and_then(|inp_id| {
                    let input_node = graph_ref.get_node(inp_id)?;
                    if input_node.output_type == node.output_type {
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
                    let input_node = graph_ref.get_node(inp_id)?;
                    let dim = node.required_attr::<usize>("dim").ok()?;
                    let start = node.required_attr::<u64>("start").ok()?;
                    let end = node.required_attr::<u64>("end").ok()?;

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
        Ok(())
    });

    if rewrites.is_empty() {
        return 0;
    }

    let replacement_map: HashMap<NodeId, NodeId> = rewrites.iter().copied().collect();
    let resolve_replacement = |mut id: NodeId| {
        let mut remaining = replacement_map.len();
        while let Some(&replacement) = replacement_map.get(&id) {
            id = replacement;
            remaining = remaining.saturating_sub(1);
            if remaining == 0 {
                break;
            }
        }
        id
    };

    // Resolve the full replacement chain before deleting anything. Rewriting
    // chained no-ops one at a time can otherwise point a consumer at an
    // intermediate node that was already removed (A -> Cast -> Cast -> B).
    for node in &mut graph.nodes {
        if replacement_map.contains_key(&node.id) {
            continue;
        }
        for input in &mut node.inputs {
            *input = resolve_replacement(*input);
        }
    }
    for output in &mut graph.outputs {
        *output = resolve_replacement(*output);
    }
    graph.required_nodes = graph
        .required_nodes
        .iter()
        .map(|&required| resolve_replacement(required))
        .collect();
    for (node_id, _) in &rewrites {
        graph.remove_node(*node_id);
    }

    graph.mark_mutated();
    rewrites.len()
}

fn shapes_equal(a: &[DimExpr], b: &[DimExpr]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(da, db)| match (da.evaluate(), db.evaluate()) {
            (Some(va), Some(vb)) => va == vb,
            _ => da == db,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ComputeGraph, Opcode, TensorType};

    #[test]
    fn cast_with_different_quantization_metadata_is_not_eliminated() {
        let mut graph = ComputeGraph::new();
        // Two U4Scaled types with different quantization metadata should
        // NOT be considered equal by the DCE pass.
        let input_rep = crate::types::ValueRepresentation::packed_affine_dequantization(
            crate::types::ScalarType::U4,
            8,
            crate::types::QuantizationGranularity::PerTensor,
            vec![0.5],
            vec![-1.0],
        )
        .unwrap();
        let output_rep = crate::types::ValueRepresentation::packed_affine_dequantization(
            crate::types::ScalarType::U4,
            8,
            crate::types::QuantizationGranularity::PerTensor,
            vec![1.0],
            vec![0.0],
        )
        .unwrap();
        let layout = crate::types::TensorStorageLayout {
            encoding: crate::types::StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            },
            row_packed: true,
            prefix_bytes: 0,
            suffix_bytes: crate::types::PACKED_SIMD_MARGIN_BYTES,
        };
        let input_type = TensorType::from_parts(vec![DimExpr::Known(8)], input_rep, layout);
        let output_type = TensorType::from_parts(vec![DimExpr::Known(8)], output_rep, layout);
        let input = graph.add_node(Opcode::Input, vec![], input_type);
        let cast = graph.add_node(Opcode::Cast, vec![input], output_type);

        assert_eq!(eliminate_noops(&mut graph), 0);
        assert!(graph.get_node(cast).is_some());
    }

    #[test]
    fn chained_noops_rewrite_consumers_to_live_terminal_input() {
        let mut graph = ComputeGraph::new();
        let ty = TensorType::new(vec![DimExpr::Known(4)], crate::ir::IrDType::F32);
        let input = graph.add_node(Opcode::Input, vec![], ty.clone());
        let cast_a = graph.add_node(Opcode::Cast, vec![input], ty.clone());
        let cast_b = graph.add_node(Opcode::Cast, vec![cast_a], ty.clone());
        let consumer = graph.add_node(Opcode::Relu, vec![cast_b], ty);
        graph.inputs = vec![input];
        graph.outputs = vec![consumer];

        eliminate_dead_code(&mut graph);

        assert!(graph.get_node(cast_a).is_none());
        assert!(graph.get_node(cast_b).is_none());
        assert_eq!(graph.get_node(consumer).unwrap().inputs, vec![input]);
        graph.validate_with_limits(&Default::default()).unwrap();
    }
}
