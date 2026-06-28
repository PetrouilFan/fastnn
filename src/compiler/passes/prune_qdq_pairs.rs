//! Prune redundant QuantizeActivations/DequantizeActivations pairs.
//!
//! When activation quantization inserts `DequantizeActivations` after a MatMul
//! and the output feeds only `QuantizeActivations` nodes (for downstream MatMuls),
//! the dequant→quant round-trip is redundant and can be eliminated.
//!
//! Pattern: `MatMul_A → DequantizeActivations → QuantizeActivations → MatMul_B`
//!   → becomes `MatMul_A → MatMul_B`
//!
//! This reduces quantization error and eliminates two kernel dispatches per chain.

use crate::ir::node::{ComputeGraph, NodeId, Opcode};
use crate::FastnnError;
use std::collections::HashSet;

/// Remove redundant DequantizeActivations→QuantizeActivations pairs.
///
/// Run this pass **after** `quantize_activations()` and **before** memory planning.
pub fn prune_qdq_pairs(graph: &mut ComputeGraph) -> Result<(), FastnnError> {
    let mut to_remove: HashSet<NodeId> = HashSet::new();
    let mut rewires: Vec<(NodeId, NodeId)> = Vec::new(); // (consumer_id, new_activation_id)

    // Collect DequantizeActivations nodes
    let dq_ids: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|n| n.opcode == Opcode::DequantizeActivations)
        .map(|n| n.id)
        .collect();

    for &dq_id in &dq_ids {
        if to_remove.contains(&dq_id) {
            continue;
        }

        // Get the source node (MatMul) feeding the DequantizeActivations
        let src_id = match graph.get_node(dq_id) {
            Some(n) => n.inputs.first().copied(),
            None => continue,
        };
        let src_id = match src_id {
            Some(id) => id,
            None => continue,
        };

        // Get all consumers of this DequantizeActivations
        let consumers: Vec<NodeId> = graph.consumers(dq_id);

        // Skip if any consumer is NOT a QuantizeActivations node
        // (those consumers need the F32 output)
        let all_are_quant = consumers.iter().all(|&c_id| {
            graph
                .get_node(c_id)
                .map(|n| n.opcode == Opcode::QuantizeActivations)
                .unwrap_or(false)
        });

        if !all_are_quant || consumers.is_empty() {
            continue;
        }

        // All consumers are QuantizeActivations — we can remove the
        // DequantizeActivations and each QuantizeActivations, routing
        // the original MatMul output directly to the downstream MatMuls.
        for &q_id in &consumers {
            if to_remove.contains(&q_id) {
                continue;
            }

            // Get MatMul consumers of this QuantizeActivations
            let q_consumers: Vec<NodeId> = graph.consumers(q_id);
            for &matmul_id in &q_consumers {
                if let Some(matmul) = graph.get_node(matmul_id) {
                    if matmul.opcode == Opcode::MatMul {
                        // Route this MatMul's first input to the original source
                        rewires.push((matmul_id, src_id));
                    }
                }
            }

            to_remove.insert(q_id);
        }

        to_remove.insert(dq_id);
    }

    // Apply rewires (must happen before node removal to preserve node references)
    for (consumer_id, new_activation_id) in &rewires {
        if let Some(consumer) = graph.get_node_mut(*consumer_id) {
            if !consumer.inputs.is_empty() {
                consumer.inputs[0] = *new_activation_id;
            }
        }
    }

    // Handle graph outputs: if a DequantizeActivations was a graph output,
    // replace it with its source
    let mut outputs_to_update: Vec<(usize, NodeId)> = Vec::new();
    for (i, &output_id) in graph.outputs.iter().enumerate() {
        if to_remove.contains(&output_id) {
            if let Some(n) = graph.get_node(output_id) {
                if let Some(&src) = n.inputs.first() {
                    outputs_to_update.push((i, src));
                }
            }
        }
    }
    for (i, new_id) in outputs_to_update {
        graph.outputs[i] = new_id;
    }

    // Remove pruned nodes
    for &id in &to_remove {
        graph.remove_node(id);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::{DimExpr, IrDType, TensorType};

    /// Build a graph with a prunable DQ→Q chain:
    /// `Input(0) → Relu(1) → Q(MatMul_A_act) → MatMul_A(3) → DQ(6)
    ///                                                          → Q(7) → MatMul_B(5) → DQ(8) → Output`
    ///
    /// This is constructed manually because the activation quantization pass's
    /// collect-then-apply ordering prevents the DQ→Q pattern from forming in a
    /// single pass (the Q for MatMul-B links to MatMul-A directly, not through DQ).
    fn build_graph_with_dq_q_chain() -> ComputeGraph {
        let mut graph = ComputeGraph::new();
        let tt_2x8 = TensorType::new(vec![DimExpr::Known(2), DimExpr::Known(8)], IrDType::F32);
        let tt_8x4 = TensorType::new(vec![DimExpr::Known(8), DimExpr::Known(4)], IrDType::F32);
        let tt_2x4 = TensorType::new(vec![DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
        let tt_2x4_i8 = TensorType::new(vec![DimExpr::Known(2), DimExpr::Known(4)], IrDType::I8);

        let input_id = graph.add_node(Opcode::Input, vec![], tt_2x8.clone());
        let relu_id = graph.add_node(Opcode::Relu, vec![input_id], tt_2x8);
        let w1_id = graph.add_node(Opcode::Input, vec![], tt_8x4);
        let mm_a = graph.add_node(Opcode::MatMul, vec![relu_id, w1_id], tt_2x4.clone());
        let w2_id = graph.add_node(Opcode::Input, vec![], tt_4x4());

        // Insert DQ after MatMul_A — represents the dequant of MatMul_A's output
        let dq_id = graph.add_node(Opcode::DequantizeActivations, vec![mm_a], tt_2x4.clone());
        // Insert Q before MatMul_B that consumes the DQ — this is the redundant pair
        let q_id = graph.add_node(Opcode::QuantizeActivations, vec![dq_id], tt_2x4_i8);
        let mm_b = graph.add_node(Opcode::MatMul, vec![q_id, w2_id], tt_2x4.clone());
        // DQ after MatMul_B for graph output
        let dq_out_id = graph.add_node(Opcode::DequantizeActivations, vec![mm_b], tt_2x4);

        graph.set_inputs(vec![input_id, w1_id, w2_id]);
        graph.set_outputs(vec![dq_out_id]);
        graph
    }

    fn tt_4x4() -> TensorType {
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32)
    }

    #[test]
    fn test_prune_redundant_qdq_chain() {
        let mut graph = build_graph_with_dq_q_chain();
        let count_before = graph.node_count();

        prune_qdq_pairs(&mut graph).unwrap();
        let count_after = graph.node_count();

        // Should have removed DQ+Q pair (2 nodes)
        assert_eq!(
            count_after,
            count_before - 2,
            "pruning should remove 2 nodes (DQ+Q), before={}, after={}",
            count_before,
            count_after
        );

        // Verify no DQ feeds a Q (the round-trip chain is broken)
        for node in &graph.nodes {
            if node.opcode == Opcode::QuantizeActivations {
                for &input_id in &node.inputs {
                    if let Some(input_node) = graph.get_node(input_id) {
                        assert_ne!(
                            input_node.opcode,
                            Opcode::DequantizeActivations,
                            "QuantizeActivations should not consume DequantizeActivations after pruning"
                        );
                    }
                }
            }
        }

        // Verify the final graph output DQ still exists
        let dq_nodes: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| n.opcode == Opcode::DequantizeActivations)
            .collect();
        assert_eq!(dq_nodes.len(), 1, "only the final output DQ should remain");
    }

    #[test]
    fn test_prune_idempotent() {
        let mut graph = build_graph_with_dq_q_chain();
        prune_qdq_pairs(&mut graph).unwrap();
        let count_1 = graph.node_count();

        // Run again — should be a no-op
        prune_qdq_pairs(&mut graph).unwrap();
        let count_2 = graph.node_count();
        assert_eq!(count_1, count_2, "pruning pass is not idempotent");
    }

    #[test]
    fn test_prune_no_qdq_no_change() {
        // Graph without any DQ/Q nodes
        let mut graph = ComputeGraph::new();
        let tt_2x8 = TensorType::new(vec![DimExpr::Known(2), DimExpr::Known(8)], IrDType::F32);
        let tt_8x4 = TensorType::new(vec![DimExpr::Known(8), DimExpr::Known(4)], IrDType::F32);
        let tt_2x4 = TensorType::new(vec![DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
        let input_id = graph.add_node(Opcode::Input, vec![], tt_2x8);
        let w1_id = graph.add_node(Opcode::Input, vec![], tt_8x4);
        let mm_id = graph.add_node(Opcode::MatMul, vec![input_id, w1_id], tt_2x4);
        graph.set_inputs(vec![input_id, w1_id]);
        graph.set_outputs(vec![mm_id]);

        let count_before = graph.node_count();
        prune_qdq_pairs(&mut graph).unwrap();
        let count_after = graph.node_count();
        assert_eq!(
            count_before, count_after,
            "no-op pass should not change node count"
        );
    }

    #[test]
    fn test_prune_partial_chain_no_change() {
        // DQ→SomeOp→Q chain should NOT be pruned (non-Q consumer exists)
        let mut graph = ComputeGraph::new();
        let tt_2x4 = TensorType::new(vec![DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
        let tt_2x4_i8 = TensorType::new(vec![DimExpr::Known(2), DimExpr::Known(4)], IrDType::I8);

        let mm_a = graph.add_node(Opcode::MatMul, vec![], tt_2x4.clone());
        let dq_id = graph.add_node(Opcode::DequantizeActivations, vec![mm_a], tt_2x4.clone());
        // Non-Q consumer (Relu) shares the DQ output
        let relu_id = graph.add_node(Opcode::Relu, vec![dq_id], tt_2x4.clone());
        // Q consumer also uses DQ output
        let q_id = graph.add_node(Opcode::QuantizeActivations, vec![dq_id], tt_2x4_i8);
        let mm_b = graph.add_node(Opcode::MatMul, vec![q_id], tt_2x4);
        graph.set_inputs(vec![]);
        graph.set_outputs(vec![relu_id, mm_b]);

        let count_before = graph.node_count();
        prune_qdq_pairs(&mut graph).unwrap();
        let count_after = graph.node_count();
        assert_eq!(
            count_before, count_after,
            "should not prune when DQ has non-Q consumers"
        );
    }
}
