//! Integration tests for the residual + add + norm fusion pass.
//!
//! Tests that the fusion pass correctly detects and fuses
//! Add(residual, main) → LayerNorm/RMSNorm patterns.

use fastnn::compiler::passes::operator_fusion::fuse_operators;
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{ComputeGraph, IrDType, Opcode};

/// Check if a graph contains a node with the given opcode.
fn has_opcode(graph: &ComputeGraph, op: Opcode) -> bool {
    graph.nodes.iter().any(|n| n.opcode == op)
}

/// Count nodes with a given opcode.
fn count_opcode(graph: &ComputeGraph, op: Opcode) -> usize {
    graph.nodes.iter().filter(|n| n.opcode == op).count()
}

// ── Structural tests ─────────────────────────────────────────────────

#[test]
fn test_fuse_add_layernorm() {
    let g = GraphBuilder::new();
    let input = g.input(&[1, 4], IrDType::F32);
    let residual = g.input(&[1, 4], IrDType::F32);
    let w = g.parameter(&[4], IrDType::F32);
    let b = g.parameter(&[4], IrDType::F32);

    let add = g.add(&input, &residual);
    let _norm = g.layer_norm(&add, &w, &b, 1e-5);

    let mut graph = g.to_graph();
    let n_inputs = graph.inputs.len();
    // Set outputs
    let last_id = graph.nodes.last().map(|n| n.id).unwrap_or(0);
    graph.set_outputs(vec![last_id]);

    fuse_operators(&mut graph).unwrap();

    assert!(
        has_opcode(&graph, Opcode::FusedResidualAddNorm),
        "FusedResidualAddNorm should exist after fusion"
    );
    // LayerNorm should be gone (fused)
    assert!(
        !has_opcode(&graph, Opcode::LayerNorm),
        "LayerNorm should be removed after fusion"
    );
    // inputs unchanged
    assert_eq!(graph.inputs.len(), n_inputs, "inputs should not change");
}

#[test]
fn test_fuse_add_rmsnorm() {
    let g = GraphBuilder::new();
    let input = g.input(&[1, 6], IrDType::F32);
    let residual = g.input(&[1, 6], IrDType::F32);
    let w = g.parameter(&[6], IrDType::F32);

    let add = g.add(&input, &residual);
    let _norm = g.rms_norm(&add, &w, 1e-5);

    let mut graph = g.to_graph();
    let last_id = graph.nodes.last().map(|n| n.id).unwrap_or(0);
    graph.set_outputs(vec![last_id]);

    fuse_operators(&mut graph).unwrap();

    assert!(
        has_opcode(&graph, Opcode::FusedResidualAddNorm),
        "FusedResidualAddNorm should exist after RMSNorm fusion"
    );
    assert!(
        !has_opcode(&graph, Opcode::RMSNorm),
        "RMSNorm should be removed after fusion"
    );
}

#[test]
fn test_no_fusion_without_add() {
    // Direct norm (no Add before it) should NOT be fused
    let g = GraphBuilder::new();
    let input = g.input(&[1, 4], IrDType::F32);
    let w = g.parameter(&[4], IrDType::F32);
    let b = g.parameter(&[4], IrDType::F32);

    let _norm = g.layer_norm(&input, &w, &b, 1e-5);

    let mut graph = g.to_graph();
    let last_id = graph.nodes.last().map(|n| n.id).unwrap_or(0);
    graph.set_outputs(vec![last_id]);

    fuse_operators(&mut graph).unwrap();

    assert!(
        !has_opcode(&graph, Opcode::FusedResidualAddNorm),
        "Should not fuse when there's no Add before Norm"
    );
    assert!(
        has_opcode(&graph, Opcode::LayerNorm),
        "LayerNorm should remain unfused"
    );
}

#[test]
fn test_fusion_idempotent() {
    let g = GraphBuilder::new();
    let input = g.input(&[1, 4], IrDType::F32);
    let residual = g.input(&[1, 4], IrDType::F32);
    let w = g.parameter(&[4], IrDType::F32);
    let b = g.parameter(&[4], IrDType::F32);

    let add = g.add(&input, &residual);
    let _norm = g.layer_norm(&add, &w, &b, 1e-5);

    let mut graph = g.to_graph();
    let last_id = graph.nodes.last().map(|n| n.id).unwrap_or(0);
    graph.set_outputs(vec![last_id]);

    // First pass
    fuse_operators(&mut graph).unwrap();
    let count1 = count_opcode(&graph, Opcode::FusedResidualAddNorm);

    // Second pass
    fuse_operators(&mut graph).unwrap();
    let count2 = count_opcode(&graph, Opcode::FusedResidualAddNorm);

    assert_eq!(count1, count2, "second fusion pass should be a no-op");
    assert_eq!(count1, 1, "exactly one fused node should exist");
}

#[test]
fn test_fusion_normalizes_its_fused_nodes() {
    // Verify that fusion doesn't touch norms that aren't preceded by Add
    let g = GraphBuilder::new();
    let input = g.input(&[1, 4], IrDType::F32);
    let residual = g.input(&[1, 4], IrDType::F32);
    let w = g.parameter(&[4], IrDType::F32);
    let b = g.parameter(&[4], IrDType::F32);
    let w2 = g.parameter(&[4], IrDType::F32);
    let b2 = g.parameter(&[4], IrDType::F32);

    // Pattern that should fuse
    let add = g.add(&input, &residual);
    let _norm1 = g.layer_norm(&add, &w, &b, 1e-5);

    // Pattern that should NOT fuse (no Add)
    let _norm2 = g.layer_norm(&input, &w2, &b2, 1e-5);

    let mut graph = g.to_graph();
    let last_id = graph.nodes.last().map(|n| n.id).unwrap_or(0);
    graph.set_outputs(vec![last_id]);

    fuse_operators(&mut graph).unwrap();

    assert!(
        has_opcode(&graph, Opcode::FusedResidualAddNorm),
        "FusedResidualAddNorm should exist for Add+Norm"
    );
    assert!(
        has_opcode(&graph, Opcode::LayerNorm),
        "Second LayerNorm (without Add) should remain unfused"
    );
    assert_eq!(
        count_opcode(&graph, Opcode::LayerNorm),
        1,
        "exactly one LayerNorm should remain (the one without Add)"
    );
    assert_eq!(
        count_opcode(&graph, Opcode::FusedResidualAddNorm),
        1,
        "exactly one fused node should exist"
    );
}

#[test]
fn test_fusion_with_other_fusions() {
    // Test that Add+LayerNorm fusion coexists with MatMulAddRelu fusion
    let g = GraphBuilder::new();
    let input = g.input(&[1, 8], IrDType::F32);
    let w_mm = g.parameter(&[8, 4], IrDType::F32);
    let b_mm = g.parameter(&[4], IrDType::F32);
    let residual = g.input(&[1, 4], IrDType::F32);
    let nw = g.parameter(&[4], IrDType::F32);
    let nb = g.parameter(&[4], IrDType::F32);

    let mm = g.matmul(&input, &w_mm);
    let biased = g.add(&mm, &b_mm);
    let relu = g.relu(&biased);
    let add = g.add(&relu, &residual);
    let _norm = g.layer_norm(&add, &nw, &nb, 1e-5);

    let mut graph = g.to_graph();
    let last_id = graph.nodes.last().map(|n| n.id).unwrap_or(0);
    graph.set_outputs(vec![last_id]);

    fuse_operators(&mut graph).unwrap();

    assert!(
        has_opcode(&graph, Opcode::FusedResidualAddNorm),
        "FusedResidualAddNorm should exist alongside MatMulAddRelu fusion"
    );
}

#[test]
fn test_fusion_preserves_outputs() {
    // Verify that fusion doesn't change the graph output node
    let g = GraphBuilder::new();
    let input = g.input(&[1, 4], IrDType::F32);
    let residual = g.input(&[1, 4], IrDType::F32);
    let w = g.parameter(&[4], IrDType::F32);
    let b = g.parameter(&[4], IrDType::F32);

    let add = g.add(&input, &residual);
    let norm = g.layer_norm(&add, &w, &b, 1e-5);

    let mut graph = g.to_graph();
    let output_id = norm.node_id();
    graph.set_outputs(vec![output_id]);

    fuse_operators(&mut graph).unwrap();

    // The norm is fused, so the fused node becomes output
    assert_eq!(
        graph.outputs,
        vec![output_id],
        "output node id should be preserved (the fused norm)"
    );
}
