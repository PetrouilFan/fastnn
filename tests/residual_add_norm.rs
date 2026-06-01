//! Integration tests for the residual + add + norm fusion pass.
//!
//! Tests that the fusion pass correctly detects and fuses
//! Add(residual, main) → LayerNorm/RMSNorm patterns.

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::backend::Instruction;
use fastnn::compiler::passes::operator_fusion::fuse_operators;
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{ComputeGraph, DimExpr, IrDType, Opcode, TensorType};

/// Check if a graph contains a node with the given opcode.
fn has_opcode(graph: &ComputeGraph, op: Opcode) -> bool {
    graph.nodes.iter().any(|n| n.opcode == op)
}

/// Count nodes with a given opcode.
fn count_opcode(graph: &ComputeGraph, op: Opcode) -> usize {
    graph.nodes.iter().filter(|n| n.opcode == op).count()
}

fn graph_from(builder: &GraphBuilder, output: &fastnn::ir::builder::GraphTensor) -> ComputeGraph {
    let mut graph = builder.to_graph();
    graph.inputs = builder.recorded_input_ids();
    graph.set_outputs(vec![output.node_id()]);
    graph
}

fn run_single_output_f32(graph: &ComputeGraph, inputs: &[&[u8]]) -> (Vec<f32>, Vec<String>) {
    let mut executor = GraphExecutor::new(CpuBackend);
    let (mut plan, memory_plan, compiled_graph) = executor
        .compile_with_plan(graph)
        .expect("graph should compile");
    let kernels = plan
        .instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::CallKernel { kernel_name, .. } => Some(kernel_name.clone()),
            _ => None,
        })
        .collect();
    let outputs = executor
        .execute(&compiled_graph, &mut plan, &memory_plan, inputs)
        .expect("graph should execute");
    (bytemuck::cast_slice(&outputs[0]).to_vec(), kernels)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at {i}: actual={a} expected={e} diff={}",
            (a - e).abs()
        );
    }
}

fn reference_rms_residual_add(
    residual: &[f32],
    main: &[f32],
    weight: &[f32],
    row_size: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; main.len()];
    for row in 0..(main.len() / row_size) {
        let start = row * row_size;
        let end = start + row_size;
        let mut sq_sum = 0.0f32;
        for i in start..end {
            let x = main[i] + residual[i];
            sq_sum += x * x;
        }
        let rms = (sq_sum / row_size as f32 + eps).sqrt();
        for i in start..end {
            out[i] = (main[i] + residual[i]) / rms * weight[i - start];
        }
    }
    out
}

fn reference_layer_residual_add(
    residual: &[f32],
    main: &[f32],
    weight: &[f32],
    bias: &[f32],
    row_size: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; main.len()];
    for row in 0..(main.len() / row_size) {
        let start = row * row_size;
        let end = start + row_size;
        let mut sum = 0.0f32;
        for i in start..end {
            sum += main[i] + residual[i];
        }
        let mean = sum / row_size as f32;
        let mut var = 0.0f32;
        for i in start..end {
            let d = main[i] + residual[i] - mean;
            var += d * d;
        }
        let inv_std = 1.0 / (var / row_size as f32 + eps).sqrt();
        for i in start..end {
            let idx = i - start;
            out[i] = (main[i] + residual[i] - mean) * inv_std * weight[idx] + bias[idx];
        }
    }
    out
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

#[test]
fn test_cpu_fused_rmsnorm_batched_uses_hidden_row_size() {
    let builder = GraphBuilder::new();
    let main_t = builder.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
    let residual_t = builder.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
    let weight = vec![0.75, -1.25, 1.5, 0.5];
    let weight_t = builder.constant(
        bytemuck::cast_slice(&weight),
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let add = builder.add(&main_t, &residual_t);
    let norm = builder.rms_norm(&add, &weight_t, 1e-5);
    let graph = graph_from(&builder, &norm);

    let main = vec![1.0, -2.0, 3.5, -4.5, 2.25, -1.0, 0.5, 3.0];
    let residual = vec![-0.25, 0.5, -1.5, 2.0, 1.25, -0.75, 2.5, -3.5];
    let main_bytes = bytemuck::cast_slice(&main).to_vec();
    let residual_bytes = bytemuck::cast_slice(&residual).to_vec();
    let (actual, kernels) = run_single_output_f32(&graph, &[&main_bytes, &residual_bytes]);
    let expected = reference_rms_residual_add(&residual, &main, &weight, 4, 1e-5);

    assert!(
        kernels
            .iter()
            .any(|name| name == "fused_residual_add_rms_norm"),
        "expected fused RMS kernel, got {kernels:?}"
    );
    assert_close(&actual, &expected, 1e-5);
}

#[test]
fn test_cpu_fused_layernorm_matches_reference() {
    let builder = GraphBuilder::new();
    let main_t = builder.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
    let residual_t = builder.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
    let weight = vec![1.25, 0.75, -0.5, 1.5];
    let bias = vec![0.1, -0.2, 0.3, -0.4];
    let weight_t = builder.constant(
        bytemuck::cast_slice(&weight),
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let bias_t = builder.constant(
        bytemuck::cast_slice(&bias),
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let add = builder.add(&main_t, &residual_t);
    let norm = builder.layer_norm(&add, &weight_t, &bias_t, 1e-5);
    let graph = graph_from(&builder, &norm);

    let main = vec![0.5, -1.0, 2.0, -3.0, 1.5, 2.5, -0.75, -1.25];
    let residual = vec![1.0, 0.25, -0.5, 1.75, -2.0, 0.5, 1.25, -0.25];
    let main_bytes = bytemuck::cast_slice(&main).to_vec();
    let residual_bytes = bytemuck::cast_slice(&residual).to_vec();
    let (actual, kernels) = run_single_output_f32(&graph, &[&main_bytes, &residual_bytes]);
    let expected = reference_layer_residual_add(&residual, &main, &weight, &bias, 4, 1e-5);

    assert!(
        kernels
            .iter()
            .any(|name| name == "fused_residual_add_layer_norm"),
        "expected fused LayerNorm kernel, got {kernels:?}"
    );
    assert_close(&actual, &expected, 1e-5);
}
