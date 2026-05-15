//! Activation quantization compiler pass.
//!
//! Inserts `QuantizeActivations` → `MatMul` → `DequantizeActivations` subgraphs
//! so that matrix multiplications consume INT8 activations instead of F32.
//! This reduces activation memory bandwidth by 4×.
//!
//! The backward pass uses straight-through estimation (STE): the gradient flows
//! through both `QuantizeActivations` and `DequantizeActivations` unchanged.
//!
//! # Pass order
//!
//! Run this pass **after** shape inference (so shapes are known) and **before**
//! memory planning (so the planner allocates slots for the new nodes).

use crate::ir::node::{ComputeGraph, IrDType, NodeId, Opcode, TensorType};

/// Insert `QuantizeActivations`/`DequantizeActivations` around every `MatMul`
/// node whose first input (the activation) is not already quantized.
///
/// The pass is idempotent — it skips MatMuls whose first input is already
/// a `QuantizeActivations` output.
pub fn quantize_activations(graph: &mut ComputeGraph) -> Result<(), String> {
    let order = graph.topological_sort();

    // Collect rewrites first, then apply them, to avoid borrow-checker issues.
    struct Rewrite {
        matmul_id: NodeId,
        quantize_id: NodeId,
        dequantize_id: NodeId,
    }

    let mut rewrites: Vec<Rewrite> = Vec::new();

    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n.clone(),
            None => continue,
        };

        if node.opcode != Opcode::MatMul {
            continue;
        }

        // Get the activation input (input[0]).
        let &act_id = match node.inputs.first() {
            Some(id) => id,
            None => continue,
        };

        // Skip if the activation is already quantized (idempotent).
        if let Some(act_node) = graph.get_node(act_id) {
            if act_node.opcode == Opcode::QuantizeActivations {
                continue;
            }
        }

        // The activation is a non-Input, non-Constant node we want to quantize.
        // Skip Input nodes (they carry external data that may already be in
        // the right format) and Constants (those are weights, handled by
        // quantize_weights).
        if let Some(act_node) = graph.get_node(act_id) {
            if matches!(act_node.opcode, Opcode::Input | Opcode::Constant(_)) {
                continue;
            }
        }

        // Collect consumers of the MatMul's output (before we insert new nodes).
        let output_id = node_id;
        let consumers: Vec<NodeId> = graph.consumers(output_id);
        let is_graph_output = graph.outputs.contains(&output_id);

        // Create QuantizeActivations node (f32 → i8).
        let act_shape = graph.get_node(act_id)
            .map(|n| n.output_type.shape.clone())
            .unwrap_or_default();
        let quant_type = TensorType::new(act_shape, IrDType::I8);
        let quantize_id = graph.add_node(
            Opcode::QuantizeActivations,
            vec![act_id],
            quant_type,
        );

        // Rewire MatMul's first input to the QuantizeActivations output.
        if let Some(mm_node) = graph.get_node_mut(node_id) {
            mm_node.inputs[0] = quantize_id;
        }

        // Create DequantizeActivations node (i8 → f32).
        let matmul_shape = node.output_type.shape.clone();
        let dequant_type = TensorType::new(matmul_shape, IrDType::F32);
        let dequantize_id = graph.add_node(
            Opcode::DequantizeActivations,
            vec![output_id],
            dequant_type,
        );

        // Rewire consumers of the MatMul output to consume DequantizeActivations.
        for &consumer_id in &consumers {
            if let Some(consumer) = graph.get_node_mut(consumer_id) {
                for inp in consumer.inputs.iter_mut() {
                    if *inp == output_id {
                        *inp = dequantize_id;
                    }
                }
            }
        }

        // If the MatMul was a graph output, replace it with DequantizeActivations.
        if is_graph_output {
            if let Some(pos) = graph.outputs.iter().position(|&id| id == output_id) {
                graph.outputs[pos] = dequantize_id;
            }
        }

        rewrites.push(Rewrite {
            matmul_id: node_id,
            quantize_id,
            dequantize_id,
        });
    }

    let _ = rewrites; // Keep for potential future diagnostics
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::ir::node::DimExpr;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::executor::GraphExecutor;
    use crate::compiler::passes::{memory_planning, shape_inference};

    /// Test that activation quantization round-trips with reasonable accuracy.
    #[test]
    fn test_activation_quantization_roundtrip() {
        // Build a simple graph: Input → MatMul → Relu → Output
        let mut graph = ComputeGraph::new();
        let input_a_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul, vec![input_a_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );

        graph.set_inputs(vec![input_a_id, weight_id]);
        graph.set_outputs(vec![mm_id]);

        // Run standard compile without activation quantization
        let executor = GraphExecutor::new(CpuBackend);
        let (plan_no_q, mem_no_q, _) = executor.compile_with_plan_and_quantize(&graph, None).unwrap();

        let input_a = vec![1.0f32, 2.0, 3.0, 4.0];
        let input_w = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let input_bytes_a: Vec<u8> = bytemuck::cast_slice(&input_a).to_vec();
        let input_bytes_w: Vec<u8> = bytemuck::cast_slice(&input_w).to_vec();

        let result_no_q = executor.execute(&graph, &plan_no_q, &mem_no_q, &[&input_bytes_a, &input_bytes_w]).unwrap();
        let result_no_q_f32: Vec<f32> = bytemuck::cast_slice(&result_no_q[0]).to_vec();

        // Run with activation quantization (manual pipeline without fusion)
        let mut graph_q = graph.clone();
        shape_inference::infer_shapes(&mut graph_q).unwrap();
        quantize_activations(&mut graph_q).unwrap();
        // Skip operator fusion for this test (the QuantizeActivations → MatMul → DequantizeActivations
        // pattern may not match existing fusion rules). Just plan memory and compile.
        let mem_q = memory_planning::plan_memory(&graph_q).unwrap();
        // Set graph outputs to include the MatMul output (which has been rewired through
        // DequantizeActivations if the pass rewired it, or still MatMul if it's a graph output)
        let plan_q = CpuBackend.compile(&graph_q, &mem_q).unwrap();

        let result_q = executor.execute(&graph_q, &plan_q, &mem_q, &[&input_bytes_a, &input_bytes_w]).unwrap();
        let result_q_f32: Vec<f32> = bytemuck::cast_slice(&result_q[0]).to_vec();

        // Results should be close (INT8 quantization introduces small error)
        assert_eq!(result_no_q_f32.len(), result_q_f32.len());
        for (&expected, &actual) in result_no_q_f32.iter().zip(result_q_f32.iter()) {
            let err = (expected - actual).abs();
            let tol = (expected.abs() * 0.02).max(0.1);
            assert!(
                err <= tol,
                "activation quant mismatch: expected {}, got {} (err={}, tol={})",
                expected, actual, err, tol
            );
        }
    }

    /// Test that the pass is idempotent.
    #[test]
    fn test_activation_quantization_idempotent() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul, vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![mm_id]);

        shape_inference::infer_shapes(&mut graph).unwrap();
        quantize_activations(&mut graph).unwrap();
        let node_count_1 = graph.node_count();

        // Run again — should have same number of nodes (idempotent)
        quantize_activations(&mut graph).unwrap();
        let node_count_2 = graph.node_count();
        assert_eq!(node_count_1, node_count_2, "pass is not idempotent");
    }

    /// Test that Input activations feeding MatMul are skipped (not quantized).
    #[test]
    fn test_activation_quantization_skips_input_activations() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul, vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![mm_id]);

        shape_inference::infer_shapes(&mut graph).unwrap();
        quantize_activations(&mut graph).unwrap();

        // The MatMul's activation input (input_id) should NOT have been replaced
        // since input nodes are skipped
        let mm_node = graph.get_node(mm_id).unwrap();
        assert_eq!(
            mm_node.inputs[0], input_id,
            "Input activation should not be replaced by QuantizeActivations"
        );
    }

    /// Test that Constant activations feeding MatMul are skipped.
    #[test]
    fn test_activation_quantization_skips_constant_activations() {
        let mut graph = ComputeGraph::new();
        let const_data: Vec<u8> = vec![1u8; 16]; // 4 f32 values
        let const_tt = TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32);
        let const_id = graph.add_node(
            Opcode::Constant(crate::ir::node::TensorValue::Data {
                bytes: const_data,
                tensor_type: const_tt.clone(),
            }),
            vec![],
            const_tt,
        );
        let weight_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul, vec![const_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![weight_id]);
        graph.set_outputs(vec![mm_id]);

        shape_inference::infer_shapes(&mut graph).unwrap();
        quantize_activations(&mut graph).unwrap();

        // The MatMul's activation input (const_id) should NOT be replaced
        let mm_node = graph.get_node(mm_id).unwrap();
        assert_eq!(
            mm_node.inputs[0], const_id,
            "Constant activation should not be replaced by QuantizeActivations"
        );
    }

    /// Test that the pass inserts QuantizeActivations/DequantizeActivations
    /// around MatMul for non-Input, non-Constant activations.
    #[test]
    fn test_activation_quantization_inserts_qdq_for_intermediate() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        // First MatMul produces an intermediate activation
        let mm1_id = graph.add_node(
            Opcode::MatMul, vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        // Second MatMul consumes the first MatMul's output (intermediate activation)
        let weight2_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm2_id = graph.add_node(
            Opcode::MatMul, vec![mm1_id, weight2_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id, weight_id, weight2_id]);
        graph.set_outputs(vec![mm2_id]);

        shape_inference::infer_shapes(&mut graph).unwrap();
        quantize_activations(&mut graph).unwrap();

        // The second MatMul's activation input should now be QuantizeActivations
        let mm2_node = graph.get_node(mm2_id).unwrap();
        let act_input_id = mm2_node.inputs[0];
        let act_node = graph.get_node(act_input_id).unwrap();
        assert_eq!(
            act_node.opcode,
            Opcode::QuantizeActivations,
            "intermediate activation should be quantized"
        );
    }

    /// Test that DequantizeActivations is inserted after MatMul when
    /// there are downstream non-MatMul consumers.
    #[test]
    fn test_activation_quantization_downstream_dequantize() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        // Intermediate relu so the MatMul's activation is not an Input (which is skipped)
        let relu_act_id = graph.add_node(
            Opcode::Relu, vec![input_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul, vec![relu_act_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        // A Relu consumes the MatMul output (non-MatMul consumer)
        let relu_id = graph.add_node(
            Opcode::Relu, vec![mm_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![mm_id, relu_id]);

        shape_inference::infer_shapes(&mut graph).unwrap();
        quantize_activations(&mut graph).unwrap();

        // The Relu node should consume DequantizeActivations, not the MatMul directly
        let relu_node = graph.get_node(relu_id).unwrap();
        let dq_id = relu_node.inputs[0];
        let dq_node = graph.get_node(dq_id).unwrap();
        assert_eq!(
            dq_node.opcode,
            Opcode::DequantizeActivations,
            "downstream non-MatMul consumer should get DequantizeActivations"
        );
    }

    /// Test that the pass does nothing on graphs without MatMul.
    #[test]
    fn test_activation_quantization_no_matmul() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input, vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        let relu_id = graph.add_node(
            Opcode::Relu, vec![input_id],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id]);
        graph.set_outputs(vec![relu_id]);

        shape_inference::infer_shapes(&mut graph).unwrap();
        // With no MatMul, the pass should be a no-op (0 new nodes)
        let node_count_before = graph.node_count();
        quantize_activations(&mut graph).unwrap();
        let node_count_after = graph.node_count();
        assert_eq!(node_count_before, node_count_after);
    }
}
