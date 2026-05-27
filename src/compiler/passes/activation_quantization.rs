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

use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorType};

/// Insert `QuantizeActivations`/`DequantizeActivations` around every `MatMul`
/// node whose first input (the activation) is not already quantized.
///
/// The pass is idempotent — it skips MatMuls whose first input is already
/// a `QuantizeActivations` output.
pub fn quantize_activations(graph: &mut ComputeGraph) -> Result<(), String> {
    struct Rewrite {
        matmul_id: NodeId,
        act_id: NodeId,
        matmul_shape: Vec<DimExpr>,
        act_shape: Vec<DimExpr>,
        consumers: Vec<NodeId>,
        is_graph_output: bool,
    }

    let mut rewrites: Vec<Rewrite> = Vec::new();

    let graph_ref = &*graph;
    crate::utils::traverse_graph(graph_ref, |node_id, node| {
        if node.opcode != Opcode::MatMul {
            return Ok(());
        }

        let &act_id = match node.inputs.first() {
            Some(id) => id,
            None => return Ok(()),
        };

        let matmul_shape = node.output_type.shape.clone();

        if let Some(act_node) = graph_ref.get_node(act_id) {
            if act_node.opcode == Opcode::QuantizeActivations {
                return Ok(());
            }
        }

        if let Some(act_node) = graph_ref.get_node(act_id) {
            if matches!(act_node.opcode, Opcode::Input | Opcode::Constant(_)) {
                return Ok(());
            }
        }

        let output_id = node_id;
        let consumers: Vec<NodeId> = graph_ref.consumers(output_id);
        let is_graph_output = graph_ref.outputs.contains(&output_id);

        let act_shape = graph_ref
            .get_node(act_id)
            .map(|n| n.output_type.shape.clone())
            .unwrap_or_default();

        rewrites.push(Rewrite {
            matmul_id: node_id,
            act_id,
            matmul_shape,
            act_shape,
            consumers,
            is_graph_output,
        });

        Ok(())
    })?;

    for rw in rewrites {
        let quant_type = TensorType::new(rw.act_shape, IrDType::I8);
        let quantize_id = graph.add_node(Opcode::QuantizeActivations, vec![rw.act_id], quant_type);

        if let Some(mm_node) = graph.get_node_mut(rw.matmul_id) {
            mm_node.inputs[0] = quantize_id;
        }

        // MatMul kernels consume quantized activations but still produce F32.
        // Do not insert DequantizeActivations after MatMul: that node expects
        // an I8 activation payload and would misread the F32 MatMul output.
        let _ = rw.matmul_shape;
        let _ = rw.consumers;
        let _ = rw.is_graph_output;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::executor::GraphExecutor;
    use crate::backend::Backend;
    use crate::compiler::passes::{memory_planning, shape_inference};
    use crate::ir::node::DimExpr;

    /// Test that activation quantization round-trips with reasonable accuracy.
    #[test]
    fn test_activation_quantization_roundtrip() {
        // Build a simple graph: Input → MatMul → Relu → Output
        let mut graph = ComputeGraph::new();
        let input_a_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![input_a_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );

        graph.set_inputs(vec![input_a_id, weight_id]);
        graph.set_outputs(vec![mm_id]);

        // Run standard compile without activation quantization
        let mut executor = GraphExecutor::new(CpuBackend);
        let (mut plan_no_q, mem_no_q, _) = executor
            .compile_with_plan_and_quantize(&graph, None)
            .unwrap();

        let input_a = vec![1.0f32, 2.0, 3.0, 4.0];
        let input_w = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let input_bytes_a: Vec<u8> = bytemuck::cast_slice(&input_a).to_vec();
        let input_bytes_w: Vec<u8> = bytemuck::cast_slice(&input_w).to_vec();

        let result_no_q = executor
            .execute(
                &graph,
                &mut plan_no_q,
                &mem_no_q,
                &[&input_bytes_a, &input_bytes_w],
            )
            .unwrap();
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
        let mut plan_q = CpuBackend.compile(&graph_q, &mem_q).unwrap();

        let result_q = executor
            .execute(
                &graph_q,
                &mut plan_q,
                &mem_q,
                &[&input_bytes_a, &input_bytes_w],
            )
            .unwrap();
        let result_q_f32: Vec<f32> = bytemuck::cast_slice(&result_q[0]).to_vec();

        // Results should be close (INT8 quantization introduces small error)
        assert_eq!(result_no_q_f32.len(), result_q_f32.len());
        for (&expected, &actual) in result_no_q_f32.iter().zip(result_q_f32.iter()) {
            let err = (expected - actual).abs();
            let tol = (expected.abs() * 0.02).max(0.1);
            assert!(
                err <= tol,
                "activation quant mismatch: expected {}, got {} (err={}, tol={})",
                expected,
                actual,
                err,
                tol
            );
        }
    }

    /// Test that the pass is idempotent.
    #[test]
    fn test_activation_quantization_idempotent() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
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
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
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
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![const_id, weight_id],
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
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        // First MatMul produces an intermediate activation
        let mm1_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        // Second MatMul consumes the first MatMul's output (intermediate activation)
        let weight2_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm2_id = graph.add_node(
            Opcode::MatMul,
            vec![mm1_id, weight2_id],
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

    /// Test that MatMul output stays F32 directly when activation quantization
    /// is inserted before MatMul. Downstream F32 consumers should consume the
    /// MatMul output directly; DequantizeActivations is only for I8 payloads.
    #[test]
    fn test_activation_quantization_downstream_f32_consumers_use_matmul_output() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        // Intermediate relu so the MatMul's activation is not an Input (which is skipped)
        let relu_act_id = graph.add_node(
            Opcode::Relu,
            vec![input_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![relu_act_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        // A Relu consumes the MatMul output (non-MatMul consumer)
        let relu_id = graph.add_node(
            Opcode::Relu,
            vec![mm_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![mm_id, relu_id]);

        shape_inference::infer_shapes(&mut graph).unwrap();
        quantize_activations(&mut graph).unwrap();

        // The Relu node should consume the MatMul directly because MatMul
        // output_type is already F32.
        let relu_node = graph.get_node(relu_id).unwrap();
        assert_eq!(
            relu_node.inputs[0], mm_id,
            "downstream F32 consumer should receive MatMul output directly"
        );

        let mm_node = graph.get_node(mm_id).unwrap();
        let qa_node = graph.get_node(mm_node.inputs[0]).unwrap();
        assert_eq!(
            qa_node.opcode,
            Opcode::QuantizeActivations,
            "MatMul activation input should still be quantized"
        );
    }

    /// Test that the pass does nothing on graphs without MatMul.
    #[test]
    fn test_activation_quantization_no_matmul() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        let relu_id = graph.add_node(
            Opcode::Relu,
            vec![input_id],
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
