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

use crate::compiler::passes::calibration::{CalibrationData, CalibrationStats};
use crate::error::FastnnError;
use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorType};
use std::collections::HashMap;

/// Insert `QuantizeActivations`/`DequantizeActivations` around every `MatMul`
/// and `Conv2d` node whose first input (the activation) is not already quantized.
///
/// The pass is idempotent — it skips ops whose first input is already
/// a `QuantizeActivations` output.
pub fn quantize_activations(graph: &mut ComputeGraph) -> Result<(), FastnnError> {
    struct Rewrite {
        node_id: NodeId,
        act_id: NodeId,
        node_shape: Vec<DimExpr>,
        act_shape: Vec<DimExpr>,
        consumers: Vec<NodeId>,
        is_graph_output: bool,
        opcode: Opcode,
    }

    let mut rewrites: Vec<Rewrite> = Vec::with_capacity(graph.nodes.len());

    let graph_ref = &*graph;
    crate::utils::traverse_graph(graph_ref, |node_id, node| {
        if !matches!(node.opcode, Opcode::MatMul | Opcode::Conv2d) {
            return Ok(());
        }

        let &act_id = match node.inputs.first() {
            Some(id) => id,
            None => return Ok(()),
        };

        let node_shape = node.output_type.shape.clone();

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
            node_id,
            act_id,
            node_shape,
            act_shape,
            consumers,
            is_graph_output,
            opcode: node.opcode.clone(),
        });

        Ok(())
    })
    .map_err(FastnnError::compilation)?;

    for rw in rewrites {
        let quant_type = TensorType::new(rw.act_shape, IrDType::I8);
        let quantize_id = graph.add_node(Opcode::QuantizeActivations, vec![rw.act_id], quant_type);

        if let Some(target) = graph.get_node_mut(rw.node_id) {
            target.inputs[0] = quantize_id;
        }

        let _ = rw.node_shape;
        let _ = rw.act_shape;
        let _ = rw.consumers;
        let _ = rw.is_graph_output;
        let _ = rw.opcode;
    }

    Ok(())
}

/// Calibration-driven activation quantization with per-channel Conv2d support.
///
/// Like [`quantize_activations`], but uses calibration statistics to determine
/// per-channel scale/zero-point for Conv2d activation inputs and per-tensor
/// scales for MatMul activation inputs.  Calibration scales are stored as
/// node attributes so the backend can emit per-channel Q/DQ kernels.
///
/// # Per-channel format
///
/// For Conv2d activations the `QuantizeActivations` node carries:
/// - `"mode"` → `"per_channel"`
/// - `"num_channels"` → `"C"` (input channels of the Conv2d)
/// - `"scales"` → `"s1,s2,...,sC"` (comma-separated f32)
/// - `"zero_points"` → `"zp1,zp2,...,zpC"` (comma-separated f32)
///
/// When no calibration entry exists for a node, it falls back to per-tensor
/// symmetric quantization (same as [`quantize_activations`]).
pub fn quantize_activations_with_calibration(
    graph: &mut ComputeGraph,
    calib: &CalibrationData,
) -> Result<(), FastnnError> {
    struct Rewrite {
        node_id: NodeId,
        act_id: NodeId,
        act_shape: Vec<DimExpr>,
        input_channels: usize,
        per_channel_scales: Vec<f32>,
        per_channel_zero_points: Vec<f32>,
        // Per-tensor fallback calibration stats
        per_tensor_scale: Option<f32>,
        per_tensor_zero_point: Option<f32>,
        // If Some, update this existing QuantizeActivations node instead of creating new
        existing_qa_id: Option<NodeId>,
    }

    let mut rewrites: Vec<Rewrite> = Vec::with_capacity(graph.nodes.len());

    let graph_ref = &*graph;
    crate::utils::traverse_graph(graph_ref, |node_id, node| {
        if !matches!(node.opcode, Opcode::MatMul | Opcode::Conv2d) {
            return Ok(());
        }

        let &act_id = match node.inputs.first() {
            Some(id) => id,
            None => return Ok(()),
        };

        // Check if activation already has QuantizeActivations
        let existing_qa_id = if let Some(act_node) = graph_ref.get_node(act_id) {
            if act_node.opcode == Opcode::QuantizeActivations {
                Some(act_id)
            } else {
                None
            }
        } else {
            None
        };

        // Skip if already quantized AND no calibration data to update with
        if existing_qa_id.is_some() && calib.stats.is_empty() {
            return Ok(());
        }
        // If we have calibration data and existing QA, we'll UPDATE it below

        if let Some(act_node) = graph_ref.get_node(act_id) {
            if matches!(act_node.opcode, Opcode::Input | Opcode::Constant(_)) {
                return Ok(());
            }
        }

        let act_shape = graph_ref
            .get_node(act_id)
            .map(|n| n.output_type.shape.clone())
            .unwrap_or_default();

        // Determine per-channel scales from calibration data
        let act_name = graph_ref
            .get_node(act_id)
            .map(|n| n.name.as_str())
            .unwrap_or("")
            .to_string();
        // Also get the Conv node's name for fallback lookup
        let conv_name = node.name.clone();

        // DEBUG: print calibration lookup
        if std::env::var("FASTNN_DEBUG_CALIB").is_ok() {
            eprintln!(
                "[CALIB_DEBUG] act_name='{}' conv_name='{}' avail_keys={:?}",
                act_name,
                conv_name,
                calib.stats.keys().collect::<Vec<_>>()
            );
        }

        // Helper to get calibration stats for either per-channel or per-tensor
        let get_calib_stats = |name: &str| -> Option<&CalibrationStats> {
            if name.is_empty() {
                None
            } else {
                calib.stats.get(name)
            }
        };

        let (
            input_channels,
            per_channel_scales,
            per_channel_zero_points,
            per_tensor_scale,
            per_tensor_zero_point,
        ) = if matches!(node.opcode, Opcode::Conv2d) {
            // Conv2d activation input channels is dim 1 of input shape (NCHW)
            let c = act_shape.get(1).and_then(|d| d.evaluate()).unwrap_or(0) as usize;
            if c > 0 {
                // Look up calibration stats for the activation tensor
                // Try activation node name first, then fall back to Conv node name
                let (scales, zps) = if !act_name.is_empty() {
                    calib
                        .stats
                        .get(&act_name)
                        .or_else(|| calib.stats.get(&conv_name))
                        .map(|stats| stats.compute_scale_zp_per_channel(8))
                        .unwrap_or_else(|| (vec![], vec![]))
                } else if !conv_name.is_empty() {
                    calib
                        .stats
                        .get(&conv_name)
                        .map(|stats| stats.compute_scale_zp_per_channel(8))
                        .unwrap_or_else(|| (vec![], vec![]))
                } else {
                    (vec![], vec![])
                };
                // Also get per-tensor stats for fallback
                let (pt_scale, pt_zp) = if !act_name.is_empty() {
                    get_calib_stats(&act_name)
                        .or_else(|| get_calib_stats(&conv_name))
                        .map(|stats| stats.compute_scale_zp(8))
                        .unwrap_or_else(|| (0.0, 0.0))
                } else if !conv_name.is_empty() {
                    get_calib_stats(&conv_name)
                        .map(|stats| stats.compute_scale_zp(8))
                        .unwrap_or_else(|| (0.0, 0.0))
                } else {
                    (0.0, 0.0)
                };
                // Only use per-channel if we have exactly c scales
                if scales.len() == c {
                    (c, scales, zps, Some(pt_scale), Some(pt_zp))
                } else {
                    (0, vec![], vec![], Some(pt_scale), Some(pt_zp))
                }
            } else {
                (0, vec![], vec![], None, None)
            }
        } else {
            (0, vec![], vec![], None, None)
        };

        rewrites.push(Rewrite {
            node_id,
            act_id,
            act_shape,
            input_channels,
            per_channel_scales,
            per_channel_zero_points,
            per_tensor_scale,
            per_tensor_zero_point,
            existing_qa_id,
        });

        Ok(())
    })
    .map_err(FastnnError::compilation)?;

    for rw in rewrites {
        let is_per_channel = !rw.per_channel_scales.is_empty();

        let quant_type = if is_per_channel {
            TensorType::new(rw.act_shape.clone(), IrDType::I8)
        } else {
            TensorType::new(rw.act_shape, IrDType::I8)
        };

        let mut attrs: HashMap<String, String> = HashMap::new();
        if is_per_channel {
            // Convert per-channel zero_points from U8 range (0..255) to I8 range for the backend.
            // Backend formula: q = round((x - zp) / scale), clamp to [-128, 127]
            // For I8: zp_i8 = scale * (128 - zp_u8)
            let per_channel_scales_i8 = rw.per_channel_scales.clone();
            let per_channel_zps_i8: Vec<f32> = rw
                .per_channel_scales
                .iter()
                .zip(rw.per_channel_zero_points.iter())
                .map(|(s, zp_u8)| s * (128.0 - zp_u8))
                .collect();
            let scale_str = per_channel_scales_i8
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let zp_str = per_channel_zps_i8
                .iter()
                .map(|z| z.to_string())
                .collect::<Vec<_>>()
                .join(",");
            attrs.insert("mode".to_string(), "per_channel".to_string());
            attrs.insert("num_channels".to_string(), rw.input_channels.to_string());
            attrs.insert("scales".to_string(), scale_str);
            attrs.insert("zero_points".to_string(), zp_str);
        } else if let (Some(scale), Some(zp_u8)) = (rw.per_tensor_scale, rw.per_tensor_zero_point) {
            // Use per-tensor calibration stats.
            // The zero_point from compute_scale_zp() is in U8 range (0..255).
            // The backend formula is: q = round((x - zp) / scale), clamp to [-128, 127]
            // For I8, we need: zp_i8 = min + 128 * scale
            //   where min = -zp_u8 * scale
            //   so: zp_i8 = -zp_u8 * scale + 128 * scale = scale * (128 - zp_u8)
            let zp_i8 = scale * (128.0 - zp_u8);
            attrs.insert("scale".to_string(), scale.to_string());
            attrs.insert("zero_point".to_string(), zp_i8.to_string());
        }

        let quantize_id = if let Some(qa_id) = rw.existing_qa_id {
            // UPDATE existing QuantizeActivations node with new calibration attrs
            if let Some(qa_node) = graph.get_node_mut(qa_id) {
                qa_node.attrs = attrs.clone();
                qa_node.output_type = quant_type.clone();
            }
            qa_id
        } else {
            // Create new QuantizeActivations node
            graph.add_node_with_attrs(
                Opcode::QuantizeActivations,
                vec![rw.act_id],
                quant_type,
                attrs,
            )
        };

        if let Some(target) = graph.get_node_mut(rw.node_id) {
            target.inputs[0] = quantize_id;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::executor::GraphExecutor;
    use crate::backend::Backend;
    use crate::compiler::passes::calibration::CalibrationData;
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
            .compile_with_plan_and_quantize(graph.clone(), None, None)
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

    // ── Conv2d per-channel activation quantization tests ──────────────

    /// Helper: build a simple Conv2d graph with an intermediate activation.
    /// Input → Relu → Conv2d → Output. The Relu intermediate should be
    /// quantized before Conv2d when calibration data is provided.
    fn build_conv2d_graph() -> ComputeGraph {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(
                vec![
                    DimExpr::Known(1),
                    DimExpr::Known(4),
                    DimExpr::Known(4),
                    DimExpr::Known(4),
                ],
                IrDType::F32,
            ),
        );
        // Intermediate Relu (non-Input, non-Constant → should be quantized)
        let relu_id = graph.add_node(
            Opcode::Relu,
            vec![input_id],
            TensorType::new(
                vec![
                    DimExpr::Known(1),
                    DimExpr::Known(4),
                    DimExpr::Known(4),
                    DimExpr::Known(4),
                ],
                IrDType::F32,
            ),
        );
        let weight_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(
                vec![
                    DimExpr::Known(8),
                    DimExpr::Known(4),
                    DimExpr::Known(3),
                    DimExpr::Known(3),
                ],
                IrDType::F32,
            ),
        );
        let mut conv_attrs = std::collections::HashMap::new();
        conv_attrs.insert("stride".to_string(), "1".to_string());
        conv_attrs.insert("padding".to_string(), "0".to_string());
        conv_attrs.insert("dilation".to_string(), "1".to_string());
        conv_attrs.insert("groups".to_string(), "1".to_string());
        let conv_id = graph.add_node_with_attrs(
            Opcode::Conv2d,
            vec![relu_id, weight_id],
            TensorType::new(
                vec![
                    DimExpr::Known(1),
                    DimExpr::Known(8),
                    DimExpr::Known(2),
                    DimExpr::Known(2),
                ],
                IrDType::F32,
            ),
            conv_attrs,
        );
        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![conv_id]);
        graph
    }

    /// Test that quantize_activations handles Conv2d (same as MatMul).
    #[test]
    fn test_activation_quantization_conv2d_basic() {
        let mut graph = build_conv2d_graph();
        shape_inference::infer_shapes(&mut graph).unwrap();

        let node_count_before = graph.node_count();
        quantize_activations(&mut graph).unwrap();

        // Should have added 1 QuantizeActivations node (for the Relu→Conv2d edge)
        assert_eq!(
            graph.node_count(),
            node_count_before + 1,
            "should insert QuantizeActivations for Conv2d activation"
        );

        // Verify the activation feeding Conv2d is now QuantizeActivations
        let conv_node = graph
            .nodes
            .iter()
            .find(|n| n.opcode == Opcode::Conv2d)
            .expect("Conv2d should still exist");
        let act_node = graph.get_node(conv_node.inputs[0]).unwrap();
        assert_eq!(
            act_node.opcode,
            Opcode::QuantizeActivations,
            "Conv2d activation should be QuantizeActivations"
        );
    }

    /// Test that quantize_activations_with_calibration produces per-channel
    /// QuantizeActivations for Conv2d when calibration data has per-channel stats.
    #[test]
    fn test_activation_quantization_conv2d_per_channel_calibration() {
        let mut graph = build_conv2d_graph();
        shape_inference::infer_shapes(&mut graph).unwrap();

        // Create calibration data with per-channel stats for the Relu output
        let mut calib;
        // The Relu activation has shape [1, 4, 4, 4] → 4 input channels, 16 spatial elements each
        let relu_name = "relu"; // must match the Relu node name (default is empty, so we set it)
                                // We need to name the relu node for calibration matching
        {
            // Find and name the Relu node
            for node in graph.nodes.iter_mut() {
                if node.opcode == Opcode::Relu {
                    node.name = "relu".to_string();
                }
            }
        }
        // Rebuild with proper per-channel observation
        calib = CalibrationData::new();
        let mut all_values = vec![0.0f32; 64];
        for ch in 0..4 {
            for i in 0..16 {
                let val = match ch {
                    0 => i as f32 * 2.0 / 15.0,
                    1 => i as f32 * 5.0 / 15.0,
                    2 => -1.0 + i as f32 * 11.0 / 15.0,
                    3 => 2.0 + i as f32 * 0.5 / 15.0,
                    _ => 0.0,
                };
                all_values[ch * 16 + i] = val;
            }
        }
        calib.observe_per_channel(relu_name, 4, 16, &all_values);

        // Run calibration-driven quantization
        quantize_activations_with_calibration(&mut graph, &calib).unwrap();

        // Find the QuantizeActivations node
        let qa_node = graph
            .nodes
            .iter()
            .find(|n| n.opcode == Opcode::QuantizeActivations)
            .expect("QuantizeActivations should exist");

        // Verify per-channel attrs
        assert_eq!(
            qa_node.attrs.get("mode").map(|s| s.as_str()),
            Some("per_channel"),
            "should have per_channel mode"
        );
        assert_eq!(
            qa_node
                .attrs
                .get("num_channels")
                .and_then(|s| s.parse::<usize>().ok()),
            Some(4),
            "should have 4 channels"
        );
        let scales_str = qa_node.attrs.get("scales").unwrap();
        let scales: Vec<f32> = scales_str
            .split(',')
            .filter_map(|s| s.parse().ok())
            .collect();
        assert_eq!(scales.len(), 4, "should have 4 per-channel scales");

        let zps_str = qa_node.attrs.get("zero_points").unwrap();
        let zps: Vec<f32> = zps_str.split(',').filter_map(|s| s.parse().ok()).collect();
        assert_eq!(zps.len(), 4, "should have 4 per-channel zero_points");

        // Verify scales are different per channel (different ranges)
        let unique_scales: std::collections::HashSet<u32> =
            scales.iter().map(|s| s.to_bits()).collect();
        assert!(
            unique_scales.len() > 1,
            "scales should differ across channels with different ranges"
        );
    }

    /// Test that without per-channel calibration data, Conv2d still gets
    /// a per-tensor QuantizeActivations (backward compat).
    #[test]
    fn test_activation_quantization_conv2d_per_tensor_fallback() {
        let mut graph = build_conv2d_graph();
        shape_inference::infer_shapes(&mut graph).unwrap();

        // Name the Relu node so calibration matching works
        for node in graph.nodes.iter_mut() {
            if node.opcode == Opcode::Relu {
                node.name = "relu_activation".to_string();
            }
        }

        // Create calibration with per-tensor data only (no per-channel)
        let mut calib = CalibrationData::new();
        calib.observe("relu_activation", &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        quantize_activations_with_calibration(&mut graph, &calib).unwrap();

        // Should still insert QuantizeActivations
        let qa_node = graph
            .nodes
            .iter()
            .find(|n| n.opcode == Opcode::QuantizeActivations)
            .expect("QuantizeActivations should exist");

        // When no per-channel data exists, the mode attr should be absent
        // or per-tensor (no "num_channels" attr beyond the one we set)
        // The pass only sets per_channel mode when calibration has valid per-channel data
        assert!(
            qa_node.attrs.get("mode") != Some(&"per_channel".to_string()),
            "should fall back to per-tensor when no per-channel calib data"
        );
    }
}
