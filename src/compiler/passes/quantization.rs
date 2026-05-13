//! Quantization compiler pass.
//!
//! Finds f32 Constant weight nodes feeding MatMul/Conv2d ops and replaces
//! them with packed U4/U8 data carrying per-channel scale/zero-point metadata.
//! The backend dispatch already selects `matmul_u4`/`matmul_u8` kernels when
//! it sees `IrDType::U4/U8` on the weight input.

use crate::dtypes::{U4x8, U8x4};
use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorType, TensorValue};
use crate::packed_tensor::PackedTensor;

/// Quantize all f32 weight constants feeding MatMul / Conv2d nodes to the
/// requested bit-width (4 or 8).
///
/// The pass is idempotent — nodes that are already quantized are skipped.
///
/// # Per-channel quantization
///
/// For a weight of shape `[out_channels, in_features]`, we compute one
/// (scale, zero_point) pair per output channel (row).  The scales and
/// zero_points are stored directly on the `IrDType::U4/U8` variant so
/// the CPU backend can feed them into `PackedTensor::from_raw(…)`.
pub fn quantize_weights(graph: &mut ComputeGraph, bit_width: u8) -> Result<(), String> {
    // ---- Phase 1: collect (constant_id, consumer_id) pairs to quantize ----
    let order = graph.topological_sort();

    // We collect first, then mutate, to avoid borrow-checker issues.
    let mut to_quantize: Vec<(NodeId, NodeId)> = Vec::new();

    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n.clone(),
            None => continue,
        };

        // Only quantize weights for MatMul-family and Conv-family ops.
        let is_consumer = matches!(
            node.opcode,
            Opcode::MatMul | Opcode::Conv1d | Opcode::Conv2d | Opcode::Conv3d
        );
        if !is_consumer {
            continue;
        }

        // The weight is typically input[1] (input[0] is the activation).
        if let Some(&weight_id) = node.inputs.get(1) {
            let weight_node = match graph.get_node(weight_id) {
                Some(n) => n.clone(),
                None => continue,
            };

            // Only quantize f32/bf16/f16 constants (skip already-quantized).
            if let Opcode::Constant(ref val) = weight_node.opcode {
                let is_float = matches!(
                    &weight_node.output_type.dtype,
                    IrDType::F32 | IrDType::F16 | IrDType::BF16
                );
                // Skip non-Data constants (scalar floats) and already-quantized.
                let is_data = matches!(val, TensorValue::Data { .. });
                if is_float && is_data {
                    to_quantize.push((weight_id, node_id));
                }
            }
        }
    }

    if to_quantize.is_empty() {
        return Ok(());
    }

    // ---- Phase 2: quantize each weight constant in-place ----
    for (const_id, _consumer_id) in to_quantize {
        // Clone needed data before mutable borrow.
        let const_node = match graph.get_node(const_id) {
            Some(n) => n.clone(),
            None => continue,
        };

        let (f32_data, orig_shape, orig_numel) = match &const_node.opcode {
            Opcode::Constant(TensorValue::Data { bytes, tensor_type }) => {
                let numel = match tensor_type.numel() {
                    Some(n) => n as usize,
                    None => continue, // skip symbolic shapes
                };
                // Reinterpret bytes as f32.
                let f32_data: Vec<f32> = if bytes.len() == numel * 4 {
                    bytemuck::cast_slice(bytes).to_vec()
                } else {
                    // Byte length mismatch — skip.
                    continue;
                };
                let shape: Vec<usize> = tensor_type
                    .shape
                    .iter()
                    .filter_map(|d| match d {
                        DimExpr::Known(v) => Some(*v as usize),
                        _ => None,
                    })
                    .collect();
                (f32_data, shape, numel)
            }
            _ => continue,
        };

        if f32_data.is_empty() {
            continue;
        }

        // Determine inner dimension for the weight tensor.
        let _out_channels = orig_shape.get(0).copied().unwrap_or(1);
        let inner_dim = if orig_shape.len() >= 2 {
            orig_shape[1..].iter().product::<usize>()
        } else {
            f32_data.len()
        };

        if inner_dim == 0 {
            continue;
        }

        // Quantize using PackedTensor and extract raw bytes + metadata.
        let (packed_bytes, _scales, _zero_points, new_dtype) = if bit_width == 4 {
            let pt = PackedTensor::<U4x8>::from_f32_per_channel(&f32_data, &orig_shape);
            let bytes = pt.as_bytes().to_vec();
            (bytes, pt.scales.clone(), pt.zeros.clone(), IrDType::U4 { scales: pt.scales.clone(), zero_points: pt.zeros.clone() })
        } else {
            let pt = PackedTensor::<U8x4>::from_f32_per_channel(&f32_data, &orig_shape);
            let bytes = pt.as_bytes().to_vec();
            (bytes, pt.scales.clone(), pt.zeros.clone(), IrDType::U8 { scales: pt.scales.clone(), zero_points: pt.zeros.clone() })
        };

        // Build the new TensorType with packed dtype.
        let new_tensor_type = TensorType {
            shape: const_node.output_type.shape.clone(),
            dtype: new_dtype,
        };

        let new_value = TensorValue::Data {
            bytes: packed_bytes,
            tensor_type: new_tensor_type.clone(),
        };

        // Mutate the node in place.
        if let Some(node_mut) = graph.get_node_mut(const_id) {
            node_mut.opcode = Opcode::Constant(new_value);
            node_mut.output_type = new_tensor_type;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorValue};
    use crate::ir::builder::GraphBuilder;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::executor::GraphExecutor;

    /// Helper: create a ComputeGraph with a MatMul and f32 Constant weight.
    fn build_matmul_graph(weight_data: &[f32], weight_shape: &[usize]) -> ComputeGraph {
        let gb = GraphBuilder::new();
        let m = weight_shape[0];
        let k = weight_shape[1];
        let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(k as u64)], IrDType::F32);
        let weight_tt = TensorType::new(
            weight_shape.iter().map(|&d| DimExpr::Known(d as u64)).collect(),
            IrDType::F32,
        );
        let weight_bytes: Vec<u8> = bytemuck::cast_slice(weight_data).to_vec();
        let weight = gb.constant(&weight_bytes, weight_tt);
        let _output = gb.matmul(&input, &weight);
        gb.to_graph()
    }

    #[test]
    fn test_quantize_f32_weight_to_u4() {
        let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                          -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let mut graph = build_matmul_graph(&weight_data, &[2, 8]);

        // Find the f32 constant node before quantization.
        let has_f32_const = graph.nodes.iter().any(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if matches!(tensor_type.dtype, IrDType::F32))
        });
        assert!(has_f32_const, "Should have an f32 constant node before quantization");

        // Quantize to U4.
        quantize_weights(&mut graph, 4).expect("quantization should succeed");

        // Verify the weight node dtype is now U4.
        let has_u4_const = graph.nodes.iter().any(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if matches!(tensor_type.dtype, IrDType::U4 { .. }))
        });
        assert!(has_u4_const, "Weight should be quantized to U4 after quantization");

        // Verify per-channel scales.
        let u4_node = graph.nodes.iter().find(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if matches!(tensor_type.dtype, IrDType::U4 { .. }))
        }).unwrap();

        if let IrDType::U4 { scales, zero_points } = &u4_node.output_type.dtype {
            assert_eq!(scales.len(), 2, "Should have 2 per-channel scales for a [2,8] weight");
            assert_eq!(zero_points.len(), 2, "Should have 2 per-channel zero_points");
            for &s in scales.iter() {
                assert!(s > 0.0, "Scale should be positive, got {}", s);
            }
        } else {
            panic!("Expected U4 dtype after quantization");
        }
    }

    #[test]
    fn test_quantize_f32_weight_to_u8() {
        let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                          -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let mut graph = build_matmul_graph(&weight_data, &[2, 8]);

        quantize_weights(&mut graph, 8).expect("quantization should succeed");

        let has_u8_const = graph.nodes.iter().any(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if matches!(tensor_type.dtype, IrDType::U8 { .. }))
        });
        assert!(has_u8_const, "Weight should be quantized to U8");

        let u8_node = graph.nodes.iter().find(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if matches!(tensor_type.dtype, IrDType::U8 { .. }))
        }).unwrap();

        if let IrDType::U8 { scales, zero_points } = &u8_node.output_type.dtype {
            assert_eq!(scales.len(), 2, "Should have 2 per-channel scales for a [2,8] weight");
            assert_eq!(zero_points.len(), 2);
        } else {
            panic!("Expected U8 dtype after quantization");
        }
    }

    #[test]
    fn test_quantize_skips_already_quantized() {
        // Build a graph with a U4 constant — it should be skipped.
        let gb = GraphBuilder::new();
        let input = gb.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
        let weight_data: Vec<u8> = vec![0u8; 4]; // dummy packed data
        let weight_tt = TensorType::new(
            vec![DimExpr::Known(2), DimExpr::Known(4)],
            IrDType::U4 { scales: vec![1.0], zero_points: vec![0.0] },
        );
        let weight = gb.constant(&weight_data, weight_tt);
        let _output = gb.matmul(&input, &weight);
        let mut graph = gb.to_graph();

        // Quantize should be a no-op (already U4).
        quantize_weights(&mut graph, 4).expect("quantization should succeed");

        // The weight should still be U4 (unchanged).
        let node = graph.nodes.iter().find(|n| matches!(&n.opcode, Opcode::Constant(TensorValue::Data { .. }))).unwrap();
        assert!(matches!(node.output_type.dtype, IrDType::U4 { .. }), "Should still be U4");
    }

    #[test]
    fn test_quantize_no_weights_is_noop() {
        let gb = GraphBuilder::new();
        let input = gb.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
        let _output = gb.relu(&input);
        let mut graph = gb.to_graph();

        quantize_weights(&mut graph, 4).expect("quantization should succeed on empty graph");
    }

    #[test]
    fn test_quantize_end_to_end_with_compile() {
        // Build a matmul graph, quantize it, and verify the compiled plan
        // includes a matmul_u4 kernel (not matmul).
        let weight_data: Vec<f32> = (0..32).map(|i| i as f32).collect(); // [8, 4] = 32 elements
        let gb = GraphBuilder::new();
        let input = gb.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(8)], IrDType::F32);
        let weight_tt = TensorType::new(
            vec![DimExpr::Known(8), DimExpr::Known(4)],
            IrDType::F32,
        );
        let weight_bytes: Vec<u8> = bytemuck::cast_slice(&weight_data).to_vec();
        let weight = gb.constant(&weight_bytes, weight_tt);
        let _output = gb.matmul(&input, &weight);

        let executor = GraphExecutor::new(CpuBackend);
        let (plan, _, compiled_graph) = executor
            .compile_with_plan_and_quantize(&gb.to_graph(), Some(4))
            .expect("compile with quantization should succeed");

        // Verify that the compiled plan contains a matmul_u4 kernel.
        let has_u4_kernel = plan.instructions.iter().any(|i| {
            match i {
                crate::backend::Instruction::CallKernel { kernel_name, .. } => kernel_name == "matmul_u4",
                _ => false,
            }
        });
        assert!(has_u4_kernel, "Compiled plan should contain a matmul_u4 kernel after quantization");

        // Verify that the weight node in the compiled graph has U4 dtype.
        let has_u4_weight = compiled_graph.nodes.iter().any(|n| {
            matches!(&n.output_type.dtype, IrDType::U4 { .. })
        });
        assert!(has_u4_weight, "Compiled graph should contain a U4 weight node");
    }
}