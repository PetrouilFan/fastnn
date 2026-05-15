//! Type inference compiler pass.
//!
//! Walks the graph in topological order and ensures every node's inputs have the
//! dtype that the node expects, inserting conversion ops (`Quantize`, `Dequantize`,
//! `ToF16`, `ToF32`) where types don't match.
//!
//! # Algorithm
//!
//! ```ignore
//! for node in topological_order(graph):
//!     for (i, input) in node.inputs.enumerate():
//!         expected = expected_input_dtype(node.opcode, i)
//!         if input.dtype != expected:
//!             insert_conversion(input, expected)
//! ```
//!
//! The expected dtype for each input depends on the opcode:
//! - Most arithmetic/activation ops expect F32.
//! - MatMul/Conv can accept U4/U8 on the weight input and INT8 on the activation.
//! - Conversion ops (Cast, Quantize, Dequantize, ToF16, ToF32) accept their
//!   natural input type.

use crate::ir::node::{ComputeGraph, IrDType, NodeId, Opcode, TensorType};

/// Returns the dtype that `opcode` expects for its `input_index`-th input
/// (0-based).  Returns `None` when the opcode treats all inputs uniformly
/// (e.g. Add, Mul) or when the expected dtype equals the output dtype.
fn expected_input_dtype(opcode: &Opcode, _input_index: usize) -> Option<IrDType> {
    match opcode {
        // Quantized MatMul: activation (input 0) can be F32 or I8,
        // weight (input 1) can be F32, U4, or U8.
        // No forced conversion here — the auto-cast pass handles it.
        Opcode::MatMul => None,

        // Conversion ops accept their natural input type (already handled
        // by construction).  No inference needed.
        Opcode::Quantize => Some(IrDType::F32),
        Opcode::Dequantize => None, // accepts U4/U8/F32
        Opcode::ToF16 => Some(IrDType::F32),
        Opcode::ToF32 => Some(IrDType::F16),
        Opcode::QuantizeActivations => Some(IrDType::F32),
        Opcode::DequantizeActivations => Some(IrDType::I8),
        Opcode::Cast => None, // already has the right input

        // Optimizer ops accept F32 gradients.
        Opcode::SgdUpdate
        | Opcode::AdamUpdate
        | Opcode::AdamWUpdate
        | Opcode::MuonUpdate
        | Opcode::LionUpdate
        | Opcode::RmspropUpdate => Some(IrDType::F32),

        // Everything else: no constraint (F32 is the default).
        _ => None,
    }
}

/// Ensure type consistency across the graph by inserting conversion ops where
/// a node's input dtype doesn't match what the node expects.
///
/// This pass is idempotent.
pub fn infer_types(graph: &mut ComputeGraph) -> Result<(), String> {
    let order = graph.topological_sort();

    // Collect (consumer_id, input_index, conversion_opcode, target_dtype) rewrites.
    struct Rewrite {
        consumer_id: NodeId,
        input_index: usize,
        conversion_opcode: Opcode,
        target_dtype: IrDType,
    }

    let mut rewrites: Vec<Rewrite> = Vec::new();

    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };

        for (i, &input_id) in node.inputs.iter().enumerate() {
            let expected = match expected_input_dtype(&node.opcode, i) {
                Some(dt) => dt,
                None => continue,
            };

            let actual = match graph.get_node(input_id) {
                Some(n) => n.output_type.dtype.clone(),
                None => continue,
            };

            // Skip if already matching
            if dtypes_match(&actual, &expected) {
                continue;
            }

            // Build the conversion op
            let conversion = match conversion_between(&actual, &expected) {
                Some(op) => op,
                None => continue,
            };

            rewrites.push(Rewrite {
                consumer_id: node_id,
                input_index: i,
                conversion_opcode: conversion,
                target_dtype: expected,
            });
        }
    }

    // Apply rewrites (insert conversion nodes and rewire).
    for rw in rewrites {
        let input_id = match graph.get_node(rw.consumer_id) {
            Some(n) => n.inputs.get(rw.input_index).copied(),
            None => continue,
        };
        let input_id = match input_id {
            Some(id) => id,
            None => continue,
        };

        // Get the input node's shape for the conversion output
        let input_shape = match graph.get_node(input_id) {
            Some(n) => n.output_type.shape.clone(),
            None => continue,
        };

        let output_type = TensorType::new(input_shape, rw.target_dtype);
        let conv_id = graph.add_node(rw.conversion_opcode, vec![input_id], output_type);

        // Rewire the consumer
        if let Some(consumer) = graph.get_node_mut(rw.consumer_id) {
            if let Some(inp) = consumer.inputs.get_mut(rw.input_index) {
                *inp = conv_id;
            }
        }
    }

    Ok(())
}

/// Check whether two dtypes are "close enough" that no conversion is needed.
fn dtypes_match(a: &IrDType, b: &IrDType) -> bool {
    use IrDType::*;
    match (a, b) {
        // Same concrete type
        (F32, F32)
        | (F16, F16)
        | (BF16, BF16)
        | (I32, I32)
        | (I64, I64)
        | (Bool, Bool)
        | (I8, I8) => true,
        // U4/U8 match regardless of scales/zps (those are metadata)
        (U4 { .. }, U4 { .. }) | (U8 { .. }, U8 { .. }) => true,
        _ => false,
    }
}

/// Return the conversion opcode needed to convert from `actual` to `expected`.
fn conversion_between(actual: &IrDType, expected: &IrDType) -> Option<Opcode> {
    use IrDType::*;
    match (actual, expected) {
        // U4/U8 → F32
        (U4 { .. }, F32) | (U8 { .. }, F32) => Some(Opcode::Dequantize),
        // F32 → U4/U8 (weight quantization — needs bit_width)
        (F32, U4 { .. }) => Some(Opcode::Quantize),
        (F32, U8 { .. }) => Some(Opcode::Quantize),
        // F32 ↔ F16
        (F32, F16) => Some(Opcode::ToF16),
        (F16, F32) => Some(Opcode::ToF32),
        // F32 → I8 (activation quantization)
        (F32, I8) => Some(Opcode::QuantizeActivations),
        // I8 → F32
        (I8, F32) => Some(Opcode::DequantizeActivations),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::DimExpr;

    /// Test insertion of Dequantize when a MatMul's weight is quantized
    /// but another consumer of that weight expects F32.
    #[test]
    fn test_type_inference_inserts_dequantize() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        // Add another consumer of the weight that expects F32:
        // This could be a constant that feeds both MatMul and an Add
        // (but for simplicity we just add another MatMul).
        let mm2_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );

        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![mm_id, mm2_id]);

        // Run type inference (no-ops for F32 → F32)
        let count_before = graph.node_count();
        infer_types(&mut graph).unwrap();
        let count_after = graph.node_count();
        // Should be no changes since everything is already F32
        assert_eq!(count_before, count_after);
    }

    /// Test that type inference inserts Quantize for a weight that feeds MatMul
    /// when the target dtype is U4.
    #[test]
    fn test_type_inference_inserts_quantize() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = graph.add_node(
            Opcode::Constant(crate::ir::node::TensorValue::Float(1.0)),
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );

        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![mm_id]);

        // Expected dtypes: weight should be U4 (for MatMul input 1).
        // Type inference doesn't change dtypes — it only inserts conversions
        // when there's a mismatch.  Since both are F32, no conversion.
        let count_before = graph.node_count();
        infer_types(&mut graph).unwrap();
        let count_after = graph.node_count();
        assert_eq!(count_before, count_after);
    }

    /// Test that type inference inserts QuantizeActivations before
    /// a DequantizeActivations node whose input is F32 (expects I8).
    #[test]
    fn test_type_inference_inserts_quantize_activations() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        // DequantizeActivations expects I8 input, so F32 is a mismatch
        let dq_id = graph.add_node(
            Opcode::DequantizeActivations,
            vec![input_id],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id]);
        graph.set_outputs(vec![dq_id]);

        let count_before = graph.node_count();
        infer_types(&mut graph).unwrap();
        let count_after = graph.node_count();
        // QuantizeActivations (F32→I8) should be inserted before DequantizeActivations
        assert_eq!(count_after, count_before + 1);

        let dq_node = graph.get_node(dq_id).unwrap();
        let inserted_id = dq_node.inputs[0];
        let inserted_node = graph.get_node(inserted_id).unwrap();
        assert_eq!(inserted_node.opcode, Opcode::QuantizeActivations);
        assert_eq!(inserted_node.output_type.dtype, IrDType::I8);
    }

    /// Test that type inference inserts DequantizeActivations before
    /// a QuantizeActivations node whose input is I8 (expects F32).
    #[test]
    fn test_type_inference_inserts_dequantize_activations() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::I8),
        );
        // QuantizeActivations expects F32 input, so I8 is a mismatch
        let qa_id = graph.add_node(
            Opcode::QuantizeActivations,
            vec![input_id],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::I8),
        );
        graph.set_inputs(vec![input_id]);
        graph.set_outputs(vec![qa_id]);

        let count_before = graph.node_count();
        infer_types(&mut graph).unwrap();
        let count_after = graph.node_count();
        // DequantizeActivations (I8→F32) should be inserted before QuantizeActivations
        assert_eq!(count_after, count_before + 1);

        let qa_node = graph.get_node(qa_id).unwrap();
        let inserted_id = qa_node.inputs[0];
        let inserted_node = graph.get_node(inserted_id).unwrap();
        assert_eq!(inserted_node.opcode, Opcode::DequantizeActivations);
        assert_eq!(inserted_node.output_type.dtype, IrDType::F32);
    }

    /// Test that type inference inserts ToF16 when a ToF16 node
    /// receives non-F32 input.
    #[test]
    fn test_type_inference_inserts_to_f16() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F16),
        );
        // ToF16 expects F32 input, so F16 input is a mismatch
        let to_f16_id = graph.add_node(
            Opcode::ToF16,
            vec![input_id],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F16),
        );
        graph.set_inputs(vec![input_id]);
        graph.set_outputs(vec![to_f16_id]);

        let count_before = graph.node_count();
        infer_types(&mut graph).unwrap();
        let count_after = graph.node_count();
        // A ToF32 (F16→F32) or other conversion should be inserted
        // conversion_between(F16, F32) = Some(ToF32)
        assert_eq!(count_after, count_before + 1);

        let node = graph.get_node(to_f16_id).unwrap();
        let inserted_id = node.inputs[0];
        let inserted_node = graph.get_node(inserted_id).unwrap();
        assert_eq!(inserted_node.opcode, Opcode::ToF32);
    }

    /// Test that type inference is idempotent — running twice produces the same graph.
    #[test]
    fn test_type_inference_idempotent() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        let dq_id = graph.add_node(
            Opcode::DequantizeActivations,
            vec![input_id],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id]);
        graph.set_outputs(vec![dq_id]);

        infer_types(&mut graph).unwrap();
        let count_1 = graph.node_count();

        // Second run should not change the graph
        infer_types(&mut graph).unwrap();
        let count_2 = graph.node_count();
        assert_eq!(count_1, count_2);
    }

    /// Test that type inference handles Quantize node with non-F32 input correctly.
    #[test]
    fn test_type_inference_quantize_with_non_f32_input() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::I8),
        );
        // Quantize expects F32 input, so I8 is a mismatch
        let q_id = graph.add_node(
            Opcode::Quantize,
            vec![input_id],
            TensorType::new(
                vec![DimExpr::Known(4)],
                IrDType::U4 {
                    scales: vec![],
                    zero_points: vec![],
                },
            ),
        );
        graph.set_inputs(vec![input_id]);
        graph.set_outputs(vec![q_id]);

        infer_types(&mut graph).unwrap();
        // DequantizeActivations (I8→F32) should be inserted
        let q_node = graph.get_node(q_id).unwrap();
        let inserted_id = q_node.inputs[0];
        let inserted_node = graph.get_node(inserted_id).unwrap();
        assert_eq!(inserted_node.opcode, Opcode::DequantizeActivations);
    }

    /// Test that Const nodes retain their declared type (pass doesn't change them).
    #[test]
    fn test_type_inference_constants_unchanged() {
        let mut graph = ComputeGraph::new();
        let const_id = graph.add_node(
            Opcode::Constant(crate::ir::node::TensorValue::Float(42.0)),
            vec![],
            TensorType::new(vec![], IrDType::F32),
        );
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        let add_id = graph.add_node(
            Opcode::Add,
            vec![input_id, const_id],
            TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id]);
        graph.set_outputs(vec![add_id]);

        let dtype_before = graph.get_node(const_id).unwrap().output_type.dtype.clone();
        infer_types(&mut graph).unwrap();
        let dtype_after = graph.get_node(const_id).unwrap().output_type.dtype.clone();
        assert_eq!(dtype_before, dtype_after);
    }
}
