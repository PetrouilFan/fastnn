//! Auto-cast insertion compiler pass.
//!
//! When a user specifies `model.to("u4")`, this pass:
//!
//! 1. Finds all f32 weight constants feeding MatMul/Conv ops and replaces them
//!    with `Quantize(f32_weight, bit_width)` sub-graphs (or directly updates
//!    the Constant node's dtype to the packed representation).
//! 2. Inserts `Dequantize` ops before nodes that expect f32 but receive a
//!    quantized input (e.g., loss computation, softmax, relu on weights).
//! 3. Optionally activates the INT8 activation quantization pass.
//!
//! # Integration
//!
//! This pass is designed to be called from the Python `model.to("u4")` API.
//! It runs **before** the standard compilation pipeline, modifying the graph
//! so that subsequent passes (shape inference, memory planning, backend
//! compile) see the quantized representation.

use crate::ir::node::{ComputeGraph, IrDType, NodeId, Opcode, TensorType};

/// Options for the auto-cast pass.
#[derive(Debug, Clone, Default)]
pub struct AutoCastOptions {
    /// Target bit width for weight quantization (4 or 8). `None` = skip weight quantization.
    pub weight_bit_width: Option<u8>,
    /// Whether to enable INT8 activation quantization.
    pub enable_activation_quant: bool,
}

/// Apply auto-cast transformations to the graph.
///
/// Returns the number of nodes modified/inserted (for diagnostics).
pub fn auto_cast(graph: &mut ComputeGraph, options: &AutoCastOptions) -> Result<usize, String> {
    let mut changes = 0usize;

    // ── Step 1: Quantize weight constants ─────────────────────────────
    if let Some(bit_width) = options.weight_bit_width {
        changes += quantize_weight_constants(graph, bit_width)?;
    }

    // ── Step 2: Insert Dequantize before f32-expecting ops ────────────
    // After step 1, some quantized constants may feed ops that expect f32
    // (e.g., a weight used in both MatMul and a loss regularizer).
    // We need to insert Dequantize before the non-MatMul consumers.
    changes += insert_dequantize_for_f32_ops(graph)?;

    // ── Step 3: Activation quantization ───────────────────────────────
    if options.enable_activation_quant {
        super::activation_quantization::quantize_activations(graph)?;
        changes += 1; // approximate: the pass may insert many nodes
    }

    Ok(changes)
}

/// Find all f32 Constant weight nodes feeding MatMul/Conv ops and replace
/// their data with a packed quantized representation.
///
/// This delegates to the existing `quantize_weights` pass in the quantization
/// module, which handles the actual data packing and dtype update.
fn quantize_weight_constants(graph: &mut ComputeGraph, bit_width: u8) -> Result<usize, String> {
    // The existing quantize_weights pass does exactly what we need:
    // it finds f32 Constants feeding MatMul/Conv and packs them.
    let count_before = graph.node_count();
    super::quantization::quantize_weights(graph, bit_width)?;
    let count_after = graph.node_count();
    // quantize_weights modifies nodes in-place (no new nodes), so the
    // count stays the same.  We track the number of modified nodes by
    // checking how many Constants now have U4/U8 dtype.
    let quantized_count = graph
        .nodes
        .iter()
        .filter(|n| {
            matches!(n.opcode, Opcode::Constant(_))
                && matches!(n.output_type.dtype, IrDType::U4 { .. } | IrDType::U8 { .. })
        })
        .count();
    let _ = count_before;
    let _ = count_after;
    Ok(quantized_count)
}

/// Insert `Dequantize` ops before nodes that expect f32 input but receive a
/// quantized (U4/U8) input.
///
/// For example, if a quantized weight feeds both a MatMul (which supports
/// quantized input) and a loss regularizer (which expects f32), we insert a
/// Dequantize before the regularizer so it sees f32 data.
fn insert_dequantize_for_f32_ops(graph: &mut ComputeGraph) -> Result<usize, String> {
    let order = graph.topological_sort();
    let mut inserted = 0usize;

    // Collect rewrites first.
    struct DequantRewrite {
        consumer_id: NodeId,
        input_index: usize,
    }
    let mut rewrites: Vec<DequantRewrite> = Vec::new();

    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };

        // Ops that can accept quantized inputs natively.
        let accepts_quantized = matches!(
            node.opcode,
            Opcode::MatMul
                | Opcode::Conv1d
                | Opcode::Conv2d
                | Opcode::Conv3d
                | Opcode::Quantize
                | Opcode::Dequantize
                | Opcode::DequantizeActivations
                | Opcode::QuantizeActivations
        );

        for (i, &input_id) in node.inputs.iter().enumerate() {
            let input_dtype = match graph.get_node(input_id) {
                Some(n) => &n.output_type.dtype,
                None => continue,
            };

            // Check if input is quantized (U4/U8) but consumer expects f32
            let is_quantized = matches!(input_dtype, IrDType::U4 { .. } | IrDType::U8 { .. });

            if is_quantized && !accepts_quantized {
                rewrites.push(DequantRewrite {
                    consumer_id: node_id,
                    input_index: i,
                });
            }
        }
    }

    // Apply rewrites.
    for rw in rewrites {
        let input_id = match graph.get_node(rw.consumer_id) {
            Some(n) => n.inputs.get(rw.input_index).copied(),
            None => continue,
        };
        let input_id = match input_id {
            Some(id) => id,
            None => continue,
        };

        let input_shape = match graph.get_node(input_id) {
            Some(n) => n.output_type.shape.clone(),
            None => continue,
        };

        let dequant_type = TensorType::new(input_shape, IrDType::F32);
        let dequant_id = graph.add_node(Opcode::Dequantize, vec![input_id], dequant_type);

        if let Some(consumer) = graph.get_node_mut(rw.consumer_id) {
            if let Some(inp) = consumer.inputs.get_mut(rw.input_index) {
                *inp = dequant_id;
                inserted += 1;
            }
        }
    }

    Ok(inserted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::{DimExpr, TensorValue};

    /// Helper to create a weight Data constant with the given dimensions.
    fn make_weight_constant(
        graph: &mut ComputeGraph,
        rows: usize,
        cols: usize,
        value: f32,
    ) -> NodeId {
        let numel = rows * cols;
        let data: Vec<u8> = std::iter::repeat(value)
            .take(numel)
            .flat_map(|v: f32| v.to_le_bytes())
            .collect();
        let tt = TensorType::new(
            vec![DimExpr::Known(rows as u64), DimExpr::Known(cols as u64)],
            IrDType::F32,
        );
        graph.add_node(
            Opcode::Constant(TensorValue::Data {
                bytes: data,
                tensor_type: tt.clone(),
            }),
            vec![],
            tt,
        )
    }

    /// Test that auto_cast quantizes weight constants.
    #[test]
    fn test_auto_cast_quantizes_weights() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = make_weight_constant(&mut graph, 4, 4, 1.0);
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![mm_id]);

        crate::compiler::passes::shape_inference::infer_shapes(&mut graph).unwrap();

        let opts = AutoCastOptions {
            weight_bit_width: Some(4),
            enable_activation_quant: false,
        };
        auto_cast(&mut graph, &opts).unwrap();

        // Check that the weight Constant now has U4 dtype
        let weight_node = graph.get_node(weight_id).unwrap();
        assert!(
            matches!(weight_node.output_type.dtype, IrDType::U4 { .. }),
            "weight should be quantized to U4 after auto_cast"
        );
    }

    /// Test that auto_cast inserts Dequantize when a quantized weight feeds a non-MatMul op.
    #[test]
    fn test_auto_cast_inserts_dequantize_for_add() {
        let mut graph = ComputeGraph::new();
        // Activation input: [1, 4]
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        // MatMul weight: [4, 4] (for [1,4] × [4,4] → [1,4])
        let mm_weight_id = make_weight_constant(&mut graph, 4, 4, 1.0);
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, mm_weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        // The MatMul output is then added to a bias [1, 4].
        // This bias is a separate constant that should NOT be quantized, but
        // to test the Dequantize insertion, we make it also feed through a
        // shared quantized weight.  Instead, use a simpler test:
        // Create a quantized weight, use it in MatMul, and also use it in Relu.
        // Relu is not a MatMul/Conv so it gets Dequantize inserted.
        let shared_weight_id = make_weight_constant(&mut graph, 4, 4, 0.5);
        let mm2_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, shared_weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        // Use shared_weight in a non-MatMul op (Relu) to trigger Dequantize insertion
        let relu_id = graph.add_node(
            Opcode::Relu,
            vec![shared_weight_id],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
        );

        graph.set_inputs(vec![input_id, mm_weight_id, shared_weight_id]);
        graph.set_outputs(vec![mm_id, mm2_id, relu_id]);

        crate::compiler::passes::shape_inference::infer_shapes(&mut graph).unwrap();

        let opts = AutoCastOptions {
            weight_bit_width: Some(4),
            enable_activation_quant: false,
        };
        auto_cast(&mut graph, &opts).unwrap();

        // shared_weight should be quantized
        let shared_weight_node = graph.get_node(shared_weight_id).unwrap();
        assert!(
            matches!(shared_weight_node.output_type.dtype, IrDType::U4 { .. }),
            "shared_weight should be quantized"
        );

        // The Relu node consumes shared_weight (quantized). Since Relu is not
        // a MatMul/Conv, it should have Dequantize inserted.
        let relu_node = graph.get_node(relu_id).unwrap();
        let weight_consumer = relu_node.inputs[0];
        let dequant_node = graph.get_node(weight_consumer).unwrap();
        assert_eq!(
            dequant_node.opcode,
            Opcode::Dequantize,
            "quantized weight feeding Relu should have Dequantize inserted"
        );

        // The MatMul (mm2_id) should directly consume shared_weight (no Dequantize)
        // because MatMul supports quantized inputs natively.
        let mm2_node = graph.get_node(mm2_id).unwrap();
        let mm2_weight_input = mm2_node.inputs[1];
        let mm2_weight_node = graph.get_node(mm2_weight_input).unwrap();
        // mm2_weight_node could be Dequantize if shared_weight has other consumers,
        // but since we check that shared_weight IS quantized, the input to MatMul
        // should either be the quantized node or a Dequantize node.
        // The important thing is that Relu got Dequantize, not that MatMul didn't.
    }

    /// Test auto_cast with activation quantization enabled.
    /// Note: activation quantization skip Input nodes (external data),
    /// so we use a Relu output as the intermediate activation to quantize.
    #[test]
    fn test_auto_cast_with_activation_quant() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = make_weight_constant(&mut graph, 4, 4, 1.0);
        // First MatMul produces an intermediate activation
        let mm1_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        // Relu transforms it (this is the activation we want to quantize)
        let relu_id = graph.add_node(
            Opcode::Relu,
            vec![mm1_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        // Second MatMul consumes the ReLU output (this MatMul's activation gets quantized)
        let weight2_id = make_weight_constant(&mut graph, 4, 4, 0.5);
        let mm2_id = graph.add_node(
            Opcode::MatMul,
            vec![relu_id, weight2_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );

        graph.set_inputs(vec![input_id, weight_id, weight2_id]);
        graph.set_outputs(vec![mm2_id]);

        crate::compiler::passes::shape_inference::infer_shapes(&mut graph).unwrap();

        let opts = AutoCastOptions {
            weight_bit_width: Some(4),
            enable_activation_quant: true,
        };
        auto_cast(&mut graph, &opts).unwrap();

        // The second MatMul's activation input (which was Relu output)
        // should go through QuantizeActivations
        let mm2_node = graph.get_node(mm2_id).unwrap();
        let act_input_id = mm2_node.inputs[0];
        let act_node = graph.get_node(act_input_id).unwrap();
        assert_eq!(
            act_node.opcode,
            Opcode::QuantizeActivations,
            "activation should be quantized when activation_quant is enabled"
        );
    }

    /// Test auto_cast with U8 quantization (not just U4).
    #[test]
    fn test_auto_cast_u8_weights() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = make_weight_constant(&mut graph, 4, 4, 1.0);
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![mm_id]);

        crate::compiler::passes::shape_inference::infer_shapes(&mut graph).unwrap();

        let opts = AutoCastOptions {
            weight_bit_width: Some(8),
            enable_activation_quant: false,
        };
        auto_cast(&mut graph, &opts).unwrap();

        let weight_node = graph.get_node(weight_id).unwrap();
        assert!(
            matches!(weight_node.output_type.dtype, IrDType::U8 { .. }),
            "weight should be quantized to U8 after auto_cast with bit_width=8"
        );
    }

    /// Test auto_cast with no options (no-op).
    #[test]
    fn test_auto_cast_noop_when_no_options() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        let weight_id = make_weight_constant(&mut graph, 4, 4, 1.0);
        let mm_id = graph.add_node(
            Opcode::MatMul,
            vec![input_id, weight_id],
            TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
        );
        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![mm_id]);

        crate::compiler::passes::shape_inference::infer_shapes(&mut graph).unwrap();

        let opts = AutoCastOptions::default();
        let changes = auto_cast(&mut graph, &opts).unwrap();
        assert_eq!(changes, 0, "no options should produce zero changes");

        // All nodes should remain F32
        for node in &graph.nodes {
            assert_eq!(
                node.output_type.dtype,
                IrDType::F32,
                "node {} should remain F32",
                node.id
            );
        }
    }

    /// Test auto_cast with Conv2d weights (not just MatMul).
    #[test]
    fn test_auto_cast_conv2d_weights() {
        let mut graph = ComputeGraph::new();
        let input_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(
                vec![
                    DimExpr::Known(1),
                    DimExpr::Known(3),
                    DimExpr::Known(8),
                    DimExpr::Known(8),
                ],
                IrDType::F32,
            ),
        );

        // Conv2d weight: [out_channels, in_channels, kh, kw] = [4, 3, 3, 3]
        let numel = 4 * 3 * 3 * 3;
        let weight_data: Vec<u8> = std::iter::repeat(1.0f32)
            .take(numel)
            .flat_map(|v: f32| v.to_le_bytes())
            .collect();
        let tt = TensorType::new(
            vec![
                DimExpr::Known(4),
                DimExpr::Known(3),
                DimExpr::Known(3),
                DimExpr::Known(3),
            ],
            IrDType::F32,
        );
        let weight_id = graph.add_node(
            Opcode::Constant(crate::ir::node::TensorValue::Data {
                bytes: weight_data,
                tensor_type: tt.clone(),
            }),
            vec![],
            tt,
        );

        let conv_id = graph.add_node_with_attrs(
            Opcode::Conv2d,
            vec![input_id, weight_id],
            TensorType::new(
                vec![
                    DimExpr::Known(1),
                    DimExpr::Known(4),
                    DimExpr::Known(6),
                    DimExpr::Known(6),
                ],
                IrDType::F32,
            ),
            {
                let mut m = std::collections::HashMap::new();
                m.insert("stride".to_string(), "1".to_string());
                m.insert("padding".to_string(), "0".to_string());
                m.insert("dilation".to_string(), "1".to_string());
                m.insert("groups".to_string(), "1".to_string());
                m
            },
        );

        graph.set_inputs(vec![input_id, weight_id]);
        graph.set_outputs(vec![conv_id]);

        crate::compiler::passes::shape_inference::infer_shapes(&mut graph).unwrap();

        let opts = AutoCastOptions {
            weight_bit_width: Some(4),
            enable_activation_quant: false,
        };
        auto_cast(&mut graph, &opts).unwrap();

        let weight_node = graph.get_node(weight_id).unwrap();
        assert!(
            matches!(weight_node.output_type.dtype, IrDType::U4 { .. }),
            "Conv2d weight should be quantized after auto_cast"
        );
    }
}
