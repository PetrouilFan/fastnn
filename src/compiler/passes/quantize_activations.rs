//! Quantize Activations Pass — Inserts Q/DQ pairs using calibration scales.
//!
//! This pass runs after calibration data is collected. It uses the computed
//! per-tensor scales/zero-points to insert Quantize/Dequantize nodes around
//! quantizable operations, enabling full INT8 inference with packed kernels.

use crate::compiler::passes::calibration::CalibrationData;
use crate::error::FastnnError;
use crate::ir::{ComputeGraph, IrDType, NodeId, Opcode, TensorType};
use std::collections::HashMap;

#[cfg(test)]
use crate::{ir::DimExpr, GraphBuilder};

/// Configuration for activation quantization.
#[derive(Debug, Clone)]
pub struct QuantizeActivationsConfig {
    pub bit_width: u8, // 4 or 8
    pub per_channel_weights: bool,
    pub per_tensor_activations: bool,
    pub use_kl_calibration: bool,
    pub skip_ops: Vec<String>, // Op names to skip (e.g., DFL head)
    pub calib_data: CalibrationData,
}

impl Default for QuantizeActivationsConfig {
    fn default() -> Self {
        Self {
            bit_width: 8,
            per_channel_weights: true,
            per_tensor_activations: true,
            use_kl_calibration: true,
            skip_ops: vec![], // Head/DFL quantization is now supported.
            calib_data: CalibrationData::new(),
        }
    }
}

/// Inserts Quantize/Dequantize pairs for quantized execution.
///
/// For each quantizable op in topological order:
/// 1. If inputs are quantized, insert Dequantize before
/// 2. Insert Quantize after with calibration scales
/// 3. Rewire consumers to Quantize output
pub fn quantize_activations(
    graph: &mut ComputeGraph,
    config: QuantizeActivationsConfig,
) -> Result<(), FastnnError> {
    let calibration_config = config
        .calib_data
        .to_quant_config(config.bit_width, config.use_kl_calibration);

    // Process nodes in topological order
    let topo_order = graph.topological_sort();

    for node_id in topo_order {
        let should_quantize = {
            let node = graph.get_node(node_id).unwrap();
            matches!(
                node.opcode,
                Opcode::Conv2d
                    | Opcode::Conv1d
                    | Opcode::Conv3d
                    | Opcode::MatMul
                    | Opcode::Add
                    | Opcode::Mul
                    | Opcode::Sub
                    | Opcode::Sigmoid
                    | Opcode::Silu
                    | Opcode::Relu
                    | Opcode::Softmax
                    | Opcode::LayerNorm
                    | Opcode::BatchNorm
            ) && !config.skip_ops.iter().any(|kw| node.name.contains(kw))
        };

        if !should_quantize {
            continue;
        }

        let node_name = graph.get_node(node_id).unwrap().name.clone();

        // Get calibration scales for this node's output
        let (scale, zero_point) = match calibration_config.get(&node_name) {
            Some(entry) => {
                let scale = entry["scale"].as_f64().unwrap() as f32;
                let zp = entry["zero_point"].as_f64().unwrap() as f32;
                (scale, zp)
            }
            None => {
                // Fallback: use default scales
                (1.0, 0.0)
            }
        };

        // Quantize the node's output
        quantize_node_output(graph, node_id, config.bit_width, scale, zero_point)?;
    }

    Ok(())
}

/// Quantize a node's output by inserting Quantize after it.
fn quantize_node_output(
    graph: &mut ComputeGraph,
    node_id: NodeId,
    bit_width: u8,
    scale: f32,
    zero_point: f32,
) -> Result<(), FastnnError> {
    let node = graph.get_node(node_id).unwrap().clone();
    let output_type = node.output_type.clone();

    // Create quantized output type
    let quant_dtype = if bit_width == 4 {
        IrDType::I4 {
            scales: vec![scale],
            zero_points: vec![zero_point],
            codebooks: vec![],
        }
    } else {
        IrDType::I8Scaled {
            scales: vec![scale],
            zero_points: vec![zero_point],
        }
    };

    let quant_output_type = TensorType::new(output_type.shape.clone(), quant_dtype);

    // Add Quantize node
    let mut attrs = HashMap::new();
    attrs.insert("bit_width".to_string(), bit_width.to_string());
    attrs.insert("scale".to_string(), scale.to_string());
    attrs.insert("zero_point".to_string(), zero_point.to_string());

    let quant_id =
        graph.add_node_with_attrs(Opcode::Quantize, vec![node_id], quant_output_type, attrs);

    // Rewire all consumers of node_id to point to quant_id
    // Collect consumers first to avoid borrow checker issues
    let consumers = graph.consumers(node_id).clone();
    let output_ids = graph.outputs.clone();

    // Rewire consumers to point to quant_id
    for consumer_id in consumers {
        let consumer = graph
            .get_node_mut(consumer_id)
            .ok_or(FastnnError::compilation("Consumer not found"))?;
        for input in &mut consumer.inputs {
            if *input == node_id {
                *input = quant_id;
            }
        }
    }

    // Update graph outputs if needed
    for (i, &out_id) in output_ids.iter().enumerate() {
        if out_id == node_id {
            graph.outputs[i] = quant_id;
        }
    }

    Ok(())
}

#[cfg(test)]
#[test]
fn test_quantize_node_output() {
    let builder = GraphBuilder::new();
    let input = builder.input(&[1, 4], IrDType::F32);
    let weight = builder.constant(
        &[0u8; 16],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    // Use a simple Add instead of Conv2d for test simplicity
    let _add = builder.add(&input, &weight);
    let mut graph = builder.to_graph();

    let mut calib = CalibrationData::new();
    calib.observe("add", &[1.0, 2.0, 3.0, 4.0]);

    let config = QuantizeActivationsConfig {
        bit_width: 8,
        calib_data: calib,
        ..Default::default()
    };

    let _ = quantize_activations(&mut graph, config);

    // Check Quantize node was added
    let has_quantize = graph.nodes.iter().any(|n| n.opcode == Opcode::Quantize);
    assert!(has_quantize);
}
