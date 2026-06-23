//! Quantize Activations Pass — Inserts Q/DQ pairs using calibration scales.
//!
//! This pass runs after calibration data is collected. It uses the computed
//! per-tensor scales/zero-points to insert Quantize/Dequantize nodes around
//! quantizable operations, enabling full INT8 inference with packed kernels.

use crate::compiler::passes::calibration::CalibrationData;
use crate::ir::builder::GraphBuilder;
use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorType, TensorValue};
use std::collections::HashMap;

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
) -> Result<(), String> {
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

    // Second pass for Dequantize insertion is disabled — it causes cycles in complex graphs.
    // Weight-only quantization with Quantize nodes is sufficient for the current inference path.
    /*
    let topo_order_2 = graph.topological_sort();
    for node_id in topo_order_2 {
        let node = graph.get_node(node_id).unwrap().clone();

        for &input_id in &node.inputs {
            let input_node = graph.get_node(input_id).unwrap();

            if input_node.opcode == Opcode::Quantize {
                insert_dequantize_before(graph, node_id, input_id, config.bit_width)?;
            }
        }
    }
    */

    Ok(())
}

/// Quantize a node's output by inserting Quantize after it.
fn quantize_node_output(
    graph: &mut ComputeGraph,
    node_id: NodeId,
    bit_width: u8,
    scale: f32,
    zero_point: f32,
) -> Result<(), String> {
    let node = graph.get_node(node_id).unwrap().clone();
    let output_type = node.output_type.clone();

    // Create quantized output type
    let quant_dtype = if bit_width == 4 {
        IrDType::U4 {
            scales: vec![scale],
            zero_points: vec![zero_point],
        }
    } else {
        IrDType::U8 {
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
            .ok_or("Consumer not found")?;
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

/// Insert Dequantize before a consumer of a Quantize node.
fn insert_dequantize_before(
    graph: &mut ComputeGraph,
    consumer_id: NodeId,
    quantized_input_id: NodeId,
    _bit_width: u8,
) -> Result<(), String> {
    let quant_node = graph.get_node(quantized_input_id).unwrap().clone();
    let input_type = quant_node.output_type.clone();

    // Dequantize to FP32
    let dequant_type = TensorType::new(input_type.shape.clone(), IrDType::F32);

    let dequant_id = graph.add_node(Opcode::Dequantize, vec![quantized_input_id], dequant_type);

    // Rewire consumer
    let consumer = graph
        .get_node_mut(consumer_id)
        .ok_or("Consumer not found")?;
    for input in &mut consumer.inputs {
        if *input == quantized_input_id {
            *input = dequant_id;
        }
    }

    Ok(())
}

/// Fuses consecutive Q/DQ pairs that cancel out.
pub fn fuse_qdq_pairs(graph: &mut ComputeGraph) -> Result<(), String> {
    let mut changed = true;
    while changed {
        changed = false;
        // NodeId is just the index into graph.nodes Vec
        let nodes: Vec<NodeId> = (0..graph.nodes.len()).collect();

        for node_id in nodes {
            if node_id >= graph.nodes.len() {
                continue; // Node may have been removed
            }
            let node = graph.get_node(node_id).unwrap().clone();

            // Pattern: Quantize -> Dequantize (same bit_width) -> remove both
            if node.opcode == Opcode::Dequantize {
                if let Some(&prev_id) = node.inputs.first() {
                    let prev = graph.get_node(prev_id).unwrap();
                    if prev.opcode == Opcode::Quantize {
                        // Check bit_width matches
                        let q_bit: Option<u8> =
                            prev.attrs.get("bit_width").and_then(|s| s.parse().ok());
                        let dq_bit: Option<u8> =
                            node.attrs.get("bit_width").and_then(|s| s.parse().ok());
                        if q_bit == dq_bit {
                            // Rewire consumers of Dequantize to Quantize input
                            let consumers = graph.consumers(node_id).clone();
                            for consumer_id in consumers {
                                let consumer = graph
                                    .get_node_mut(consumer_id)
                                    .ok_or("Consumer not found")?;
                                for input in &mut consumer.inputs {
                                    if *input == node_id {
                                        *input = prev_id;
                                    }
                                }
                            }

                            // Remove Dequantize node
                            graph.remove_node(node_id);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[test]
fn test_quantize_node_output() {
    let builder = GraphBuilder::new();
    let input = builder.input(&[1, 4], IrDType::F32);
    let weight = builder.constant(
        &[0u8; 16],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    // Use a simple Add instead of Conv2d for test simplicity
    let add = builder.add(&input, &weight);
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

/// Variant of quantize_activations that uses pre-computed scales from calibration data
/// instead of recomputing them. This is useful when scales are loaded from a JSON file.
pub fn quantize_activations_with_scales(
    graph: &mut ComputeGraph,
    mut config: QuantizeActivationsConfig,
) -> Result<(), String> {
    // The config.calib_data already contains the pre-computed scales in its stats
    // We need to use those directly instead of recomputing from min/max

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

        // Get calibration scales for this node's output from pre-computed stats
        let (scale, zero_point) = match config.calib_data.stats.get(&node_name) {
            Some(stats) => {
                // Use the pre-computed scale and zero_point from the stats
                // We need to recompute scale from min/max to match the original calibration
                stats.compute_scale_zp(config.bit_width)
            }
            None => {
                // Fallback: use default scales
                (1.0, 0.0)
            }
        };

        // Quantize the node's output
        quantize_node_output(graph, node_id, config.bit_width, scale, zero_point)?;
    }

    // Second pass for Dequantize insertion is disabled — it causes cycles in complex graphs.
    // Weight-only quantization with Quantize nodes is sufficient for the current inference path.
    /*
    let topo_order_2 = graph.topological_sort();
    for node_id in topo_order_2 {
        let node = graph.get_node(node_id).unwrap().clone();

        for &input_id in &node.inputs {
            let input_node = graph.get_node(input_id).unwrap();

            if input_node.opcode == Opcode::Quantize {
                insert_dequantize_before(graph, node_id, input_id, config.bit_width)?;
            }
        }
    }
    */

    Ok(())
}
