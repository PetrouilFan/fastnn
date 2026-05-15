//! ONNX model export — walks a [`ComputeGraph`] and writes a portable JSON
//! representation compatible with the [`OnnxConverter`](crate::onnx::converter::OnnxConverter)
//! import format.
//!
//! # Output Format
//!
//! The JSON output has the same structure that the Python ONNX importer produces:
//!
//! ```json
//! {
//!   "nodes": [ { "op_type": "MatMul", "inputs": "...", "outputs": "...", ... } ],
//!   "params": { "name": { "data": [f32...], "shape": [dims], "dtype": "f32" } },
//!   "input_names": ["x"],
//!   "output_names": ["y"]
//! }
//! ```

use crate::ir::node::{ComputeGraph, IrDType, NodeId, Opcode, TensorValue};
use std::collections::{HashMap, HashSet};

/// ONNX node representation for export JSON.
#[derive(Debug, Clone, serde::Serialize)]
struct OnnxExportNode {
    op_type: String,
    name: String,
    inputs: String,
    outputs: String,
    #[serde(flatten)]
    attrs: HashMap<String, String>,
}

/// Weight parameter for export — supports both f32 arrays and raw byte data.
#[derive(Debug, Clone, serde::Serialize)]
struct OnnxExportParam {
    data: serde_json::Value,
    shape: Vec<u64>,
    dtype: String,
}

// ── Quantized pattern detection ───────────────────────────────────────

/// Describes a detected quantized linear/conv pattern that should be fused
/// into a single ONNX QLinear op during export.
#[derive(Debug, Clone)]
struct QLinearPattern {
    /// "QLinearMatMul" or "QLinearConv"
    op_type: String,
    /// The Quantize node consuming the output.
    q_output_id: NodeId,
    /// Weight per-channel scales (from packed dtype).
    weight_scales: Vec<f32>,
    /// Weight per-channel zero points.
    weight_zero_points: Vec<f32>,
    /// Activation scale from DequantizeActivations (None if unquantized).
    activation_scale: Option<f32>,
}

/// Detect all QLinearMatMul and QLinearConv patterns in the graph.
///
/// Patterns detected:
///   Dequantize(weight) ──┐
///                        ├── MatMul/Conv2d ── Quantize
///   activations ─────────┘
///
/// Returns a set of node IDs to skip during normal export (the Dequantize
/// and Quantize nodes), and a map from the main op node ID to its pattern.
fn detect_qlinear_patterns(
    graph: &ComputeGraph,
    order: &[NodeId],
) -> (HashSet<NodeId>, HashMap<NodeId, QLinearPattern>) {
    let mut skip_nodes: HashSet<NodeId> = HashSet::new();
    let mut patterns: HashMap<NodeId, QLinearPattern> = HashMap::new();

    for &node_id in order {
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };

        let is_main = matches!(node.opcode, Opcode::MatMul | Opcode::Conv2d);
        if !is_main {
            continue;
        }

        // The weight is typically input[1] (input[0] is the activation).
        let weight_input = match node.inputs.get(1) {
            Some(&id) => id,
            None => continue,
        };

        // Check if weight input is a Dequantize node.
        let deq_node = match graph.get_node(weight_input) {
            Some(n) if matches!(n.opcode, Opcode::Dequantize) => n,
            _ => continue,
        };

        // The Dequantize consumes a packed tensor (Constant with U4/U8 dtype).
        let packed_id = match deq_node.inputs.first() {
            Some(&id) => id,
            None => continue,
        };
        let packed_node = match graph.get_node(packed_id) {
            Some(n) => n,
            None => continue,
        };

        // Extract scales/zero_points from the packed weight's dtype.
        let (weight_scales, weight_zero_points) = match &packed_node.output_type.dtype {
            IrDType::U4 {
                scales,
                zero_points,
            } => (scales.clone(), zero_points.clone()),
            IrDType::U8 {
                scales,
                zero_points,
            } => (scales.clone(), zero_points.clone()),
            _ => continue,
        };

        // Check if output is consumed by a Quantize node.
        let consumers: Vec<NodeId> = graph.consumers(node_id);
        let q_output_id = match consumers.iter().find(|&&cid| {
            graph
                .get_node(cid)
                .is_some_and(|c| matches!(c.opcode, Opcode::Quantize))
        }) {
            Some(&id) => id,
            None => continue,
        };

        // Check if activation input comes from DequantizeActivations.
        let activation_scale = match node.inputs.first() {
            Some(&act_id) => graph.get_node(act_id).and_then(|act_node| {
                if matches!(act_node.opcode, Opcode::DequantizeActivations) {
                    // Look for QuantizeActivations feeding this DequantizeActivations
                    act_node.inputs.first().and_then(|&qa_id| {
                        graph.get_node(qa_id).and_then(|qa_node| {
                            if matches!(qa_node.opcode, Opcode::QuantizeActivations) {
                                qa_node
                                    .attrs
                                    .get("scale")
                                    .and_then(|s| s.parse::<f32>().ok())
                            } else {
                                None
                            }
                        })
                    })
                } else {
                    None
                }
            }),
            None => None,
        };

        skip_nodes.insert(weight_input); // Dequantize
        skip_nodes.insert(q_output_id); // Quantize

        patterns.insert(
            node_id,
            QLinearPattern {
                op_type: match node.opcode {
                    Opcode::MatMul => "QLinearMatMul".to_string(),
                    Opcode::Conv2d => "QLinearConv".to_string(),
                    _ => unreachable!(),
                },
                q_output_id,
                weight_scales,
                weight_zero_points,
                activation_scale,
            },
        );
    }

    (skip_nodes, patterns)
}

/// Extract a per-tensor scale from per-channel scales.
/// Uses the first channel's scale as the representative value.
fn per_tensor_scale(scales: &[f32]) -> f32 {
    scales.first().copied().unwrap_or(1.0)
}

/// Extract a per-tensor zero_point from per-channel zero_points.
fn per_tensor_zero_point(zero_points: &[f32]) -> i32 {
    zero_points.first().copied().unwrap_or(0.0) as i32
}

/// Generate unique param entry names for the scale/zero_point scalars
/// needed by a QLinear op node.
fn scale_zp_param_names(pattern_id: NodeId) -> [String; 6] {
    [
        format!("a_scale_{}", pattern_id),
        format!("a_zp_{}", pattern_id),
        format!("b_scale_{}", pattern_id),
        format!("b_zp_{}", pattern_id),
        format!("y_scale_{}", pattern_id),
        format!("y_zp_{}", pattern_id),
    ]
}

/// Extract output scale and zero_point from a Quantize node.
/// Returns (scale, zero_point) — falls back to (1.0, 0) if unavailable.
fn extract_output_scale_zp(graph: &ComputeGraph, q_node_id: NodeId) -> (f32, i32) {
    let q_node = match graph.get_node(q_node_id) {
        Some(n) => n,
        None => return (1.0, 0),
    };
    match &q_node.output_type.dtype {
        IrDType::U4 {
            scales,
            zero_points,
        } => {
            let s = scales.first().copied().unwrap_or(1.0);
            let zp = zero_points.first().copied().unwrap_or(0.0) as i32;
            // If scales is empty, try computing from bit_width
            if scales.is_empty() {
                // Default U4 scale: max_val = 15, symmetric → scale = 1.0
                (1.0, 0)
            } else {
                (s, zp)
            }
        }
        IrDType::U8 {
            scales,
            zero_points,
        } => {
            let s = scales.first().copied().unwrap_or(1.0);
            let zp = zero_points.first().copied().unwrap_or(0.0) as i32;
            if scales.is_empty() {
                (1.0, 0)
            } else {
                (s, zp)
            }
        }
        _ => (1.0, 0),
    }
}

// ── Main export function ──────────────────────────────────────────────

/// Export a ComputeGraph to ONNX JSON format.
///
/// Returns a JSON string that can be re-imported via [`OnnxConverter`].
pub fn export_to_onnx_json(graph: &ComputeGraph) -> Result<String, String> {
    let order = graph.topological_sort();

    // Phase 1: detect quantized patterns
    let (skip_nodes, qlinear_patterns) = detect_qlinear_patterns(graph, &order);

    // Collect weight parameters from Constant(Data) nodes
    let mut params: HashMap<String, OnnxExportParam> = HashMap::new();
    let mut onnx_nodes: Vec<OnnxExportNode> = Vec::new();

    // Map from NodeId to ONNX tensor name for wiring
    let mut node_output_names: HashMap<NodeId, String> = HashMap::new();
    for &node_id in &order {
        let name = format!("t_{}", node_id);
        node_output_names.insert(node_id, name);
    }

    // Reserve unique names for scale/zp scalars (used in QLinear inputs).
    // We store the actual values when we encounter the pattern.
    let mut scale_zp_scalars: HashMap<NodeId, [f32; 6]> = HashMap::new(); // pattern_id → [a_s, a_zp, b_s, b_zp, y_s, y_zp]

    for &node_id in &order {
        let node = graph
            .get_node(node_id)
            .ok_or_else(|| format!("export: node {} not found", node_id))?;

        // Skip nodes that are part of quantized patterns (Dequantize, Quantize).
        if skip_nodes.contains(&node_id) {
            continue;
        }

        let output_name = &node_output_names[&node_id];
        let input_names: Vec<String> = node
            .inputs
            .iter()
            .map(|id| {
                node_output_names
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| format!("t_{}", id))
            })
            .collect();

        // Check if this node is a pattern main node (MatMul or Conv2d that matched).
        if let Some(pattern) = qlinear_patterns.get(&node_id) {
            // ── Emit QLinearMatMul or QLinearConv ────────────────
            let a_src = node_output_names[&node.inputs[0]].clone();
            let b_src = node_output_names[&node.inputs[1]].clone();

            let b_scale = per_tensor_scale(&pattern.weight_scales);
            let b_zp = per_tensor_zero_point(&pattern.weight_zero_points);
            let a_scale = pattern.activation_scale.unwrap_or(1.0);
            let a_zp = 0;
            let (y_scale, y_zp) = extract_output_scale_zp(graph, pattern.q_output_id);

            // Generate unique param names for the 6 scale/zp scalars
            let [a_s_name, a_z_name, b_s_name, b_z_name, y_s_name, y_z_name] =
                scale_zp_param_names(node_id);

            // Store scalar values for later insertion into params
            scale_zp_scalars.insert(
                node_id,
                [
                    a_scale,
                    a_zp as f32,
                    b_scale,
                    b_zp as f32,
                    y_scale,
                    y_zp as f32,
                ],
            );

            // Build the 8-input string (format expected by the converter)
            let q_inputs = vec![
                a_src, // A (quantized activations) — may be a_scale placeholder
                node_output_names[&node.inputs[0]].clone(), // A (real activation)
                a_s_name, // A_scale
                a_z_name, // A_zp
                b_src, // B (weight input of Dequantize → the packed constant)
                b_s_name, // B_scale
                b_z_name, // B_zp
                y_s_name, // Y_scale
                y_z_name, // Y_zp
            ];

            let mut attrs = HashMap::new();
            attrs.insert("a_scale".to_string(), a_scale.to_string());
            attrs.insert("a_zero_point".to_string(), a_zp.to_string());
            attrs.insert("b_scale".to_string(), b_scale.to_string());
            attrs.insert("b_zero_point".to_string(), b_zp.to_string());
            attrs.insert("y_scale".to_string(), y_scale.to_string());
            attrs.insert("y_zero_point".to_string(), y_zp.to_string());

            // For QLinearConv, also forward conv attributes.
            if pattern.op_type == "QLinearConv" {
                if let Some(s) = node.attrs.get("stride") {
                    attrs.insert("strides".to_string(), s.clone());
                }
                if let Some(s) = node.attrs.get("padding") {
                    attrs.insert("pads".to_string(), s.clone());
                }
                if let Some(s) = node.attrs.get("dilation") {
                    attrs.insert("dilations".to_string(), s.clone());
                }
                if let Some(s) = node.attrs.get("group") {
                    attrs.insert("group".to_string(), s.clone());
                }
            }

            let inputs_str = q_inputs.join(",");

            onnx_nodes.push(OnnxExportNode {
                op_type: pattern.op_type.clone(),
                name: format!("node_{}", node_id),
                inputs: inputs_str,
                outputs: output_name.clone(),
                attrs,
            });

            // Continue to next node — don't fall through to the regular match
            continue;
        }

        // Handle fused residual add norm: decompose into Add + LayerNorm/RMSNorm
        if node.opcode == Opcode::FusedResidualAddNorm {
            let norm_type = node
                .attrs
                .get("norm_type")
                .map(|s| s.as_str())
                .unwrap_or("layer_norm");
            let norm_op = match norm_type {
                "rms_norm" => "RMSNorm",
                _ => "LayerNormalization",
            };
            let eps = node
                .attrs
                .get("eps")
                .cloned()
                .unwrap_or_else(|| "1e-5".to_string());
            let intermediate = format!("t_fused_add_{}", node_id);

            // Add: only the first 2 inputs (x and residual)
            onnx_nodes.push(OnnxExportNode {
                op_type: "Add".to_string(),
                name: format!("node_{}_add", node_id),
                inputs: input_names[..2.min(input_names.len())].join(","),
                outputs: intermediate.clone(),
                attrs: HashMap::new(),
            });

            // Norm: intermediate plus optional weight (at index 2)
            let norm_input = if input_names.len() >= 3 {
                let mut parts = vec![intermediate];
                parts.push(input_names[2].clone());
                parts.join(",")
            } else {
                intermediate
            };

            let mut attrs = HashMap::new();
            attrs.insert("epsilon".to_string(), eps);
            onnx_nodes.push(OnnxExportNode {
                op_type: norm_op.to_string(),
                name: format!("node_{}", node_id),
                inputs: norm_input,
                outputs: output_name.clone(),
                attrs,
            });
            continue;
        }

        let (op_type, extra_attrs): (String, HashMap<String, String>) = match &node.opcode {
            Opcode::Constant(val) => {
                // Constant nodes become weight params, not ONNX ops
                match val {
                    TensorValue::Data { bytes, tensor_type } => {
                        let is_packed =
                            matches!(&tensor_type.dtype, IrDType::U4 { .. } | IrDType::U8 { .. });
                        if is_packed {
                            // Export quantized packed weights as raw byte params.
                            let shape: Vec<u64> = tensor_type
                                .shape
                                .iter()
                                .filter_map(|d| d.evaluate())
                                .collect();
                            // Represent packed bytes as array of integers in JSON.
                            let data: Vec<u64> = bytes.iter().map(|&b| b as u64).collect();
                            params.insert(
                                output_name.clone(),
                                OnnxExportParam {
                                    data: serde_json::json!(data),
                                    shape,
                                    dtype: match &tensor_type.dtype {
                                        IrDType::U4 { .. } => "u4".to_string(),
                                        IrDType::U8 { .. } => "u8".to_string(),
                                        _ => unreachable!(),
                                    },
                                },
                            );
                        } else if tensor_type.dtype.byte_size() == 4 {
                            // Only export F32 weights; skip Float(0) fill constants
                            let f32_data: Vec<f32> = bytes
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            let shape: Vec<u64> = tensor_type
                                .shape
                                .iter()
                                .filter_map(|d| d.evaluate())
                                .collect();
                            params.insert(
                                output_name.clone(),
                                OnnxExportParam {
                                    data: serde_json::json!(f32_data),
                                    shape,
                                    dtype: "f32".to_string(),
                                },
                            );
                        }
                    }
                    TensorValue::Float(_) | TensorValue::Int(_) => {
                        // Scalar constants: export as small weights
                        let val_f32 = match val {
                            TensorValue::Float(v) => vec![*v],
                            TensorValue::Int(v) => vec![*v as f32],
                            _ => unreachable!(),
                        };
                        params.insert(
                            output_name.clone(),
                            OnnxExportParam {
                                data: serde_json::json!(val_f32),
                                shape: vec![],
                                dtype: "f32".to_string(),
                            },
                        );
                    }
                }
                // Constant with no params → skip (it's either a weight or we already handled it)
                if !params.contains_key(output_name) {
                    continue;
                }
                // Don't emit an ONNX node for constants — they're params
                continue;
            }
            Opcode::Input => {
                // Input nodes are graph inputs, not ONNX ops
                continue;
            }
            Opcode::MatMul => ("MatMul".to_string(), HashMap::new()),
            Opcode::Conv2d => ("Conv2d".to_string(), {
                let mut a = HashMap::new();
                if let Some(s) = node.attrs.get("stride") {
                    a.insert("stride".to_string(), s.clone());
                }
                if let Some(s) = node.attrs.get("padding") {
                    a.insert("padding".to_string(), s.clone());
                }
                if let Some(s) = node.attrs.get("dilation") {
                    a.insert("dilation".to_string(), s.clone());
                }
                if let Some(s) = node.attrs.get("group") {
                    a.insert("group".to_string(), s.clone());
                }
                a
            }),
            Opcode::Add => ("Add".to_string(), HashMap::new()),
            Opcode::Sub => ("Sub".to_string(), HashMap::new()),
            Opcode::Mul => ("Mul".to_string(), HashMap::new()),
            Opcode::Div => ("Div".to_string(), HashMap::new()),
            Opcode::Relu => ("Relu".to_string(), HashMap::new()),
            Opcode::Sigmoid => ("Sigmoid".to_string(), HashMap::new()),
            Opcode::Tanh => ("Tanh".to_string(), HashMap::new()),
            Opcode::Exp => ("Exp".to_string(), HashMap::new()),
            Opcode::Log => ("Log".to_string(), HashMap::new()),
            Opcode::Neg => ("Neg".to_string(), HashMap::new()),
            Opcode::Sqrt => ("Sqrt".to_string(), HashMap::new()),
            Opcode::Abs => ("Abs".to_string(), HashMap::new()),
            Opcode::Transpose => ("Transpose".to_string(), {
                let mut a = HashMap::new();
                if let Some(perm) = node.attrs.get("perm") {
                    a.insert("perm".to_string(), perm.clone());
                }
                a
            }),
            Opcode::Reshape => ("Reshape".to_string(), HashMap::new()),
            Opcode::Flatten => ("Flatten".to_string(), HashMap::new()),
            Opcode::Squeeze => ("Squeeze".to_string(), {
                let mut a = HashMap::new();
                if let Some(axes) = node.attrs.get("axes") {
                    a.insert("axes".to_string(), axes.clone());
                }
                a
            }),
            Opcode::Unsqueeze => ("Unsqueeze".to_string(), {
                let mut a = HashMap::new();
                if let Some(axes) = node.attrs.get("axes") {
                    a.insert("axes".to_string(), axes.clone());
                }
                a
            }),
            Opcode::Concat => ("Concat".to_string(), {
                let mut a = HashMap::new();
                if let Some(axis) = node.attrs.get("axis") {
                    a.insert("axis".to_string(), axis.clone());
                }
                a
            }),
            Opcode::ReduceSum => ("ReduceSum".to_string(), {
                let mut a = HashMap::new();
                if let Some(axis) = node.attrs.get("axis") {
                    a.insert("axes".to_string(), axis.clone());
                }
                a
            }),
            Opcode::ReduceMean => ("ReduceMean".to_string(), {
                let mut a = HashMap::new();
                if let Some(axis) = node.attrs.get("axis") {
                    a.insert("axes".to_string(), axis.clone());
                }
                a
            }),
            Opcode::ReduceMax => ("ReduceMax".to_string(), {
                let mut a = HashMap::new();
                if let Some(axis) = node.attrs.get("axis") {
                    a.insert("axes".to_string(), axis.clone());
                }
                a
            }),
            Opcode::BatchNorm => ("BatchNormalization".to_string(), {
                let mut a = HashMap::new();
                if let Some(eps) = node.attrs.get("eps") {
                    a.insert("epsilon".to_string(), eps.clone());
                }
                a
            }),
            Opcode::Softmax => ("Softmax".to_string(), {
                let mut a = HashMap::new();
                if let Some(axis) = node.attrs.get("axis") {
                    a.insert("axis".to_string(), axis.clone());
                }
                a
            }),
            Opcode::Gelu => ("Gelu".to_string(), HashMap::new()),
            Opcode::LeakyRelu => ("LeakyRelu".to_string(), {
                let mut a = HashMap::new();
                if let Some(slope) = node.attrs.get("negative_slope") {
                    a.insert("alpha".to_string(), slope.clone());
                }
                a
            }),
            Opcode::Pad => ("Pad".to_string(), {
                let mut a = HashMap::new();
                if let Some(pads) = node.attrs.get("pads") {
                    a.insert("pads".to_string(), pads.clone());
                }
                if let Some(mode) = node.attrs.get("mode") {
                    a.insert("mode".to_string(), mode.clone());
                }
                a
            }),
            Opcode::Slice => ("Slice".to_string(), {
                let mut a = HashMap::new();
                if let Some(axes) = node.attrs.get("axes") {
                    a.insert("axes".to_string(), axes.clone());
                }
                if let Some(starts) = node.attrs.get("starts") {
                    a.insert("starts".to_string(), starts.clone());
                }
                if let Some(ends) = node.attrs.get("ends") {
                    a.insert("ends".to_string(), ends.clone());
                }
                a
            }),
            Opcode::Expand => ("Expand".to_string(), HashMap::new()),
            Opcode::Tile => ("Tile".to_string(), HashMap::new()),
            Opcode::Where => ("Where".to_string(), HashMap::new()),
            Opcode::Cast => ("Cast".to_string(), {
                let mut a = HashMap::new();
                if let Some(to) = node.attrs.get("to") {
                    a.insert("to".to_string(), to.clone());
                }
                a
            }),
            Opcode::Gather => ("Gather".to_string(), {
                let mut a = HashMap::new();
                if let Some(axis) = node.attrs.get("axis") {
                    a.insert("axis".to_string(), axis.clone());
                }
                a
            }),
            // Fallback: use the opcode name directly
            other => {
                let name = format!("{:?}", other);
                // Strip the "Opcode::" prefix
                let onnx_name = name.strip_prefix("Opcode::").unwrap_or(&name);
                // Skip unsupported ops (training-only ops like SgdUpdate, GradientScale)
                if matches!(
                    other,
                    Opcode::SgdUpdate
                        | Opcode::AdamUpdate
                        | Opcode::AdamWUpdate
                        | Opcode::MuonUpdate
                        | Opcode::LionUpdate
                        | Opcode::RmspropUpdate
                        | Opcode::GradientScale
                        | Opcode::Quantize
                        | Opcode::Dequantize
                        | Opcode::QuantizeActivations
                        | Opcode::DequantizeActivations
                        | Opcode::ToF16
                        | Opcode::ToF32
                        | Opcode::MulScalar
                        | Opcode::AddScalar
                        | Opcode::DivScalar
                        | Opcode::Input
                        | Opcode::Constant(_)
                ) {
                    continue;
                }
                (onnx_name.to_string(), HashMap::new())
            }
        };

        let inputs_str = input_names.join(",");
        let outputs_str = output_name.clone();

        onnx_nodes.push(OnnxExportNode {
            op_type,
            name: format!("node_{}", node_id),
            inputs: inputs_str,
            outputs: outputs_str,
            attrs: extra_attrs,
        });
    }

    // Phase 3: Add scale/zp scalar params for each QLinear pattern.
    for (&pattern_id, scalars) in &scale_zp_scalars {
        let [a_s, a_zp, b_s, b_zp, y_s, y_zp] = *scalars;
        let [a_s_name, a_z_name, b_s_name, b_z_name, y_s_name, y_z_name] =
            scale_zp_param_names(pattern_id);

        let insert_scalar =
            |params: &mut HashMap<String, OnnxExportParam>, name: String, val: f32| {
                params.entry(name).or_insert_with(|| OnnxExportParam {
                    data: serde_json::json!(vec![val]),
                    shape: vec![],
                    dtype: "f32".to_string(),
                });
            };

        insert_scalar(&mut params, a_s_name, a_s);
        insert_scalar(&mut params, a_z_name, a_zp);
        insert_scalar(&mut params, b_s_name, b_s);
        insert_scalar(&mut params, b_z_name, b_zp);
        insert_scalar(&mut params, y_s_name, y_s);
        insert_scalar(&mut params, y_z_name, y_zp);
    }

    // Collect input and output names
    let input_names: Vec<String> = graph
        .inputs
        .iter()
        .map(|id| {
            node_output_names
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("t_{}", id))
        })
        .collect();

    let output_names: Vec<String> = graph
        .outputs
        .iter()
        .map(|id| {
            node_output_names
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("t_{}", id))
        })
        .collect();

    let export_obj = serde_json::json!({
        "nodes": onnx_nodes,
        "params": params,
        "input_names": input_names,
        "output_names": output_names,
    });

    serde_json::to_string_pretty(&export_obj)
        .map_err(|e| format!("ONNX export JSON serialization: {}", e))
}

/// Export a ComputeGraph to an ONNX JSON file.
pub fn export_to_onnx_file(graph: &ComputeGraph, path: &str) -> Result<(), String> {
    let json = export_to_onnx_json(graph)?;
    std::fs::write(path, &json).map_err(|e| format!("ONNX export file write: {}", e))
}
