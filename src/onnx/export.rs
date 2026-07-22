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

use crate::ir::{ComputeGraph, NodeId, Opcode, TensorType, TensorValue};
use crate::types::{RepresentationTransform, ScalarType, StorageEncoding};
use std::collections::{HashMap, HashSet};

fn packed_param_dtype(tensor_type: &TensorType) -> Option<&'static str> {
    let representation = tensor_type.value_representation().ok()?;
    if !matches!(representation.encoding, StorageEncoding::Packed { .. }) {
        return None;
    }
    match representation.storage {
        ScalarType::I4 => Some("i4"),
        ScalarType::U4 => Some("u4"),
        ScalarType::I8 => Some("i8"),
        ScalarType::U8 => Some("u8"),
        ScalarType::Fp4E2M1 => Some("f4"),
        ScalarType::Fp8E4M3 => Some("f8"),
        ScalarType::Fp8E5M2 => Some("f8r"),
        _ => None,
    }
}

/// Configuration for ONNX export behavior.
pub struct ExportConfig {
    /// When `true`, export will fail with an explicit error if the graph
    /// contains training-only opcodes (optimizer updates, gradient scaling).
    ///
    /// When `false`, training opcodes are intentionally dropped from the export
    /// and the output metadata records that omission. This preserves an opt-in
    /// inference-only export path, but **can produce a wrong graph** if used on
    /// a graph where the training outputs matter.
    ///
    /// # Recommendation
    ///
    /// Always set `fail_on_training_ops: true` unless you have verified that
    /// the training opcodes in the graph are intentionally irrelevant.
    pub fail_on_training_ops: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            fail_on_training_ops: true,
        }
    }
}

/// Returns the list of opcodes that are **only meaningful in a training graph**
/// and have no standard ONNX representation.
///
/// If any of these appear in a `ComputeGraph`, default export fails rather than
/// dropping them and producing a graph that may execute but produces wrong
/// results (missing optimizer steps, missing gradient scaling, etc.).
const TRAINING_ONLY_OPCODES: &[fn(&Opcode) -> bool] = &[
    |op| matches!(op, Opcode::SgdUpdate),
    |op| matches!(op, Opcode::AdamUpdate),
    |op| matches!(op, Opcode::AdamWUpdate),
    |op| matches!(op, Opcode::MuonUpdate),
    |op| matches!(op, Opcode::LionUpdate),
    |op| matches!(op, Opcode::RmspropUpdate),
    |op| matches!(op, Opcode::GradientScale),
    |op| matches!(op, Opcode::QuantizeGradient),
    |op| matches!(op, Opcode::DequantizeGradient),
];

/// Detect training-only opcodes present in the graph.
///
/// Returns a list of `(node_id, opcode_name)` pairs for every node whose
/// opcode has no standard ONNX representation and is only meaningful during
/// training (optimizer updates, gradient scaling).
///
/// An empty return value means the graph is safe for ONNX inference export.
pub fn detect_training_ops(graph: &ComputeGraph) -> Result<Vec<(NodeId, String)>, String> {
    let order = graph
        .try_topological_sort()
        .map_err(|error| error.to_string())?;
    let mut found = Vec::new();

    for &node_id in &order {
        if let Some(node) = graph.get_node(node_id) {
            for check in TRAINING_ONLY_OPCODES {
                if check(&node.opcode) {
                    let name = format!("{:?}", node.opcode);
                    let onnx_name = name.strip_prefix("Opcode::").unwrap_or(&name);
                    found.push((node_id, onnx_name.to_string()));
                    break;
                }
            }
        }
    }

    Ok(found)
}

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
    weight_dequant_offsets: Vec<f32>,
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

        // The Dequantize consumes a packed tensor (Constant with I4/I8/F4/F8/F8R dtype).
        let packed_id = match deq_node.inputs.first() {
            Some(&id) => id,
            None => continue,
        };
        let packed_node = match graph.get_node(packed_id) {
            Some(n) => n,
            None => continue,
        };

        // Extract affine metadata from the canonical representation rather
        // than maintaining another list of packed IR dtype variants.
        let representation = match packed_node.output_type.value_representation() {
            Ok(representation) => representation,
            Err(_) => continue,
        };
        let (weight_scales, weight_dequant_offsets) = match representation.transform {
            RepresentationTransform::AffineDequantization {
                scales, offsets, ..
            }
            | RepresentationTransform::ScaledAffine {
                scales, offsets, ..
            }
            | RepresentationTransform::Codebook {
                scales, offsets, ..
            } => (
                scales,
                if offsets.is_empty() {
                    vec![0.0]
                } else {
                    offsets
                },
            ),
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
                weight_dequant_offsets,
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

/// Convert `real = q * scale + offset` metadata to ONNX's
/// `real = (q - zero_point) * scale` convention.
fn per_tensor_zero_point(scales: &[f32], dequant_offsets: &[f32]) -> i32 {
    let scale = scales.first().copied().unwrap_or(1.0);
    let offset = dequant_offsets.first().copied().unwrap_or(0.0);
    if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
        return 0;
    }
    (-offset / scale)
        .round()
        .clamp(i32::MIN as f32, i32::MAX as f32) as i32
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
    let representation = match q_node.output_type.value_representation() {
        Ok(representation) => representation,
        Err(_) => return (1.0, 0),
    };
    match representation.transform {
        RepresentationTransform::AffineDequantization {
            scales, offsets, ..
        }
        | RepresentationTransform::ScaledAffine {
            scales, offsets, ..
        }
        | RepresentationTransform::Codebook {
            scales, offsets, ..
        } if !scales.is_empty() => {
            let scale = scales[0];
            (scale, per_tensor_zero_point(&scales, &offsets))
        }
        RepresentationTransform::AffineQuantization(spec) if !spec.scales.is_empty() => {
            let zero_point = spec.zero_points.first().copied().unwrap_or(0);
            (spec.scales[0], zero_point)
        }
        _ => (1.0, 0),
    }
}

// ── Main export function ──────────────────────────────────────────────

/// Export a ComputeGraph to ONNX JSON format.
///
/// Returns a JSON string that can be re-imported via [`OnnxConverter`].
///
/// # Training Safety
///
/// By default, this function returns an error if the graph contains
/// training-only opcodes (optimizer updates, gradient scaling). Pass
/// `ExportConfig { fail_on_training_ops: false }` to silently drop them,
/// but be aware this **can produce a wrong graph**.
pub fn export_to_onnx_json(graph: &ComputeGraph) -> Result<String, String> {
    export_to_onnx_json_with_config(graph, &ExportConfig::default())
}

/// Export a ComputeGraph to ONNX JSON format with explicit configuration.
///
/// This is the primary export entry point. It validates the graph against
/// the provided configuration and returns a JSON string suitable for
/// re-import via [`OnnxConverter`].
pub fn export_to_onnx_json_with_config(
    graph: &ComputeGraph,
    config: &ExportConfig,
) -> Result<String, String> {
    // ── Phase 0: Training safety check ──────────────────────────────
    let training_ops = detect_training_ops(graph)?;
    if config.fail_on_training_ops && !training_ops.is_empty() {
        let opcodes: Vec<String> = training_ops
            .iter()
            .map(|(id, name)| format!("node {} ({})", id, name))
            .collect();
        return Err(format!(
            "ONNX export failed: graph contains {} training-only opcode(s) \
             that have no standard ONNX representation and would be silently \
             dropped, producing a wrong graph: {}.\n\n\
             Training opcodes in fastnn (SgdUpdate, AdamUpdate, AdamWUpdate, \
             MuonUpdate, LionUpdate, RmspropUpdate, GradientScale) are executed \
             by the fastnn compiled training pipeline and cannot be faithfully \
             represented in the ONNX inference format.\n\n\
             To export this graph for inference only, remove training nodes \
             (optimizer steps, gradient scaling) before export, or set \
             `ExportConfig {{ fail_on_training_ops: false }}` to explicitly \
             acknowledge that training nodes will be dropped.",
            training_ops.len(),
            opcodes.join(", ")
        ));
    }

    let order = graph
        .try_topological_sort()
        .map_err(|error| error.to_string())?;

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
            let b_zp =
                per_tensor_zero_point(&pattern.weight_scales, &pattern.weight_dequant_offsets);
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
                        if let Some(dtype) = packed_param_dtype(tensor_type) {
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
                                    dtype: dtype.to_string(),
                                },
                            );
                        } else if tensor_type.is_native_scalar(ScalarType::F32) {
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
            other => {
                match crate::onnx::opcode_map::opcode_to_onnx(other, &node.attrs) {
                    Some(result) => result,
                    None => {
                        // Fallback: use the opcode name directly
                        let name = format!("{:?}", other);
                        let onnx_name = name.strip_prefix("Opcode::").unwrap_or(&name);
                        if matches!(
                            other,
                            Opcode::SgdUpdate
                                | Opcode::AdamUpdate
                                | Opcode::AdamWUpdate
                                | Opcode::MuonUpdate
                                | Opcode::LionUpdate
                                | Opcode::RmspropUpdate
                                | Opcode::GradientScale
                                | Opcode::QuantizeGradient
                                | Opcode::DequantizeGradient
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
                }
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
        "metadata": {
            "export_mode": "inference",
            "training_ops_dropped": !training_ops.is_empty(),
            "training_ops_count": training_ops.len(),
        },
    });

    serde_json::to_string_pretty(&export_obj)
        .map_err(|e| format!("ONNX export JSON serialization: {}", e))
}

/// Export a ComputeGraph to an ONNX JSON file.
pub fn export_to_onnx_file(graph: &ComputeGraph, path: &str) -> Result<(), String> {
    let json = export_to_onnx_json(graph)?;
    std::fs::write(path, &json).map_err(|e| format!("ONNX export file write: {}", e))
}

/// Export a ComputeGraph to an ONNX JSON file with explicit configuration.
pub fn export_to_onnx_file_with_config(
    graph: &ComputeGraph,
    path: &str,
    config: &ExportConfig,
) -> Result<(), String> {
    let json = export_to_onnx_json_with_config(graph, config)?;
    std::fs::write(path, &json).map_err(|e| format!("ONNX export file write: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IrDType;

    #[test]
    fn packed_param_dtype_uses_canonical_storage_identity() {
        let i4 = TensorType::new(vec![crate::ir::DimExpr::Known(8)], IrDType::I4);
        let u4 = TensorType::new(vec![crate::ir::DimExpr::Known(8)], IrDType::U4Scaled);
        let f32 = TensorType::new(vec![crate::ir::DimExpr::Known(8)], IrDType::F32);
        assert_eq!(packed_param_dtype(&i4), Some("i4"));
        assert_eq!(packed_param_dtype(&u4), Some("u4"));
        assert_eq!(packed_param_dtype(&f32), None);
    }

    #[test]
    fn dequantization_offsets_are_converted_to_onnx_zero_points() {
        assert_eq!(per_tensor_zero_point(&[0.5], &[-1.0]), 2);
        assert_eq!(per_tensor_zero_point(&[0.25], &[1.0]), -4);
        assert_eq!(per_tensor_zero_point(&[], &[]), 0);
        assert_eq!(per_tensor_zero_point(&[0.0], &[1.0]), 0);
    }
}
