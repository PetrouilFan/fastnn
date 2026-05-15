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

use crate::ir::node::{ComputeGraph, NodeId, Opcode, TensorType, TensorValue};
use std::collections::HashMap;

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

/// Weight parameter for export.
#[derive(Debug, Clone, serde::Serialize)]
struct OnnxExportParam {
    data: Vec<f32>,
    shape: Vec<u64>,
    dtype: String,
}

/// Export a ComputeGraph to ONNX JSON format.
///
/// Returns a JSON string that can be re-imported via [`OnnxConverter`].
pub fn export_to_onnx_json(graph: &ComputeGraph) -> Result<String, String> {
    let order = graph.topological_sort();

    // Collect weight parameters from Constant(Data) nodes
    let mut params: HashMap<String, OnnxExportParam> = HashMap::new();
    let mut onnx_nodes: Vec<OnnxExportNode> = Vec::new();

    // Map from NodeId to ONNX tensor name for wiring
    let mut node_output_names: HashMap<NodeId, String> = HashMap::new();
    for &node_id in &order {
        let name = format!("t_{}", node_id);
        node_output_names.insert(node_id, name);
    }

    for &node_id in &order {
        let node = graph.get_node(node_id).ok_or_else(|| {
            format!("export: node {} not found", node_id)
        })?;

        let output_name = &node_output_names[&node_id];
        let input_names: Vec<String> = node
            .inputs
            .iter()
            .map(|id| node_output_names.get(id).cloned().unwrap_or_else(|| format!("t_{}", id)))
            .collect();

        let (op_type, extra_attrs): (String, HashMap<String, String>) = match &node.opcode {
            Opcode::Constant(val) => {
                // Constant nodes become weight params, not ONNX ops
                match val {
                    TensorValue::Data { bytes, tensor_type } => {
                        // Only export F32 weights; skip Float(0) fill constants
                        if tensor_type.dtype.byte_size() == 4 {
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
                                    data: f32_data,
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
                                data: val_f32,
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
                if matches!(other, Opcode::SgdUpdate | Opcode::AdamUpdate | Opcode::AdamWUpdate
                    | Opcode::GradientScale | Opcode::Quantize | Opcode::Dequantize
                    | Opcode::ToF16 | Opcode::ToF32 | Opcode::MulScalar | Opcode::AddScalar
                    | Opcode::DivScalar | Opcode::Input | Opcode::Constant(_))
                {
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

    // Collect input and output names
    let input_names: Vec<String> = graph
        .inputs
        .iter()
        .map(|id| node_output_names.get(id).cloned().unwrap_or_else(|| format!("t_{}", id)))
        .collect();

    let output_names: Vec<String> = graph
        .outputs
        .iter()
        .map(|id| node_output_names.get(id).cloned().unwrap_or_else(|| format!("t_{}", id)))
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
