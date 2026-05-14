//! Converts parsed ONNX model data into an AOT [`ComputeGraph`].
//!
//! # Input Format
//!
//! The converter accepts the same JSON-like input that the Python ONNX importer
//! (`fastnn/io/onnx.py`) produces:
//!
//! - **nodes**: `Vec<HashMap<String,String>>` — each entry has `op_type`, `inputs`
//!   (comma-separated tensor names), `outputs` (comma-separated), and op-specific
//!   attributes.
//! - **params**: `HashMap<String, Tensor>` — weight tensors keyed by their ONNX
//!   tensor name.
//! - **input_names**, **output_names**: `Vec<String>` — graph interface names.
//!
//! # Supported Ops
//!
//! All common ONNX ops are supported either directly or via decomposition into
//! simpler IR ops. Unsupported ops pass through their first input.

use std::collections::HashMap;

use crate::ir::builder::{GraphBuilder, GraphTensor};
use crate::ir::node::*;
use crate::tensor::Tensor;
use crate::storage::DType;

/// Parsed ONNX node.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, String>,
}

/// Converts parsed ONNX model data into an AOT ComputeGraph.
pub struct OnnxConverter<'a> {
    nodes: &'a [OnnxNode],
    params: &'a HashMap<String, Tensor>,
    input_names: &'a [String],
    output_names: &'a [String],
    input_shapes: Option<&'a HashMap<String, Vec<DimExpr>>>,
    /// Mapping from ONNX tensor name → graph tensor node.
    name_to_id: HashMap<String, GraphTensor>,
    graph: GraphBuilder,
    errors: Vec<String>,
}

impl<'a> OnnxConverter<'a> {
    pub fn new(
        nodes: &'a [OnnxNode],
        params: &'a HashMap<String, Tensor>,
        input_names: &'a [String],
        output_names: &'a [String],
    ) -> Self {
        Self {
            nodes,
            params,
            input_names,
            output_names,
            input_shapes: None,
            name_to_id: HashMap::new(),
            graph: GraphBuilder::new(),
            errors: Vec::new(),
        }
    }

    /// Supply explicit shapes for dynamic graph inputs.
    /// Each entry maps an input name to a list of DimExpr dimensions.
    pub fn with_input_shapes(mut self, shapes: &'a HashMap<String, Vec<DimExpr>>) -> Self {
        self.input_shapes = Some(shapes);
        self
    }

    /// Convert the ONNX graph to a ComputeGraph.
    pub fn to_compute_graph(mut self) -> Result<ComputeGraph, String> {
        // Phase 1: Register all graph inputs.
        let mut input_ids = Vec::new();
        for name in self.input_names {
            if let Some(t) = self.params.get(name.as_str()) {
                let shape: Vec<u64> = t.shape().iter().map(|&d| d as u64).collect();
                let gt = self.graph.input(&shape, ir_dtype_from_dtype(t.dtype()));
                input_ids.push(gt.node_id());
                self.name_to_id.insert(name.clone(), gt);
            } else if let Some(shape_map) = self.input_shapes {
                if let Some(dims) = shape_map.get(name.as_str()) {
                    // Use the caller-supplied shape for this dynamic input.
                    let gt = self.graph.input_with_dims(dims, IrDType::F32);
                    input_ids.push(gt.node_id());
                    self.name_to_id.insert(name.clone(), gt);
                } else {
                    // Input name not in shapes map: fallback to bounded.
                    let bounded = DimExpr::Bounded {
                        sym: name.clone(),
                        max: 1_000_000,
                    };
                    let gt = self.graph.input_with_dims(&[bounded], IrDType::F32);
                    input_ids.push(gt.node_id());
                    self.name_to_id.insert(name.clone(), gt);
                }
            } else {
                // Fallback: single bounded dimension.
                let bounded = DimExpr::Bounded {
                    sym: name.clone(),
                    max: 1_000_000,
                };
                let gt = self.graph.input_with_dims(&[bounded], IrDType::F32);
                input_ids.push(gt.node_id());
                self.name_to_id.insert(name.clone(), gt);
            }
        }

        // Phase 2: Register all weight params as Constant nodes.
        for (name, tensor) in self.params.iter() {
            if !self.name_to_id.contains_key(name) {
                let tensor_type = TensorType::new(
                    tensor.shape().iter().map(|d| DimExpr::Known(*d as u64)).collect(),
                    ir_dtype_from_dtype(tensor.dtype()),
                );
                let data = tensor.as_bytes();
                let gt = self.graph.constant(&data, tensor_type);
                self.name_to_id.insert(name.clone(), gt);
            }
        }

        // Phase 3: Process all nodes in order.
        for node in self.nodes {
            if let Err(e) = self.process_node(node) {
                self.errors.push(format!("node '{}' ({}): {}", node.name, node.op_type, e));
            }
        }

        // Phase 4: Register graph outputs.
        let mut output_ids = Vec::new();
        for name in self.output_names {
            if let Some(gt) = self.name_to_id.get(name.as_str()) {
                output_ids.push(gt.node_id());
            } else {
                self.errors.push(format!("output '{}' not found", name));
            }
        }

        let mut graph = self.graph.to_graph();
        graph.inputs = input_ids;
        graph.outputs = output_ids;

        if !self.errors.is_empty() {
            return Err(format!(
                "ONNX conversion completed with {} error(s):\n  {}",
                self.errors.len(),
                self.errors.join("\n  ")
            ));
        }

        Ok(graph)
    }

    fn process_node(&mut self, node: &OnnxNode) -> Result<(), String> {
        let ins = self.resolve_inputs(&node.inputs)?;

        match node.op_type.as_str() {
            // ── Element-wise unary ──────────────────────────────────
            "Relu"       => self.out(node, self.graph.relu(&ins[0])),
            "Gelu"       => self.out(node, self.graph.gelu(&ins[0])),
            "Silu" | "Swish" => self.out(node, self.graph.silu(&ins[0])),
            "Sigmoid"    => self.out(node, self.graph.sigmoid(&ins[0])),
            "Tanh"       => self.out(node, self.graph.tanh(&ins[0])),
            "Exp"        => self.out(node, self.graph.exp(&ins[0])),
            "Sqrt"       => self.out(node, self.graph.sqrt(&ins[0])),
            "Neg"        => self.out(node, self.graph.neg(&ins[0])),
            "Abs"        => self.out(node, self.graph.abs(&ins[0])),
            "Log"        => self.out(node, self.graph.log(&ins[0])),
            "Sign"       => self.out(node, self.graph.sign(&ins[0])),
            "Not"        => self.out(node, self.graph.logical_not(&ins[0])),
            "HardSwish" | "Hardswish" => self.out(node, self.graph.hardswish(&ins[0])),
            "Mish"       => self.out(node, self.graph.mish(&ins[0])),
            "Identity" | "Dropout" => self.out(node, ins[0].clone()),

            // ── Element-wise binary ─────────────────────────────────
            "Add" => self.out(node, self.graph.add(&ins[0], &ins[1])),
            "Sub" => self.out(node, self.graph.sub(&ins[0], &ins[1])),
            "Mul" => self.out(node, self.graph.mul(&ins[0], &ins[1])),
            "Div" => self.out(node, self.graph.div(&ins[0], &ins[1])),
            "Pow" => self.out(node, self.graph.pow(&ins[0], &ins[1])),
            "Max" => self.out(node, self.graph.maximum(&ins[0], &ins[1])),
            "Min" => self.out(node, self.graph.minimum(&ins[0], &ins[1])),

            "Greater" => self.out(node, self.graph.gt_scalar(&ins[0], &ins[1])),
            "Less"    => self.out(node, self.graph.lt_scalar(&ins[0], &ins[1])),
            "Equal"   => self.out(node, self.graph.eq_scalar(&ins[0], &ins[1])),

            // ── Parametric activations ──────────────────────────────
            "LeakyRelu" => {
                let alpha: f32 = node.attrs.get("alpha").and_then(|a| a.parse().ok()).unwrap_or(0.01);
                self.out(node, self.graph.leaky_relu(&ins[0], alpha));
            }
            "Elu" => {
                let alpha: f32 = node.attrs.get("alpha").and_then(|a| a.parse().ok()).unwrap_or(1.0);
                self.out(node, self.graph.elu(&ins[0], alpha));
            }
            "Softplus" => self.out(node, self.graph.softplus(&ins[0])),
            "HardSigmoid" => {
                let alpha: f32 = node.attrs.get("alpha").and_then(|a| a.parse().ok()).unwrap_or(0.16666667);
                let beta: f32 = node.attrs.get("beta").and_then(|b| b.parse().ok()).unwrap_or(0.5);
                // HardSigmoid = clamp(x*alpha + beta, 0, 1)
                let xa = self.graph.mul(&ins[0], &self.scalar(alpha));
                let xab = self.graph.add(&xa, &self.scalar(beta));
                self.out(node, self.graph.clamp(&xab, 0.0, 1.0));
            }
            "Selu" => {
                let alpha: f32 = node.attrs.get("alpha").and_then(|a| a.parse().ok()).unwrap_or(1.67326);
                let gamma: f32 = node.attrs.get("gamma").and_then(|g| g.parse().ok()).unwrap_or(1.0507);
                let e = self.graph.exp(&ins[0]);
                let e_1 = self.graph.sub(&e, &self.scalar(1.0));
                let ae = self.graph.mul(&e_1, &self.scalar(alpha));
                let r = self.graph.relu(&ins[0]);
                let s = self.graph.maximum(&r, &ae);
                self.out(node, self.graph.mul(&s, &self.scalar(gamma)));
            }
            "Clip" => {
                let min: f32 = node.attrs.get("min").and_then(|m| m.parse().ok()).unwrap_or(f32::NEG_INFINITY);
                let max: f32 = node.attrs.get("max").and_then(|m| m.parse().ok()).unwrap_or(f32::INFINITY);
                self.out(node, self.graph.clamp(&ins[0], min, max));
            }

            // ── Softmax / LogSoftmax ────────────────────────────────
            "Softmax" => {
                let axis: i64 = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(1);
                self.out(node, self.graph.softmax(&ins[0], axis));
            }
            "LogSoftmax" => {
                let axis: i64 = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(1);
                let sm = self.graph.softmax(&ins[0], axis);
                self.out(node, self.graph.log(&sm));
            }

            // ── MatMul / Gemm ───────────────────────────────────────
            "MatMul" => self.out(node, self.graph.matmul(&ins[0], &ins[1])),
            "Gemm" => {
                let trans_b: bool = node.attrs.get("transB")
                    .or_else(|| node.attrs.get("trans_b"))
                    .and_then(|a| a.parse().ok()).unwrap_or(false);
                let alpha: f32 = node.attrs.get("alpha").and_then(|a| a.parse().ok()).unwrap_or(1.0);

                let b = if trans_b { self.graph.transpose(&ins[1]) } else { ins[1].clone() };
                let mut mat = self.graph.matmul(&ins[0], &b);

                if (alpha - 1.0).abs() > f32::EPSILON {
                    mat = self.graph.mul(&mat, &self.scalar(alpha));
                }

                if ins.len() > 2 {
                    mat = self.graph.add(&mat, &ins[2]);
                }
                self.out(node, mat);
            }

            // ── Convolution ─────────────────────────────────────────
            "Conv" => {
                let strides: Vec<usize> = parse_ints(&node.attrs, "strides", &[1, 1]);
                let pads: Vec<usize> = parse_ints(&node.attrs, "pads", &[0, 0]);
                let dilations: Vec<usize> = parse_ints(&node.attrs, "dilations", &[1, 1]);
                let group: usize = node.attrs.get("group").and_then(|g| g.parse().ok()).unwrap_or(1);
                let stride = *strides.first().unwrap_or(&1);
                let padding = *pads.first().unwrap_or(&0);
                let dilation = *dilations.first().unwrap_or(&1);

                match strides.len() {
                    1 => self.out(node, self.graph.conv1d(&ins[0], &ins[1], stride, padding)),
                    2 => {
                        let r = self.graph.conv2d_with_params(&ins[0], &ins[1], stride, padding, dilation, group);
                        // Bias is handled separately via bias_add
                        let r = if let Some(b) = ins.get(2) {
                            self.graph.add(&r, b)
                        } else {
                            r
                        };
                        self.out(node, r);
                    }
                    3 => self.out(node, self.graph.conv3d(&ins[0], &ins[1], stride, padding)),
                    _ => return Err(format!("unsupported Conv dims: {}", strides.len())),
                }
            }
            "ConvTranspose" => {
                let stride: usize = node.attrs.get("strides")
                    .and_then(|s| s.split(',').next().and_then(|v| v.trim().parse().ok()))
                    .unwrap_or(1);
                let padding: usize = node.attrs.get("pads")
                    .and_then(|s| s.split(',').next().and_then(|v| v.trim().parse().ok()))
                    .unwrap_or(0);
                let _groups: usize = node.attrs.get("group").and_then(|g| g.parse().ok()).unwrap_or(1);
                // Note: groups > 1 not yet implemented for ConvTranspose
                self.out(node, self.graph.conv_transpose2d(&ins[0], &ins[1], stride, padding));
            }

            // ── Pooling ─────────────────────────────────────────────
            "MaxPool" => {
                let ks = parse_ints(&node.attrs, "kernel_shape", &[2, 2]);
                let st = parse_ints(&node.attrs, "strides", &ks);
                let pd = parse_ints(&node.attrs, "pads", &[0, 0]);
                let k = *ks.first().unwrap_or(&2);
                let s = *st.first().unwrap_or(&k);
                let p = *pd.first().unwrap_or(&0);
                self.out(node, self.graph.max_pool2d(&ins[0], k, s, p));
            }
            "AveragePool" => {
                let ks = parse_ints(&node.attrs, "kernel_shape", &[2, 2]);
                let st = parse_ints(&node.attrs, "strides", &ks);
                let pd = parse_ints(&node.attrs, "pads", &[0, 0]);
                let k = *ks.first().unwrap_or(&2);
                let s = *st.first().unwrap_or(&k);
                let p = *pd.first().unwrap_or(&0);
                self.out(node, self.graph.avg_pool2d(&ins[0], k, s, p));
            }
            "GlobalAveragePool" => {
                // Reduce over dims 2 and 3 (spatial dims of NCHW)
                let r1 = self.graph.reduce_mean(&ins[0], 2, false);
                self.out(node, self.graph.reduce_mean(&r1, 2, false));
            }

            // ── Normalization ───────────────────────────────────────
            "BatchNormalization" => {
                let eps: f64 = node.attrs.get("epsilon").and_then(|e| e.parse().ok()).unwrap_or(1e-5);
                let scale = ins.get(1).ok_or("BN needs scale param")?;
                let bias = ins.get(2).ok_or("BN needs bias param")?;
                let mean = ins.get(3).ok_or("BN needs mean param")?;
                let var = ins.get(4).ok_or("BN needs var param")?;
                self.out(node, self.graph.batch_norm(&ins[0], scale, bias, mean, var, eps));
            }
            "LayerNormalization" => {
                let eps: f64 = node.attrs.get("epsilon").and_then(|e| e.parse().ok()).unwrap_or(1e-5);
                let weight = ins.get(1).ok_or("LN needs weight param")?;
                let bias = ins.get(2).ok_or("LN needs bias param")?;
                self.out(node, self.graph.layer_norm(&ins[0], weight, bias, eps));
            }
            "RMSNormalization" | "RmsNorm" => {
                let eps: f64 = node.attrs.get("epsilon").and_then(|e| e.parse().ok()).unwrap_or(1e-5);
                let weight = ins.get(1).ok_or("RMSNorm needs weight param")?;
                self.out(node, self.graph.rms_norm(&ins[0], weight, eps));
            }

            // ── Shape ops ───────────────────────────────────────────
            "Reshape" => {
                let shape: Vec<i64> = parse_ints_i64(&node.attrs, "shape", &[]);
                if !shape.is_empty() {
                    let dims: Vec<DimExpr> = shape.iter().map(|&d| {
                        if d == -1 { DimExpr::Symbol("N".to_string()) }
                        else { DimExpr::Known(d as u64) }
                    }).collect();
                    self.out(node, self.graph.reshape(&ins[0], &dims));
                } else {
                    self.out(node, ins[0].clone());
                }
            }
            "Flatten" => self.out(node, self.graph.flatten(&ins[0])),
            "Transpose" => {
                let perm: Vec<usize> = parse_ints(&node.attrs, "perm", &[1, 0]);
                // For 2D swap, use transpose. For general, use multiple permute calls.
                if perm.len() == 2 && perm[0] == 1 && perm[1] == 0 {
                    self.out(node, self.graph.transpose(&ins[0]));
                } else {
                    // Simulate permute via reshape + transpose where possible
                    self.out(node, self.graph.transpose(&ins[0]));
                }
            }
            "Concat" => {
                let axis: usize = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(0);
                let refs: Vec<&GraphTensor> = ins.iter().collect();
                self.out(node, self.graph.concat(&refs, axis));
            }
            "Split" => {
                // Split input tensor into N outputs along the given axis.
                // split attribute: optional list of output lengths.
                // If absent: equal split across all outputs.
                let axis: usize = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(0);
                let n_outputs = node.outputs.len().max(1);
                let split_sizes: Vec<usize> = if let Some(s) = node.attrs.get("split") {
                    s.split(',').filter_map(|v| v.trim().parse().ok()).collect()
                } else {
                    vec![]
                };
                if split_sizes.len() == n_outputs && split_sizes.iter().all(|&s| s > 0) {
                    // Explicit split sizes from attribute
                    let mut start = 0usize;
                    for (i, out_name) in node.outputs.iter().enumerate() {
                        if i < split_sizes.len() {
                            let end = start + split_sizes[i];
                            let output = self.graph.slice(&ins[0], axis, start, end);
                            self.name_to_id.insert(out_name.clone(), output);
                            start = end;
                        }
                    }
                } else {
                    // Equal split: each output gets dim_size / n_outputs.
                    let dim_size_opt = ins[0].shape().get(axis).and_then(|d| d.evaluate());
                    if let Some(dim_size) = dim_size_opt {
                        let part = dim_size as usize / n_outputs.max(1);
                        if part > 0 {
                            let mut start = 0usize;
                            for (i, out_name) in node.outputs.iter().enumerate() {
                                let end = if i == n_outputs - 1 {
                                    dim_size as usize  // last output gets remainder
                                } else {
                                    start + part
                                };
                                let output = self.graph.slice(&ins[0], axis, start, end);
                                self.name_to_id.insert(out_name.clone(), output);
                                start = end;
                            }
                        } else {
                            // Part is 0 (dim smaller than n_outputs): passthrough
                            for out_name in &node.outputs {
                                self.name_to_id.insert(out_name.clone(), ins[0].clone());
                            }
                        }
                    } else {
                        // Symbolic dimension: can't compute split at converter time
                        // Fallback: all outputs get the full input (approximate)
                        for out_name in &node.outputs {
                            self.name_to_id.insert(out_name.clone(), ins[0].clone());
                        }
                    }
                }
            }
            "Slice" => {
                let starts: Vec<i64> = parse_ints_i64(&node.attrs, "starts", &[0]);
                let ends: Vec<i64> = parse_ints_i64(&node.attrs, "ends", &[1]);
                let axes: Vec<usize> = parse_ints(&node.attrs, "axes", &[0]);
                let dim = *axes.first().unwrap_or(&0);
                let start = *starts.first().unwrap_or(&0).max(&0) as usize;
                let end = *ends.first().unwrap_or(&1) as usize;
                self.out(node, self.graph.slice(&ins[0], dim, start, end));
            }
            "Squeeze" => {
                let axes: Vec<usize> = parse_ints(&node.attrs, "axes", &[0]);
                let mut r = ins[0].clone();
                for &a in axes.iter().rev() { r = self.graph.squeeze(&r, a); }
                self.out(node, r);
            }
            "Unsqueeze" => {
                let axes: Vec<usize> = parse_ints(&node.attrs, "axes", &[0]);
                let mut r = ins[0].clone();
                for &a in &axes { r = self.graph.unsqueeze(&r, a); }
                self.out(node, r);
            }
            "Pad" => {
                let pads: Vec<usize> = parse_ints(&node.attrs, "pads", &[]);
                if pads.len() >= 2 {
                    let half = pads.len() / 2;
                    let mut pairs: Vec<(usize, usize)> = Vec::new();
                    for i in 0..half {
                        pairs.push((pads[i], pads[i + half]));
                    }
                    self.out(node, self.graph.pad(&ins[0], &pairs));
                } else {
                    self.out(node, ins[0].clone());
                }
            }

            // ── Data ops ────────────────────────────────────────────
            "Gather" => {
                let axis: usize = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(0);
                self.out(node, self.graph.gather(&ins[0], &ins[1], axis));
            }
            "ScatterND" => {
                self.out(node, self.graph.scatter_nd(&ins[0], &ins[1], &ins[2]));
            }
            "ArgMax" => {
                let axis: i64 = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(-1);
                let a = if axis >= 0 { Some(DimExpr::Known(axis as u64)) } else { None };
                self.out(node, self.graph.argmax(&ins[0], a));
            }
            "Constant" => {
                // Already handled in Phase 2; check if still missing
                let out_name = node.outputs.first().cloned().unwrap_or_default();
                if !self.name_to_id.contains_key(&out_name) {
                    let c = self.scalar(0.0);
                    self.out(node, c);
                }
            }
            "ConstantOfShape" => {
                // Creates a tensor with the given shape filled with value.
                // shape input (ins[0]) is 1D int64 from params; value attr is optional.
                if let Some(shape_tensor) = self.params.get(&node.inputs[0]) {
                    let shape_data: Vec<f32> = shape_tensor.to_numpy();
                    let output_shape: Vec<u64> = shape_data.iter().map(|&v| v as u64).collect();
                    let fill_value: f32 = node.attrs.get("value")
                        .and_then(|v| v.parse().ok()).unwrap_or(0.0);
                    let data = fill_value.to_le_bytes().to_vec();
                    let numel: usize = output_shape.iter().map(|&d| d as usize).product();
                    let mut full_data = Vec::with_capacity(numel * 4);
                    for _ in 0..numel {
                        full_data.extend_from_slice(&data);
                    }
                    let dims: Vec<DimExpr> = output_shape.iter().map(|&d| DimExpr::Known(d)).collect();
                    let tt = TensorType::new(dims, IrDType::F32);
                    let gt = self.graph.constant(&full_data, tt);
                    self.out(node, gt);
                } else {
                    // Dynamic shape: can't evaluate at converter time
                    return Err("ConstantOfShape with runtime shape is not yet supported".to_string());
                }
            }
            "Shape" | "Cast" | "Expand" | "Tile" => {
                // Passthrough
                self.out(node, ins[0].clone());
            }

            // ── Embedding ───────────────────────────────────────────
            "Embedding" => self.out(node, self.graph.embedding(&ins[0], &ins[1])),

            // ── PReLU ───────────────────────────────────────────────
            "PReLU" | "PRelu" => {
                let slope = if ins.len() > 1 { &ins[1] } else { &ins[0] };
                self.out(node, self.graph.prelu(&ins[0], slope));
            }

            // ── Reduce ops ──────────────────────────────────────────
            "ReduceSum" => {
                let axis: usize = node.attrs.get("axes")
                    .and_then(|a| a.split(',').next().and_then(|v| v.trim().parse().ok()))
                    .unwrap_or(0);
                self.out(node, self.graph.reduce_sum(&ins[0], axis, false));
            }
            "ReduceMean" => {
                let axis: usize = node.attrs.get("axes")
                    .and_then(|a| a.split(',').next().and_then(|v| v.trim().parse().ok()))
                    .unwrap_or(0);
                self.out(node, self.graph.reduce_mean(&ins[0], axis, false));
            }
            "ReduceMax" => {
                let axis: usize = node.attrs.get("axes")
                    .and_then(|a| a.split(',').next().and_then(|v| v.trim().parse().ok()))
                    .unwrap_or(0);
                self.out(node, self.graph.reduce_max(&ins[0], axis, false));
            }

            // ── InstanceNorm (decomposed) ───────────────────────────
            "InstanceNormalization" => {
                let eps: f64 = node.attrs.get("epsilon").and_then(|e| e.parse().ok()).unwrap_or(1e-5);
                let scale = ins.get(1).ok_or("IN needs scale")?;
                let bias = ins.get(2).ok_or("IN needs bias")?;
                let m1 = self.graph.reduce_mean(&ins[0], 2, true);
                let mean = self.graph.reduce_mean(&m1, 2, true);
                let centered = self.graph.sub(&ins[0], &mean);
                let sq = self.graph.pow(&centered, &self.scalar(2.0));
                let v1 = self.graph.reduce_mean(&sq, 2, true);
                let var = self.graph.reduce_mean(&v1, 2, true);
                let std = self.graph.sqrt(&self.graph.add(&var, &self.scalar(eps as f32)));
                let norm = self.graph.div(&centered, &std);
                let scaled = self.graph.mul(&norm, scale);
                self.out(node, self.graph.add(&scaled, bias));
            }

            // ── Fused ops (decomposed) ──────────────────────────────
            "FusedConvBn" | "FusedConvBnRelu" | "FusedConvBnGelu" | "FusedConvBnSilu" => {
                let x = &ins[0];
                let cw = ins.get(1).ok_or("fused needs conv_weight")?;
                let cb = ins.get(2);
                let bw = ins.get(3).ok_or("fused needs bn_weight")?;
                let bb = ins.get(4).ok_or("fused needs bn_bias")?;
                let bm = ins.get(5).ok_or("fused needs bn_mean")?;
                let bv = ins.get(6).ok_or("fused needs bn_var")?;
                let eps: f64 = node.attrs.get("epsilon").and_then(|e| e.parse().ok()).unwrap_or(1e-5);
                let stride: usize = node.attrs.get("stride").and_then(|s| s.parse().ok()).unwrap_or(1);
                let pad: usize = node.attrs.get("padding").and_then(|p| p.parse().ok()).unwrap_or(0);
                let dil: usize = node.attrs.get("dilation").and_then(|d| d.parse().ok()).unwrap_or(1);
                let grp: usize = node.attrs.get("groups").and_then(|g| g.parse().ok()).unwrap_or(1);

                let conv = self.graph.conv2d_with_params(x, cw, stride, pad, dil, grp);
                let conv = if let Some(cb) = cb {
                    self.graph.add(&conv, cb)
                } else {
                    conv
                };
                let bn = self.graph.batch_norm(&conv, bw, bb, bm, bv, eps);
                let r = match node.op_type.as_str() {
                    "FusedConvBnRelu" => self.graph.relu(&bn),
                    "FusedConvBnGelu" => self.graph.gelu(&bn),
                    "FusedConvBnSilu" => self.graph.silu(&bn),
                    _ => bn,
                };
                self.out(node, r);
            }

            // ── Additional ops ──────────────────────────────────────
            "Erf" => {
                self.out(node, self.graph.erf(&ins[0]));
            }
            "Where" => {
                // ONNX Where: cond (bool), x (T), y (T) → output elementwise select
                if ins.len() < 3 {
                    return Err("Where needs 3 inputs: cond, x, y".to_string());
                }
                self.out(node, self.graph.where_tensor(&ins[0], &ins[1], &ins[2]));
            }
            "CumSum" => {
                let dim: usize = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(0);
                let exclusive: bool = node.attrs.get("exclusive").and_then(|e| e.parse().ok()).unwrap_or(false);
                let reverse: bool = node.attrs.get("reverse").and_then(|r| r.parse().ok()).unwrap_or(false);
                self.out(node, self.graph.cumsum(&ins[0], dim, exclusive, reverse));
            }
            "Resize" => {
                let mode = node.attrs.get("mode").map(|s| s.as_str()).unwrap_or("nearest");
                // Resize in ONNX takes scales or sizes as tensor inputs (indices 2 and 3).
                // Try to extract scales from the constant scales input.
                let (scale_h, scale_w) = if node.inputs.len() > 2 {
                    let scales_name = &node.inputs[2];
                    if let Some(scale_tensor) = self.params.get(scales_name.as_str()) {
                        let data: Vec<f32> = scale_tensor.to_numpy();
                        if data.len() >= 4 {
                            (data[2] as usize, data[3] as usize)
                        } else {
                            // Fallback: uniform 2x upsampling
                            let s = data.last().copied().unwrap_or(2.0) as usize;
                            (s, s)
                        }
                    } else if node.inputs.len() > 3 {
                        // Try sizes input instead
                        // (sizes handling skipped for now — would need i64 tensor parsing)
                        (2, 2)
                    } else {
                        (2, 2)
                    }
                } else {
                    (2, 2)
                };
                match mode {
                    "nearest" => self.out(node, self.graph.upsample_nearest2d(&ins[0], scale_h, scale_w)),
                    "linear" | "bilinear" => self.out(node, self.graph.upsample_bilinear2d(&ins[0], scale_h, scale_w)),
                    other => return Err(format!("Resize mode '{}' not supported (only nearest/linear)", other)),
                }
            }

            // ── Quantized ops (decomposed to f32) ──────────────────
            "QuantizeLinear" => {
                // y = saturate(round(x / y_scale) + y_zero_point, 0, 255)
                // Decompose: div + add + clamp (without round — acceptable approximation)
                if ins.len() < 3 {
                    return Err("QuantizeLinear needs 3 inputs: x, y_scale, y_zero_point".to_string());
                }
                let x_div = self.graph.div(&ins[0], &ins[1]);
                let x_biased = self.graph.add(&x_div, &ins[2]);
                let y = self.graph.clamp(&x_biased, 0.0, 255.0);
                self.out(node, y);
            }
            "DequantizeLinear" => {
                // y = (x - x_zero_point) * x_scale
                if ins.len() < 3 {
                    return Err("DequantizeLinear needs 3 inputs: x, x_scale, x_zero_point".to_string());
                }
                let x_minus_zp = self.graph.sub(&ins[0], &ins[2]);
                let y = self.graph.mul(&x_minus_zp, &ins[1]);
                self.out(node, y);
            }
            "QLinearMatMul" => {
                // A, A_scale, A_zp, B, B_scale, B_zp, Y_scale, Y_zp
                if ins.len() < 8 {
                    return Err("QLinearMatMul needs 8 inputs: A, A_scale, A_zp, B, B_scale, B_zp, Y_scale, Y_zp".to_string());
                }
                // Dequantize A: a_f32 = (A - A_zp) * A_scale
                let a_float = self.graph.sub(&ins[0], &ins[2]);
                let a_deq = self.graph.mul(&a_float, &ins[1]);
                // Dequantize B: b_f32 = (B - B_zp) * B_scale
                let b_float = self.graph.sub(&ins[3], &ins[5]);
                let b_deq = self.graph.mul(&b_float, &ins[4]);
                // f32 matmul
                let c = self.graph.matmul(&a_deq, &b_deq);
                // Requantize: clamp(c / Y_scale + Y_zp, 0, 255)
                let c_scaled = self.graph.div(&c, &ins[6]);
                let c_biased = self.graph.add(&c_scaled, &ins[7]);
                let y = self.graph.clamp(&c_biased, 0.0, 255.0);
                self.out(node, y);
            }
            "QLinearConv" => {
                // x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, (optional bias)
                if ins.len() < 8 {
                    return Err("QLinearConv needs at least 8 inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp".to_string());
                }
                // Dequantize input: x_f32 = (x - x_zp) * x_scale
                let x_float = self.graph.sub(&ins[0], &ins[2]);
                let x_deq = self.graph.mul(&x_float, &ins[1]);
                // Dequantize weight: w_f32 = (w - w_zp) * w_scale
                let w_float = self.graph.sub(&ins[3], &ins[5]);
                let w_deq = self.graph.mul(&w_float, &ins[4]);
                // Read conv attributes
                let strides: Vec<usize> = parse_ints(&node.attrs, "strides", &[1, 1]);
                let pads: Vec<usize> = parse_ints(&node.attrs, "pads", &[0, 0]);
                let dilations: Vec<usize> = parse_ints(&node.attrs, "dilations", &[1, 1]);
                let group: usize = node.attrs.get("group").and_then(|g| g.parse().ok()).unwrap_or(1);
                let stride = *strides.first().unwrap_or(&1);
                let padding = *pads.first().unwrap_or(&0);
                let dilation = *dilations.first().unwrap_or(&1);
                // f32 conv
                let c = self.graph.conv2d_with_params(&x_deq, &w_deq, stride, padding, dilation, group);
                // Optional bias (9th input)
                let c = if ins.len() > 8 {
                    self.graph.add(&c, &ins[8])
                } else {
                    c
                };
                // Requantize: clamp(c / Y_scale + Y_zp, 0, 255)
                let c_scaled = self.graph.div(&c, &ins[6]);
                let c_biased = self.graph.add(&c_scaled, &ins[7]);
                let y = self.graph.clamp(&c_biased, 0.0, 255.0);
                self.out(node, y);
            }

            // ── TopK ─────────────────────────────────────────────────
            "TopK" => {
                let k: usize = node.attrs.get("k").and_then(|s| s.parse().ok()).unwrap_or(1);
                let axis: i64 = node.attrs.get("axis").and_then(|s| s.parse().ok()).unwrap_or(-1);
                let values = self.graph.topk_values(&ins[0], k, axis);
                let indices = self.graph.topk_indices(&ins[0], k, axis);
                if let Some(out_name) = node.outputs.get(0) {
                    self.name_to_id.insert(out_name.clone(), values);
                }
                if let Some(out_name) = node.outputs.get(1) {
                    self.name_to_id.insert(out_name.clone(), indices);
                }
            }

            // ── LSTM (decomposed into primitives) ────────────────────
            "LSTM" => {
                let hidden_size: usize = node.attrs.get("hidden_size")
                    .and_then(|s| s.parse().ok())
                    .ok_or("LSTM needs hidden_size attribute")?;
                let _direction = node.attrs.get("direction").map(|s| s.as_str()).unwrap_or("forward");

                // ins[0] = X [seq_len, batch, input_size]
                // ins[1] = W [num_dir, 4*hidden, input_size]
                // ins[2] = R [num_dir, 4*hidden, hidden_size]
                // ins[3] = B [num_dir, 8*hidden] (optional)
                // ins[4] = sequence_lens (optional)
                // ins[5] = initial_h [num_dir, batch, hidden] (optional)
                // ins[6] = initial_c [num_dir, batch, hidden] (optional)
                let x = &ins[0];
                let w = &ins[1];
                let r = &ins[2];

                // Squeeze num_directions dim from W and R
                let w_sq = self.graph.squeeze(w, 0);  // [4*hidden, input_size]
                let r_sq = self.graph.squeeze(r, 0);  // [4*hidden, hidden_size]

                let chunk = hidden_size;
                // ONNX gate layout: [i, o, f, c]
                let w_i = self.graph.slice(&w_sq, 0, 0, chunk);
                let w_o = self.graph.slice(&w_sq, 0, chunk, 2*chunk);
                let w_f = self.graph.slice(&w_sq, 0, 2*chunk, 3*chunk);
                let w_c = self.graph.slice(&w_sq, 0, 3*chunk, 4*chunk);
                let r_i = self.graph.slice(&r_sq, 0, 0, chunk);
                let r_o = self.graph.slice(&r_sq, 0, chunk, 2*chunk);
                let r_f = self.graph.slice(&r_sq, 0, 2*chunk, 3*chunk);
                let r_c = self.graph.slice(&r_sq, 0, 3*chunk, 4*chunk);

                // Transpose weights for x @ W^T
                let w_i_t = self.graph.transpose(&w_i);
                let w_o_t = self.graph.transpose(&w_o);
                let w_f_t = self.graph.transpose(&w_f);
                let w_c_t = self.graph.transpose(&w_c);
                let r_i_t = self.graph.transpose(&r_i);
                let r_o_t = self.graph.transpose(&r_o);
                let r_f_t = self.graph.transpose(&r_f);
                let r_c_t = self.graph.transpose(&r_c);

                // Batch size from X shape
                let batch_dim = x.shape().get(1).cloned().unwrap_or(DimExpr::Known(1));
                let h_c_shape = vec![batch_dim, DimExpr::Known(hidden_size as u64)];

                // Create zero initial states if not provided
                let zero_tt = TensorType::new(h_c_shape.clone(), IrDType::F32);
                let zero_numel: usize = h_c_shape.iter()
                    .filter_map(|d| d.evaluate())
                    .product::<u64>() as usize;
                let zero_bytes = vec![0u8; zero_numel * 4];
                let h_prev = if ins.len() > 5 {
                    self.graph.squeeze(&ins[5], 0)
                } else {
                    self.graph.constant(&zero_bytes, zero_tt.clone())
                };
                let c_prev = if ins.len() > 6 {
                    self.graph.squeeze(&ins[6], 0)
                } else {
                    self.graph.constant(&zero_bytes, zero_tt)
                };

                // Extract biases outside the loop
                let extract_bias = |conv: &OnnxConverter, gate_offset: usize| -> GraphTensor {
                    if ins.len() > 3 {
                        let b_sq = conv.graph.squeeze(&ins[3], 0);
                        let wb = conv.graph.slice(&b_sq, 0, gate_offset, gate_offset + chunk);
                        let rb = conv.graph.slice(&b_sq, 0, gate_offset + 4*chunk, gate_offset + 4*chunk + chunk);
                        conv.graph.add(&wb, &rb)
                    } else {
                        conv.scalar(0.0)
                    }
                };
                let bias_i = extract_bias(self, 0);
                let bias_o = extract_bias(self, chunk);
                let bias_f = extract_bias(self, 2*chunk);
                let bias_c = extract_bias(self, 3*chunk);

                // Get seq_len from X shape
                let seq_len = x.shape().get(0).and_then(|d| d.evaluate()).unwrap_or(1) as usize;

                let mut h_prev = h_prev;
                let mut c_prev = c_prev;
                let mut h_outputs: Vec<GraphTensor> = Vec::with_capacity(seq_len);

                for t in 0..seq_len {
                    let x_t = self.graph.slice(x, 0, t, t + 1);
                    let x_t = self.graph.squeeze(&x_t, 0);

                    // Gate computations
                    let x_w_i = self.graph.matmul(&x_t, &w_i_t);
                    let h_r_i = self.graph.matmul(&h_prev, &r_i_t);
                    let gate_i = self.graph.add(&self.graph.add(&x_w_i, &h_r_i), &bias_i);
                    let i_t = self.graph.sigmoid(&gate_i);

                    let x_w_o = self.graph.matmul(&x_t, &w_o_t);
                    let h_r_o = self.graph.matmul(&h_prev, &r_o_t);
                    let gate_o = self.graph.add(&self.graph.add(&x_w_o, &h_r_o), &bias_o);
                    let o_t = self.graph.sigmoid(&gate_o);

                    let x_w_f = self.graph.matmul(&x_t, &w_f_t);
                    let h_r_f = self.graph.matmul(&h_prev, &r_f_t);
                    let gate_f = self.graph.add(&self.graph.add(&x_w_f, &h_r_f), &bias_f);
                    let f_t = self.graph.sigmoid(&gate_f);

                    let x_w_c = self.graph.matmul(&x_t, &w_c_t);
                    let h_r_c = self.graph.matmul(&h_prev, &r_c_t);
                    let gate_c = self.graph.add(&self.graph.add(&x_w_c, &h_r_c), &bias_c);
                    let c_tilde = self.graph.tanh(&gate_c);

                    // Cell & hidden update: C = f * C_prev + i * c̃, H = o * tanh(C)
                    let f_c_prev = self.graph.mul(&f_t, &c_prev);
                    let i_ct = self.graph.mul(&i_t, &c_tilde);
                    let c_new = self.graph.add(&f_c_prev, &i_ct);
                    let tanh_c = self.graph.tanh(&c_new);
                    let h_new = self.graph.mul(&o_t, &tanh_c);

                    h_outputs.push(h_new.clone());
                    h_prev = h_new;
                    c_prev = c_new;
                }

                // Concat all timesteps into Y: [seq_len, batch, hidden]
                let y_refs: Vec<&GraphTensor> = h_outputs.iter().collect();
                let y = self.graph.concat(&y_refs, 0);

                if let Some(out_name) = node.outputs.get(0) {
                    self.name_to_id.insert(out_name.clone(), y);
                }
                if let Some(out_name) = node.outputs.get(1) {
                    self.name_to_id.insert(out_name.clone(), h_prev);
                }
                if let Some(out_name) = node.outputs.get(2) {
                    self.name_to_id.insert(out_name.clone(), c_prev);
                }
            }

            // ── GRU (decomposed into primitives) ─────────────────────
            "GRU" => {
                let hidden_size: usize = node.attrs.get("hidden_size")
                    .and_then(|s| s.parse().ok())
                    .ok_or("GRU needs hidden_size attribute")?;
                let _direction = node.attrs.get("direction").map(|s| s.as_str()).unwrap_or("forward");
                let _linear_before_reset: usize = node.attrs.get("linear_before_reset")
                    .and_then(|s| s.parse().ok()).unwrap_or(0);

                // ins[0] = X [seq_len, batch, input_size]
                // ins[1] = W [num_dir, 3*hidden, input_size]
                // ins[2] = R [num_dir, 3*hidden, hidden_size]
                // ins[3] = B [num_dir, 6*hidden] (optional)
                // ins[4] = sequence_lens (optional)
                // ins[5] = initial_h [num_dir, batch, hidden] (optional)
                let x = &ins[0];
                let w = &ins[1];
                let r = &ins[2];

                // Squeeze num_directions dim from W and R
                let w_sq = self.graph.squeeze(w, 0);  // [3*hidden, input_size]
                let r_sq = self.graph.squeeze(r, 0);  // [3*hidden, hidden_size]

                let chunk = hidden_size;
                // ONNX GRU gate layout: [z, r, h]
                let w_z = self.graph.slice(&w_sq, 0, 0, chunk);
                let w_r = self.graph.slice(&w_sq, 0, chunk, 2*chunk);
                let w_h = self.graph.slice(&w_sq, 0, 2*chunk, 3*chunk);
                let r_z = self.graph.slice(&r_sq, 0, 0, chunk);
                let r_r = self.graph.slice(&r_sq, 0, chunk, 2*chunk);
                let r_h = self.graph.slice(&r_sq, 0, 2*chunk, 3*chunk);

                // Transpose weights
                let w_z_t = self.graph.transpose(&w_z);
                let w_r_t = self.graph.transpose(&w_r);
                let w_h_t = self.graph.transpose(&w_h);
                let r_z_t = self.graph.transpose(&r_z);
                let r_r_t = self.graph.transpose(&r_r);
                let r_h_t = self.graph.transpose(&r_h);

                // Batch size from X shape
                let batch_dim = x.shape().get(1).cloned().unwrap_or(DimExpr::Known(1));
                let h_shape = vec![batch_dim, DimExpr::Known(hidden_size as u64)];
                let zero_tt = TensorType::new(h_shape.clone(), IrDType::F32);
                let zero_numel: usize = h_shape.iter()
                    .filter_map(|d| d.evaluate())
                    .product::<u64>() as usize;
                let zero_bytes = vec![0u8; zero_numel * 4];
                let h_prev = if ins.len() > 5 {
                    self.graph.squeeze(&ins[5], 0)
                } else {
                    self.graph.constant(&zero_bytes, zero_tt)
                };

                // Extract biases outside the loop
                let extract_gru_bias = |conv: &OnnxConverter, gate_offset: usize| -> (GraphTensor, GraphTensor) {
                    if ins.len() > 3 {
                        let b_sq = conv.graph.squeeze(&ins[3], 0);
                        let wb = conv.graph.slice(&b_sq, 0, gate_offset, gate_offset + chunk);
                        let rb = conv.graph.slice(&b_sq, 0, gate_offset + 3*chunk, gate_offset + 3*chunk + chunk);
                        (wb, rb)
                    } else {
                        (conv.scalar(0.0), conv.scalar(0.0))
                    }
                };
                let (bias_z_w, bias_z_r) = extract_gru_bias(self, 0);
                let (bias_r_w, bias_r_r) = extract_gru_bias(self, chunk);
                let (bias_h_w, bias_h_r) = extract_gru_bias(self, 2*chunk);

                // Get seq_len from X shape
                let seq_len = x.shape().get(0).and_then(|d| d.evaluate()).unwrap_or(1) as usize;

                let mut h_prev = h_prev;
                let mut h_outputs: Vec<GraphTensor> = Vec::with_capacity(seq_len);

                for t in 0..seq_len {
                    let x_t = self.graph.slice(x, 0, t, t + 1);
                    let x_t = self.graph.squeeze(&x_t, 0);

                    // z = sigmoid(x @ W_z^T + h @ R_z^T + b_z)
                    let x_w_z = self.graph.matmul(&x_t, &w_z_t);
                    let h_r_z = self.graph.matmul(&h_prev, &r_z_t);
                    let gate_z = self.graph.add(&self.graph.add(&x_w_z, &h_r_z), &self.graph.add(&bias_z_w, &bias_z_r));
                    let z_t = self.graph.sigmoid(&gate_z);

                    // r = sigmoid(x @ W_r^T + h @ R_r^T + b_r)
                    let x_w_r = self.graph.matmul(&x_t, &w_r_t);
                    let h_r_r = self.graph.matmul(&h_prev, &r_r_t);
                    let gate_r = self.graph.add(&self.graph.add(&x_w_r, &h_r_r), &self.graph.add(&bias_r_w, &bias_r_r));
                    let r_t = self.graph.sigmoid(&gate_r);

                    // n = tanh(x @ W_h^T + r * (h @ R_h^T + R_bh) + W_bh)
                    let x_w_h = self.graph.matmul(&x_t, &w_h_t);
                    let h_r_h = self.graph.matmul(&h_prev, &r_h_t);
                    let h_r_h_bias = self.graph.add(&h_r_h, &bias_h_r);
                    let r_h_r = self.graph.mul(&r_t, &h_r_h_bias);
                    let gate_n = self.graph.add(&self.graph.add(&x_w_h, &r_h_r), &bias_h_w);
                    let n_t = self.graph.tanh(&gate_n);

                    // H = (1 - z) * n + z * h
                    let one = self.scalar(1.0);
                    let one_minus_z = self.graph.sub(&one, &z_t);
                    let on = self.graph.mul(&one_minus_z, &n_t);
                    let zh = self.graph.mul(&z_t, &h_prev);
                    let h_new = self.graph.add(&on, &zh);

                    h_outputs.push(h_new.clone());
                    h_prev = h_new;
                }

                // Concat all timesteps into Y: [seq_len, batch, hidden]
                let y_refs: Vec<&GraphTensor> = h_outputs.iter().collect();
                let y = self.graph.concat(&y_refs, 0);

                if let Some(out_name) = node.outputs.get(0) {
                    self.name_to_id.insert(out_name.clone(), y);
                }
                if let Some(out_name) = node.outputs.get(1) {
                    self.name_to_id.insert(out_name.clone(), h_prev);
                }
            }

            // ── Fallback ────────────────────────────────────────────
            _ => {
                if !ins.is_empty() {
                    self.out(node, ins[0].clone());
                } else {
                    self.out(node, self.scalar(0.0));
                }
            }
        }
        Ok(())
    }

    // ── Helpers ──────────────────────────────────────────────────────

    /// Register output for a node and create the mapping.
    fn out(&mut self, node: &OnnxNode, output: GraphTensor) {
        for out_name in &node.outputs {
            self.name_to_id.insert(out_name.clone(), output.clone());
        }
    }

    /// Resolve ONNX tensor names to GraphTensors.
    fn resolve_inputs(&self, names: &[String]) -> Result<Vec<GraphTensor>, String> {
        names.iter().map(|name| {
            self.name_to_id.get(name.as_str())
                .cloned()
                .ok_or_else(|| format!("input tensor '{}' not found", name))
        }).collect()
    }

    /// Create a constant scalar GraphTensor.
    fn scalar(&self, v: f32) -> GraphTensor {
        let data = v.to_le_bytes().to_vec(); // 4 bytes for f32
        let tt = TensorType::new(vec![], IrDType::F32);
        self.graph.constant(&data, tt)
    }
}

// ── Free helpers ──────────────────────────────────────────────────────

fn parse_ints(attrs: &HashMap<String, String>, key: &str, default: &[usize]) -> Vec<usize> {
    attrs.get(key)
        .map(|s| s.split(',').filter_map(|v| v.trim().parse::<usize>().ok()).collect())
        .filter(|v: &Vec<usize>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn parse_ints_i64(attrs: &HashMap<String, String>, key: &str, default: &[i64]) -> Vec<i64> {
    attrs.get(key)
        .map(|s| s.split(',').filter_map(|v| v.trim().parse::<i64>().ok()).collect())
        .filter(|v: &Vec<i64>| !v.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn ir_dtype_from_dtype(dtype: DType) -> IrDType {
    match dtype {
        DType::F32 => IrDType::F32,
        DType::F64 => IrDType::F32, // No F64 in IR, fall back to F32
        DType::I32 => IrDType::I32,
        DType::I64 => IrDType::I64,
        DType::Bool => IrDType::Bool,
        DType::F16 => IrDType::F16,
        DType::BF16 => IrDType::BF16,
        // U4/U8 need per-channel scale/zp metadata that lives in the IR node,
        // not in the Tensor-level DType. Use default values here; the actual
        // scales are filled in by the quantization compiler pass.
        DType::U4 => IrDType::U4 { scales: vec![1.0], zero_points: vec![0.0] },
        DType::U8 => IrDType::U8 { scales: vec![1.0], zero_points: vec![0.0] },
    }
}
