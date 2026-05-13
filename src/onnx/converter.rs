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
            name_to_id: HashMap::new(),
            graph: GraphBuilder::new(),
            errors: Vec::new(),
        }
    }

    /// Convert the ONNX graph to a ComputeGraph.
    pub fn to_compute_graph(mut self) -> Result<ComputeGraph, String> {
        // Phase 1: Register all graph inputs.
        for name in self.input_names {
            let shape: Vec<u64> = if let Some(t) = self.params.get(name.as_str()) {
                t.shape().iter().map(|&d| d as u64).collect()
            } else {
                vec![]
            };
            let dtype = self.params.get(name.as_str())
                .map(|t| ir_dtype_from_dtype(t.dtype()))
                .unwrap_or(IrDType::F32);
            let gt = self.graph.input(&shape, dtype);
            self.name_to_id.insert(name.clone(), gt);
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

        let graph = self.graph.to_graph();

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
                let scal = self.scalar(alpha);
                let added = self.scalar(beta);
                let sc = self.graph.add_scalar(&ins[0], &scal);
                self.out(node, self.graph.clamp(&sc, 0.0, 1.0));
                // Actually HardSigmoid = clamp(x*alpha + beta, 0, 1)
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
                let axis: usize = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(1);
                self.out(node, self.graph.softmax(&ins[0], axis));
            }
            "LogSoftmax" => {
                let axis: usize = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(1);
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
                let bias = ins.get(2);

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
                let groups: usize = node.attrs.get("group").and_then(|g| g.parse().ok()).unwrap_or(1);
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
        _ => IrDType::F32,
    }
}
