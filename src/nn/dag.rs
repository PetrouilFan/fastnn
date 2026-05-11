#![allow(dead_code)]
use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::storage::{Device, DType};
use crate::tensor::Tensor;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpCode {
    Relu,
    Sigmoid,
    Tanh,
    Silu,
    Gelu,
    LeakyRelu,
    Elu,
    Softmax,
    Hardswish,
    Softplus,
    HardSigmoid,
    Prelu,
    Selu,
    Swish,
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Pow,
    BiasAdd,
    BiasSub,
    Exp,
    Sqrt,
    Neg,
    Log,
    Abs,
    Ceil,
    Floor,
    Round,
    Sign,
    Reciprocal,
    Erf,
    Not,
    And,
    Or,
    Xor,
    Less,
    Greater,
    Equal,
    IsNan,
    IsInf,
    ReduceMean,
    ReduceSum,
    ArgMax,
    ArgMin,
    Min,
    Max,
    Reshape,
    Flatten,
    Transpose,
    Concat,
    Squeeze,
    Unsqueeze,
    Slice,
    Split,
    Expand,
    Tile,
    Pad,
    Reverse,
    Conv,
    Gemm,
    BatchNorm,
    MaxPool,
    AvgPool,
    GlobalAvgPool,
    ConvTranspose,
    InstanceNorm,
    LayerNorm,
    RmsNorm,
    FusedConvBn,
    FusedConvBnRelu,
    FusedConvBnGelu,
    Identity,
    Shape,
    Cast,
    TopK,
    Gather,
    GatherNd,
    ScatterNd,
    Where,
    Compress,
    DepthToSpace,
    SpaceToDepth,
    EyeLike,
    LogSoftmax,
    Range,
    Constant,
    ConstantOfShape,
    NonMaxSuppression,
    Resize,
    Clip,
    CumSum,
    OneHot,
    RandomNormal,
    RandomUniform,
    Unknown(String),
}

pub fn op_type_to_code(op_type: &str) -> OpCode {
    match op_type.to_lowercase().as_str() {
        "relu" => OpCode::Relu,
        "sigmoid" => OpCode::Sigmoid,
        "tanh" => OpCode::Tanh,
        "silu" => OpCode::Silu,
        "gelu" => OpCode::Gelu,
        "leakyrelu" | "leaky_relu" => OpCode::LeakyRelu,
        "elu" => OpCode::Elu,
        "softmax" => OpCode::Softmax,
        "hardswish" => OpCode::Hardswish,
        "softplus" => OpCode::Softplus,
        "hardsigmoid" | "hard_sigmoid" => OpCode::HardSigmoid,
        "prelu" => OpCode::Prelu,
        "selu" => OpCode::Selu,
        "swish" => OpCode::Swish,
        "add" | "elementwiseadd" => OpCode::Add,
        "sub" | "elementwisesub" => OpCode::Sub,
        "mul" | "elementwisemul" => OpCode::Mul,
        "div" | "elementwisediv" => OpCode::Div,
        "matmul" => OpCode::MatMul,
        "pow" | "elementwisepow" => OpCode::Pow,
        "biasadd" | "bias_add" => OpCode::BiasAdd,
        "biassub" | "bias_sub" => OpCode::BiasSub,
        "exp" | "expop" => OpCode::Exp,
        "sqrt" | "sqrtop" => OpCode::Sqrt,
        "neg" | "negop" => OpCode::Neg,
        "log" | "logop" => OpCode::Log,
        "abs" => OpCode::Abs,
        "ceil" | "ceilop" => OpCode::Ceil,
        "floor" | "floorop" => OpCode::Floor,
        "round" | "roundop" => OpCode::Round,
        "sign" | "signop" => OpCode::Sign,
        "reciprocal" | "reciprocalop" => OpCode::Reciprocal,
        "erf" | "erfop" => OpCode::Erf,
        "not" | "notop" => OpCode::Not,
        "and" | "andop" => OpCode::And,
        "or" | "orop" => OpCode::Or,
        "xor" | "xorop" => OpCode::Xor,
        "less" | "lessop" => OpCode::Less,
        "greater" | "greaterop" => OpCode::Greater,
        "equal" | "equalop" => OpCode::Equal,
        "isnan" | "isnanop" => OpCode::IsNan,
        "isinf" | "isinfop" => OpCode::IsInf,
        "reducemean" | "reduce_mean" => OpCode::ReduceMean,
        "reducesum" | "reduce_sum" => OpCode::ReduceSum,
        "argmax" => OpCode::ArgMax,
        "argmin" => OpCode::ArgMin,
        "min" | "minop" => OpCode::Min,
        "max" | "maxop" => OpCode::Max,
        "reshape" => OpCode::Reshape,
        "flatten" => OpCode::Flatten,
        "transpose" => OpCode::Transpose,
        "concat" => OpCode::Concat,
        "squeeze" | "squeezeop" => OpCode::Squeeze,
        "unsqueeze" | "unsqueezeop" => OpCode::Unsqueeze,
        "slice" | "sliceop" => OpCode::Slice,
        "split" => OpCode::Split,
        "expand" => OpCode::Expand,
        "tileop" | "tile" => OpCode::Tile,
        "pad" => OpCode::Pad,
        "reverse" => OpCode::Reverse,
        "conv" => OpCode::Conv,
        "gemm" | "linear" => OpCode::Gemm,
        "batchnormalization" | "batch_norm" | "batchnorm2d" => OpCode::BatchNorm,
        "maxpool" | "max_pool2d" => OpCode::MaxPool,
        "averagepool" | "avg_pool" | "avg_pool2d" => OpCode::AvgPool,
        "globalaveragepool" | "global_avg_pool" => OpCode::GlobalAvgPool,
        "convtranspose" | "conv_transpose" => OpCode::ConvTranspose,
        "instancenormalization" | "instance_norm" => OpCode::InstanceNorm,
        "layernormalization" | "layer_norm" | "layernorm" => OpCode::LayerNorm,
        "rmsnormalization" | "rms_norm" => OpCode::RmsNorm,
        "fusedconvbn" | "fused_conv_bn" => OpCode::FusedConvBn,
        "fusedconvbnrelu" | "fused_conv_bn_relu" => OpCode::FusedConvBnRelu,
        "fusedconvbngelu" | "fused_conv_bn_gelu" => OpCode::FusedConvBnGelu,
        "identity" | "identityop" | "dropout" => OpCode::Identity,
        "shape" | "shapeop" => OpCode::Shape,
        "cast" | "castop" => OpCode::Cast,
        "topk" | "topkop" => OpCode::TopK,
        "gatherop" | "gather" => OpCode::Gather,
        "gathernd" | "gatherndop" => OpCode::GatherNd,
        "scatternd" | "scatterndop" => OpCode::ScatterNd,
        "whereop" | "where" => OpCode::Where,
        "compress" => OpCode::Compress,
        "depthtospace" | "depth_to_space" => OpCode::DepthToSpace,
        "spacetodepth" | "space_to_depth" => OpCode::SpaceToDepth,
        "eyelike" | "eyelikeop" => OpCode::EyeLike,
        "logsoftmax" => OpCode::LogSoftmax,
        "range" | "rangeop" => OpCode::Range,
        "constant" | "constantop" => OpCode::Constant,
        "constantofshape" => OpCode::ConstantOfShape,
        "nonmaxsuppression" => OpCode::NonMaxSuppression,
        "resize" => OpCode::Resize,
        "clip" => OpCode::Clip,
        "cumsum" => OpCode::CumSum,
        "onehot" | "onehotop" => OpCode::OneHot,
        "randomnormal" => OpCode::RandomNormal,
        "randomuniform" => OpCode::RandomUniform,
        _ => OpCode::Unknown(op_type.to_string()),
    }
}

/// A single node in the DAG computation graph.
#[derive(Clone)]
pub struct DAGNode {
    pub name: String,
    pub op_type: String,
    pub op_code: OpCode,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, String>,
}

/// The DAG graph executor.
pub struct DAGExecutor {
    nodes: Vec<DAGNode>,
    params: HashMap<String, Tensor>,
    transposed_weights: HashMap<String, Tensor>,
    scalar_cache: HashMap<i64, Tensor>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    name_to_id: HashMap<String, usize>,
    total_tensors: usize,
}

fn parse_int_list(s: &str) -> Vec<i64> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter_map(|x| x.trim().parse::<i64>().ok())
        .collect()
}

impl DAGExecutor {
    pub fn new(
        nodes: Vec<DAGNode>,
        params: HashMap<String, Tensor>,
        input_names: Vec<String>,
        output_names: Vec<String>,
    ) -> Self {
        let mut name_to_id = HashMap::new();
        let mut next_id = 0;

        for name in &input_names {
            if !name_to_id.contains_key(name) {
                name_to_id.insert(name.clone(), next_id);
                next_id += 1;
            }
        }
        for name in params.keys() {
            if !name_to_id.contains_key(name) {
                name_to_id.insert(name.clone(), next_id);
                next_id += 1;
            }
        }
        for node in &nodes {
            for name in &node.inputs {
                if !name.is_empty() && !name_to_id.contains_key(name) {
                    name_to_id.insert(name.clone(), next_id);
                    next_id += 1;
                }
            }
            for name in &node.outputs {
                if !name.is_empty() && !name_to_id.contains_key(name) {
                    name_to_id.insert(name.clone(), next_id);
                    next_id += 1;
                }
            }
        }
        let total_tensors = next_id;

        let mut scalar_cache = HashMap::new();
        for i in 0..=32i64 {
            scalar_cache.insert(i, Tensor::from_scalar(i as f32));
        }

        let mut transposed_weights = HashMap::new();
        for node in &nodes {
            if node.op_code == OpCode::Conv {
                let weight_key = format!("{}.weight", node.name);
                if let Some(w) = params.get(&weight_key) {
                    let w_shape = w.shape_ref();
                    if w_shape.len() >= 4 {
                        let oc = w_shape[0] as usize;
                        let ic = w_shape[1] as usize;
                        let kh = w_shape[2] as usize;
                        let kw = w_shape[3] as usize;

                        if kh == 1 && kw == 1 {
                            // 1x1: pre-transpose [oc, ic] -> [ic, oc]
                            let reshaped = w.reshape(vec![oc as i64, ic as i64]);
                            let transposed = reshaped.transpose(0, 1).contiguous();
                            transposed_weights.insert(weight_key.clone(), transposed);
                        } else if kh == 3 && kw == 3 {
                            // 3x3: pre-transpose to [ic*9, oc] (WT_TRANS_BUF layout)
                            let w_data = w.as_f32_slice();
                            let mut t_data = vec![0.0f32; ic * 9 * oc];
                            for ic_idx in 0..ic {
                                for kh_idx in 0..3 {
                                    for kw_idx in 0..3 {
                                        let k_idx = ic_idx * 9 + kh_idx * 3 + kw_idx;
                                        for oc_idx in 0..oc {
                                            let w_idx = ((oc_idx * ic + ic_idx) * 3 + kh_idx) * 3 + kw_idx;
                                            t_data[k_idx * oc + oc_idx] = w_data[w_idx];
                                        }
                                    }
                                }
                            }
                            let transposed = Tensor::from_vec(t_data, vec![(ic * 9) as i64, oc as i64]);
                            transposed_weights.insert(weight_key.clone(), transposed);
                        }
                    }
                }
            }
        }

        DAGExecutor {
            nodes,
            params,
            transposed_weights,
            scalar_cache,
            input_names,
            output_names,
            name_to_id,
            total_tensors,
        }
    }

    pub fn forward(&self, inputs: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        let mut buffer: Vec<Option<Tensor>> = (0..self.total_tensors).map(|_| None).collect();

        for name in &self.input_names {
            if let Some(id) = self.name_to_id.get(name) {
                let t = if let Some(t) = inputs.get(name) {
                    t.clone()
                } else if let Some(p) = self.params.get(name) {
                    p.clone()
                } else {
                    continue;
                };
                buffer[*id] = Some(t);
            }
        }

        for node in &self.nodes {
            let mut args: Vec<Tensor> = Vec::new();
            for in_name in &node.inputs {
                if in_name.is_empty() {
                    continue;
                }
                if let Some(id) = self.name_to_id.get(in_name) {
                    if let Some(Some(t)) = buffer.get(*id) {
                        args.push(t.clone());
                    } else if let Some(p) = self.params.get(in_name) {
                        args.push(p.clone());
                    }
                } else if let Some(p) = self.params.get(in_name) {
                    args.push(p.clone());
                }
            }
            if args.is_empty() && node.op_code != OpCode::Constant && node.op_code != OpCode::ConstantOfShape {
                continue;
            }
            let result = match node.op_code {
                OpCode::Relu => self.dispatch_unary("relu", &args),
                OpCode::Sigmoid => self.dispatch_unary("sigmoid", &args),
                OpCode::Tanh => self.dispatch_unary("tanh", &args),
                OpCode::Silu => self.dispatch_unary("silu", &args),
                OpCode::Gelu => self.dispatch_unary("gelu", &args),
                OpCode::LeakyRelu => {
                    let slope = node.attrs.get("alpha")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(0.01);
                    let slope_t = Tensor::from_scalar(slope);
                    dispatch("leaky_relu", DispatchKey::Cpu, &[&args[0], &slope_t]).ok()
                }
                OpCode::Elu => {
                    let alpha = node.attrs.get("alpha")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(1.0);
                    let alpha_t = Tensor::from_scalar(alpha);
                    dispatch("elu", DispatchKey::Cpu, &[&args[0], &alpha_t]).ok()
                }
                OpCode::Softmax => {
                    let axis = node.attrs.get("axis")
                        .and_then(|a| a.parse::<i32>().ok())
                        .unwrap_or(1);
                    Some(vec![args[0].softmax(axis)])
                }
                OpCode::Hardswish => self.dispatch_unary("hardswish", &args),
                OpCode::Softplus => {
                    let beta = node.attrs.get("beta")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(1.0);
                    let threshold = node.attrs.get("threshold")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(20.0);
                    let beta_t = Tensor::from_scalar(beta);
                    let threshold_t = Tensor::from_scalar(threshold);
                    dispatch("softplus", DispatchKey::Cpu, &[&args[0], &beta_t, &threshold_t]).ok()
                }

                OpCode::Conv => self.dispatch_conv(node, &args),
                OpCode::Gemm => self.dispatch_gemm(node, &args),
                OpCode::BatchNorm => {
                    self.dispatch_batch_norm(node, &args)
                }
                OpCode::MaxPool => self.dispatch_max_pool(node, &args),
                OpCode::AvgPool => self.dispatch_avg_pool(node, &args),
                OpCode::GlobalAvgPool => {
                    let x = &args[0];
                    let shape = x.shape_ref();
                    let h = shape[2];
                    let _w = shape[3];
                    let k_t = Tensor::from_scalar(h as f32);
                    let s_t = Tensor::from_scalar(1.0f32);
                    let p_t = Tensor::from_scalar(0.0f32);
                    dispatch("avg_pool2d", DispatchKey::Cpu, &[x, &k_t, &s_t, &p_t]).ok()
                }

                OpCode::Add => Some(vec![args[0].add(&args[1])]),
                OpCode::Sub => Some(vec![args[0].sub(&args[1])]),
                OpCode::Mul => Some(vec![args[0].mul(&args[1])]),
                OpCode::Div => Some(vec![args[0].div(&args[1])]),
                OpCode::MatMul => Some(vec![args[0].matmul(&args[1])]),
                OpCode::Pow => {
                    dispatch("pow", DispatchKey::Cpu, &[&args[0], &args[1]]).ok()
                }

                OpCode::Exp => Some(vec![args[0].exp()]),
                OpCode::Sqrt => Some(vec![args[0].sqrt()]),
                OpCode::Neg => Some(vec![-&args[0]]),
                OpCode::Log => Some(vec![args[0].ln()]),
                OpCode::Abs => Some(vec![args[0].abs()]),

                OpCode::Reshape => self.dispatch_reshape(node, &args),
                OpCode::Flatten => self.dispatch_flatten(node, &args),
                OpCode::Transpose => self.dispatch_transpose(node, &args),
                OpCode::Concat => self.dispatch_concat(node, &args),
                OpCode::Squeeze => self.dispatch_squeeze(node, &args),
                OpCode::Unsqueeze => self.dispatch_unsqueeze(node, &args),
                OpCode::Slice => {
                    let x = &args[0];

                    let starts_str = node.attrs.get("starts");
                    let ends_str = node.attrs.get("ends");

                    let slice_res: Option<Vec<Tensor>> =
                    if let (Some(starts_str), Some(ends_str)) = (starts_str, ends_str) {
                        let starts: Vec<i64> = parse_int_list(starts_str);
                        let ends: Vec<i64> = parse_int_list(ends_str);
                        let axes: Vec<i64> = node.attrs.get("axes")
                            .map(|s| parse_int_list(s))
                            .unwrap_or_else(|| (0..starts.len() as i64).collect());
                        let steps: Vec<i64> = node.attrs.get("steps")
                            .map(|s| parse_int_list(s))
                            .unwrap_or_else(|| vec![1; starts.len()]);
                        let mut result = x.clone();
                        for (i, ((&ax, &st), &en)) in axes.iter().zip(starts.iter()).zip(ends.iter()).enumerate() {
                            let step = *steps.get(i).unwrap_or(&1);
                            result = result.slice(ax as usize, st, en, step);
                        }
                        Some(vec![result])
                    } else if args.len() >= 3 {
                        let starts: Vec<i64> = args[1].as_f32_slice().iter().map(|&v| v as i64).collect();
                        let ends: Vec<i64> = args[2].as_f32_slice().iter().map(|&v| v as i64).collect();
                        let axes: Vec<i64> = if args.len() >= 4 {
                            args[3].as_f32_slice().iter().map(|&v| v as i64).collect()
                        } else {
                            (0..starts.len() as i64).collect()
                        };
                        let steps: Vec<i64> = if args.len() >= 5 {
                            args[4].as_f32_slice().iter().map(|&v| v as i64).collect()
                        } else {
                            vec![1; starts.len()]
                        };
                        let mut result = x.clone();
                        for (i, ((&ax, &st), &en)) in axes.iter().zip(starts.iter()).zip(ends.iter()).enumerate() {
                            let step = *steps.get(i).unwrap_or(&1);
                            result = result.slice(ax as usize, st, en, step);
                        }
                        Some(vec![result])
                    } else {
                        Some(vec![x.clone()])
                    };
                    slice_res
                }

                OpCode::Pad => {
                    let x = &args[0];
                    let pads_str = node.attrs.get("pads");
                    let pads: Vec<i64> = if let Some(pads_str) = pads_str {
                        parse_int_list(pads_str)
                    } else if args.len() >= 2 {
                        args[1].as_f32_slice().iter().map(|&v| v as i64).collect()
                    } else {
                        vec![]
                    };
                    if pads.len() == 8 {
                        let shape = x.shape_ref();
                        let pad_t = pads[2] as usize;
                        let pad_b = pads[6] as usize;
                        let pad_l = pads[3] as usize;
                        let pad_r = pads[7] as usize;
                        let new_h = shape[2] as usize + pad_t + pad_b;
                        let new_w = shape[3] as usize + pad_l + pad_r;
                        let mut out_data = vec![0.0f32; (shape[0] * shape[1] * new_h as i64 * new_w as i64) as usize];
                        let x_data = x.as_f32_slice();
                        for b in 0..shape[0] as usize {
                            for c in 0..shape[1] as usize {
                                for h in 0..shape[2] as usize {
                                    for w in 0..shape[3] as usize {
                                        let src_idx = ((b * shape[1] as usize + c) * shape[2] as usize + h) * shape[3] as usize + w;
                                        let dst_idx = ((b * shape[1] as usize + c) * new_h + (h + pad_t)) * new_w + (w + pad_l);
                                        out_data[dst_idx] = x_data[src_idx];
                                    }
                                }
                            }
                        }
                        Some(vec![Tensor::from_vec(out_data, vec![shape[0], shape[1], new_h as i64, new_w as i64])])
                    } else {
                        Some(vec![x.clone()])
                    }
                }

                OpCode::Identity => Some(vec![args[0].clone()]),

                OpCode::Shape => {
                    let x = &args[0];
                    let shape = x.shape_ref();
                    let shape_f32: Vec<f32> = shape.iter().map(|&d| d as f32).collect();
                    let dims = vec![shape.len() as i64];
                    Some(vec![Tensor::from_vec(shape_f32, dims)])
                }

                OpCode::Cast => {
                    let to_dtype_val = node.attrs.get("to")
                        .and_then(|a| a.parse::<i32>().ok())
                        .unwrap_or(1);
                    let target_dtype = match to_dtype_val {
                        1 => DType::F32,
                        9 => DType::Bool,
                        10 => DType::I32,
                        11 => DType::I64,
                        _ => DType::F32,
                    };
                    Some(vec![args[0].to_dtype(target_dtype)])
                }

                OpCode::TopK => {
                    let k_val = node.attrs.get("k")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(1);
                    let axis = node.attrs.get("axis")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(-1);

                    let rank = args[0].shape_ref().len() as i64;
                    let axis = if axis < 0 { axis + rank } else { axis };

                    if k_val == 1 && axis >= 0 {
                        let values = args[0].max(axis as i32, false);
                        let indices = args[0].clone();
                        Some(vec![values, indices])
                    } else {
                        Some(vec![args[0].clone(), args[0].clone()])
                    }
                }

                OpCode::ReduceMean => {
                    let axes_str = node.attrs.get("axes");
                    let keepdim = node.attrs.get("keepdims")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(1) != 0;
                    if let Some(axes_str) = axes_str {
                        let axes: Vec<i64> = parse_int_list(axes_str);
                        let mut result = args[0].clone();
                        for &ax in axes.iter().rev() {
                            result = result.mean(ax as i32, keepdim);
                        }
                        Some(vec![result])
                    } else {
                        let shape = args[0].shape_ref().to_vec();
                        let mut result = args[0].clone();
                        for ax in (0..shape.len() as i64).rev() {
                            result = result.mean(ax as i32, true);
                        }
                        if !keepdim {
                            result = result.reshape(vec![1]);
                        }
                        Some(vec![result])
                    }
                }

                OpCode::ReduceSum => {
                    let axes_str = node.attrs.get("axes");
                    let keepdim = node.attrs.get("keepdims")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(1) != 0;
                    if let Some(axes_str) = axes_str {
                        let axes: Vec<i64> = parse_int_list(axes_str);
                        let mut result = args[0].clone();
                        for &ax in axes.iter().rev() {
                            result = result.sum(ax as i32, keepdim);
                        }
                        Some(vec![result])
                    } else {
                        let shape = args[0].shape_ref().to_vec();
                        let mut result = args[0].clone();
                        for ax in (0..shape.len() as i64).rev() {
                            result = result.sum(ax as i32, true);
                        }
                        if !keepdim {
                            result = result.reshape(vec![1]);
                        }
                        Some(vec![result])
                    }
                }

                OpCode::HardSigmoid => {
                    let alpha = node.attrs.get("alpha")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(0.2);
                    let beta = node.attrs.get("beta")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(0.5);
                    let result = args[0].clone().mul(&Tensor::from_scalar(alpha));
                    let result = result.add(&Tensor::from_scalar(beta));
                    Some(vec![result.clamp(0.0, 1.0)])
                }

                OpCode::Prelu => {
                    dispatch("prelu", DispatchKey::Cpu, &[&args[0], &args[1]]).ok()
                }

                OpCode::LayerNorm => {
                    let eps = node.attrs.get("epsilon")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(1e-5);
                    let _axis = node.attrs.get("axis")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(-1);
                    let _normalized_shape = args[0].shape_ref().to_vec();
                    let weight = if args.len() > 1 { Some(&args[1]) } else { None };
                    let bias = if args.len() > 2 { Some(&args[2]) } else { None };

                    let eps_t = Tensor::from_scalar(eps);
                    let mut dispatch_args = vec![&args[0], &eps_t];
                    if let Some(w) = weight { dispatch_args.push(w); }
                    if let Some(b) = bias { dispatch_args.push(b); }

                    match dispatch("layer_norm", DispatchKey::Cpu, &dispatch_args) {
                        Ok(r) => Some(r),
                        Err(_) => Some(vec![args[0].clone()]),
                    }
                }

                OpCode::Erf => {
                    let x = &args[0];
                    let data = x.as_f32_slice();
                    let result_data: Vec<f32> = data.iter().map(|&v| {
                        let sign = if v >= 0.0 { 1.0f32 } else { -1.0f32 };
                        let x_abs = v.abs();
                        let t = 1.0 / (1.0 + 0.3275911 * x_abs);
                        let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x_abs * x_abs).exp();
                        sign * y
                    }).collect();
                    let shape = x.shape_ref().to_vec();
                    Some(vec![Tensor::from_vec(result_data, shape)])
                }

                OpCode::Gather => {
                    let x = &args[0];
                    let axis = node.attrs.get("axis")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(0);
                    if args.len() >= 2 {
                        Some(vec![x.gather(axis, &args[1])])
                    } else {
                        Some(vec![x.clone()])
                    }
                }

                OpCode::Tile => {
                    let x = &args[0];
                    if args.len() >= 2 {
                        let repeats_data = args[1].as_f32_slice();
                        let repeats: Vec<i64> = repeats_data.iter().map(|&v| v as i64).collect();
                        Some(vec![x.repeat(&repeats)])
                    } else {
                        Some(vec![x.clone()])
                    }
                }

                OpCode::Where => {
                    if args.len() >= 3 {
                        Some(vec![args[0].where_tensor(&args[1], &args[2])])
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }

                OpCode::Expand => {
                    let x = &args[0];
                    if args.len() >= 2 {
                        let shape_data = args[1].as_f32_slice();
                        let shape: Vec<i64> = shape_data.iter().map(|&v| v as i64).collect();
                        Some(vec![x.expand(shape)])
                    } else {
                        Some(vec![x.clone()])
                    }
                }

                OpCode::Split => {
                    let num_outputs = node.outputs.len();
                    let x = &args[0];
                    let axis = node.attrs.get("axis")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(1) as usize;
                    let shape = x.shape_ref();
                    let total_dim = shape[axis] as usize;

                    let splits: Vec<usize> = if args.len() >= 2 {
                        let splits_data = args[1].as_f32_slice();
                        splits_data.iter().map(|&v| v as usize).collect()
                    } else if let Some(s) = node.attrs.get("split") {
                        parse_int_list(s).iter().map(|&v| v as usize).collect()
                    } else {
                        let n = num_outputs.max(1);
                        let part = total_dim / n;
                        vec![part; n]
                    };

                    let mut results = Vec::new();
                    let mut start = 0i64;
                    for &sz in &splits {
                        let end = start + sz as i64;
                        let sliced = x.slice(axis, start, end, 1);
                        results.push(sliced);
                        start = end;
                    }
                    Some(results)
                }

                OpCode::Constant => {
                    let key = node.outputs[0].clone();
                    let value_key = format!("{}.value", node.name);
                    if let Some(t) = self.params.get(&key) {
                        Some(vec![t.clone()])
                    } else if let Some(t) = self.params.get(&value_key) {
                        Some(vec![t.clone()])
                    } else {
                        Some(vec![Tensor::from_scalar(0.0f32)])
                    }
                }

                OpCode::NonMaxSuppression => {
                    Some(vec![args[0].clone()])
                }

                OpCode::Resize => {
                    let x = &args[0];
                    let mode = node.attrs.get("mode").map(|s| s.to_lowercase()).unwrap_or_else(|| "nearest".to_string());
                    let scales_str = node.attrs.get("scales");

                    let scale_factor = if let Some(scales_str) = scales_str {
                        let scales: Vec<f64> = scales_str.trim_matches(|c| c == '[' || c == ']')
                            .split(',')
                            .filter_map(|s| s.trim().parse::<f64>().ok())
                            .collect();
                        if scales.len() >= 4 {
                            scales[2].max(scales[3])
                        } else if scales.len() >= 2 {
                            scales[scales.len() - 1]
                        } else if scales.len() == 1 {
                            scales[0]
                        } else {
                            2.0
                        }
                    } else if args.len() >= 2 {
                        let scales_data = args[1].as_f32_slice();
                        if scales_data.len() >= 4 {
                            scales_data[2].max(scales_data[3]) as f64
                        } else if scales_data.len() >= 2 {
                            scales_data[scales_data.len() - 1] as f64
                        } else if scales_data.len() == 1 {
                            scales_data[0] as f64
                        } else {
                            2.0
                        }
                    } else {
                        2.0
                    };

                    let upsampler = crate::nn::upsample::Upsample::new(scale_factor, mode);
                    Some(vec![upsampler.forward(x)])
                }

                OpCode::FusedConvBn => {
                    if args.len() >= 2 {
                        dispatch("fused_conv_bn", DispatchKey::Cpu, &[&args[0], &args[1]]).ok()
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }
                OpCode::FusedConvBnRelu => {
                    if args.len() >= 2 {
                        dispatch("fused_conv_bn_relu", DispatchKey::Cpu, &[&args[0], &args[1]]).ok()
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }
                OpCode::FusedConvBnGelu => {
                    if args.len() >= 2 {
                        dispatch("fused_conv_bn_gelu", DispatchKey::Cpu, &[&args[0], &args[1]]).ok()
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }
                OpCode::Clip => {
                    let x = &args[0];
                    let min_val = node.attrs.get("min")
                        .and_then(|a| a.parse::<f32>().ok());
                    let max_val = node.attrs.get("max")
                        .and_then(|a| a.parse::<f32>().ok());
                    let result = match (min_val, max_val) {
                        (Some(min), Some(max)) => x.clamp(min, max),
                        (Some(min), None) => x.clamp(min, f32::MAX),
                        (None, Some(max)) => x.clamp(f32::MIN, max),
                        (None, None) => x.clone(),
                    };
                    Some(vec![result])
                }

                OpCode::Ceil => {
                    let x = &args[0];
                    let data = x.as_f32_slice();
                    let result: Vec<f32> = data.iter().map(|&v| v.ceil()).collect();
                    Some(vec![Tensor::from_vec(result, x.shape_ref().to_vec())])
                }
                OpCode::Floor => {
                    let x = &args[0];
                    let data = x.as_f32_slice();
                    let result: Vec<f32> = data.iter().map(|&v| v.floor()).collect();
                    Some(vec![Tensor::from_vec(result, x.shape_ref().to_vec())])
                }
                OpCode::Round => {
                    let x = &args[0];
                    let data = x.as_f32_slice();
                    let result: Vec<f32> = data.iter().map(|&v| v.round()).collect();
                    Some(vec![Tensor::from_vec(result, x.shape_ref().to_vec())])
                }
                OpCode::Sign => {
                    self.dispatch_unary("sign", &args)
                }
                OpCode::Reciprocal => {
                    let x = &args[0];
                    let data = x.as_f32_slice();
                    let result: Vec<f32> = data.iter().map(|&v| if v == 0.0 { 0.0 } else { 1.0 / v }).collect();
                    Some(vec![Tensor::from_vec(result, x.shape_ref().to_vec())])
                }
                OpCode::IsNan => {
                    let x = &args[0];
                    let data = x.as_f32_slice();
                    let result: Vec<f32> = data.iter().map(|&v| if v.is_nan() { 1.0 } else { 0.0 }).collect();
                    Some(vec![Tensor::from_vec(result, x.shape_ref().to_vec())])
                }
                OpCode::IsInf => {
                    let x = &args[0];
                    let data = x.as_f32_slice();
                    let result: Vec<f32> = data.iter().map(|&v| if v.is_infinite() { 1.0 } else { 0.0 }).collect();
                    Some(vec![Tensor::from_vec(result, x.shape_ref().to_vec())])
                }

                OpCode::Not => {
                    dispatch("logical_not", DispatchKey::Cpu, &[&args[0]]).ok()
                }

                OpCode::And => {
                    let a = args[0].as_f32_slice();
                    let b = args[1].as_f32_slice();

                    let shape = args[0].shape_ref().to_vec();
                    let result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| if x != 0.0 && y != 0.0 { 1.0 } else { 0.0 }).collect();
                    Some(vec![Tensor::from_vec(result, shape)])
                }

                OpCode::Or => {
                    let a = args[0].as_f32_slice();
                    let b = args[1].as_f32_slice();

                    let shape = args[0].shape_ref().to_vec();
                    let result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| if x != 0.0 || y != 0.0 { 1.0 } else { 0.0 }).collect();
                    Some(vec![Tensor::from_vec(result, shape)])
                }

                OpCode::Xor => {
                    let a = args[0].as_f32_slice();
                    let b = args[1].as_f32_slice();

                    let shape = args[0].shape_ref().to_vec();
                    let result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| if (x != 0.0) != (y != 0.0) { 1.0 } else { 0.0 }).collect();
                    Some(vec![Tensor::from_vec(result, shape)])
                }

                OpCode::Less => {
                    let a = args[0].as_f32_slice();
                    let b = args[1].as_f32_slice();

                    let shape = args[0].shape_ref().to_vec();
                    let result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| if x < y { 1.0 } else { 0.0 }).collect();
                    Some(vec![Tensor::from_vec(result, shape)])
                }

                OpCode::Greater => {
                    let a = args[0].as_f32_slice();
                    let b = args[1].as_f32_slice();

                    let shape = args[0].shape_ref().to_vec();
                    let result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| if x > y { 1.0 } else { 0.0 }).collect();
                    Some(vec![Tensor::from_vec(result, shape)])
                }

                OpCode::Equal => {
                    let a = args[0].as_f32_slice();
                    let b = args[1].as_f32_slice();

                    let shape = args[0].shape_ref().to_vec();
                    let result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| if x == y { 1.0 } else { 0.0 }).collect();
                    Some(vec![Tensor::from_vec(result, shape)])
                }

                OpCode::CumSum => {
                    let dim = node.attrs.get("axis")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(0);
                    let exclusive = node.attrs.get("exclusive")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(0) != 0;
                    let reverse = node.attrs.get("reverse")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(0) != 0;
                    Some(vec![args[0].cumsum(dim, exclusive, reverse)])
                }

                OpCode::OneHot => {
                    let indices = &args[0];
                    let depth = if let Some(d) = node.attrs.get("depth") {
                        d.parse::<i64>().unwrap_or(1)
                    } else if args.len() >= 2 {
                        args[1].as_f32_slice()[0] as i64
                    } else {
                        1
                    };
                    let indices_shape = indices.shape_ref().to_vec();
                    let mut out_shape = indices_shape.clone();
                    out_shape.push(depth);
                    let out_size = out_shape.iter().product::<i64>() as usize;
                    let indices_data = indices.as_f32_slice();
                    let mut out_data = vec![0.0f32; out_size];
                    for i in 0..indices_data.len() {
                        let idx = indices_data[i] as i64;
                        if idx >= 0 && idx < depth {
                            out_data[i * depth as usize + idx as usize] = 1.0;
                        }
                    }
                    Some(vec![Tensor::from_vec(out_data, out_shape)])
                }

                OpCode::GatherNd => {
                    if args.len() >= 2 {
                        let indices = &args[1];
                        let indices_shape = indices.shape_ref().to_vec();
                        if indices_shape.len() == 1 {
                            Some(vec![args[0].gather(0, &args[1])])
                        } else {
                            Some(vec![args[0].gather(0, &args[1])])
                        }
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }

                OpCode::ScatterNd => {
                    Some(vec![args[0].clone()])
                }

                OpCode::Compress => {
                    Some(vec![args[0].clone()])
                }

                OpCode::DepthToSpace => {
                    let x = &args[0];
                    let blocksize = node.attrs.get("blocksize")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(2);
                    let _mode = node.attrs.get("mode")
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "DCR".to_string());
                    let shape = x.shape_ref().to_vec();
                    if shape.len() == 4 {
                        let n = shape[0];
                        let c = shape[1];
                        let h = shape[2];
                        let w = shape[3];
                        let bs = blocksize;
                        let out_c = c / (bs * bs);
                        let reshaped = x.reshape(vec![n, bs, bs, out_c, h, w]);
                        let permuted = reshaped.permute(vec![0, 3, 4, 1, 5, 2]);
                        let result = permuted.reshape(vec![n, out_c, h * bs, w * bs]);
                        Some(vec![result])
                    } else {
                        Some(vec![x.clone()])
                    }
                }

                OpCode::SpaceToDepth => {
                    let x = &args[0];
                    let blocksize = node.attrs.get("blocksize")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(2);
                    let shape = x.shape_ref().to_vec();
                    if shape.len() == 4 {
                        let n = shape[0];
                        let c = shape[1];
                        let h = shape[2];
                        let w = shape[3];
                        let bs = blocksize;
                        let out_h = h / bs;
                        let out_w = w / bs;
                        let reshaped = x.reshape(vec![n, c, out_h, bs, out_w, bs]);
                        let permuted = reshaped.permute(vec![0, 3, 5, 1, 2, 4]);
                        let result = permuted.reshape(vec![n, c * bs * bs, out_h, out_w]);
                        Some(vec![result])
                    } else {
                        Some(vec![x.clone()])
                    }
                }

                OpCode::EyeLike => {
                    let x = &args[0];
                    let shape = x.shape_ref().to_vec();
                    let k = node.attrs.get("k")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(0);
                    let n = shape.get(0).copied().unwrap_or(1);
                    let m = shape.get(1).copied().unwrap_or(n);
                    let mut data = vec![0.0f32; (n * m) as usize];
                    let offset = k.abs() as usize;
                    for i in 0..n as usize {
                        let j = if k >= 0 { i + offset } else { i };
                        let idx = if k >= 0 {
                            if j < m as usize { i * m as usize + j } else { continue; }
                        } else {
                            if i < m as usize { (i + offset) * m as usize + i } else { continue; }
                        };
                        data[idx] = 1.0;
                    }
                    Some(vec![Tensor::from_vec(data, vec![n, m])])
                }

                OpCode::ConvTranspose => self.dispatch_conv_transpose(node, &args),

                OpCode::InstanceNorm => {
                    let x = &args[0];
                    let shape = x.shape_ref().to_vec();
                    let eps = node.attrs.get("epsilon")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(1e-5);
                    let weight = if let Some(w) = self.params.get(&format!("{}.weight", node.name)) {
                        w.clone()
                    } else if args.len() > 1 { args[1].clone() } else { Tensor::from_scalar(1.0) };
                    let bias = if let Some(b) = self.params.get(&format!("{}.bias", node.name)) {
                        b.clone()
                    } else if args.len() > 2 { args[2].clone() } else { Tensor::from_scalar(0.0) };

                    if shape.len() == 4 {
                        let n = shape[0] as usize;
                        let c = shape[1] as usize;
                        let h = shape[2] as usize;
                        let w = shape[3] as usize;
                        let data = x.as_f32_slice();
                        let weight_data = weight.as_f32_slice();
                        let bias_data = bias.as_f32_slice();
                        let mut out_data = vec![0.0f32; n * c * h * w];
                        for ni in 0..n {
                            for ci in 0..c {
                                let start = (ni * c + ci) * h * w;
                                let mut sum = 0.0f32;
                                for i in start..start + h * w {
                                    sum += data[i];
                                }
                                let mean = sum / (h * w) as f32;
                                let mut var_sum = 0.0f32;
                                for i in start..start + h * w {
                                    let diff = data[i] - mean;
                                    var_sum += diff * diff;
                                }
                                let variance = var_sum / (h * w) as f32;
                                let inv_std = 1.0 / (variance + eps).sqrt();
                                let g = weight_data[ci.min(weight_data.len() - 1)];
                                let b = bias_data[ci.min(bias_data.len() - 1)];
                                for i in start..start + h * w {
                                    out_data[i] = (data[i] - mean) * inv_std * g + b;
                                }
                            }
                        }
                        Some(vec![Tensor::from_vec(out_data, shape)])
                    } else {
                        Some(vec![x.clone()])
                    }
                }

                OpCode::LogSoftmax => {
                    let axis = node.attrs.get("axis")
                        .and_then(|a| a.parse::<i32>().ok())
                        .unwrap_or(1);
                    Some(vec![args[0].log_softmax(axis)])
                }

                OpCode::Selu => {
                    let x = &args[0];
                    let alpha = node.attrs.get("alpha")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(1.6732632423543772);
                    let gamma = node.attrs.get("gamma")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(1.0507009873554805);
                    let data = x.as_f32_slice();
                    let result: Vec<f32> = data.iter().map(|&v| {
                        if v > 0.0 { gamma * v } else { gamma * alpha * (v.exp() - 1.0) }
                    }).collect();
                    Some(vec![Tensor::from_vec(result, x.shape_ref().to_vec())])
                }

                OpCode::RmsNorm => {
                    let eps = node.attrs.get("epsilon")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(1e-5);
                    let eps_t = Tensor::from_scalar(eps);
                    let mut dispatch_args = vec![&args[0], &eps_t];
                    if args.len() > 1 { dispatch_args.push(&args[1]); }
                    dispatch("rms_norm", DispatchKey::Cpu, &dispatch_args).ok()
                }

                OpCode::Range => {
                    let start = if args.len() >= 1 { args[0].as_f32_slice()[0] } else { 0.0f32 };
                    let limit = if args.len() >= 2 { args[1].as_f32_slice()[0] } else { 1.0f32 };
                    let step = if args.len() >= 3 { args[2].as_f32_slice()[0] } else { 1.0f32 };
                    let mut data = Vec::new();
                    let mut v = start;
                    if step > 0.0 {
                        while v < limit {
                            data.push(v);
                            v += step;
                        }
                    } else if step < 0.0 {
                        while v > limit {
                            data.push(v);
                            v += step;
                        }
                    }
                    let n = data.len() as i64;
                    Some(vec![Tensor::from_vec(data, vec![n])])
                }

                OpCode::RandomNormal => {
                    let shape_str = node.attrs.get("shape");
                    let shape: Vec<i64> = if let Some(s) = shape_str {
                        s.trim_matches(|c| c == '[' || c == ']')
                            .split(',')
                            .filter_map(|x| x.trim().parse::<i64>().ok())
                            .collect()
                    } else {
                        args[0].shape_ref().to_vec()
                    };
                    let _mean = node.attrs.get("mean").and_then(|a| a.parse::<f32>().ok()).unwrap_or(0.0);
                    let _scale = node.attrs.get("scale").and_then(|a| a.parse::<f32>().ok()).unwrap_or(1.0);
                    let shape_t = Tensor::empty(shape, DType::F32, Device::Cpu);
                    dispatch("randn", DispatchKey::Cpu, &[&shape_t]).ok()
                }

                OpCode::RandomUniform => {
                    let shape_str = node.attrs.get("shape");
                    let shape: Vec<i64> = if let Some(s) = shape_str {
                        s.trim_matches(|c| c == '[' || c == ']')
                            .split(',')
                            .filter_map(|x| x.trim().parse::<i64>().ok())
                            .collect()
                    } else {
                        args[0].shape_ref().to_vec()
                    };
                    let _low = node.attrs.get("low").and_then(|a| a.parse::<f32>().ok()).unwrap_or(0.0);
                    let _high = node.attrs.get("high").and_then(|a| a.parse::<f32>().ok()).unwrap_or(1.0);
                    let shape_t = Tensor::empty(shape, DType::F32, Device::Cpu);
                    dispatch("rand", DispatchKey::Cpu, &[&shape_t]).ok()
                }

                OpCode::ArgMax => {
                    let x = &args[0];
                    let axis = node.attrs.get("axis").and_then(|a| a.parse::<i64>().ok());
                    let keepdims = node.attrs.get("keepdims")
                        .and_then(|a| a.parse::<i64>().ok()).unwrap_or(1) != 0;
                    let shape = x.shape_ref().to_vec();
                    let data = x.as_f32_slice();
                    let rank = shape.len() as i64;

                    let result = if let Some(ax) = axis {
                        let ax = if ax < 0 { (ax + rank) as usize } else { ax as usize };
                        let outer: usize = shape[..ax].iter().product::<i64>() as usize;
                        let dim = shape[ax] as usize;
                        let inner: usize = shape[ax + 1..].iter().product::<i64>() as usize;
                        let mut out_data = vec![0.0f32; outer * inner];
                        for o in 0..outer {
                            for i in 0..inner {
                                let mut best_idx = 0usize;
                                let mut best_val = f32::NEG_INFINITY;
                                for d in 0..dim {
                                    let idx = (o * dim + d) * inner + i;
                                    if data[idx] > best_val {
                                        best_val = data[idx];
                                        best_idx = d;
                                    }
                                }
                                out_data[o * inner + i] = best_idx as f32;
                            }
                        }
                        let mut out_shape: Vec<i64> = shape.clone();
                        out_shape.remove(ax);
                        if keepdims { out_shape.insert(ax, 1); }
                        Tensor::from_vec(out_data, out_shape)
                    } else {
                        let mut best_idx = 0usize;
                        let mut best_val = f32::NEG_INFINITY;
                        for (i, &v) in data.iter().enumerate() {
                            if v > best_val { best_val = v; best_idx = i; }
                        }
                        Tensor::from_vec(vec![best_idx as f32], vec![1])
                    };
                    Some(vec![result])
                }

                OpCode::ArgMin => {
                    let x = &args[0];
                    let axis = node.attrs.get("axis").and_then(|a| a.parse::<i64>().ok());
                    let keepdims = node.attrs.get("keepdims")
                        .and_then(|a| a.parse::<i64>().ok()).unwrap_or(1) != 0;
                    let shape = x.shape_ref().to_vec();
                    let data = x.as_f32_slice();
                    let rank = shape.len() as i64;

                    let result = if let Some(ax) = axis {
                        let ax = if ax < 0 { (ax + rank) as usize } else { ax as usize };
                        let outer: usize = shape[..ax].iter().product::<i64>() as usize;
                        let dim = shape[ax] as usize;
                        let inner: usize = shape[ax + 1..].iter().product::<i64>() as usize;
                        let mut out_data = vec![0.0f32; outer * inner];
                        for o in 0..outer {
                            for i in 0..inner {
                                let mut best_idx = 0usize;
                                let mut best_val = f32::INFINITY;
                                for d in 0..dim {
                                    let idx = (o * dim + d) * inner + i;
                                    if data[idx] < best_val {
                                        best_val = data[idx];
                                        best_idx = d;
                                    }
                                }
                                out_data[o * inner + i] = best_idx as f32;
                            }
                        }
                        let mut out_shape: Vec<i64> = shape.clone();
                        out_shape.remove(ax);
                        if keepdims { out_shape.insert(ax, 1); }
                        Tensor::from_vec(out_data, out_shape)
                    } else {
                        let mut best_idx = 0usize;
                        let mut best_val = f32::INFINITY;
                        for (i, &v) in data.iter().enumerate() {
                            if v < best_val { best_val = v; best_idx = i; }
                        }
                        Tensor::from_vec(vec![best_idx as f32], vec![1])
                    };
                    Some(vec![result])
                }

                OpCode::Min => {
                    if args.len() >= 2 {
                        Some(vec![args[0].minimum(&args[1])])
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }

                OpCode::Max => {
                    if args.len() >= 2 {
                        Some(vec![args[0].maximum(&args[1])])
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }

                OpCode::Swish => self.dispatch_unary("silu", &args),

                OpCode::Reverse => {
                    let x = &args[0];
                    let axes_str = node.attrs.get("axes");
                    let data = x.as_f32_slice();
                    let shape = x.shape_ref().to_vec();

                    let axes: Vec<usize> = if let Some(s) = axes_str {
                        parse_int_list(s).iter().map(|&v| {
                            if v < 0 { (v + shape.len() as i64) as usize } else { v as usize }
                        }).collect()
                    } else {
                        (0..shape.len()).collect()
                    };

                    let mut out_data = data.to_vec();
                    let rank = shape.len();

                    for &ax in &axes {
                        let ax = ax.min(rank - 1);
                        let dim = shape[ax] as usize;
                        let outer: usize = shape[..ax].iter().product::<i64>() as usize;
                        let inner: usize = shape[ax + 1..].iter().product::<i64>() as usize;

                        let mut new_data = out_data.clone();
                        for o in 0..outer {
                            for i in 0..inner {
                                for d in 0..dim {
                                    let src_idx = (o * dim + d) * inner + i;
                                    let dst_idx = (o * dim + (dim - 1 - d)) * inner + i;
                                    new_data[dst_idx] = out_data[src_idx];
                                }
                            }
                        }
                        out_data = new_data;
                    }
                    Some(vec![Tensor::from_vec(out_data, shape)])
                }

                OpCode::ConstantOfShape => {
                    let shape_arr: Vec<i64> = if !args.is_empty() {
                        args[0].as_f32_slice().iter().map(|&v| v as i64).collect()
                    } else {
                        vec![1]
                    };
                    let value = node.attrs.get("value")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(0.0);
                    let n: usize = shape_arr.iter().product::<i64>() as usize;
                    let data = vec![value; n];
                    Some(vec![Tensor::from_vec(data, shape_arr)])
                }

                OpCode::BiasAdd => {
                    if args.len() >= 2 {
                        let x = &args[0];
                        let bias = &args[1];
                        Some(vec![x.add(bias)])
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }

                OpCode::BiasSub => {
                    if args.len() >= 2 {
                        let x = &args[0];
                        let bias = &args[1];
                        Some(vec![x.sub(bias)])
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }

                _ => {
                    args.first().cloned().map(|t| vec![t])
                }
            };

            if let Some(tensors) = result {
                for (i, name) in node.outputs.iter().enumerate() {
                    if let Some(t) = tensors.get(i) {
                        if let Some(id) = self.name_to_id.get(name) {
                            buffer[*id] = Some(t.clone());
                        }
                    }
                }
            }
        }

        let mut outputs = HashMap::new();
        for name in &self.output_names {
            if let Some(id) = self.name_to_id.get(name) {
                if let Some(Some(t)) = buffer.get(*id) {
                    outputs.insert(name.clone(), t.clone());
                }
            }
        }
        outputs
    }

    // ---- Dispatch helpers ----

    fn dispatch_unary(&self, op_name: &str, args: &[Tensor]) -> Option<Vec<Tensor>> {
        dispatch(op_name, DispatchKey::Cpu, &[&args[0]]).ok()
    }

    fn scalar_tensor(&self, v: i64) -> Option<&Tensor> {
        self.scalar_cache.get(&v)
    }

    fn dispatch_conv(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let weight_name = format!("{}.weight", node.name);
        let weight = self.params.get(&weight_name)?;
        let has_bias = self.params.contains_key(&format!("{}.bias", node.name));

        let stride = node.attrs.get("stride")
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(1);
        let padding = node.attrs.get("padding")
            .and_then(|p| p.parse::<i64>().ok())
            .unwrap_or(0);
        let dilation = node.attrs.get("dilation")
            .and_then(|d| d.parse::<i64>().ok())
            .unwrap_or(1);
        let groups = node.attrs.get("groups")
            .and_then(|g| g.parse::<i64>().ok())
            .unwrap_or(1);

        let stride_t = self.scalar_tensor(stride).cloned().unwrap_or_else(|| Tensor::from_scalar(stride as f32));
        let pad_t = self.scalar_tensor(padding).cloned().unwrap_or_else(|| Tensor::from_scalar(padding as f32));
        let dilation_t = self.scalar_tensor(dilation).cloned().unwrap_or_else(|| Tensor::from_scalar(dilation as f32));
        let groups_t = self.scalar_tensor(groups).cloned().unwrap_or_else(|| Tensor::from_scalar(groups as f32));

        let pre_t_weight = self.transposed_weights.get(&weight_name);

        if has_bias {
            let bias_name = format!("{}.bias", node.name);
            let bias = self.params.get(&bias_name)?;
            if let Some(wt) = pre_t_weight {
                dispatch("conv2d", DispatchKey::Cpu, &[x, weight, bias, &stride_t, &pad_t, &dilation_t, &groups_t, wt]).ok()
            } else {
                dispatch("conv2d", DispatchKey::Cpu, &[x, weight, bias, &stride_t, &pad_t, &dilation_t, &groups_t]).ok()
            }
        } else {
            let zero_bias = Tensor::from_scalar(0.0f32);
            if let Some(wt) = pre_t_weight {
                dispatch("conv2d", DispatchKey::Cpu, &[x, weight, &zero_bias, &stride_t, &pad_t, &dilation_t, &groups_t, wt]).ok()
            } else {
                dispatch("conv2d", DispatchKey::Cpu, &[x, weight, &zero_bias, &stride_t, &pad_t, &dilation_t, &groups_t]).ok()
            }
        }
    }

    fn dispatch_conv_transpose(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let weight_name = format!("{}.weight", node.name);
        let weight = self.params.get(&weight_name)?;
        let has_bias = self.params.contains_key(&format!("{}.bias", node.name));

        let stride = node.attrs.get("stride").and_then(|s| s.parse::<i64>().ok()).unwrap_or(1);
        let padding = node.attrs.get("padding").and_then(|p| p.parse::<i64>().ok()).unwrap_or(0);
        let dilation = node.attrs.get("dilation").and_then(|d| d.parse::<i64>().ok()).unwrap_or(1);
        let groups = node.attrs.get("groups").and_then(|g| g.parse::<i64>().ok()).unwrap_or(1);
        let output_padding = node.attrs.get("output_padding").and_then(|o| o.parse::<i64>().ok()).unwrap_or(0);

        let stride_t = Tensor::from_scalar(stride as f32);
        let pad_t = Tensor::from_scalar(padding as f32);
        let dilation_t = Tensor::from_scalar(dilation as f32);
        let groups_t = Tensor::from_scalar(groups as f32);
        let op_t = Tensor::from_scalar(output_padding as f32);

        if has_bias {
            let bias_name = format!("{}.bias", node.name);
            let bias = self.params.get(&bias_name)?;
            dispatch("conv_transpose2d", DispatchKey::Cpu, &[x, weight, bias, &stride_t, &pad_t, &dilation_t, &groups_t, &op_t]).ok()
        } else {
            let zero_bias = Tensor::from_scalar(0.0f32);
            dispatch("conv_transpose2d", DispatchKey::Cpu, &[x, weight, &zero_bias, &stride_t, &pad_t, &dilation_t, &groups_t, &op_t]).ok()
        }
    }

    fn dispatch_gemm(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let weight_name = format!("{}.weight", node.name);
        let weight = self.params.get(&weight_name)?;

        let bias_name = format!("{}.bias", node.name);
        let has_bias = self.params.contains_key(&bias_name);

        let trans_b = node.attrs.get("trans_b")
            .and_then(|t| t.parse::<i64>().ok())
            .unwrap_or(0);

        let w = if trans_b != 0 {
            weight.transpose(0, 1).contiguous()
        } else {
            weight.clone()
        };

        let result = x.matmul(&w);

        let result = if has_bias {
            let bias = &self.params[&bias_name];
            result.add(bias)
        } else {
            result
        };

        Some(vec![result])
    }

    fn dispatch_batch_norm(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let weight_name = format!("{}.weight", node.name);
        let bias_name = format!("{}.bias", node.name);
        let mean_name = format!("{}.running_mean", node.name);
        let var_name = format!("{}.running_var", node.name);

        let weight = self.params.get(&weight_name)?;
        let bias = self.params.get(&bias_name)?;
        let mean = self.params.get(&mean_name)?;
        let var = self.params.get(&var_name)?;

        dispatch("batch_norm", DispatchKey::Cpu, &[x, weight, bias, mean, var]).ok()
    }

    fn dispatch_max_pool(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let kernel = node.attrs.get("kernel_size")
            .and_then(|k| k.parse::<i64>().ok())
            .unwrap_or(2);
        let stride = node.attrs.get("stride")
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(kernel);
        let padding = node.attrs.get("padding")
            .and_then(|p| p.parse::<i64>().ok())
            .unwrap_or(0);
        let dilation = node.attrs.get("dilation")
            .and_then(|d| d.parse::<i64>().ok())
            .unwrap_or(1);

        let k_t = Tensor::from_scalar(kernel as f32);
        let s_t = Tensor::from_scalar(stride as f32);
        let p_t = Tensor::from_scalar(padding as f32);
        let d_t = Tensor::from_scalar(dilation as f32);

        dispatch("max_pool2d", DispatchKey::Cpu, &[x, &k_t, &s_t, &p_t, &d_t]).ok()
    }

    fn dispatch_avg_pool(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let kernel = node.attrs.get("kernel_size")
            .and_then(|k| k.parse::<i64>().ok())
            .unwrap_or(2);
        let stride = node.attrs.get("stride")
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(kernel);
        let padding = node.attrs.get("padding")
            .and_then(|p| p.parse::<i64>().ok())
            .unwrap_or(0);

        let k_t = Tensor::from_scalar(kernel as f32);
        let s_t = Tensor::from_scalar(stride as f32);
        let p_t = Tensor::from_scalar(padding as f32);

        dispatch("avg_pool2d", DispatchKey::Cpu, &[x, &k_t, &s_t, &p_t]).ok()
    }

    fn dispatch_reshape(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        if args.len() >= 2 {
            let shape_data = args[1].as_f32_slice();
            let shape: Vec<i64> = shape_data.iter().map(|&v| v as i64).collect();
            Some(vec![x.reshape(shape)])
        } else if let Some(shape_str) = node.attrs.get("shape") {
            let shape: Vec<i64> = shape_str.trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|s| s.trim().parse::<i64>().ok())
                .collect();
            Some(vec![x.reshape(shape)])
        } else {
            Some(vec![x.clone()])
        }
    }

    fn dispatch_flatten(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let axis = node.attrs.get("axis")
            .and_then(|a| a.parse::<i64>().ok())
            .unwrap_or(1);
        let shape = x.shape_ref();
        let outer: i64 = shape.iter().take(axis as usize).product();
        let inner: i64 = shape.iter().skip(axis as usize).product();
        Some(vec![x.reshape(vec![outer, inner])])
    }

    fn dispatch_transpose(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let perm_str = node.attrs.get("perm")?;
        let perm: Vec<usize> = perm_str.trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        if perm.len() == 2 {
            Some(vec![x.transpose(perm[0], perm[1])])
        } else if perm.len() == 4 {
            let mut result = x.clone();
            let mut current = vec![0usize, 1, 2, 3];
            for target in 0..4 {
                if current[target] != perm[target] {
                    let swap_idx = current.iter().position(|&c| c == perm[target]).unwrap();
                    result = result.transpose(target, swap_idx);
                    current.swap(target, swap_idx);
                }
            }
            Some(vec![result.contiguous()])
        } else {
            Some(vec![x.clone()])
        }
    }

    fn dispatch_concat(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let axis = node.attrs.get("axis")
            .and_then(|a| a.parse::<i32>().ok())
            .unwrap_or(1);
        Some(vec![Tensor::cat(&args, axis)])
    }

    fn dispatch_squeeze(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let axes_str = node.attrs.get("axes");
        if let Some(axes_str) = axes_str {
            let axes: Vec<usize> = axes_str.trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            let shape = x.shape_ref();
            let mut new_shape: Vec<i64> = Vec::new();
            for (i, &dim) in shape.iter().enumerate() {
                if !axes.contains(&i) || dim != 1 {
                    new_shape.push(dim);
                }
            }
            Some(vec![x.reshape(new_shape)])
        } else {
            let shape = x.shape_ref();
            let new_shape: Vec<i64> = shape.iter().filter(|&&d| d != 1).copied().collect();
            if new_shape.is_empty() {
                Some(vec![x.reshape(vec![1])])
            } else {
                Some(vec![x.reshape(new_shape)])
            }
        }
    }

    fn dispatch_unsqueeze(&self, node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        let x = &args[0];
        let axes_str = node.attrs.get("axes");
        let shape = x.shape_ref();
        if let Some(axes_str) = axes_str {
            // Parse as i64 to handle negative axes (e.g., axes=[-1] means "add at end")
            let mut axes: Vec<i64> = axes_str.trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|s| s.trim().parse::<i64>().ok())
                .collect();
            // Normalize negative axes: convert to positive indices
            // Output rank = input rank + number of axes to insert
            let ndim = shape.len() as i64;
            let num_axes = axes.len() as i64;
            for axis in axes.iter_mut() {
                if *axis < 0 {
                    *axis += ndim + num_axes;
                }
            }
            axes.sort();
            let mut new_shape: Vec<i64> = shape.to_vec();
            for (offset, &axis) in axes.iter().enumerate() {
                new_shape.insert(axis as usize + offset, 1);
            }
            Some(vec![x.reshape(new_shape)])
        } else {
            Some(vec![x.clone()])
        }
    }

    fn dispatch_slice(&self, _node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        Some(vec![args[0].clone()])
    }

    fn dispatch_pad(&self, _node: &DAGNode, args: &[Tensor]) -> Option<Vec<Tensor>> {
        Some(vec![args[0].clone()])
    }
}

impl Module for DAGExecutor {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut inputs = HashMap::new();
        if !self.input_names.is_empty() {
            inputs.insert(self.input_names[0].clone(), x.clone());
        } else {
            inputs.insert("input".to_string(), x.clone());
        }
        let outputs = self.forward(&inputs);
        if !self.output_names.is_empty() {
            outputs.get(&self.output_names[0]).cloned().unwrap_or_else(|| x.clone())
        } else {
            outputs.into_values().next().unwrap_or_else(|| x.clone())
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.params.values().cloned().collect()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.params.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    fn zero_grad(&self) {
        for t in self.params.values() {
            crate::nn::clear_grad(t);
        }
    }

    fn train_mode(&self) {}
    fn eval_mode(&self) {}
    fn is_training(&self) -> bool { false }
}
