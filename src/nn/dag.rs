#![allow(dead_code)]
use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::storage::DType;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// A single node in the DAG computation graph.
#[derive(Clone)]
pub struct DAGNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, String>,
}

/// The DAG graph executor.
pub struct DAGExecutor {
    nodes: Vec<DAGNode>,
    params: HashMap<String, Tensor>,
    input_names: Vec<String>,
    output_names: Vec<String>,
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
        DAGExecutor { nodes, params, input_names, output_names }
    }

    pub fn forward(&self, inputs: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        let mut buffer: HashMap<String, Tensor> = HashMap::new();

        for name in &self.input_names {
            if let Some(t) = inputs.get(name) {
                buffer.insert(name.clone(), t.clone());
            } else if let Some(p) = self.params.get(name) {
                buffer.insert(name.clone(), p.clone());
            }
        }

        for node in &self.nodes {
            let mut args: Vec<Tensor> = Vec::new();
            let mut all_resolved = true;
            for in_name in &node.inputs {
                if let Some(t) = buffer.get(in_name) {
                    args.push(t.clone());
                } else if let Some(p) = self.params.get(in_name) {
                    args.push(p.clone());
                } else {
                    all_resolved = false;
                    break;
                }
            }
            if !all_resolved {
                continue;
            }

            let op_lower = node.op_type.to_lowercase();
            let result = match op_lower.as_str() {
                "relu" => self.dispatch_unary("relu", &args),
                "sigmoid" => self.dispatch_unary("sigmoid", &args),
                "tanh" => self.dispatch_unary("tanh", &args),
                "silu" => self.dispatch_unary("silu", &args),
                "gelu" => self.dispatch_unary("gelu", &args),
                "leakyrelu" | "leaky_relu" => {
                    let slope = node.attrs.get("alpha")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(0.01);
                    let slope_t = Tensor::from_scalar(slope);
                    dispatch("leaky_relu", DispatchKey::Cpu, &[&args[0], &slope_t]).ok()
                }
                "elu" => {
                    let alpha = node.attrs.get("alpha")
                        .and_then(|a| a.parse::<f32>().ok())
                        .unwrap_or(1.0);
                    let alpha_t = Tensor::from_scalar(alpha);
                    dispatch("elu", DispatchKey::Cpu, &[&args[0], &alpha_t]).ok()
                }
                "softmax" => {
                    let axis = node.attrs.get("axis")
                        .and_then(|a| a.parse::<i32>().ok())
                        .unwrap_or(1);
                    Some(vec![args[0].softmax(axis)])
                }
                "hardswish" => self.dispatch_unary("hardswish", &args),
                "softplus" => {
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

                "conv" => self.dispatch_conv(node, &args),
                "gemm" | "linear" => self.dispatch_gemm(node, &args),
                "batchnormalization" | "batch_norm" | "batchnorm2d" => {
                    self.dispatch_batch_norm(node, &args)
                }
                "maxpool" | "max_pool2d" => self.dispatch_max_pool(node, &args),
                "averagepool" | "avg_pool" | "avg_pool2d" => self.dispatch_avg_pool(node, &args),
                "globalaveragepool" | "global_avg_pool" => {
                    let x = &args[0];
                    let shape = x.shape_ref();
                    let h = shape[2];
                    let _w = shape[3];
                    let k_t = Tensor::from_scalar(h as f32);
                    let s_t = Tensor::from_scalar(1.0f32);
                    let p_t = Tensor::from_scalar(0.0f32);
                    dispatch("avg_pool2d", DispatchKey::Cpu, &[x, &k_t, &s_t, &p_t]).ok()
                }

                "add" | "elementwiseadd" => Some(vec![args[0].add(&args[1])]),
                "sub" | "elementwisesub" => Some(vec![args[0].sub(&args[1])]),
                "mul" | "elementwisemul" => Some(vec![args[0].mul(&args[1])]),
                "div" | "elementwisediv" => Some(vec![args[0].div(&args[1])]),
                "matmul" => Some(vec![args[0].matmul(&args[1])]),
                "pow" | "elementwisepow" => {
                    dispatch("pow", DispatchKey::Cpu, &[&args[0], &args[1]]).ok()
                }

                "exp" | "expop" => Some(vec![args[0].exp()]),
                "sqrt" | "sqrtop" => Some(vec![args[0].sqrt()]),
                "neg" | "negop" => Some(vec![-&args[0]]),
                "log" | "logop" => Some(vec![args[0].ln()]),
                "abs" => Some(vec![args[0].abs()]),

                "reshape" => self.dispatch_reshape(node, &args),
                "flatten" => self.dispatch_flatten(node, &args),
                "transpose" => self.dispatch_transpose(node, &args),
                "concat" => self.dispatch_concat(node, &args),
                "squeeze" | "squeezeop" => self.dispatch_squeeze(node, &args),
                "unsqueeze" | "unsqueezeop" => self.dispatch_unsqueeze(node, &args),
                "slice" | "sliceop" => {
                    let x = &args[0];
                    let starts_str = node.attrs.get("starts");
                    let ends_str = node.attrs.get("ends");
                    let axes_str = node.attrs.get("axes");
                    let steps_str = node.attrs.get("steps");

                    if let (Some(starts_str), Some(ends_str)) = (starts_str, ends_str) {
                        let starts: Vec<i64> = parse_int_list(starts_str);
                        let ends: Vec<i64> = parse_int_list(ends_str);
                        let axes: Vec<i64> = axes_str.map(|s| parse_int_list(s)).unwrap_or_else(|| (0..starts.len() as i64).collect());
                        let steps: Vec<i64> = steps_str.map(|s| parse_int_list(s)).unwrap_or_else(|| vec![1; starts.len()]);

                        let mut result = x.clone();
                        for (i, ((&ax, &st), &en)) in axes.iter().zip(starts.iter()).zip(ends.iter()).enumerate() {
                            let step = *steps.get(i).unwrap_or(&1);
                            result = result.slice(ax as usize, st, en, step);
                        }
                        Some(vec![result])
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }

                "pad" => {
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

                "identity" | "identityop" | "dropout" => Some(vec![args[0].clone()]),

                "shapeop" => {
                    let x = &args[0];
                    let shape = x.shape_ref();
                    let shape_f32: Vec<f32> = shape.iter().map(|&d| d as f32).collect();
                    let dims = vec![shape.len() as i64];
                    Some(vec![Tensor::from_vec(shape_f32, dims)])
                }

                "castop" => {
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

                "topkop" => {
                    let mut axis = node.attrs.get("axis")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(-1) as usize;
                    let rank = args[0].shape_ref().len();
                    if axis >= rank { axis = rank - 1; }
                    let keepdim = node.attrs.get("keepdims")
                        .and_then(|a| a.parse::<i64>().ok())
                        .unwrap_or(1) != 0;
                    Some(vec![args[0].max(axis as i32, keepdim), args[0].clone()])
                }

                "reducemean" | "reduce_mean" => {
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

                "reducesum" | "reduce_sum" => {
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

                "hardsigmoid" | "hard_sigmoid" => {
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

                "prelu" => {
                    dispatch("prelu", DispatchKey::Cpu, &[&args[0], &args[1]]).ok()
                }

                "layernormalization" | "layer_norm" | "layernorm" => {
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

                "erf" | "erfop" => {
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

                "gatherop" | "gather" => {
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

                "tileop" | "tile" => {
                    let x = &args[0];
                    if args.len() >= 2 {
                        let repeats_data = args[1].as_f32_slice();
                        let repeats: Vec<i64> = repeats_data.iter().map(|&v| v as i64).collect();
                        Some(vec![x.repeat(&repeats)])
                    } else {
                        Some(vec![x.clone()])
                    }
                }

                "whereop" | "where" => {
                    if args.len() >= 3 {
                        Some(vec![args[0].where_tensor(&args[1], &args[2])])
                    } else {
                        Some(vec![args[0].clone()])
                    }
                }

                "expand" => {
                    let x = &args[0];
                    if args.len() >= 2 {
                        let shape_data = args[1].as_f32_slice();
                        let shape: Vec<i64> = shape_data.iter().map(|&v| v as i64).collect();
                        Some(vec![x.expand(shape)])
                    } else {
                        Some(vec![x.clone()])
                    }
                }

                "split" => {
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

                "constantop" => {
                    if let Some(t) = self.params.get(&node.outputs[0]) {
                        Some(vec![t.clone()])
                    } else {
                        Some(vec![Tensor::from_scalar(0.0f32)])
                    }
                }

                "nonmaxsuppression" => {
                    // NonMaxSuppression is complex; keep as pass-through for now
                    // Real NMS is done in Python post-processing
                    Some(vec![args[0].clone()])
                }

                "resize" => {
                    // Resize/interpolation is not yet implemented in dispatcher
                    // Keep as pass-through for now
                    Some(vec![args[0].clone()])
                }

                _ => {
                    args.first().cloned().map(|t| vec![t])
                }
            };

            if let Some(outputs) = result {
                for (i, output_name) in node.outputs.iter().enumerate() {
                    if i < outputs.len() {
                        buffer.insert(output_name.clone(), outputs[i].clone());
                    }
                }
            }
        }

        let mut outputs = HashMap::new();
        for name in &self.output_names {
            if let Some(t) = buffer.get(name) {
                outputs.insert(name.clone(), t.clone());
            }
        }
        outputs
    }

    // ---- Dispatch helpers ----

    fn dispatch_unary(&self, op_name: &str, args: &[Tensor]) -> Option<Vec<Tensor>> {
        dispatch(op_name, DispatchKey::Cpu, &[&args[0]]).ok()
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

        let stride_t = Tensor::from_scalar(stride as f32);
        let pad_t = Tensor::from_scalar(padding as f32);

        if has_bias {
            let bias_name = format!("{}.bias", node.name);
            let bias = self.params.get(&bias_name)?;
            dispatch("conv2d", DispatchKey::Cpu, &[x, weight, bias, &stride_t, &pad_t]).ok()
        } else {
            let zero_bias = Tensor::from_scalar(0.0f32);
            dispatch("conv2d", DispatchKey::Cpu, &[x, weight, &zero_bias, &stride_t, &pad_t]).ok()
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
            let mut axes: Vec<usize> = axes_str.trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            axes.sort();
            let mut new_shape: Vec<i64> = shape.to_vec();
            for (offset, &axis) in axes.iter().enumerate() {
                new_shape.insert(axis + offset, 1);
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
