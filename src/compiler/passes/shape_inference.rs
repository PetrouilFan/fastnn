#![allow(dead_code)]

use crate::ir::node::{ComputeGraph, DimExpr, IRNode, Opcode};
use std::collections::HashMap;

pub fn infer_shapes(graph: &mut ComputeGraph) -> Result<(), String> {
    let order = graph.topological_sort();

    for &node_id in &order {
        let node = graph.get_node(node_id).unwrap().clone();
        let inputs: Vec<&IRNode> = node
            .inputs
            .iter()
            .filter_map(|&id| graph.get_node(id))
            .collect();

        let inferred = match node.opcode {
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div
            | Opcode::Maximum | Opcode::Minimum => {
                if inputs.len() >= 2 {
                    Some(broadcast_shapes(
                        &inputs[0].output_type.shape,
                        &inputs[1].output_type.shape,
                    ).map_err(|e| format!("node {} ({:?}) shapes {:?} vs {:?}: {}", 
                        node_id, node.opcode, 
                        inputs[0].output_type.shape, 
                        inputs[1].output_type.shape, 
                        e))?)
                } else if inputs.len() == 1 {
                    Some(inputs[0].output_type.shape.clone())
                } else {
                    None
                }
            }
            Opcode::Relu
            | Opcode::Gelu
            | Opcode::Silu
            | Opcode::Sigmoid
            | Opcode::Tanh
            | Opcode::Exp
            | Opcode::Log
            | Opcode::Sqrt
            | Opcode::Neg
            | Opcode::Abs
            | Opcode::LeakyRelu
            | Opcode::Elu
            | Opcode::Softplus
            | Opcode::Hardswish
            | Opcode::Clamp
            | Opcode::Sign
            | Opcode::LogicalNot
            | Opcode::LogSoftmax
            | Opcode::Mish => inputs.first().map(|i| i.output_type.shape.clone()),
            Opcode::MatMul => {
                if inputs.len() >= 2 {
                    Some(matmul_output_shape(
                        &inputs[0].output_type.shape,
                        &inputs[1].output_type.shape,
                    )?)
                } else {
                    None
                }
            }
            Opcode::Conv2d => {
                if inputs.len() >= 2 {
                    Some(conv2d_output_shape(
                        &inputs[0].output_type.shape,
                        &inputs[1].output_type.shape,
                        &node.attrs,
                    )?)
                } else {
                    None
                }
            }
            Opcode::Reshape => node.attrs.get("shape").map(|s| parse_shape_attr(s)),
            Opcode::Transpose => inputs.first().map(|i| {
                let mut s = i.output_type.shape.clone();
                s.reverse();
                s
            }),
            Opcode::Flatten => inputs.first().map(|i| {
                let shape = &i.output_type.shape;
                if shape.len() >= 2 {
                    let mut dims = Vec::new();
                    if let Some(first) = shape.first() {
                        dims.push(first.clone());
                    }
                    let product: DimExpr = shape[1..].iter()
                        .cloned()
                        .reduce(|a, b| a.mul(&b))
                        .unwrap_or(DimExpr::Known(1));
                    dims.push(product);
                    dims
                } else {
                    shape.clone()
                }
            }),
            Opcode::TopKValues | Opcode::TopKIndices | Opcode::TopK => inputs.first().map(|i| {
                i.output_type.shape.clone()
            }),
            Opcode::ReduceSum | Opcode::ReduceMean | Opcode::ReduceMax | Opcode::ArgMax => inputs.first().map(|i| {
                let mut s = i.output_type.shape.clone();
                if let Some(axis_str) = node.attrs.get("axis") {
                    if let Ok(axis) = axis_str.parse::<usize>() {
                        if axis < s.len() {
                            s.remove(axis);
                        }
                    }
                }
                s
            }),
            Opcode::Concat => {
                if !inputs.is_empty() {
                    let axis: usize = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    let mut shape = inputs[0].output_type.shape.clone();
                    if axis < shape.len() {
                        let mut total = DimExpr::Known(0);
                        for inp in &inputs {
                            if axis < inp.output_type.shape.len() {
                                total = total.add(&inp.output_type.shape[axis]);
                            }
                        }
                        shape[axis] = total;
                    }
                    Some(shape)
                } else {
                    None
                }
            }
            Opcode::BatchNorm | Opcode::LayerNorm | Opcode::Softmax | Opcode::BiasAdd => {
                inputs.first().map(|i| i.output_type.shape.clone())
            }
            Opcode::MaxPool | Opcode::AvgPool => {
                if inputs.len() >= 1 {
                    let input_shape = &inputs[0].output_type.shape;
                    let kernel: i64 = node
                        .attrs
                        .get("kernel_size")
                        .and_then(|k| k.parse().ok())
                        .unwrap_or(2);
                    let stride: i64 = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2);
                    Some(
                        input_shape
                            .iter()
                            .enumerate()
                            .map(|(i, dim)| {
                                if i >= 2 {
                                    match dim {
                                        DimExpr::Known(w) => {
                                            DimExpr::Known(((*w as i64 - kernel) / stride + 1) as u64)
                                        }
                                        other => other.clone(),
                                    }
                                } else {
                                    dim.clone()
                                }
                            })
                            .collect(),
                    )
                } else {
                    None
                }
            }
            Opcode::Slice => {
                if inputs.len() >= 1 {
                    let input_shape = &inputs[0].output_type.shape;
                    let mut shape = input_shape.clone();
                    if let Some(dim_str) = node.attrs.get("dim") {
                        if let Ok(dim) = dim_str.parse::<usize>() {
                            let start: u64 = node
                                .attrs
                                .get("start")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0);
                            let end: u64 = node
                                .attrs
                                .get("end")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0);
                            if dim < shape.len() && end > start {
                                shape[dim] = DimExpr::Known(end - start);
                            }
                        }
                    }
                    Some(shape)
                } else {
                    None
                }
            }
            Opcode::Squeeze => inputs.first().map(|i| {
                let mut shape = i.output_type.shape.clone();
                if let Some(dim_str) = node.attrs.get("dim") {
                    if let Ok(dim) = dim_str.parse::<usize>() {
                        if dim < shape.len() {
                            shape.remove(dim);
                        }
                    }
                } else {
                    shape.retain(|d| !matches!(d, DimExpr::Known(1)));
                }
                shape
            }),
            Opcode::Unsqueeze => inputs.first().map(|i| {
                let mut shape = i.output_type.shape.clone();
                if let Some(dim_str) = node.attrs.get("dim") {
                    if let Ok(dim) = dim_str.parse::<usize>() {
                        if dim <= shape.len() {
                            shape.insert(dim, DimExpr::Known(1));
                        }
                    }
                }
                shape
            }),
            Opcode::Pad => {
                if inputs.len() >= 1 {
                    let input_shape = &inputs[0].output_type.shape;
                    let mut shape = input_shape.clone();
                    if let Some(pads_str) = node.attrs.get("pads") {
                        let pads: Vec<u64> = pads_str
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        let n_dims = shape.len().min(pads.len() / 2);
                        for i in 0..n_dims {
                            let lo = DimExpr::Known(pads[2 * i]);
                            let hi = DimExpr::Known(pads[2 * i + 1]);
                            shape[i] = shape[i].add(&lo).add(&hi);
                        }
                    }
                    Some(shape)
                } else {
                    None
                }
            }
            Opcode::Gather | Opcode::ScatterNd => {
                inputs.first().map(|i| i.output_type.shape.clone())
            }
            Opcode::Conv1d => {
                if inputs.len() >= 2 {
                    Some(conv1d_output_shape(
                        &inputs[0].output_type.shape,
                        &inputs[1].output_type.shape,
                        &node.attrs,
                    )?)
                } else {
                    None
                }
            }
            Opcode::Conv3d => {
                if inputs.len() >= 2 {
                    Some(conv3d_output_shape(
                        &inputs[0].output_type.shape,
                        &inputs[1].output_type.shape,
                        &node.attrs,
                    )?)
                } else {
                    None
                }
            }
            Opcode::ConvTranspose2d => {
                if inputs.len() >= 2 {
                    Some(conv_transpose2d_output_shape(
                        &inputs[0].output_type.shape,
                        &inputs[1].output_type.shape,
                        &node.attrs,
                    )?)
                } else {
                    None
                }
            }
            Opcode::Prelu | Opcode::RMSNorm => {
                inputs.first().map(|i| i.output_type.shape.clone())
            }
            Opcode::Embedding => {
                if inputs.len() >= 2 {
                    let weight_shape = &inputs[0].output_type.shape;
                    let indices_shape = &inputs[1].output_type.shape;
                    let mut shape = indices_shape.clone();
                    if weight_shape.len() >= 2 {
                        shape.push(weight_shape[1].clone());
                    }
                    Some(shape)
                } else {
                    None
                }
            }
            Opcode::Pow => {
                if inputs.len() >= 2 {
                    Some(broadcast_shapes(
                        &inputs[0].output_type.shape,
                        &inputs[1].output_type.shape,
                    )?)
                } else if inputs.len() == 1 {
                    Some(inputs[0].output_type.shape.clone())
                } else {
                    None
                }
            }
            Opcode::GtScalar
            | Opcode::LtScalar
            | Opcode::EqScalar
            | Opcode::AddScalar
            | Opcode::MulScalar
            | Opcode::DivScalar => inputs.first().map(|i| i.output_type.shape.clone()),
            Opcode::Constant(_) | Opcode::Input
            | Opcode::UpsampleNearest2d | Opcode::UpsampleBilinear2d
            | Opcode::AdaptiveAvgPool2d | Opcode::Repeat
            | Opcode::CumSum | Opcode::Erf | Opcode::Flip | Opcode::Where
            | Opcode::SgdUpdate | Opcode::AdamUpdate | Opcode::AdamWUpdate => None,
        };

        if let Some(shape) = inferred {
            if let Some(node_mut) = graph.get_node_mut(node_id) {
                node_mut.output_type.shape = shape;
            }
        }
    }

    Ok(())
}

fn broadcast_shapes(a: &[DimExpr], b: &[DimExpr]) -> Result<Vec<DimExpr>, String> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let da = if i < a.len() {
            &a[a.len() - 1 - i]
        } else {
            &DimExpr::Known(1)
        };
        let db = if i < b.len() {
            &b[b.len() - 1 - i]
        } else {
            &DimExpr::Known(1)
        };

        let dim = match (da, db) {
            (DimExpr::Known(1), other) => other.clone(),
            (other, DimExpr::Known(1)) => other.clone(),
            (DimExpr::Known(va), DimExpr::Known(vb)) if va == vb => DimExpr::Known(*va),
            (DimExpr::Known(va), DimExpr::Known(vb)) => {
                return Err(format!(
                    "Cannot broadcast dimensions: {} vs {}",
                    va, vb
                ));
            }
            _ => {
                let va = da.evaluate();
                let vb = db.evaluate();
                match (va, vb) {
                    (Some(1), _) => db.clone(),
                    (_, Some(1)) => da.clone(),
                    _ if da == db => da.clone(),
                    _ => {
                        return Err(format!(
                            "Cannot broadcast dimensions: {} vs {}",
                            da, db
                        ));
                    }
                }
            }
        };
        result.push(dim);
    }

    result.reverse();
    Ok(result)
}

fn matmul_output_shape(a: &[DimExpr], b: &[DimExpr]) -> Result<Vec<DimExpr>, String> {
    if a.len() < 2 {
        return Err(format!(
            "MatMul: first input must have at least 2 dimensions, got {}",
            a.len()
        ));
    }
    if b.len() < 2 {
        return Err(format!(
            "MatMul: second input must have at least 2 dimensions, got {}",
            b.len()
        ));
    }

    let m = &a[a.len() - 2];
    let k_a = &a[a.len() - 1];
    let k_b = &b[b.len() - 2];
    let n = &b[b.len() - 1];

    match (k_a.evaluate(), k_b.evaluate()) {
        (Some(va), Some(vb)) if va != vb => {
            return Err(format!(
                "MatMul: inner dimensions must match, got {} vs {}",
                va, vb
            ));
        }
        (Some(_), Some(_)) => {}
        _ => {
            if k_a != k_b {
                return Err(format!(
                    "MatMul: inner dimensions must match, got {} vs {}",
                    k_a, k_b
                ));
            }
        }
    }

    let batch_a = &a[..a.len() - 2];
    let batch_b = &b[..b.len() - 2];
    let batch = broadcast_shapes(batch_a, batch_b)?;

    let mut result = batch;
    result.push(m.clone());
    result.push(n.clone());
    Ok(result)
}

fn conv2d_output_shape(
    input_shape: &[DimExpr],
    weight_shape: &[DimExpr],
    attrs: &HashMap<String, String>,
) -> Result<Vec<DimExpr>, String> {
    if input_shape.len() < 4 {
        return Err(format!(
            "Conv2d: input must have at least 4 dimensions [N,C,H,W], got {}",
            input_shape.len()
        ));
    }
    if weight_shape.len() < 4 {
        return Err(format!(
            "Conv2d: weight must have 4 dimensions [F,C,KH,KW], got {}",
            weight_shape.len()
        ));
    }

    let stride: i64 = attrs
        .get("stride")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let padding: i64 = attrs
        .get("padding")
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);

    let n = input_shape[0].clone();
    let f = weight_shape[0].clone();

    let h_out = spatial_output_dim(&input_shape[2], &weight_shape[2], stride, padding)?;
    let w_out = spatial_output_dim(&input_shape[3], &weight_shape[3], stride, padding)?;

    Ok(vec![n, f, h_out, w_out])
}

fn spatial_output_dim(
    input_dim: &DimExpr,
    kernel_dim: &DimExpr,
    stride: i64,
    padding: i64,
) -> Result<DimExpr, String> {
    match (input_dim, kernel_dim) {
        (DimExpr::Known(h), DimExpr::Known(k)) => {
            let result = (*h as i64 + 2 * padding - *k as i64) / stride + 1;
            if result <= 0 {
                return Err(format!(
                    "Conv2d: output spatial dimension is non-positive ({})",
                    result
                ));
            }
            Ok(DimExpr::Known(result as u64))
        }
        _ => {
            let h_val = input_dim.evaluate();
            let k_val = kernel_dim.evaluate();
            match (h_val, k_val) {
                (Some(h), Some(k)) => {
                    let result = (h as i64 + 2 * padding - k as i64) / stride + 1;
                    if result <= 0 {
                        return Err(format!(
                            "Conv2d: output spatial dimension is non-positive ({})",
                            result
                        ));
                    }
                    Ok(DimExpr::Known(result as u64))
                }
                _ => Ok(DimExpr::Symbol("?".to_string())),
            }
        }
    }
}

fn conv1d_output_shape(
    input_shape: &[DimExpr],
    weight_shape: &[DimExpr],
    attrs: &HashMap<String, String>,
) -> Result<Vec<DimExpr>, String> {
    if input_shape.len() < 3 {
        return Err(format!(
            "Conv1d: input must have 3 dimensions [N,C,W], got {}",
            input_shape.len()
        ));
    }
    if weight_shape.len() < 3 {
        return Err(format!(
            "Conv1d: weight must have 3 dimensions [F,C,KW], got {}",
            weight_shape.len()
        ));
    }

    let stride: i64 = attrs
        .get("stride")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let padding: i64 = attrs
        .get("padding")
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);

    let n = input_shape[0].clone();
    let f = weight_shape[0].clone();
    let w_out = spatial_output_dim(&input_shape[2], &weight_shape[2], stride, padding)?;

    Ok(vec![n, f, w_out])
}

fn conv3d_output_shape(
    input_shape: &[DimExpr],
    weight_shape: &[DimExpr],
    attrs: &HashMap<String, String>,
) -> Result<Vec<DimExpr>, String> {
    if input_shape.len() < 5 {
        return Err(format!(
            "Conv3d: input must have 5 dimensions [N,C,D,H,W], got {}",
            input_shape.len()
        ));
    }
    if weight_shape.len() < 5 {
        return Err(format!(
            "Conv3d: weight must have 5 dimensions [F,C,KD,KH,KW], got {}",
            weight_shape.len()
        ));
    }

    let stride: i64 = attrs
        .get("stride")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let padding: i64 = attrs
        .get("padding")
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);

    let n = input_shape[0].clone();
    let f = weight_shape[0].clone();
    let d_out = spatial_output_dim(&input_shape[2], &weight_shape[2], stride, padding)?;
    let h_out = spatial_output_dim(&input_shape[3], &weight_shape[3], stride, padding)?;
    let w_out = spatial_output_dim(&input_shape[4], &weight_shape[4], stride, padding)?;

    Ok(vec![n, f, d_out, h_out, w_out])
}

fn conv_transpose2d_output_shape(
    input_shape: &[DimExpr],
    weight_shape: &[DimExpr],
    attrs: &HashMap<String, String>,
) -> Result<Vec<DimExpr>, String> {
    if input_shape.len() < 4 {
        return Err(format!(
            "ConvTranspose2d: input must have 4 dimensions [N,C,H,W], got {}",
            input_shape.len()
        ));
    }
    if weight_shape.len() < 4 {
        return Err(format!(
            "ConvTranspose2d: weight must have 4 dimensions [C,F,KH,KW], got {}",
            weight_shape.len()
        ));
    }

    let stride: i64 = attrs
        .get("stride")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let padding: i64 = attrs
        .get("padding")
        .and_then(|p| p.parse().ok())
        .unwrap_or(0);

    let n = input_shape[0].clone();
    let f = weight_shape[1].clone();

    let h_out = conv_transpose_spatial_dim(&input_shape[2], &weight_shape[2], stride, padding)?;
    let w_out = conv_transpose_spatial_dim(&input_shape[3], &weight_shape[3], stride, padding)?;

    Ok(vec![n, f, h_out, w_out])
}

fn conv_transpose_spatial_dim(
    input_dim: &DimExpr,
    kernel_dim: &DimExpr,
    stride: i64,
    padding: i64,
) -> Result<DimExpr, String> {
    let result = |input: i64, kernel: i64| -> DimExpr {
        DimExpr::Known(((input - 1) * stride - 2 * padding + kernel) as u64)
    };
    match (input_dim, kernel_dim) {
        (DimExpr::Known(h), DimExpr::Known(k)) => {
            let out = result(*h as i64, *k as i64);
            Ok(out)
        }
        _ => {
            let h_val = input_dim.evaluate();
            let k_val = kernel_dim.evaluate();
            match (h_val, k_val) {
                (Some(h), Some(k)) => Ok(result(h as i64, k as i64)),
                _ => Ok(DimExpr::Symbol("?".to_string())),
            }
        }
    }
}

fn parse_shape_attr(s: &str) -> Vec<DimExpr> {
    let trimmed = s.trim();
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(trimmed);
    if inner.is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|part| {
            let p = part.trim();
            if let Ok(v) = p.parse::<u64>() {
                DimExpr::Known(v)
            } else if p.is_empty() {
                DimExpr::Known(1)
            } else {
                DimExpr::Symbol(p.to_string())
            }
        })
        .collect()
}

fn add_dim_exprs(a: DimExpr, b: DimExpr) -> DimExpr {
    a.add(&b)
}

fn max_u64_for_symbol(_v: &u64) -> u64 {
    u64::MAX
}
