use crate::error::FastnnError;
use crate::ir::{ComputeGraph, DimExpr, IRNode, Opcode};
use crate::utils::{parse_conv_attrs, parse_shape_attr, spatial_output_dim};
use std::collections::{HashMap, HashSet};

pub fn infer_shapes(graph: &mut ComputeGraph) -> Result<(), FastnnError> {
    let order = graph.try_topological_sort()?;

    for &node_id in &order {
        let node = graph.get_node(node_id).unwrap();
        let inputs: Vec<&IRNode> = node
            .inputs
            .iter()
            .filter_map(|&id| graph.get_node(id))
            .collect();

        let inferred = match node.opcode {
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Maximum
            | Opcode::Minimum => {
                if inputs.len() >= 2 {
                    Some(
                        broadcast_shapes(
                            &inputs[0].output_type.shape,
                            &inputs[1].output_type.shape,
                        )
                        .map_err(|e| {
                            FastnnError::compilation(format!(
                                "node {} ({:?} '{}') shapes {:?} vs {:?}: {}",
                                node_id,
                                node.opcode,
                                node.name,
                                inputs[0].output_type.shape,
                                inputs[1].output_type.shape,
                                e
                            ))
                        })?,
                    )
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
            | Opcode::Round
            | Opcode::LogicalNot
            | Opcode::LogSoftmax
            | Opcode::Mish => inputs.first().map(|i| i.output_type.shape.clone()),
            Opcode::MatMul => {
                if inputs.len() >= 2 {
                    Some(
                        matmul_output_shape(
                            &inputs[0].output_type.shape,
                            &inputs[1].output_type.shape,
                        )
                        .map_err(|e| {
                            FastnnError::compilation(format!(
                                "node {} (MatMul '{}') shapes {:?} vs {:?}: {}",
                                node_id,
                                node.name,
                                inputs[0].output_type.shape,
                                inputs[1].output_type.shape,
                                e
                            ))
                        })?,
                    )
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
            Opcode::Transpose => {
                if let Some(input) = inputs.first() {
                    let rank = input.output_type.shape.len();
                    let perm = node
                        .optional_attr_list::<usize>("perm")?
                        .filter(|perm| !perm.is_empty())
                        .unwrap_or_else(|| (0..rank).rev().collect());
                    let valid = perm.len() == rank && perm.iter().all(|&axis| axis < rank) && {
                        let unique: HashSet<_> = perm.iter().copied().collect();
                        unique.len() == rank
                    };
                    if !valid {
                        return Err(FastnnError::shape(format!(
                            "Transpose node {node_id} has invalid permutation {perm:?} for rank {rank}"
                        )));
                    }
                    Some(
                        perm.iter()
                            .map(|&axis| input.output_type.shape[axis].clone())
                            .collect(),
                    )
                } else {
                    None
                }
            }
            Opcode::Flatten => inputs.first().map(|i| {
                let shape = &i.output_type.shape;
                if shape.len() >= 2 {
                    let mut dims = Vec::new();
                    if let Some(first) = shape.first() {
                        dims.push(first.clone());
                    }
                    let product: DimExpr = shape[1..]
                        .iter()
                        .cloned()
                        .reduce(|a, b| a.mul(&b))
                        .unwrap_or(DimExpr::Known(1));
                    dims.push(product);
                    dims
                } else {
                    shape.clone()
                }
            }),
            Opcode::TopK => inputs.first().map(|i| i.output_type.shape.clone()),
            Opcode::ReduceSum | Opcode::ReduceMean | Opcode::ReduceMax | Opcode::ArgMax => {
                if let Some(input) = inputs.first() {
                    let mut shape = input.output_type.shape.clone();
                    let keepdim = node.optional_bool_attr("keepdim")?.unwrap_or(true);
                    if let Some(raw_axis) = node.optional_attr::<i64>("axis")? {
                        let rank = i64::try_from(shape.len()).map_err(|_| {
                            FastnnError::shape(format!(
                                "{:?} node {node_id} rank does not fit i64",
                                node.opcode
                            ))
                        })?;
                        let axis = if raw_axis < 0 {
                            rank.checked_add(raw_axis)
                        } else {
                            Some(raw_axis)
                        }
                        .filter(|&axis| axis >= 0 && axis < rank)
                        .ok_or_else(|| {
                            FastnnError::shape(format!(
                                "{:?} node {node_id} axis {raw_axis} is out of range for rank {rank}",
                                node.opcode
                            ))
                        })? as usize;
                        if keepdim {
                            shape[axis] = DimExpr::Known(1);
                        } else {
                            shape.remove(axis);
                        }
                    }
                    Some(shape)
                } else {
                    None
                }
            }
            Opcode::Concat => {
                if !inputs.is_empty() {
                    let axis = node.optional_attr::<usize>("axis")?.unwrap_or(0);
                    let rank = inputs[0].output_type.shape.len();
                    if axis >= rank {
                        return Err(FastnnError::shape(format!(
                            "Concat node {node_id} axis {axis} is out of range for rank {rank}"
                        )));
                    }
                    let mut shape = inputs[0].output_type.shape.clone();
                    let mut total = DimExpr::Known(0);
                    for input in &inputs {
                        if axis >= input.output_type.shape.len() {
                            return Err(FastnnError::shape(format!(
                                "Concat node {node_id} input rank is too small for axis {axis}"
                            )));
                        }
                        total = total.add(&input.output_type.shape[axis]);
                    }
                    shape[axis] = total;
                    Some(shape)
                } else {
                    None
                }
            }
            Opcode::BatchNorm | Opcode::LayerNorm | Opcode::Softmax | Opcode::BiasAdd => {
                inputs.first().map(|i| i.output_type.shape.clone())
            }
            Opcode::MaxPool | Opcode::AvgPool => {
                if !inputs.is_empty() {
                    let input_shape = &inputs[0].output_type.shape;
                    let kernel = node.optional_attr::<i64>("kernel_size")?.unwrap_or(2);
                    let stride = node.optional_attr::<i64>("stride")?.unwrap_or(2);
                    if kernel <= 0 || stride <= 0 {
                        return Err(FastnnError::shape(format!(
                            "{:?} node {node_id} requires positive kernel_size and stride",
                            node.opcode
                        )));
                    }
                    // Parse padding: ONNX pads = [top, left, bottom, right] for 2D.
                    // If absent, assume 0 (valid padding).
                    // The builder stores symmetric `padding` as a single int; fall back
                    // to the ONNX `pads` CSV format when `padding` is absent.
                    let symmetric_pad = node.optional_attr::<i64>("padding")?.unwrap_or(0);
                    if symmetric_pad < 0 {
                        return Err(FastnnError::shape(format!(
                            "{:?} node {node_id} requires non-negative padding",
                            node.opcode
                        )));
                    }
                    let pads: Vec<i64> = if symmetric_pad > 0 {
                        // Symmetric padding: total pad = 2 * per-side padding
                        vec![symmetric_pad; 4]
                    } else {
                        node.optional_attr_list::<i64>("pads")?.unwrap_or_default()
                    };
                    if pads.iter().any(|&pad| pad < 0) {
                        return Err(FastnnError::shape(format!(
                            "{:?} node {node_id} requires non-negative pads",
                            node.opcode
                        )));
                    }
                    let pad_h =
                        pads.first().copied().unwrap_or(0) + pads.get(2).copied().unwrap_or(0);
                    let pad_w =
                        pads.get(1).copied().unwrap_or(0) + pads.get(3).copied().unwrap_or(0);
                    Some(
                        input_shape
                            .iter()
                            .enumerate()
                            .map(|(i, dim)| {
                                if i >= 2 {
                                    let total_pad = if i == 2 { pad_h } else { pad_w };
                                    match dim {
                                        DimExpr::Known(w) => DimExpr::Known(
                                            ((*w as i64 + total_pad - kernel) / stride + 1).max(1)
                                                as u64,
                                        ),
                                        DimExpr::Bounded { sym, max } => {
                                            // Apply spatial reduction to Bounded dims
                                            let reduced =
                                                ((*max as i64 + total_pad - kernel) / stride + 1)
                                                    .max(1)
                                                    as u64;
                                            DimExpr::Bounded {
                                                sym: format!("pool({})", sym),
                                                max: reduced,
                                            }
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
                if !inputs.is_empty() {
                    let input_shape = &inputs[0].output_type.shape;
                    let mut shape = input_shape.clone();
                    let dim = node.required_attr::<usize>("dim")?;
                    let start = node.required_attr::<u64>("start")?;
                    let end = node.required_attr::<u64>("end")?;
                    if dim >= shape.len() {
                        return Err(FastnnError::shape(format!(
                            "Slice node {} dimension {dim} is out of range for rank {}",
                            node.id,
                            shape.len()
                        )));
                    }
                    if end <= start {
                        return Err(FastnnError::shape(format!(
                            "Slice node {} requires end greater than start, got {start}..{end}",
                            node.id
                        )));
                    }
                    shape[dim] = DimExpr::Known(end - start);
                    Some(shape)
                } else {
                    None
                }
            }
            Opcode::Squeeze => {
                if let Some(input) = inputs.first() {
                    let mut shape = input.output_type.shape.clone();
                    if let Some(dim) = node.optional_attr::<usize>("dim")? {
                        if dim >= shape.len() {
                            return Err(FastnnError::shape(format!(
                                "Squeeze node {} dimension {dim} is out of range for rank {}",
                                node.id,
                                shape.len()
                            )));
                        }
                        shape.remove(dim);
                    } else {
                        shape.retain(|d| !matches!(d, DimExpr::Known(1)));
                    }
                    Some(shape)
                } else {
                    None
                }
            }
            Opcode::Unsqueeze => {
                if let Some(input) = inputs.first() {
                    let mut shape = input.output_type.shape.clone();
                    if let Some(dim) = node.optional_attr::<usize>("dim")? {
                        if dim > shape.len() {
                            return Err(FastnnError::shape(format!(
                                "Unsqueeze node {} dimension {dim} is out of range for rank {}",
                                node.id,
                                shape.len()
                            )));
                        }
                        shape.insert(dim, DimExpr::Known(1));
                    }
                    Some(shape)
                } else {
                    None
                }
            }
            Opcode::Pad => {
                if !inputs.is_empty() {
                    let input_shape = &inputs[0].output_type.shape;
                    let mut shape = input_shape.clone();
                    if let Some(pads) = node.optional_attr_list::<u64>("pads")? {
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
            Opcode::Gather => {
                // ONNX Gather output shape: data_shape[:axis] + indices_shape + data_shape[axis+1:]
                if inputs.len() >= 2 {
                    let axis = node.optional_attr::<usize>("axis")?.unwrap_or(0);
                    let data_shape = &inputs[0].output_type.shape;
                    let indices_shape = &inputs[1].output_type.shape;
                    if axis >= data_shape.len() {
                        return Err(FastnnError::shape(format!(
                            "Gather node {node_id} axis {axis} is out of range for rank {}",
                            data_shape.len()
                        )));
                    }
                    let mut new_shape: Vec<DimExpr> = data_shape[..axis].to_vec();
                    new_shape.extend_from_slice(indices_shape);
                    new_shape.extend_from_slice(&data_shape[axis + 1..]);
                    Some(new_shape)
                } else {
                    inputs.first().map(|i| i.output_type.shape.clone())
                }
            }
            Opcode::ScatterNd => inputs.first().map(|i| i.output_type.shape.clone()),
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
            Opcode::Prelu | Opcode::RMSNorm | Opcode::FusedResidualAddNorm => {
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
            Opcode::Shape => {
                // Shape output: 1D tensor of length = rank(input)
                inputs.first().map(|i| {
                    let rank = i.output_type.shape.len();
                    vec![DimExpr::Known(rank as u64)]
                })
            }
            Opcode::Cast => {
                // Cast preserves shape, changes dtype (handled by output_type)
                inputs.first().map(|i| i.output_type.shape.clone())
            }
            Opcode::Quantize | Opcode::Dequantize => {
                // Quantize/Dequantize preserve shape (only dtype changes)
                inputs.first().map(|i| i.output_type.shape.clone())
            }
            Opcode::ToF16 | Opcode::ToF32 => {
                // Half-precision conversion preserves shape
                inputs.first().map(|i| i.output_type.shape.clone())
            }
            Opcode::QuantizeActivations | Opcode::DequantizeActivations => {
                // Activation quantization/dequantization preserve shape
                inputs.first().map(|i| i.output_type.shape.clone())
            }
            Opcode::Expand => {
                // Expand output shape: try to use the DAG builder's pre-computed
                // "expand_shape" attr, which contains concrete target dims.
                if inputs.len() >= 2 {
                    if let Some(target) = node.optional_attr_list::<u64>("expand_shape")? {
                        if !target.is_empty() {
                            let data_shape = &inputs[0].output_type.shape;
                            let data_rank = data_shape.len();
                            let target_rank = target.len();
                            let max_rank = data_rank.max(target_rank);
                            let mut broadcast_shape = Vec::with_capacity(max_rank);
                            for i in 0..max_rank {
                                let data_dim = if i < max_rank - data_rank {
                                    1u64
                                } else {
                                    let idx = i - (max_rank - data_rank);
                                    match data_shape[idx] {
                                        DimExpr::Known(v) => v,
                                        _ => 1u64,
                                    }
                                };
                                let target_dim = if i < max_rank - target_rank {
                                    1u64
                                } else {
                                    target[i - (max_rank - target_rank)]
                                };
                                broadcast_shape.push(DimExpr::Known(data_dim.max(target_dim)));
                            }
                            Some(broadcast_shape)
                        } else {
                            // Fallback: use data shape
                            Some(inputs[0].output_type.shape.clone())
                        }
                    } else {
                        // No expand_shape attr: fallback to data shape.
                        // The expand kernel will read the target shape at runtime.
                        Some(inputs[0].output_type.shape.clone())
                    }
                } else {
                    inputs.first().map(|i| i.output_type.shape.clone())
                }
            }
            Opcode::Tile => {
                // Tile output shape: data_shape[i] * repeats[i]
                // For now, just use data shape (repeats are usually 1 for most dims)
                inputs.first().map(|i| i.output_type.shape.clone())
            }
            Opcode::Range => {
                // Range(start, limit, step) — always produces a 1D F32 tensor.
                // The length is dynamic (set by the builder as Symbol("N")).
                inputs
                    .first()
                    .map(|_| vec![DimExpr::Symbol("N".to_string())])
            }
            Opcode::Constant(_)
            | Opcode::Input
            | Opcode::UpsampleNearest2d
            | Opcode::UpsampleBilinear2d
            | Opcode::AdaptiveAvgPool2d
            | Opcode::Repeat
            | Opcode::CumSum
            | Opcode::Erf
            | Opcode::Flip
            | Opcode::Where
            | Opcode::SgdUpdate
            | Opcode::AdamUpdate
            | Opcode::AdamWUpdate
            | Opcode::MuonUpdate
            | Opcode::LionUpdate
            | Opcode::RmspropUpdate
            | Opcode::GradientScale
            | Opcode::QuantizeGradient
            | Opcode::DequantizeGradient => None,
        };

        if let Some(shape) = inferred {
            if let Some(node_mut) = graph.get_node_mut(node_id) {
                node_mut.output_type.shape = shape;
            }
        }
    }

    Ok(())
}

fn broadcast_shapes(a: &[DimExpr], b: &[DimExpr]) -> Result<Vec<DimExpr>, FastnnError> {
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
                return Err(FastnnError::compilation(format!(
                    "Cannot broadcast dimensions: {} vs {}",
                    va, vb
                )));
            }
            _ => {
                let va = da.evaluate();
                let vb = db.evaluate();
                match (va, vb) {
                    (Some(1), _) => db.clone(),
                    (_, Some(1)) => da.clone(),
                    _ if da == db => da.clone(),
                    _ => {
                        return Err(FastnnError::compilation(format!(
                            "Cannot broadcast dimensions: {} vs {}",
                            da, db
                        )));
                    }
                }
            }
        };
        result.push(dim);
    }

    result.reverse();
    Ok(result)
}

fn matmul_output_shape(a: &[DimExpr], b: &[DimExpr]) -> Result<Vec<DimExpr>, FastnnError> {
    if a.len() < 2 {
        return Err(FastnnError::compilation(format!(
            "MatMul: first input must have at least 2 dimensions, got {}",
            a.len()
        )));
    }
    if b.len() < 2 {
        return Err(FastnnError::compilation(format!(
            "MatMul: second input must have at least 2 dimensions, got {}",
            b.len()
        )));
    }

    let m = &a[a.len() - 2];
    let k_a = &a[a.len() - 1];
    let k_b = &b[b.len() - 2];
    let n = &b[b.len() - 1];

    match (k_a.evaluate(), k_b.evaluate()) {
        (Some(va), Some(vb)) if va != vb => {
            return Err(FastnnError::compilation(format!(
                "MatMul: inner dimensions must match, got {} vs {}",
                va, vb
            )));
        }
        (Some(_), Some(_)) => {}
        _ => {
            if k_a != k_b {
                return Err(FastnnError::compilation(format!(
                    "MatMul: inner dimensions must match, got {} vs {}",
                    k_a, k_b
                )));
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
) -> Result<Vec<DimExpr>, FastnnError> {
    if input_shape.len() < 4 {
        return Err(FastnnError::compilation(format!(
            "Conv2d: input must have at least 4 dimensions [N,C,H,W], got {}",
            input_shape.len()
        )));
    }
    if weight_shape.len() < 4 {
        return Err(FastnnError::compilation(format!(
            "Conv2d: weight must have 4 dimensions [F,C,KH,KW], got {}",
            weight_shape.len()
        )));
    }

    let (stride, padding, dilation) = parse_conv_attrs(attrs);

    let n = input_shape[0].clone();
    let f = weight_shape[0].clone();

    let h_out = spatial_output_dim(&input_shape[2], &weight_shape[2], stride, padding, dilation)
        .map_err(FastnnError::compilation)?;
    let w_out = spatial_output_dim(&input_shape[3], &weight_shape[3], stride, padding, dilation)
        .map_err(FastnnError::compilation)?;

    Ok(vec![n, f, h_out, w_out])
}

fn conv1d_output_shape(
    input_shape: &[DimExpr],
    weight_shape: &[DimExpr],
    attrs: &HashMap<String, String>,
) -> Result<Vec<DimExpr>, FastnnError> {
    if input_shape.len() < 3 {
        return Err(FastnnError::compilation(format!(
            "Conv1d: input must have 3 dimensions [N,C,W], got {}",
            input_shape.len()
        )));
    }
    if weight_shape.len() < 3 {
        return Err(FastnnError::compilation(format!(
            "Conv1d: weight must have 3 dimensions [F,C,KW], got {}",
            weight_shape.len()
        )));
    }

    let (stride, padding, dilation) = parse_conv_attrs(attrs);

    let n = input_shape[0].clone();
    let f = weight_shape[0].clone();
    let w_out = spatial_output_dim(&input_shape[2], &weight_shape[2], stride, padding, dilation)
        .map_err(FastnnError::compilation)?;

    Ok(vec![n, f, w_out])
}

fn conv3d_output_shape(
    input_shape: &[DimExpr],
    weight_shape: &[DimExpr],
    attrs: &HashMap<String, String>,
) -> Result<Vec<DimExpr>, FastnnError> {
    if input_shape.len() < 5 {
        return Err(FastnnError::compilation(format!(
            "Conv3d: input must have 5 dimensions [N,C,D,H,W], got {}",
            input_shape.len()
        )));
    }
    if weight_shape.len() < 5 {
        return Err(FastnnError::compilation(format!(
            "Conv3d: weight must have 5 dimensions [F,C,KD,KH,KW], got {}",
            weight_shape.len()
        )));
    }

    let (stride, padding, dilation) = parse_conv_attrs(attrs);

    let n = input_shape[0].clone();
    let f = weight_shape[0].clone();
    let d_out = spatial_output_dim(&input_shape[2], &weight_shape[2], stride, padding, dilation)
        .map_err(FastnnError::compilation)?;
    let h_out = spatial_output_dim(&input_shape[3], &weight_shape[3], stride, padding, dilation)
        .map_err(FastnnError::compilation)?;
    let w_out = spatial_output_dim(&input_shape[4], &weight_shape[4], stride, padding, dilation)
        .map_err(FastnnError::compilation)?;

    Ok(vec![n, f, d_out, h_out, w_out])
}

fn conv_transpose2d_output_shape(
    input_shape: &[DimExpr],
    weight_shape: &[DimExpr],
    attrs: &HashMap<String, String>,
) -> Result<Vec<DimExpr>, FastnnError> {
    if input_shape.len() < 4 {
        return Err(FastnnError::compilation(format!(
            "ConvTranspose2d: input must have 4 dimensions [N,C,H,W], got {}",
            input_shape.len()
        )));
    }
    if weight_shape.len() < 4 {
        return Err(FastnnError::compilation(format!(
            "ConvTranspose2d: weight must have 4 dimensions [C,F,KH,KW], got {}",
            weight_shape.len()
        )));
    }

    let (stride, padding, dilation) = parse_conv_attrs(attrs);

    let n = input_shape[0].clone();
    let f = weight_shape[1].clone();

    let h_out =
        conv_transpose_spatial_dim(&input_shape[2], &weight_shape[2], stride, padding, dilation)?;
    let w_out =
        conv_transpose_spatial_dim(&input_shape[3], &weight_shape[3], stride, padding, dilation)?;

    Ok(vec![n, f, h_out, w_out])
}

fn conv_transpose_spatial_dim(
    input_dim: &DimExpr,
    kernel_dim: &DimExpr,
    stride: i64,
    padding: i64,
    dilation: i64,
) -> Result<DimExpr, FastnnError> {
    let result = |input: i64, kernel: i64| -> DimExpr {
        DimExpr::Known(((input - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1) as u64)
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
