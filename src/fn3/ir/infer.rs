//! Forward shape inference over the typed graph.
//!
//! # Why a separate pass
//!
//! A graph that reaches a backend with the wrong output type wastes the
//! allocation and fails in a kernel that has to describe the same rule again.
//! This pass resolves every output type once, in node order, and rejects the
//! first node it cannot resolve with a message that names the source node and
//! the op. It never guesses: an op whose rule is not implemented here returns
//! [`Fn3Error::UnsupportedOp`], which a caller can report as "this model is not
//! supported yet" before execution starts.
//!
//! # Symbolic rules
//!
//! A [`DimExpr`] is either a constant or a named symbol; there is no "unknown"
//! dimension. Therefore a rule that would need an unprovable fact is an error:
//!
//! - broadcasting accepts equal extents or an extent of `1`; two different
//!   symbols, or a symbol against a constant other than `1`, are rejected
//!   because the symbols could resolve to different values;
//! - `Concat` sums constant extents and lets a zero contribute nothing; it
//!   cannot express `batch + batch`, so it rejects a symbolic sum;
//! - `Reshape` resolves a `-1` only when the input is fully static, and accepts
//!   a symbolic target only when it repeats the input extents exactly;
//! - `Flatten` and `Squeeze`/`Unsqueeze` fail when the result would depend on a
//!   symbol's value (a symbolic product, or a squeeze without axes).
//!
//! The alternative — passing the operand shape through and hoping — is what the
//! crate forbids: a backend must never receive a shape the graph cannot justify.
//!
//! # What is inferred
//!
//! Every output type of the registry: elementwise binary (broadcasting) and
//! unary, `MatMul`, `Transpose`, `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten`,
//! `Concat`, `Slice`, `Split`, `Softmax`, `LayerNorm`, `ReduceMean`,
//! `ReduceSum`, `Gather`, `Shape`, and the CNN set (`Conv`, `MaxPool`,
//! `AveragePool`, `GlobalAveragePool`, `BatchNorm`, `Resize`, `Pad`, `Clip`,
//! `Sigmoid`, `Maximum`, `Minimum`, `TopK`).
//! Everything else is reported as not implemented, with the source node index.
//! The pass propagates dtype where the op's contract pins it (`MatMul` is f32,
//! `Gather` indices are integer, `LayerNorm` is f32, `Shape` and `TopK` indices
//! are i64) and otherwise preserves the input dtype; per-kernel dtype support
//! remains a backend question
//! ([`crate::inference::backend::Backend::admits`]), not a shape rule.
//!
//! # Constants live in attributes, not in the graph
//!
//! A [`TensorType`] carries no element values, so a rule that needs a constant
//! — a `Reshape` target, `Slice` bounds, `Resize` scales, `Pad` extents, a
//! `TopK` k — reads it from [`crate::inference::backend::OpParams`].
//! The importer folds ONNX initializers and `Constant` nodes there. When a rule
//! finds the operand but not the folded value it returns
//! [`Fn3Error::UnsupportedOp`] naming the node, rather than reading payload
//! bytes the IR does not have.

use std::fmt;

use crate::inference::backend::{
    AutoPad, CoordinateTransformationMode, NearestMode, OpParams, PadMode, ResizeMode,
};
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;

use super::{DimExpr, Graph, Node, NodeId, TensorType, MAX_RANK};

/// Resolve every node output type in execution order.
///
/// The graph's builder keeps nodes topologically ordered, so one forward pass
/// is enough. Each resolved type is validated with [`TensorType::validate`]
/// before it is written, and a failure leaves the earlier nodes' types in place
/// (there is no partially written value). Calling this twice is deterministic:
/// the second pass writes the same types.
pub fn infer_shapes(graph: &mut Graph) -> Result<(), Fn3Error> {
    for index in 0..graph.node_count() {
        let node_id = u32::try_from(index).map_err(|_| Fn3Error::CapacityExceeded {
            what: "ir node table",
            requested: (index as u64).saturating_add(1),
            capacity: u64::from(u32::MAX),
        })?;
        let node = graph
            .node(NodeId::new(node_id))
            .ok_or(Fn3Error::OutOfBounds {
                what: "ir node id",
                index: u64::from(node_id),
                len: graph.node_count() as u64,
            })?
            .clone();
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for &input in &node.inputs {
            let value = graph.value(input).ok_or(Fn3Error::OutOfBounds {
                what: "ir node input value",
                index: u64::try_from(input.index()).unwrap_or(u64::MAX),
                len: graph.value_count() as u64,
            })?;
            inputs.push(value.ty.clone());
        }
        let outputs = infer_node(&node, &inputs)?;
        if outputs.len() != node.outputs.len() {
            return Err(invalid(
                &node,
                format!(
                    "shape inference produced {} result(s) for {} output value(s)",
                    outputs.len(),
                    node.outputs.len()
                ),
            ));
        }
        for (&output, ty) in node.outputs.iter().zip(outputs) {
            ty.validate().map_err(|error| annotate(error, &node))?;
            graph.set_value_type(output, ty)?;
        }
    }
    Ok(())
}

fn infer_node(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    match node.op {
        OpId::Add
        | OpId::Sub
        | OpId::Mul
        | OpId::Div
        | OpId::Pow
        | OpId::Maximum
        | OpId::Minimum => binary(node, inputs),
        OpId::Neg
        | OpId::Sqrt
        | OpId::Erf
        | OpId::Tanh
        | OpId::Gelu
        | OpId::Relu
        | OpId::Sin
        | OpId::Cos
        | OpId::Sigmoid => unary(node, inputs),
        OpId::MatMul => matmul(node, inputs),
        OpId::Transpose => transpose(node, inputs),
        OpId::Reshape => reshape(node, inputs),
        OpId::Squeeze => squeeze(node, inputs),
        OpId::Unsqueeze => unsqueeze(node, inputs),
        OpId::Flatten => flatten(node, inputs),
        OpId::Concat => concat(node, inputs),
        OpId::Slice => slice(node, inputs),
        OpId::Split => split(node, inputs),
        OpId::Softmax => softmax(node, inputs),
        OpId::LayerNorm => layer_norm(node, inputs),
        OpId::ReduceMean | OpId::ReduceSum => reduce(node, inputs),
        OpId::Gather => gather(node, inputs),
        OpId::Shape => shape_op(node, inputs),
        OpId::Conv => conv(node, inputs),
        OpId::MaxPool | OpId::AveragePool => pool(node, inputs),
        OpId::GlobalAveragePool => global_average_pool(node, inputs),
        OpId::BatchNorm => batch_norm(node, inputs),
        OpId::Resize => resize(node, inputs),
        OpId::Pad => pad(node, inputs),
        OpId::Clip => clip(node, inputs),
        OpId::TopK => top_k(node, inputs),
        _ => Err(unsupported(node)),
    }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

fn unary(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_params(node, Req::NONE)?;
    Ok(vec![inputs[0].clone()])
}

fn binary(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 2)?;
    require_params(node, Req::NONE)?;
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    if lhs.dtype != rhs.dtype {
        return Err(invalid(
            node,
            format!(
                "op {} needs both operands in one dtype, found {} and {}",
                node.op, lhs.dtype, rhs.dtype
            ),
        ));
    }
    let shape = broadcast(node, lhs.dims(), rhs.dims())?;
    Ok(vec![lhs.with_shape(shape)])
}

/// Right-aligned NumPy broadcasting over symbolic extents.
fn broadcast(node: &Node, lhs: &[DimExpr], rhs: &[DimExpr]) -> Result<Vec<DimExpr>, Fn3Error> {
    let rank = lhs.len().max(rhs.len());
    let mut result = Vec::with_capacity(rank);
    // Walk from the rightmost axis; every axis of the shorter operand aligns
    // there. The result is reversed at the end so it stays outermost-first.
    for depth in 1..=rank {
        let left = extent_at_depth(lhs, depth);
        let right = extent_at_depth(rhs, depth);
        let extent = match (left, right) {
            (Some(left), Some(right)) => broadcast_dim(node, lhs, rhs, depth, left, right)?,
            (Some(extent), None) | (None, Some(extent)) => extent,
            (None, None) => {
                // `depth <= max(lhs.len(), rhs.len())` makes this impossible;
                // report it as a typed error rather than an `unreachable!()`.
                return Err(invalid(
                    node,
                    format!("internal: axis {depth} from the right names no operand"),
                ));
            }
        };
        result.push(extent);
    }
    result.reverse();
    Ok(result)
}

fn extent_at_depth(dims: &[DimExpr], depth: usize) -> Option<DimExpr> {
    dims.len()
        .checked_sub(depth)
        .and_then(|index| dims.get(index))
        .copied()
}

fn broadcast_dim(
    node: &Node,
    lhs: &[DimExpr],
    rhs: &[DimExpr],
    depth: usize,
    left: DimExpr,
    right: DimExpr,
) -> Result<DimExpr, Fn3Error> {
    if left == right {
        return Ok(left);
    }
    if left == DimExpr::Const(1) {
        return Ok(right);
    }
    if right == DimExpr::Const(1) {
        return Ok(left);
    }
    Err(shape_mismatch(
        node,
        format!("a shape broadcast-compatible with {}", render(lhs)),
        format!(
            "{} (axis {depth} from the right: {left} vs {right})",
            render(rhs)
        ),
    ))
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

fn matmul(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 2)?;
    require_params(node, Req::NONE)?;
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    if lhs.dtype != rhs.dtype {
        return Err(invalid(
            node,
            format!(
                "op matmul needs both operands in one dtype, found {} and {}",
                lhs.dtype, rhs.dtype
            ),
        ));
    }
    if !lhs.dtype.is_compute() {
        return Err(invalid(
            node,
            format!(
                "op matmul is a rank-2 {} contract, found {}",
                DType::F32,
                lhs.dtype
            ),
        ));
    }
    if lhs.rank() < 2 || rhs.rank() < 2 {
        return Err(invalid(
            node,
            format!(
                "op matmul needs rank >= 2 with the matrix axes last; operands are {} and {}",
                render(lhs.dims()),
                render(rhs.dims())
            ),
        ));
    }
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    let (rows, inner) = (lhs_dims[lhs_dims.len() - 2], lhs_dims[lhs_dims.len() - 1]);
    let (rhs_inner, columns) = (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1]);
    if inner != rhs_inner {
        return Err(shape_mismatch(
            node,
            format!("inner extents to match in {}", render(lhs_dims)),
            format!(
                "{} contracts {} vs {}",
                render(rhs_dims),
                rhs_inner,
                render(&[inner])
            ),
        ));
    }
    // ONNX MatMul is NumPy matmul: the leading dimensions broadcast right
    // aligned, and only the last two axes are the matrix. A transformer
    // attention block multiplies [1, heads, seq, head_dim] by
    // [1, heads, head_dim, seq] this way, which is why a rank-2-only rule
    // could not import a current YOLO export.
    let lhs_batch = &lhs_dims[..lhs_dims.len() - 2];
    let rhs_batch = &rhs_dims[..rhs_dims.len() - 2];
    let batch_len = lhs_batch.len().max(rhs_batch.len());
    let mut out_dims = Vec::with_capacity(batch_len + 2);
    for position in 0..batch_len {
        let left = lhs_batch
            .len()
            .checked_sub(batch_len - position)
            .map(|index| lhs_batch[index]);
        let right = rhs_batch
            .len()
            .checked_sub(batch_len - position)
            .map(|index| rhs_batch[index]);
        let extent = match (left, right) {
            (Some(left), Some(right)) => matmul_batch_dim(node, left, right)?,
            (Some(only), None) | (None, Some(only)) => only,
            (None, None) => unreachable!("position is inside max(len)"),
        };
        out_dims.push(extent);
    }
    out_dims.push(rows);
    out_dims.push(columns);
    Ok(vec![lhs.with_shape(out_dims)])
}

/// Broadcast one pair of extents the way NumPy does.
///
/// A symbolic extent only broadcasts against a literal one, which is the
/// literal `1`; two different symbols describe two batch sizes that this
/// graph cannot serve with one kernel, so they are rejected rather than
/// guessed.
fn matmul_batch_dim(node: &Node, left: DimExpr, right: DimExpr) -> Result<DimExpr, Fn3Error> {
    if left == right {
        return Ok(left);
    }
    match (left, right) {
        (DimExpr::Const(1), other) | (other, DimExpr::Const(1)) => Ok(other),
        (left, right) => Err(invalid(
            node,
            format!(
                "batch extents {} and {} do not broadcast (only a literal 1 does)",
                render(&[left]),
                render(&[right])
            ),
        )),
    }
}

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

fn transpose(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_params(node, Req::AXES)?;
    let input = &inputs[0];
    let rank = input.rank();
    let perm: Vec<usize> = if node.params.axes.is_empty() {
        (0..rank).rev().collect()
    } else {
        let mut perm = Vec::with_capacity(node.params.axes.len());
        for &axis in &node.params.axes {
            perm.push(normalize_axis(node, axis, rank)?);
        }
        perm
    };
    if perm.len() != rank {
        return Err(invalid(
            node,
            format!(
                "op transpose permutation has {} axes for a rank-{rank} operand",
                perm.len()
            ),
        ));
    }
    let mut seen = vec![false; rank];
    for &axis in &perm {
        if seen[axis] {
            return Err(invalid(
                node,
                format!("op transpose permutation repeats axis {axis}"),
            ));
        }
        seen[axis] = true;
    }
    Ok(vec![input.with_shape(
        perm.iter().map(|&axis| input.dims()[axis]).collect(),
    )])
}

fn reshape(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if inputs.len() > 2 {
        return Err(invalid(
            node,
            format!("op reshape takes at most 2 inputs, found {}", inputs.len()),
        ));
    }
    if inputs.len() == 2 && node.params.axes.is_empty() {
        return Err(unsupported_variant(
            node,
            "shape inference is not implemented for a runtime reshape target; the IR resolves \
             only the constant form carried in OpParams::axes",
        ));
    }
    if inputs.len() != 1 && !(inputs.len() == 2 && !node.params.axes.is_empty()) {
        return Err(invalid(
            node,
            format!(
                "op reshape takes 1 input, or 2 with the constant target in \
                 OpParams::axes, found {} input(s) and {} target extent(s)",
                inputs.len(),
                node.params.axes.len()
            ),
        ));
    }
    require_params(node, Req::AXES)?;
    let input = &inputs[0];
    let target = &node.params.axes;
    if target.is_empty() {
        return Err(invalid(
            node,
            "op reshape needs the constant target shape in OpParams::axes",
        ));
    }
    let mut resolved: Vec<Option<DimExpr>> = Vec::with_capacity(target.len());
    let mut inferred_at: Option<usize> = None;
    for (position, &raw) in target.iter().enumerate() {
        match raw {
            -1 => {
                if inferred_at.is_some() {
                    return Err(invalid(
                        node,
                        "op reshape allows at most one -1 target extent",
                    ));
                }
                inferred_at = Some(position);
                resolved.push(None);
            }
            0 => {
                let dim = input.dims().get(position).copied().ok_or_else(|| {
                    invalid(
                        node,
                        format!(
                            "op reshape copies input axis {position}, but the operand has rank {}",
                            input.rank()
                        ),
                    )
                })?;
                resolved.push(Some(dim));
            }
            extent if extent > 0 => resolved.push(Some(DimExpr::Const(extent as u64))),
            extent => {
                return Err(invalid(
                    node,
                    format!("op reshape target extent {extent} is negative other than -1"),
                ))
            }
        }
    }

    let input_numel = const_numel(node, input.dims(), "reshape input")?;
    match inferred_at {
        Some(position) => {
            let mut product = 1u64;
            for (index, dim) in resolved.iter().enumerate() {
                if index == position {
                    continue;
                }
                let Some(DimExpr::Const(extent)) = dim else {
                    return Err(invalid(
                        node,
                        "op reshape cannot infer -1 unless every other target extent is constant",
                    ));
                };
                product = product
                    .checked_mul(*extent)
                    .ok_or_else(|| overflow(node, "reshape target element count"))?;
            }
            let Some(input_numel) = input_numel else {
                return Err(invalid(
                    node,
                    "op reshape cannot infer -1 for a symbolic input; the input element count \
                     is not a constant",
                ));
            };
            let extent = if product == 0 {
                if input_numel == 0 {
                    0
                } else {
                    return Err(shape_mismatch(
                        node,
                        "a non-zero target product when -1 is present",
                        format!("{input_numel} input elements and a zero target product"),
                    ));
                }
            } else if input_numel % product == 0 {
                input_numel / product
            } else {
                return Err(shape_mismatch(
                    node,
                    format!("{input_numel} elements to divide by the target product {product}"),
                    format!("{input_numel} is not divisible by {product}"),
                ));
            };
            resolved[position] = Some(DimExpr::Const(extent));
        }
        None => {
            let target_numel = const_numel_from_options(node, &resolved, "reshape target")?;
            match (input_numel, target_numel) {
                (Some(input_numel), Some(target_numel)) => {
                    if input_numel != target_numel {
                        return Err(shape_mismatch(
                            node,
                            format!("{input_numel} elements to match {}", render(input.dims())),
                            format!("the target has {target_numel} elements"),
                        ));
                    }
                }
                _ => {
                    // With a symbolic extent the element counts cannot be
                    // compared, and this IR has no "unknown" dimension. The
                    // only honest symbolic reshape is one that repeats the
                    // input extents, after `0` copies them.
                    let repeated = resolved.len() == input.rank()
                        && resolved
                            .iter()
                            .zip(input.dims())
                            .all(|(target, input)| target.as_ref() == Some(input));
                    if !repeated {
                        return Err(invalid(
                            node,
                            format!(
                                "op reshape cannot prove {} -> {}; with a symbolic extent the \
                                 target must repeat the input extents exactly",
                                render(input.dims()),
                                render_options(&resolved)
                            ),
                        ));
                    }
                }
            }
        }
    }
    let shape = resolved
        .into_iter()
        .map(|dim| dim.ok_or_else(|| invalid(node, "op reshape left -1 unresolved")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(vec![input.with_shape(shape)])
}

fn squeeze(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if inputs.len() == 2 {
        return Err(unsupported_variant(
            node,
            "shape inference is not implemented for an axes input tensor; supply the axes in \
             OpParams::axes",
        ));
    }
    expect_inputs(node, inputs, 1)?;
    require_params(node, Req::AXES)?;
    let input = &inputs[0];
    let rank = input.rank();
    let mut removed = vec![false; rank];
    if node.params.axes.is_empty() {
        for axis in 0..rank {
            if input.dims()[axis].as_const() == Some(1) {
                removed[axis] = true;
            } else if input.dims()[axis].as_const().is_none() {
                return Err(unsupported_variant(
                    node,
                    format!(
                        "Squeeze without axes over symbolic axis {axis} ({}); the output rank \
                         would depend on the bound value",
                        input.dims()[axis]
                    ),
                ));
            }
        }
    } else {
        for &raw in &node.params.axes {
            let axis = normalize_axis(node, raw, rank)?;
            if removed[axis] {
                return Err(invalid(node, format!("op squeeze lists axis {axis} twice")));
            }
            match input.dims()[axis].as_const() {
                Some(1) => removed[axis] = true,
                Some(extent) => {
                    return Err(shape_mismatch(
                        node,
                        format!("extent 1 on squeeze axis {axis}"),
                        format!("extent {extent}"),
                    ))
                }
                None => {
                    return Err(unsupported_variant(
                        node,
                        format!(
                            "cannot prove that symbolic axis {axis} ({}) has extent 1",
                            input.dims()[axis]
                        ),
                    ))
                }
            }
        }
    }
    let shape = input
        .dims()
        .iter()
        .enumerate()
        .filter(|(axis, _)| !removed[*axis])
        .map(|(_, dim)| *dim)
        .collect();
    Ok(vec![input.with_shape(shape)])
}

fn unsqueeze(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if inputs.len() == 2 {
        return Err(unsupported_variant(
            node,
            "shape inference is not implemented for an axes input tensor; supply the axes in \
             OpParams::axes",
        ));
    }
    expect_inputs(node, inputs, 1)?;
    require_params(node, Req::AXES)?;
    let input = &inputs[0];
    if node.params.axes.is_empty() {
        return Err(invalid(
            node,
            "op unsqueeze needs at least one axis to insert",
        ));
    }
    let out_rank = input
        .rank()
        .checked_add(node.params.axes.len())
        .ok_or_else(|| overflow(node, "unsqueeze rank"))?;
    if out_rank > MAX_RANK {
        return Err(invalid(
            node,
            format!("op unsqueeze would produce rank {out_rank}, above the {MAX_RANK}-axis limit"),
        ));
    }
    let mut inserted = vec![false; out_rank];
    for &raw in &node.params.axes {
        let axis = normalize_axis(node, raw, out_rank)?;
        if inserted[axis] {
            return Err(invalid(node, format!("op unsqueeze repeats axis {axis}")));
        }
        inserted[axis] = true;
    }
    let mut shape = Vec::with_capacity(out_rank);
    let mut source = 0;
    for axis in 0..out_rank {
        if inserted[axis] {
            shape.push(DimExpr::Const(1));
        } else {
            let dim = input.dims().get(source).copied().ok_or_else(|| {
                invalid(
                    node,
                    "internal: unsqueeze consumed more axes than the operand has",
                )
            })?;
            shape.push(dim);
            source += 1;
        }
    }
    Ok(vec![input.with_shape(shape)])
}

fn flatten(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_params(node, Req::AXES)?;
    let input = &inputs[0];
    let rank = input.rank();
    let axis = match node.params.axes.as_slice() {
        [] => 1.min(rank),
        [axis] => {
            // Flatten's axis may equal the rank (a trailing empty product),
            // which `normalize_axis` deliberately does not allow. The ONNX
            // range is `[-rank, rank]`: a negative axis counts from the back,
            // so `-1` names the last existing axis.
            let rank_i64 = i64::try_from(rank)
                .map_err(|_| invalid(node, format!("rank {rank} does not fit in i64")))?;
            let resolved = if *axis < 0 {
                axis.checked_add(rank_i64).ok_or_else(|| {
                    invalid(node, format!("op flatten axis {axis} is out of range"))
                })?
            } else {
                *axis
            };
            let Ok(resolved) = usize::try_from(resolved) else {
                return Err(invalid(
                    node,
                    format!("op flatten axis {axis} is outside 0..={rank}"),
                ));
            };
            if resolved > rank {
                return Err(invalid(
                    node,
                    format!("op flatten axis {axis} is outside 0..={rank}"),
                ));
            }
            resolved
        }
        axes => {
            return Err(invalid(
                node,
                format!("op flatten takes exactly one axis; the node supplies {axes:?}"),
            ))
        }
    };
    let leading = product_of(node, &input.dims()[..axis], "flatten leading axes")?;
    let trailing = product_of(node, &input.dims()[axis..], "flatten trailing axes")?;
    Ok(vec![input.with_shape(vec![leading, trailing])])
}

fn concat(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    require_params(node, Req::AXES)?;
    let first = inputs
        .first()
        .ok_or_else(|| invalid(node, "op concat needs at least one input"))?;
    let rank = first.rank();
    let axis = single_axis(node, rank, 0)?;
    let mut total = first.dims()[axis];
    for (index, input) in inputs.iter().enumerate().skip(1) {
        if input.dtype != first.dtype {
            return Err(invalid(
                node,
                format!(
                    "op concat needs one dtype; operand 0 is {} and operand {index} is {}",
                    first.dtype, input.dtype
                ),
            ));
        }
        if input.rank() != rank {
            return Err(invalid(
                node,
                format!(
                    "op concat needs one rank; operand 0 is rank {rank} and operand {index} is \
                     rank {}",
                    input.rank()
                ),
            ));
        }
        for position in 0..rank {
            if position != axis && input.dims()[position] != first.dims()[position] {
                return Err(shape_mismatch(
                    node,
                    format!("every axis but {axis} to match {}", render(first.dims())),
                    format!(
                        "operand {index} is {} (differs on axis {position})",
                        render(input.dims())
                    ),
                ));
            }
        }
        total = concat_dim(node, axis, total, input.dims()[axis])?;
    }
    let mut shape = first.dims().to_vec();
    shape[axis] = total;
    Ok(vec![first.with_shape(shape)])
}

/// Sum one concatenated extent, when the sum is representable.
fn concat_dim(
    node: &Node,
    axis: usize,
    left: DimExpr,
    right: DimExpr,
) -> Result<DimExpr, Fn3Error> {
    match (left, right) {
        (DimExpr::Const(left), DimExpr::Const(right)) => left
            .checked_add(right)
            .map(DimExpr::Const)
            .ok_or_else(|| overflow(node, "concat extent")),
        (DimExpr::Const(0), other) | (other, DimExpr::Const(0)) => Ok(other),
        (left, right) => Err(invalid(
            node,
            format!(
                "op concat cannot express the symbolic sum {left} + {right} on axis {axis}; \
                 only constant extents and a zero extent are representable"
            ),
        )),
    }
}

/// `Slice(data, starts, ends, axes?, steps?)` with constant bounds.
///
/// The bounded constants live in [`OpParams`] because a [`TensorType`] carries
/// no element values; the importer folds them there. A bound that cannot be
/// folded is reported rather than guessed.
fn slice(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if !(1..=5).contains(&inputs.len()) {
        return Err(invalid(
            node,
            format!("op slice takes 1 to 5 inputs, found {}", inputs.len()),
        ));
    }
    require_params(node, Req::AXES.union(Req::SLICE_BOUNDS))?;
    let input = &inputs[0];
    let rank = input.rank();
    let starts = &node.params.slice_starts;
    let ends = &node.params.slice_ends;
    if starts.is_empty() || ends.is_empty() {
        return Err(unsupported_variant(
            node,
            "Slice needs constant starts and ends; the importer folds them into \
             OpParams::slice_starts and OpParams::slice_ends, and this pass does not read \
             tensor payloads",
        ));
    }
    if starts.len() != ends.len() {
        return Err(invalid(
            node,
            format!(
                "op slice has {} starts and {} ends",
                starts.len(),
                ends.len()
            ),
        ));
    }
    let steps: Vec<i64> = if node.params.slice_steps.is_empty() {
        vec![1; starts.len()]
    } else {
        if node.params.slice_steps.len() != starts.len() {
            return Err(invalid(
                node,
                format!(
                    "op slice has {} starts and {} steps",
                    starts.len(),
                    node.params.slice_steps.len()
                ),
            ));
        }
        node.params.slice_steps.clone()
    };
    let axes: Vec<usize> = if node.params.axes.is_empty() {
        (0..starts.len()).collect()
    } else {
        if node.params.axes.len() != starts.len() {
            return Err(invalid(
                node,
                format!(
                    "op slice has {} starts and {} axes",
                    starts.len(),
                    node.params.axes.len()
                ),
            ));
        }
        let mut axes = Vec::with_capacity(node.params.axes.len());
        for &raw in &node.params.axes {
            let axis = normalize_axis(node, raw, rank)?;
            if axes.contains(&axis) {
                return Err(invalid(node, format!("op slice lists axis {axis} twice")));
            }
            axes.push(axis);
        }
        axes
    };
    let mut shape = input.dims().to_vec();
    for (position, axis) in axes.iter().enumerate() {
        shape[*axis] = slice_dim(
            node,
            shape[*axis],
            starts[position],
            ends[position],
            steps[position],
            *axis,
        )?;
    }
    Ok(vec![input.with_shape(shape)])
}

/// The extent of one sliced axis, under ONNX clamp semantics.
fn slice_dim(
    node: &Node,
    dim: DimExpr,
    start: i64,
    end: i64,
    step: i64,
    axis: usize,
) -> Result<DimExpr, Fn3Error> {
    if step == 0 {
        return Err(invalid(
            node,
            format!("op slice step 0 on axis {axis} is not a sequence"),
        ));
    }
    match dim.as_const() {
        Some(extent) => Ok(DimExpr::Const(onnx_slice_count(
            u128::from(extent),
            start,
            end,
            step,
        ))),
        None => {
            // A symbolic axis keeps its symbol only for the whole-axis form;
            // any other bound would depend on the extent's value.
            if step == 1 && start == 0 && end == i64::MAX {
                Ok(dim)
            } else {
                Err(unsupported_variant(
                    node,
                    format!(
                        "slice of the symbolic axis {axis} ({dim}) from {start} to {end} with \
                         step {step} needs the extent's value; only the whole-axis form is \
                         representable"
                    ),
                ))
            }
        }
    }
}

/// ONNX `Slice` clamp rule: how many indices one axis selects.
fn onnx_slice_count(extent: u128, start: i64, end: i64, step: i64) -> u64 {
    if extent == 0 {
        return 0;
    }
    let extent = extent as i128;
    let step = i128::from(step);
    let (first, last) = if step > 0 {
        let first = if start < 0 {
            (extent + i128::from(start)).max(0)
        } else {
            i128::from(start).min(extent)
        };
        let last = if end < 0 {
            (extent + i128::from(end)).max(0)
        } else {
            i128::from(end).min(extent)
        };
        (first, last)
    } else {
        let first = if start < 0 {
            (extent + i128::from(start)).clamp(0, extent - 1)
        } else {
            i128::from(start).min(extent - 1)
        };
        let last = if end < 0 {
            (extent + i128::from(end)).clamp(-1, extent - 1)
        } else {
            i128::from(end).min(extent - 1)
        };
        (first, last)
    };
    let count = if step > 0 {
        if last <= first {
            0
        } else {
            (last - first + step - 1) / step
        }
    } else if first <= last {
        0
    } else {
        (first - last + (-step) - 1) / (-step)
    };
    u64::try_from(count).unwrap_or(0)
}

/// `Split(data, split?)` into `node.outputs.len()` parts along one axis.
fn split(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if !(1..=2).contains(&inputs.len()) {
        return Err(invalid(
            node,
            format!("op split takes 1 or 2 inputs, found {}", inputs.len()),
        ));
    }
    require_params(node, Req::AXES.union(Req::SPLIT_SIZES))?;
    if node.params.split_sizes.is_empty() && inputs.len() > 1 {
        return Err(unsupported_variant(
            node,
            "the split-sizes operand is not a folded constant; the IR cannot assume an equal \
             split when the node supplies sizes",
        ));
    }
    let input = &inputs[0];
    let rank = input.rank();
    let axis = single_axis(node, rank, 0)?;
    let outputs = node.outputs.len();
    if outputs == 0 {
        return Err(invalid(node, "op split writes no outputs"));
    }
    let extent = input.dims()[axis];
    let sizes: Vec<Option<u64>> = if node.params.split_sizes.is_empty() {
        if extent.as_const().is_none() {
            if outputs == 1 {
                return Ok(vec![input.clone()]);
            }
            return Err(unsupported_variant(
                node,
                format!(
                    "an equal split of the symbolic extent {extent} into {outputs} parts needs \
                     the extent's value"
                ),
            ));
        }
        let extent = extent.as_const().expect("the symbolic case returned above");
        if extent % outputs as u64 != 0 {
            return Err(shape_mismatch(
                node,
                format!("an equal split of extent {extent} into {outputs} parts"),
                format!("{extent} is not divisible by {outputs}"),
            ));
        }
        let each = extent / outputs as u64;
        vec![Some(each); outputs]
    } else {
        if node.params.split_sizes.len() != outputs {
            return Err(invalid(
                node,
                format!(
                    "op split has {} sizes for {outputs} outputs",
                    node.params.split_sizes.len()
                ),
            ));
        }
        let mut sizes = Vec::with_capacity(outputs);
        let mut total = 0u64;
        for &raw in &node.params.split_sizes {
            let size = u64::try_from(raw)
                .map_err(|_| invalid(node, format!("op split size {raw} is negative")))?;
            total = total
                .checked_add(size)
                .ok_or_else(|| overflow(node, "split sizes"))?;
            sizes.push(Some(size));
        }
        if let Some(extent) = extent.as_const() {
            if total != extent {
                return Err(shape_mismatch(
                    node,
                    format!("split sizes to sum to the axis extent {extent}"),
                    format!("they sum to {total}"),
                ));
            }
        }
        sizes
    };
    let mut shape = input.dims().to_vec();
    let mut results = Vec::with_capacity(outputs);
    for size in sizes {
        let Some(size) = size else {
            return Err(unsupported_variant(
                node,
                "an equal split of a symbolic extent",
            ));
        };
        shape[axis] = DimExpr::Const(size);
        results.push(input.with_shape(shape.clone()));
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Convolution and pooling
// ---------------------------------------------------------------------------

/// `Conv(X, W, B?)` over `[N, C, spatial...]`, with groups and auto_pad.
fn conv(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if !(2..=3).contains(&inputs.len()) {
        return Err(invalid(
            node,
            format!("op conv takes 2 or 3 inputs, found {}", inputs.len()),
        ));
    }
    require_params(
        node,
        Req::KERNEL
            .union(Req::STRIDES)
            .union(Req::PADS)
            .union(Req::DILATIONS)
            .union(Req::GROUP)
            .union(Req::AUTO_PAD),
    )?;
    let input = &inputs[0];
    let weight = &inputs[1];
    if input.dtype != weight.dtype {
        return Err(invalid(
            node,
            format!(
                "op conv needs the data and the weight in one dtype, found {} and {}",
                input.dtype, weight.dtype
            ),
        ));
    }
    let rank = input.rank();
    if rank < 3 {
        return Err(invalid(
            node,
            format!(
                "op conv needs an [N, C, spatial...] operand of rank 3 or more, found {}",
                input
            ),
        ));
    }
    if weight.rank() != rank {
        return Err(shape_mismatch(
            node,
            format!("the weight to have rank {rank} (one kernel axis per spatial axis)"),
            format!("{}", weight),
        ));
    }
    let spatial = rank - 2;
    let kernel = kernel_extents(node, spatial)?;
    let strides = axis_extents(node, &node.params.strides, spatial, 1, "strides")?;
    let dilations = axis_extents(node, &node.params.dilations, spatial, 1, "dilations")?;
    let group = match node.params.group {
        0 | 1 => 1usize,
        positive if positive > 0 => usize::try_from(positive)
            .map_err(|_| invalid(node, format!("op conv group {positive} is too large")))?,
        negative => {
            return Err(invalid(
                node,
                format!("op conv group {negative} is negative"),
            ))
        }
    };
    let in_channels = const_dim(node, input.dims()[1], "conv input channels")?;
    let weight_in = const_dim(node, weight.dims()[1], "conv weight input channels")?;
    let out_channels = const_dim(node, weight.dims()[0], "conv output channels")?;
    if weight_in
        .checked_mul(group as u64)
        .ok_or_else(|| overflow(node, "conv grouped input channels"))?
        != in_channels
    {
        return Err(shape_mismatch(
            node,
            format!(
                "the weight to have {in_channels}/{group} = {} input channels for group = {group}",
                in_channels / group as u64
            ),
            format!("it has {weight_in}"),
        ));
    }
    if out_channels % group as u64 != 0 {
        return Err(invalid(
            node,
            format!("op conv output channels {out_channels} are not divisible by group {group}"),
        ));
    }
    for (axis, &extent) in kernel.iter().enumerate() {
        let declared = weight.dims()[axis + 2];
        if declared != DimExpr::Const(extent as u64) {
            return Err(shape_mismatch(
                node,
                format!("the kernel extent {extent} on spatial axis {axis}"),
                format!("the weight declares {declared}"),
            ));
        }
    }
    if let Some(bias) = inputs.get(2) {
        if bias.dtype != input.dtype {
            return Err(invalid(
                node,
                format!(
                    "op conv needs the bias in the data dtype {}, found {}",
                    input.dtype, bias.dtype
                ),
            ));
        }
        if bias.rank() != 1 || bias.dims()[0] != DimExpr::Const(out_channels) {
            return Err(shape_mismatch(
                node,
                format!("a rank-1 [{out_channels}] bias"),
                format!("{bias}"),
            ));
        }
    }
    let spatial_in = spatial_extents(node, input, "conv input")?;
    let pads = resolve_pads(node, spatial, &spatial_in, &kernel, &strides, &dilations)?;
    let mut shape = Vec::with_capacity(rank);
    shape.push(input.dims()[0]);
    shape.push(DimExpr::Const(out_channels));
    for axis in 0..spatial {
        let (begin, end) = pads[axis];
        shape.push(DimExpr::Const(window_output(
            node,
            "conv",
            spatial_in[axis],
            kernel[axis],
            strides[axis],
            dilations[axis],
            begin,
            end,
            false,
        )?));
    }
    Ok(vec![input.with_shape(shape)])
}

/// `MaxPool`/`AveragePool` over `[N, C, spatial...]`.
fn pool(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    let mut allowed = Req::KERNEL
        .union(Req::STRIDES)
        .union(Req::PADS)
        .union(Req::DILATIONS)
        .union(Req::AUTO_PAD)
        .union(Req::CEIL_MODE);
    if node.op == OpId::AveragePool {
        allowed = allowed.union(Req::COUNT_INCLUDE_PAD);
    }
    require_params(node, allowed)?;
    let input = &inputs[0];
    let rank = input.rank();
    if rank < 3 {
        return Err(invalid(
            node,
            format!(
                "op {} needs an [N, C, spatial...] operand of rank 3 or more, found {input}",
                node.op
            ),
        ));
    }
    let spatial = rank - 2;
    let kernel = kernel_extents(node, spatial)?;
    let strides = axis_extents(node, &node.params.strides, spatial, 1, "strides")?;
    let dilations = axis_extents(node, &node.params.dilations, spatial, 1, "dilations")?;
    let spatial_in = spatial_extents(node, input, node.op.name())?;
    let pads = resolve_pads(node, spatial, &spatial_in, &kernel, &strides, &dilations)?;
    let mut shape = Vec::with_capacity(rank);
    shape.push(input.dims()[0]);
    shape.push(input.dims()[1]);
    for axis in 0..spatial {
        let (begin, end) = pads[axis];
        shape.push(DimExpr::Const(window_output(
            node,
            node.op.name(),
            spatial_in[axis],
            kernel[axis],
            strides[axis],
            dilations[axis],
            begin,
            end,
            node.params.ceil_mode,
        )?));
    }
    Ok(vec![input.with_shape(shape)])
}

/// `GlobalAveragePool`: every spatial axis collapses to extent 1.
fn global_average_pool(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_params(node, Req::NONE)?;
    let input = &inputs[0];
    let rank = input.rank();
    if rank < 3 {
        return Err(invalid(
            node,
            format!(
                "op global_average_pool needs an [N, C, spatial...] operand of rank 3 or more, \
                 found {input}"
            ),
        ));
    }
    let mut shape = Vec::with_capacity(rank);
    shape.push(input.dims()[0]);
    shape.push(input.dims()[1]);
    shape.extend(std::iter::repeat_n(DimExpr::Const(1), rank - 2));
    Ok(vec![input.with_shape(shape)])
}

/// `BatchNormalization(X, scale, B, mean, var)`: identity on the data shape.
fn batch_norm(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 5)?;
    require_params(node, Req::EPSILON)?;
    let input = &inputs[0];
    let rank = input.rank();
    if rank < 2 {
        return Err(invalid(
            node,
            format!("op batch_norm needs an [N, C, ...] operand of rank 2 or more, found {input}"),
        ));
    }
    if !node.params.epsilon.is_finite() || node.params.epsilon < 0.0 {
        return Err(invalid(
            node,
            format!(
                "op batch_norm epsilon {} must be finite and non-negative",
                node.params.epsilon
            ),
        ));
    }
    let channels = input.dims()[1];
    for (position, name) in ["scale", "B", "mean", "var"].iter().enumerate() {
        let operand = &inputs[position + 1];
        if operand.dtype != input.dtype {
            return Err(invalid(
                node,
                format!(
                    "op batch_norm {name} is {}, the data is {}",
                    operand.dtype, input.dtype
                ),
            ));
        }
        if operand.rank() != 1 || operand.dims()[0] != channels {
            return Err(shape_mismatch(
                node,
                format!("a rank-1 [{channels}] {name}"),
                format!("{operand}"),
            ));
        }
    }
    Ok(vec![input.clone()])
}

/// `Resize(X, roi?, scales?, sizes?)` in its 4-D nearest/linear form.
fn resize(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if !(1..=4).contains(&inputs.len()) {
        return Err(invalid(
            node,
            format!("op resize takes 1 to 4 inputs, found {}", inputs.len()),
        ));
    }
    require_params(
        node,
        Req::RESIZE_MODES
            .union(Req::RESIZE_SCALES)
            .union(Req::RESIZE_SIZES),
    )?;
    let input = &inputs[0];
    if input.rank() != 4 {
        return Err(unsupported_variant(
            node,
            format!(
                "the Resize rule implements the 4-D [N, C, H, W] nearest/linear form, found \
                 {input}"
            ),
        ));
    }
    if node.params.resize_mode == ResizeMode::Cubic {
        return Err(unsupported_variant(
            node,
            "cubic Resize has no shape rule or kernel in this revision",
        ));
    }
    if node.params.coordinate_transformation_mode == CoordinateTransformationMode::TfCropAndResize {
        return Err(unsupported_variant(
            node,
            "tf_crop_and_resize is an ROI crop, which this runtime does not represent",
        ));
    }
    // Positional form: input 1 is `roi`, which is accepted and ignored. ONNX
    // defines its values only for `tf_crop_and_resize` — rejected above — so a
    // four-zero placeholder and an omitted operand describe the same resize.
    // The kernel reads input 0 and the constant scales/sizes in the params, so
    // the roi operand neither changes the output nor needs a known shape.
    if !node.params.resize_scales.is_empty() && !node.params.resize_sizes.is_empty() {
        return Err(invalid(
            node,
            "op resize may carry either scales or sizes, but the node supplies both",
        ));
    }
    let mut shape = Vec::with_capacity(4);
    if !node.params.resize_sizes.is_empty() {
        if node.params.resize_sizes.len() != 4 {
            return Err(invalid(
                node,
                format!(
                    "op resize sizes has {} extents for a rank-4 operand",
                    node.params.resize_sizes.len()
                ),
            ));
        }
        for &size in &node.params.resize_sizes {
            if size <= 0 {
                return Err(invalid(
                    node,
                    format!("op resize output extent {size} must be positive"),
                ));
            }
            shape.push(DimExpr::Const(size as u64));
        }
    } else if !node.params.resize_scales.is_empty() {
        if node.params.resize_scales.len() != 4 {
            return Err(invalid(
                node,
                format!(
                    "op resize scales has {} extents for a rank-4 operand",
                    node.params.resize_scales.len()
                ),
            ));
        }
        for (axis, &scale) in node.params.resize_scales.iter().enumerate() {
            if !scale.is_finite() || scale <= 0.0 {
                return Err(invalid(
                    node,
                    format!("op resize scale {scale} on axis {axis} must be finite and positive"),
                ));
            }
            let Some(extent) = input.dims()[axis].as_const() else {
                return Err(unsupported_variant(
                    node,
                    format!(
                        "resizing the symbolic axis {axis} ({}) by {scale} needs the extent's \
                         value",
                        input.dims()[axis]
                    ),
                ));
            };
            let scaled = (extent as f64) * f64::from(scale);
            if scaled > u64::MAX as f64 {
                return Err(overflow(node, "resize extent"));
            }
            shape.push(DimExpr::Const(scaled.floor() as u64));
        }
    } else {
        return Err(unsupported_variant(
            node,
            "op resize needs constant scales or constant sizes; the importer folds them into \
             OpParams and this pass does not read tensor payloads",
        ));
    }
    Ok(vec![input.with_shape(shape)])
}

/// `Pad(data, pads?, constant_value?)`.
fn pad(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if !(1..=3).contains(&inputs.len()) {
        return Err(invalid(
            node,
            format!("op pad takes 1 to 3 inputs, found {}", inputs.len()),
        ));
    }
    require_params(node, Req::PADS.union(Req::PAD_MODE).union(Req::PAD_VALUE))?;
    let input = &inputs[0];
    let rank = input.rank();
    if node.params.pad_mode == PadMode::Wrap {
        return Err(unsupported_variant(
            node,
            "wrap padding is not implemented; constant, reflect and edge are",
        ));
    }
    if node.params.pad_mode != PadMode::Constant && node.params.pad_value != 0.0 {
        return Err(invalid(
            node,
            format!(
                "op pad in {} mode cannot use the fill value {}",
                node.params.pad_mode.name(),
                node.params.pad_value
            ),
        ));
    }
    if node.params.pads.is_empty() {
        return Err(unsupported_variant(
            node,
            "Pad needs constant pads; the importer folds them into OpParams::pads, and this pass \
             does not read tensor payloads",
        ));
    }
    if node.params.pads.len() != 2 * rank {
        return Err(invalid(
            node,
            format!(
                "op pad has {} pad extents for a rank-{rank} operand; ONNX supplies {rank} begin \
                 extents followed by {rank} end extents",
                node.params.pads.len()
            ),
        ));
    }
    let mut shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        let begin = i128::from(node.params.pads[axis]);
        let end = i128::from(node.params.pads[rank + axis]);
        match input.dims()[axis] {
            DimExpr::Const(extent) => {
                let padded = i128::from(extent) + begin + end;
                if padded < 0 {
                    return Err(shape_mismatch(
                        node,
                        format!(
                            "axis {axis} of extent {extent} with pads {begin} + {end} to stay \
                             non-negative"
                        ),
                        format!("it would have {padded} elements"),
                    ));
                }
                shape.push(DimExpr::Const(padded as u64));
            }
            symbol => {
                if begin + end != 0 {
                    return Err(unsupported_variant(
                        node,
                        format!(
                            "padding the symbolic axis {axis} ({symbol}) by {begin} + {end} \
                             needs the extent's value"
                        ),
                    ));
                }
                shape.push(symbol);
            }
        }
    }
    Ok(vec![input.with_shape(shape)])
}

/// `Clip(data, min?, max?)` with constant bounds.
fn clip(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if !(1..=3).contains(&inputs.len()) {
        return Err(invalid(
            node,
            format!("op clip takes 1 to 3 inputs, found {}", inputs.len()),
        ));
    }
    require_params(node, Req::CLIP)?;
    if inputs.len() >= 2 && node.params.clip_min.is_none() {
        return Err(unsupported_variant(
            node,
            "the Clip min operand is not a folded constant; the importer folds it into \
             OpParams::clip_min",
        ));
    }
    if inputs.len() >= 3 && node.params.clip_max.is_none() {
        return Err(unsupported_variant(
            node,
            "the Clip max operand is not a folded constant; the importer folds it into \
             OpParams::clip_max",
        ));
    }
    if let Some(min) = node.params.clip_min {
        if !min.is_finite() {
            return Err(invalid(node, format!("op clip min {min} is not finite")));
        }
    }
    if let Some(max) = node.params.clip_max {
        if !max.is_finite() {
            return Err(invalid(node, format!("op clip max {max} is not finite")));
        }
    }
    if let (Some(min), Some(max)) = (node.params.clip_min, node.params.clip_max) {
        if min > max {
            return Err(invalid(
                node,
                format!("op clip min {min} is above max {max}"),
            ));
        }
    }
    Ok(vec![inputs[0].clone()])
}

/// `TopK(data, k?)` producing values and i64 indices.
fn top_k(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    if !(1..=2).contains(&inputs.len()) {
        return Err(invalid(
            node,
            format!("op top_k takes 1 or 2 inputs, found {}", inputs.len()),
        ));
    }
    require_params(
        node,
        Req::TOPK_AXIS.union(Req::TOPK_K).union(Req::TOPK_ORDER),
    )?;
    let input = &inputs[0];
    let rank = input.rank();
    if rank == 0 {
        return Err(invalid(
            node,
            "op top_k needs a rank-1 or higher operand to name an axis",
        ));
    }
    let axis = normalize_axis(node, node.params.topk_axis, rank)?;
    if node.params.topk_k < 0 {
        return Err(unsupported_variant(
            node,
            "TopK needs a constant k; the importer folds it into OpParams::topk_k and this pass \
             does not read tensor payloads",
        ));
    }
    let k = node.params.topk_k as u64;
    if let Some(extent) = input.dims()[axis].as_const() {
        if k > extent {
            return Err(shape_mismatch(
                node,
                format!("k <= the axis {axis} extent {extent}"),
                format!("k = {k}"),
            ));
        }
    }
    let mut shape = input.dims().to_vec();
    shape[axis] = DimExpr::Const(k);
    Ok(vec![
        input.with_shape(shape.clone()),
        TensorType::new(DType::I64, shape),
    ])
}

// ---------------------------------------------------------------------------
// Normalization and reduction
// ---------------------------------------------------------------------------

fn softmax(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_params(node, Req::AXES)?;
    let input = &inputs[0];
    if input.rank() == 0 {
        return Err(invalid(
            node,
            "op softmax needs a rank-1 or higher operand to name an axis",
        ));
    }
    single_axis(node, input.rank(), -1)?;
    Ok(vec![input.clone()])
}

fn layer_norm(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 3)?;
    require_params(node, Req::AXES.union(Req::EPSILON))?;
    let input = &inputs[0];
    let weight = &inputs[1];
    let bias = &inputs[2];
    let rank = input.rank();
    if rank == 0 {
        return Err(invalid(
            node,
            "op layernorm needs a rank-1 or higher operand to name an axis",
        ));
    }
    let axis = match node.params.axes.as_slice() {
        [] => rank - 1,
        [axis] => normalize_axis(node, *axis, rank)?,
        axes => {
            return Err(unsupported_variant(
                node,
                format!(
                    "shape inference is implemented for the trailing-axis LayerNorm form only; \
                     axes {axes:?} were supplied"
                ),
            ))
        }
    };
    if axis != rank - 1 {
        return Err(unsupported_variant(
            node,
            format!(
                "shape inference is implemented for the trailing-axis LayerNorm form only; \
                 axis {axis} of rank {rank} was supplied"
            ),
        ));
    }
    if !node.params.epsilon.is_finite() || node.params.epsilon < 0.0 {
        return Err(invalid(
            node,
            format!(
                "op layernorm epsilon {} must be finite and non-negative",
                node.params.epsilon
            ),
        ));
    }
    for (position, operand) in [input, weight, bias].iter().enumerate() {
        if !operand.dtype.is_compute() {
            return Err(invalid(
                node,
                format!(
                    "op layernorm is an f32 contract; operand {position} is {}",
                    operand.dtype
                ),
            ));
        }
    }
    let hidden = input.dims()[rank - 1];
    for (operand, what) in [(weight, "weight"), (bias, "bias")] {
        if operand.rank() != 1 {
            return Err(invalid(
                node,
                format!(
                    "op layernorm {what} must be rank 1 [hidden], found {}",
                    render(operand.dims())
                ),
            ));
        }
        if operand.dims()[0] != hidden {
            return Err(shape_mismatch(
                node,
                format!("{what} extent to match the trailing extent {hidden}"),
                format!("{what} is {}", render(operand.dims())),
            ));
        }
    }
    Ok(vec![input.clone()])
}

fn reduce(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_params(node, Req::AXES.union(Req::KEEP_DIMS))?;
    let input = &inputs[0];
    let rank = input.rank();
    let axes = resolve_axes(node, rank)?;
    let mut shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        if axes.contains(&axis) {
            if node.params.keep_dims {
                shape.push(DimExpr::Const(1));
            }
        } else {
            shape.push(input.dims()[axis]);
        }
    }
    Ok(vec![input.with_shape(shape)])
}

/// Resolve reduction axes: empty means every axis, matching the CPU kernel.
fn resolve_axes(node: &Node, rank: usize) -> Result<Vec<usize>, Fn3Error> {
    if node.params.axes.is_empty() {
        return Ok((0..rank).collect());
    }
    let mut axes = Vec::with_capacity(node.params.axes.len());
    for &raw in &node.params.axes {
        let axis = normalize_axis(node, raw, rank)?;
        if axes.contains(&axis) {
            return Err(invalid(
                node,
                format!("op {} lists axis {axis} twice", node.op),
            ));
        }
        axes.push(axis);
    }
    axes.sort_unstable();
    Ok(axes)
}

// ---------------------------------------------------------------------------
// Gather and Shape
// ---------------------------------------------------------------------------

/// `Shape(data)`: a rank-length `i64` vector of the operand's extents.
fn shape_op(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_params(node, Req::NONE)?;
    let rank = inputs[0].rank();
    Ok(vec![TensorType::new(
        DType::I64,
        [DimExpr::Const(rank as u64)],
    )])
}

fn gather(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 2)?;
    require_params(node, Req::AXES)?;
    let data = &inputs[0];
    let indices = &inputs[1];
    let axis = single_axis(node, data.rank(), 0)?;
    if !indices.dtype.is_integer() {
        return Err(invalid(
            node,
            format!(
                "op gather indices must be i32 or i64, found {}",
                indices.dtype
            ),
        ));
    }
    let mut shape = Vec::with_capacity(data.rank().saturating_sub(1) + indices.rank());
    shape.extend_from_slice(&data.dims()[..axis]);
    shape.extend_from_slice(indices.dims());
    shape.extend_from_slice(&data.dims()[axis + 1..]);
    Ok(vec![data.with_shape(shape)])
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn expect_inputs(node: &Node, inputs: &[TensorType], expected: usize) -> Result<(), Fn3Error> {
    if inputs.len() == expected {
        Ok(())
    } else {
        Err(invalid(
            node,
            format!(
                "op {} takes {expected} input(s), found {}",
                node.op,
                inputs.len()
            ),
        ))
    }
}

/// A constant extent, named for the diagnostic when it is symbolic.
fn const_dim(node: &Node, dim: DimExpr, what: &'static str) -> Result<u64, Fn3Error> {
    dim.as_const().ok_or_else(|| {
        unsupported_variant(
            node,
            format!("{what} is the symbolic extent {dim}; this rule needs a constant"),
        )
    })
}

/// Kernel extents per spatial axis.
fn kernel_extents(node: &Node, spatial: usize) -> Result<Vec<u64>, Fn3Error> {
    if node.params.kernel_shape.is_empty() {
        return Err(invalid(
            node,
            format!(
                "op {} needs kernel_shape for its {spatial} spatial axis/axes",
                node.op
            ),
        ));
    }
    axis_extents(node, &node.params.kernel_shape, spatial, 1, "kernel_shape")
}

/// A per-spatial-axis attribute vector, defaulted or validated.
fn axis_extents(
    node: &Node,
    values: &[i64],
    spatial: usize,
    default: u64,
    what: &'static str,
) -> Result<Vec<u64>, Fn3Error> {
    if values.is_empty() {
        return Ok(vec![default; spatial]);
    }
    if values.len() != spatial {
        return Err(invalid(
            node,
            format!(
                "op {} has {} {what} extents for a {spatial}-axis spatial rank",
                node.op,
                values.len()
            ),
        ));
    }
    let mut extents = Vec::with_capacity(spatial);
    for &raw in values {
        let extent = u64::try_from(raw).map_err(|_| {
            invalid(
                node,
                format!("op {} has the non-positive {what} extent {raw}", node.op),
            )
        })?;
        if extent == 0 {
            return Err(invalid(
                node,
                format!("op {} has a zero {what} extent", node.op),
            ));
        }
        extents.push(extent);
    }
    Ok(extents)
}

/// The constant spatial extents `[2..]` of a convolution or pooling operand.
fn spatial_extents(
    node: &Node,
    input: &TensorType,
    what: &'static str,
) -> Result<Vec<u64>, Fn3Error> {
    let mut extents = Vec::with_capacity(input.rank().saturating_sub(2));
    for axis in 2..input.rank() {
        extents.push(const_dim(
            node,
            input.dims()[axis],
            match what {
                "conv input" => "the conv input spatial extent",
                _ => "the pooling input spatial extent",
            },
        )?);
    }
    Ok(extents)
}

/// Resolve `auto_pad` and explicit `pads` into begin/end pairs per spatial axis.
fn resolve_pads(
    node: &Node,
    spatial: usize,
    input: &[u64],
    kernel: &[u64],
    strides: &[u64],
    dilations: &[u64],
) -> Result<Vec<(i64, i64)>, Fn3Error> {
    let pads = &node.params.pads;
    match node.params.auto_pad {
        AutoPad::NotSet => {
            if pads.is_empty() {
                return Ok(vec![(0, 0); spatial]);
            }
            if pads.len() != 2 * spatial {
                return Err(invalid(
                    node,
                    format!(
                        "op {} has {} pads for a {spatial}-axis spatial rank; ONNX supplies \
                         {spatial} begin extents followed by {spatial} end extents",
                        node.op,
                        pads.len()
                    ),
                ));
            }
            let mut pairs = Vec::with_capacity(spatial);
            for axis in 0..spatial {
                let begin = pads[axis];
                let end = pads[spatial + axis];
                if begin < 0 || end < 0 {
                    return Err(invalid(
                        node,
                        format!(
                            "op {} has the negative pads ({begin}, {end}) on spatial axis \
                             {axis}; convolution and pooling pads are non-negative",
                            node.op
                        ),
                    ));
                }
                pairs.push((begin, end));
            }
            Ok(pairs)
        }
        AutoPad::Valid => {
            if !pads.is_empty() {
                return Err(invalid(
                    node,
                    format!(
                        "op {} sets auto_pad = valid and explicit pads; ONNX forbids combining \
                         them",
                        node.op
                    ),
                ));
            }
            Ok(vec![(0, 0); spatial])
        }
        upper @ (AutoPad::SameUpper | AutoPad::SameLower) => {
            if !pads.is_empty() {
                return Err(invalid(
                    node,
                    format!(
                        "op {} sets auto_pad = {} and explicit pads; ONNX forbids combining them",
                        node.op,
                        upper.name()
                    ),
                ));
            }
            let mut pairs = Vec::with_capacity(spatial);
            for axis in 0..spatial {
                let effective = dilations[axis] * (kernel[axis] - 1) + 1;
                let out = input[axis].div_ceil(strides[axis]);
                let total = ((out.saturating_sub(1)) * strides[axis] + effective)
                    .saturating_sub(input[axis]);
                let (begin, end) = if upper == AutoPad::SameUpper {
                    (total / 2, total - total / 2)
                } else {
                    (total - total / 2, total / 2)
                };
                let begin =
                    i64::try_from(begin).map_err(|_| overflow(node, "auto_pad begin extent"))?;
                let end = i64::try_from(end).map_err(|_| overflow(node, "auto_pad end extent"))?;
                pairs.push((begin, end));
            }
            Ok(pairs)
        }
    }
}

/// One output extent of a convolution or window, under ONNX rules.
///
/// `ceil_mode` is pooling-only; convolution passes `false`.
fn window_output(
    node: &Node,
    what: &'static str,
    input: u64,
    kernel: u64,
    stride: u64,
    dilation: u64,
    begin: i64,
    end: i64,
    ceil_mode: bool,
) -> Result<u64, Fn3Error> {
    let effective = dilation * (kernel - 1) + 1;
    let total = i128::from(input) + i128::from(begin) + i128::from(end);
    let stride = i128::from(stride);
    let delta = total - i128::from(effective);
    let mut out = if ceil_mode {
        -((-delta).div_euclid(stride))
    } else {
        delta.div_euclid(stride)
    } + 1;
    if ceil_mode && out > 0 && (out - 1) * stride >= i128::from(input) + i128::from(begin) {
        out -= 1;
    }
    if out < 0 {
        out = 0;
    }
    u64::try_from(out).map_err(|_| overflow(node, what))
}

/// The `OpParams` fields one op reads, as a bit set.
///
/// A rule names exactly the fields its op defines; every other field must be
/// neutral. That is how the pass rejects a semantic flag it would otherwise
/// silently drop, without a per-op struct and without a boolean per helper.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Req(u32);

impl Req {
    /// No attributes.
    const NONE: Req = Req(0);
    const AXES: Req = Req(1 << 0);
    const KEEP_DIMS: Req = Req(1 << 1);
    const EPSILON: Req = Req(1 << 2);
    const STRIDES: Req = Req(1 << 3);
    const PADS: Req = Req(1 << 4);
    const DILATIONS: Req = Req(1 << 5);
    const KERNEL: Req = Req(1 << 6);
    const GROUP: Req = Req(1 << 7);
    const CEIL_MODE: Req = Req(1 << 8);
    const COUNT_INCLUDE_PAD: Req = Req(1 << 9);
    const AUTO_PAD: Req = Req(1 << 10);
    const RESIZE_MODES: Req = Req(1 << 11);
    const RESIZE_SCALES: Req = Req(1 << 12);
    const RESIZE_SIZES: Req = Req(1 << 13);
    const PAD_MODE: Req = Req(1 << 14);
    const PAD_VALUE: Req = Req(1 << 15);
    const CLIP: Req = Req(1 << 16);
    const TOPK_AXIS: Req = Req(1 << 17);
    const TOPK_K: Req = Req(1 << 18);
    const TOPK_ORDER: Req = Req(1 << 19);
    const SPLIT_SIZES: Req = Req(1 << 20);
    const SLICE_BOUNDS: Req = Req(1 << 21);

    /// The union of two field sets, as an op that reads both spells it.
    const fn union(self, other: Req) -> Req {
        Req(self.0 | other.0)
    }

    const fn has(self, field: Req) -> bool {
        self.0 & field.0 != 0
    }
}

/// Every field a rule may name, with the name used in diagnostics.
const PARAM_FIELDS: [(Req, &str); 22] = [
    (Req::AXES, "axes"),
    (Req::KEEP_DIMS, "keep_dims"),
    (Req::EPSILON, "epsilon"),
    (Req::STRIDES, "strides"),
    (Req::PADS, "pads"),
    (Req::DILATIONS, "dilations"),
    (Req::KERNEL, "kernel_shape"),
    (Req::GROUP, "group"),
    (Req::CEIL_MODE, "ceil_mode"),
    (Req::COUNT_INCLUDE_PAD, "count_include_pad"),
    (Req::AUTO_PAD, "auto_pad"),
    (
        Req::RESIZE_MODES,
        "resize mode/coordinate_transformation_mode/nearest_mode",
    ),
    (Req::RESIZE_SCALES, "resize_scales"),
    (Req::RESIZE_SIZES, "resize_sizes"),
    (Req::PAD_MODE, "pad_mode"),
    (Req::PAD_VALUE, "pad_value"),
    (Req::CLIP, "clip_min/clip_max"),
    (Req::TOPK_AXIS, "topk_axis"),
    (Req::TOPK_K, "topk_k"),
    (Req::TOPK_ORDER, "largest/sorted"),
    (Req::SPLIT_SIZES, "split_sizes"),
    (Req::SLICE_BOUNDS, "slice_starts/slice_ends/slice_steps"),
];

/// `true` when `params` carries the field, at a value the op would read.
///
/// A vector field counts as set when it is non-empty, a flag when it is `true`,
/// an optional bound when it is `Some`, and an enum when it differs from its
/// `Default` variant. `pad_value` alone is special: zero is both the default and
/// a meaningful fill, so it is reported only for ops that did not name
/// [`Req::PAD_VALUE`] and only when it is non-zero (a zero fill for a non-pad op
/// cannot change its meaning).
fn field_is_set(params: &OpParams, field: Req) -> bool {
    match field {
        Req::AXES => !params.axes.is_empty(),
        Req::KEEP_DIMS => params.keep_dims,
        Req::EPSILON => params.epsilon != 0.0,
        Req::STRIDES => !params.strides.is_empty(),
        Req::PADS => !params.pads.is_empty(),
        Req::DILATIONS => !params.dilations.is_empty(),
        Req::KERNEL => !params.kernel_shape.is_empty(),
        Req::GROUP => params.group != 0,
        Req::CEIL_MODE => params.ceil_mode,
        Req::COUNT_INCLUDE_PAD => params.count_include_pad,
        Req::AUTO_PAD => params.auto_pad != AutoPad::default(),
        Req::RESIZE_MODES => {
            params.resize_mode != ResizeMode::default()
                || params.coordinate_transformation_mode != CoordinateTransformationMode::default()
                || params.nearest_mode != NearestMode::default()
        }
        Req::RESIZE_SCALES => !params.resize_scales.is_empty(),
        Req::RESIZE_SIZES => !params.resize_sizes.is_empty(),
        Req::PAD_MODE => params.pad_mode != PadMode::default(),
        Req::PAD_VALUE => params.pad_value != 0.0,
        Req::CLIP => params.clip_min.is_some() || params.clip_max.is_some(),
        Req::TOPK_AXIS => params.topk_axis != 0,
        Req::TOPK_K => params.topk_k != 0,
        Req::TOPK_ORDER => params.largest || params.sorted,
        Req::SPLIT_SIZES => !params.split_sizes.is_empty(),
        Req::SLICE_BOUNDS => {
            !params.slice_starts.is_empty()
                || !params.slice_ends.is_empty()
                || !params.slice_steps.is_empty()
        }
        _ => false,
    }
}

/// Reject an attribute field the op does not define.
///
/// The crate's rule is that an op rejects a field it does not understand
/// rather than ignoring it, so a plan cannot silently drop a semantic flag.
fn require_params(node: &Node, allowed: Req) -> Result<(), Fn3Error> {
    let params = &node.params;
    // Every attribute vector a handler reads is bounded by the rank cap, so a
    // malformed `OpParams` cannot make the pass allocate without a limit.
    for (name, len, limit) in [
        ("axes", params.axes.len(), MAX_RANK),
        ("strides", params.strides.len(), MAX_RANK),
        ("pads", params.pads.len(), 2 * MAX_RANK),
        ("dilations", params.dilations.len(), MAX_RANK),
        ("kernel_shape", params.kernel_shape.len(), MAX_RANK),
        ("resize_scales", params.resize_scales.len(), MAX_RANK),
        ("resize_sizes", params.resize_sizes.len(), MAX_RANK),
        ("split_sizes", params.split_sizes.len(), MAX_RANK),
        ("slice_starts", params.slice_starts.len(), MAX_RANK),
        ("slice_ends", params.slice_ends.len(), MAX_RANK),
        ("slice_steps", params.slice_steps.len(), MAX_RANK),
    ] {
        if len > limit {
            return Err(invalid(
                node,
                format!(
                    "op {} carries {len} {name} attributes; the IR accepts at most {limit}",
                    node.op
                ),
            ));
        }
    }
    let mut unexpected = Vec::new();
    let mut accepted = Vec::new();
    for (field, name) in PARAM_FIELDS {
        if allowed.has(field) {
            accepted.push(name);
        } else if field_is_set(params, field) {
            unexpected.push(name);
        }
    }
    if unexpected.is_empty() {
        Ok(())
    } else {
        Err(invalid(
            node,
            format!(
                "op {} accepts {}, but the node supplies {}",
                node.op,
                if accepted.is_empty() {
                    "no attributes".to_string()
                } else {
                    accepted.join(", ")
                },
                unexpected.join(", ")
            ),
        ))
    }
}

fn single_axis(node: &Node, rank: usize, default: i64) -> Result<usize, Fn3Error> {
    match node.params.axes.as_slice() {
        [] => normalize_axis(node, default, rank),
        [axis] => normalize_axis(node, *axis, rank),
        axes => Err(invalid(
            node,
            format!(
                "op {} takes exactly one axis; the node supplies {axes:?}",
                node.op
            ),
        )),
    }
}

/// Normalise a signed axis in `-rank..rank`.
///
/// Checked addition keeps `i64::MIN` from wrapping; the result is converted to
/// `usize` only after it is known to be non-negative.
fn normalize_axis(node: &Node, axis: i64, rank: usize) -> Result<usize, Fn3Error> {
    let rank_i64 = i64::try_from(rank)
        .map_err(|_| invalid(node, format!("rank {rank} does not fit in i64")))?;
    let resolved = if axis < 0 {
        axis.checked_add(rank_i64).ok_or_else(|| {
            invalid(
                node,
                format!(
                    "op {}: axis {axis} is outside the rank-{rank} operand",
                    node.op
                ),
            )
        })?
    } else {
        axis
    };
    if resolved < 0 || resolved >= rank_i64 {
        return Err(invalid(
            node,
            format!(
                "op {}: axis {axis} is outside the rank-{rank} operand; legal axes are \
                 -{rank}..={}",
                node.op,
                rank.saturating_sub(1)
            ),
        ));
    }
    usize::try_from(resolved).map_err(|_| invalid(node, "resolved axis does not fit in usize"))
}

/// The constant product of a dim slice, or an explanation of why not.
///
/// An empty slice is 1 and a single dim is itself, which is exactly what
/// `Flatten` needs. A zero short-circuits before multiplication, matching
/// `Shape::numel`.
fn product_of(node: &Node, dims: &[DimExpr], what: &'static str) -> Result<DimExpr, Fn3Error> {
    if dims.is_empty() {
        return Ok(DimExpr::Const(1));
    }
    if dims.len() == 1 {
        return Ok(dims[0]);
    }
    if dims.iter().any(|dim| dim.as_const() == Some(0)) {
        return Ok(DimExpr::Const(0));
    }
    let mut product = 1u64;
    for dim in dims {
        match dim.as_const() {
            Some(extent) => {
                product = product
                    .checked_mul(extent)
                    .ok_or_else(|| overflow(node, what))?;
            }
            None => {
                return Err(invalid(
                    node,
                    format!(
                        "{what} spans more than one axis and includes symbolic dimension {dim}; \
                         this IR cannot express a symbolic product"
                    ),
                ))
            }
        }
    }
    Ok(DimExpr::Const(product))
}

/// The constant element count of a dim slice, or `None` when a symbol appears.
fn const_numel(node: &Node, dims: &[DimExpr], what: &'static str) -> Result<Option<u64>, Fn3Error> {
    if dims.iter().any(|dim| dim.as_const() == Some(0)) {
        return Ok(Some(0));
    }
    let mut product = 1u64;
    for dim in dims {
        match dim.as_const() {
            Some(extent) => {
                product = product
                    .checked_mul(extent)
                    .ok_or_else(|| overflow(node, what))?;
            }
            None => return Ok(None),
        }
    }
    Ok(Some(product))
}

/// [`const_numel`] for a reshape target where `-1` has already been resolved.
fn const_numel_from_options(
    node: &Node,
    dims: &[Option<DimExpr>],
    what: &'static str,
) -> Result<Option<u64>, Fn3Error> {
    let mut product = 1u64;
    for dim in dims {
        match dim {
            Some(DimExpr::Const(0)) => return Ok(Some(0)),
            Some(DimExpr::Const(extent)) => {
                product = product
                    .checked_mul(*extent)
                    .ok_or_else(|| overflow(node, what))?;
            }
            _ => return Ok(None),
        }
    }
    Ok(Some(product))
}

fn render(dims: &[DimExpr]) -> String {
    let mut text = String::from("[");
    for (index, dim) in dims.iter().enumerate() {
        if index > 0 {
            text.push_str(", ");
        }
        text.push_str(&dim.to_string());
    }
    text.push(']');
    text
}

fn render_options(dims: &[Option<DimExpr>]) -> String {
    let mut text = String::from("[");
    for (index, dim) in dims.iter().enumerate() {
        if index > 0 {
            text.push_str(", ");
        }
        match dim {
            Some(dim) => text.push_str(&dim.to_string()),
            None => text.push('?'),
        }
    }
    text.push(']');
    text
}

fn invalid(node: &Node, detail: impl fmt::Display) -> Fn3Error {
    Fn3Error::InvalidShape {
        context: "ir shape inference",
        detail: format!("{}: {detail}", node.label()),
    }
}

fn shape_mismatch(node: &Node, expected: impl Into<String>, actual: impl Into<String>) -> Fn3Error {
    Fn3Error::ShapeMismatch {
        op: Some(node.op),
        expected: expected.into(),
        actual: format!("{} ({})", actual.into(), node.label()),
    }
}

fn unsupported(node: &Node) -> Fn3Error {
    unsupported_variant(
        node,
        format!(
            "shape inference is not implemented for op {} ({}); the pass rejects the node \
             instead of guessing a shape",
            node.op,
            node.op.description()
        ),
    )
}

fn unsupported_variant(node: &Node, detail: impl fmt::Display) -> Fn3Error {
    Fn3Error::UnsupportedOp {
        op: node.op,
        backend: "shape inference",
        reason: format!("{}: {detail}", node.label()),
    }
}

fn overflow(node: &Node, what: &'static str) -> Fn3Error {
    Fn3Error::ShapeOverflow {
        context: "ir shape inference",
        axis: None,
        detail: format!("{}: {what} exceeds u64::MAX", node.label()),
    }
}

/// Add the node label to an error a type check produced.
fn annotate(error: Fn3Error, node: &Node) -> Fn3Error {
    match error {
        Fn3Error::ShapeOverflow {
            context,
            axis,
            detail,
        } => Fn3Error::ShapeOverflow {
            context,
            axis,
            detail: format!("{}: {detail}", node.label()),
        },
        Fn3Error::InvalidShape { context, detail } => Fn3Error::InvalidShape {
            context,
            detail: format!("{}: {detail}", node.label()),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::backend::OpParams;
    use crate::inference::ir::{SourceLoc, ValueId};
    use crate::inference::shape::Symbol;

    fn consts(extents: &[u64]) -> Vec<DimExpr> {
        extents.iter().copied().map(DimExpr::constant).collect()
    }

    fn symbol(symbol: Symbol) -> DimExpr {
        DimExpr::symbol(symbol)
    }

    fn add_value(graph: &mut Graph, dtype: DType, shape: Vec<DimExpr>) -> ValueId {
        graph
            .add_value(None, TensorType::new(dtype, shape))
            .expect("the test graph is valid")
    }

    fn add_node(
        graph: &mut Graph,
        op: OpId,
        inputs: &[ValueId],
        outputs: &[ValueId],
        params: OpParams,
        node_index: usize,
    ) {
        graph
            .add_node(
                op,
                inputs.to_vec(),
                outputs.to_vec(),
                params,
                SourceLoc::new(node_index, op.name()),
            )
            .expect("the test graph is valid");
    }

    fn ty(graph: &Graph, id: ValueId) -> TensorType {
        graph.value(id).expect("the value was added").ty.clone()
    }

    fn tensor(shape: Vec<DimExpr>) -> TensorType {
        TensorType::new(DType::F32, shape)
    }

    #[test]
    fn cnn_shaped_chain_infers_every_value() {
        let mut graph = Graph::new();
        let x = add_value(
            &mut graph,
            DType::F32,
            vec![
                symbol(Symbol::Batch),
                DimExpr::constant(3),
                DimExpr::constant(8),
                DimExpr::constant(8),
            ],
        );
        let flattened = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Flatten,
            &[x],
            &[flattened],
            OpParams::new().with_axes([1]),
            0,
        );

        let weight = add_value(&mut graph, DType::F32, consts(&[192, 64]));
        let bias = add_value(&mut graph, DType::F32, consts(&[64]));
        let projected = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::MatMul,
            &[flattened, weight],
            &[projected],
            OpParams::new(),
            1,
        );
        let shifted = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Add,
            &[projected, bias],
            &[shifted],
            OpParams::new(),
            2,
        );
        let activated = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Relu,
            &[shifted],
            &[activated],
            OpParams::new(),
            3,
        );

        let weight2 = add_value(&mut graph, DType::F32, consts(&[64, 64]));
        let projected2 = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::MatMul,
            &[activated, weight2],
            &[projected2],
            OpParams::new(),
            4,
        );
        let residual = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Add,
            &[projected2, activated],
            &[residual],
            OpParams::new(),
            5,
        );

        let transposed = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Transpose,
            &[residual],
            &[transposed],
            OpParams::new().with_axes([1, 0]),
            6,
        );
        let pooled = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::ReduceSum,
            &[transposed],
            &[pooled],
            OpParams::new().with_axes([0]).with_keep_dims(true),
            7,
        );
        let logits = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Softmax,
            &[pooled],
            &[logits],
            OpParams::new().with_axes([0]),
            8,
        );

        infer_shapes(&mut graph).unwrap();

        let batch = symbol(Symbol::Batch);
        assert_eq!(
            ty(&graph, flattened),
            tensor(vec![batch, DimExpr::constant(192)])
        );
        assert_eq!(
            ty(&graph, projected),
            tensor(vec![batch, DimExpr::constant(64)])
        );
        assert_eq!(
            ty(&graph, shifted),
            tensor(vec![batch, DimExpr::constant(64)])
        );
        assert_eq!(
            ty(&graph, activated),
            tensor(vec![batch, DimExpr::constant(64)])
        );
        assert_eq!(
            ty(&graph, projected2),
            tensor(vec![batch, DimExpr::constant(64)])
        );
        assert_eq!(
            ty(&graph, residual),
            tensor(vec![batch, DimExpr::constant(64)])
        );
        assert_eq!(
            ty(&graph, transposed),
            tensor(vec![DimExpr::constant(64), batch])
        );
        assert_eq!(
            ty(&graph, pooled),
            tensor(vec![DimExpr::constant(1), batch])
        );
        assert_eq!(
            ty(&graph, logits),
            tensor(vec![DimExpr::constant(1), batch])
        );
    }

    #[test]
    fn broadcasting_is_right_aligned() {
        let mut graph = Graph::new();
        let left = add_value(&mut graph, DType::F32, consts(&[2, 3]));
        let right = add_value(&mut graph, DType::F32, consts(&[3]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Add,
            &[left, right],
            &[out],
            OpParams::new(),
            0,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(ty(&graph, out).dims(), consts(&[2, 3]).as_slice());

        let mut symbolic = Graph::new();
        let batch = symbol(Symbol::Batch);
        let left = add_value(&mut symbolic, DType::F32, vec![batch, DimExpr::constant(4)]);
        let right = add_value(
            &mut symbolic,
            DType::F32,
            vec![DimExpr::constant(1), DimExpr::constant(4)],
        );
        let out = add_value(&mut symbolic, DType::F32, Vec::new());
        add_node(
            &mut symbolic,
            OpId::Add,
            &[left, right],
            &[out],
            OpParams::new(),
            0,
        );
        infer_shapes(&mut symbolic).unwrap();
        assert_eq!(
            ty(&symbolic, out).dims(),
            vec![batch, DimExpr::constant(4)].as_slice()
        );
    }

    #[test]
    fn incompatible_broadcast_names_the_node() {
        let mut graph = Graph::new();
        let left = add_value(&mut graph, DType::F32, consts(&[2, 3]));
        let right = add_value(&mut graph, DType::F32, consts(&[2]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Add,
            &[left, right],
            &[out],
            OpParams::new(),
            0,
        );
        let error = infer_shapes(&mut graph).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("node 0"), "{text}");
        assert!(text.contains("op add"), "{text}");
    }

    #[test]
    fn unsupported_op_reports_the_node_index_and_op() {
        let mut graph = Graph::new();
        let data = add_value(&mut graph, DType::F32, consts(&[2, 2]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::ArgMax,
            &[data],
            &[out],
            OpParams::new(),
            3,
        );
        let error = infer_shapes(&mut graph).unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::UnsupportedOp {
                op: OpId::ArgMax,
                ..
            }
        ));
        let text = error.to_string();
        assert!(text.contains("node 3"), "{text}");
        assert!(text.contains("argmax"), "{text}");
        assert!(text.contains("not implemented"), "{text}");
    }

    #[test]
    fn transpose_defaults_to_reverse_and_honours_a_permutation() {
        let mut reversed = Graph::new();
        let input = add_value(&mut reversed, DType::F32, consts(&[2, 3, 4]));
        let out = add_value(&mut reversed, DType::F32, Vec::new());
        add_node(
            &mut reversed,
            OpId::Transpose,
            &[input],
            &[out],
            OpParams::new(),
            0,
        );
        infer_shapes(&mut reversed).unwrap();
        assert_eq!(ty(&reversed, out).dims(), consts(&[4, 3, 2]).as_slice());

        let mut permuted = Graph::new();
        let batch = symbol(Symbol::Batch);
        let input = add_value(
            &mut permuted,
            DType::F32,
            vec![batch, DimExpr::constant(3), DimExpr::constant(4)],
        );
        let out = add_value(&mut permuted, DType::F32, Vec::new());
        add_node(
            &mut permuted,
            OpId::Transpose,
            &[input],
            &[out],
            OpParams::new().with_axes([1, 2, 0]),
            0,
        );
        infer_shapes(&mut permuted).unwrap();
        assert_eq!(
            ty(&permuted, out).dims(),
            vec![DimExpr::constant(3), DimExpr::constant(4), batch].as_slice()
        );
    }

    #[test]
    fn reshape_resolves_a_constant_target() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[2, 3, 4]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Reshape,
            &[input],
            &[out],
            OpParams::new().with_axes([-1, 4]),
            0,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(ty(&graph, out).dims(), consts(&[6, 4]).as_slice());

        let wrong = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Reshape,
            &[input],
            &[wrong],
            OpParams::new().with_axes([5, 5]),
            1,
        );
        let error = infer_shapes(&mut graph).unwrap_err();
        assert!(error.to_string().contains("node 1"), "{error}");
    }

    #[test]
    fn reshape_rejects_a_runtime_target() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[2, 3]));
        let target = add_value(&mut graph, DType::I64, consts(&[2]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Reshape,
            &[input, target],
            &[out],
            OpParams::new(),
            0,
        );
        let error = infer_shapes(&mut graph).unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::UnsupportedOp {
                op: OpId::Reshape,
                ..
            }
        ));
        assert!(error.to_string().contains("node 0"), "{error}");
    }

    #[test]
    fn squeeze_and_unsqueeze_follow_the_axes_attributes() {
        let mut squeezed = Graph::new();
        let input = add_value(&mut squeezed, DType::F32, consts(&[2, 1, 3, 1]));
        let out = add_value(&mut squeezed, DType::F32, Vec::new());
        add_node(
            &mut squeezed,
            OpId::Squeeze,
            &[input],
            &[out],
            OpParams::new().with_axes([1, 3]),
            0,
        );
        infer_shapes(&mut squeezed).unwrap();
        assert_eq!(ty(&squeezed, out).dims(), consts(&[2, 3]).as_slice());

        let mut unsqueezed = Graph::new();
        let input = add_value(&mut unsqueezed, DType::F32, consts(&[2, 3]));
        let out = add_value(&mut unsqueezed, DType::F32, Vec::new());
        add_node(
            &mut unsqueezed,
            OpId::Unsqueeze,
            &[input],
            &[out],
            OpParams::new().with_axes([1, 3]),
            0,
        );
        infer_shapes(&mut unsqueezed).unwrap();
        assert_eq!(
            ty(&unsqueezed, out).dims(),
            consts(&[2, 1, 3, 1]).as_slice()
        );
    }

    #[test]
    fn concat_sums_one_axis_and_checks_the_rest() {
        let mut graph = Graph::new();
        let batch = symbol(Symbol::Batch);
        let left = add_value(
            &mut graph,
            DType::F32,
            vec![batch, DimExpr::constant(4), DimExpr::constant(8)],
        );
        let right = add_value(
            &mut graph,
            DType::F32,
            vec![batch, DimExpr::constant(3), DimExpr::constant(8)],
        );
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Concat,
            &[left, right],
            &[out],
            OpParams::new().with_axes([1]),
            0,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(
            ty(&graph, out).dims(),
            vec![batch, DimExpr::constant(7), DimExpr::constant(8)].as_slice()
        );

        let mismatched = add_value(
            &mut graph,
            DType::F32,
            vec![batch, DimExpr::constant(3), DimExpr::constant(9)],
        );
        let bad = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Concat,
            &[left, mismatched],
            &[bad],
            OpParams::new().with_axes([1]),
            1,
        );
        let error = infer_shapes(&mut graph).unwrap_err();
        assert!(error.to_string().contains("node 1"), "{error}");
    }

    #[test]
    fn gather_replaces_the_indexed_axis_with_the_indices_shape() {
        let mut graph = Graph::new();
        let batch = symbol(Symbol::Batch);
        let table = add_value(&mut graph, DType::F32, consts(&[50257, 128]));
        let indices = add_value(&mut graph, DType::I64, vec![batch, DimExpr::constant(4)]);
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Gather,
            &[table, indices],
            &[out],
            OpParams::new().with_axes([0]),
            0,
        );
        infer_shapes(&mut graph).unwrap();
        let expected = vec![batch, DimExpr::constant(4), DimExpr::constant(128)];
        assert_eq!(ty(&graph, out).dims(), expected.as_slice());
    }

    #[test]
    fn reduction_and_layer_norm_keep_the_expected_rank() {
        let mut reduced = Graph::new();
        let batch = symbol(Symbol::Batch);
        let input = add_value(
            &mut reduced,
            DType::F32,
            vec![batch, DimExpr::constant(4), DimExpr::constant(8)],
        );
        let mean = add_value(&mut reduced, DType::F32, Vec::new());
        add_node(
            &mut reduced,
            OpId::ReduceMean,
            &[input],
            &[mean],
            OpParams::new().with_axes([2]),
            0,
        );
        let kept = add_value(&mut reduced, DType::F32, Vec::new());
        add_node(
            &mut reduced,
            OpId::ReduceMean,
            &[input],
            &[kept],
            OpParams::new().with_axes([2]).with_keep_dims(true),
            1,
        );
        infer_shapes(&mut reduced).unwrap();
        assert_eq!(
            ty(&reduced, mean).dims(),
            vec![batch, DimExpr::constant(4)].as_slice()
        );
        assert_eq!(
            ty(&reduced, kept).dims(),
            vec![batch, DimExpr::constant(4), DimExpr::constant(1)].as_slice()
        );

        let mut normalized = Graph::new();
        let batch = symbol(Symbol::Batch);
        let input = add_value(
            &mut normalized,
            DType::F32,
            vec![batch, DimExpr::constant(8)],
        );
        let weight = add_value(&mut normalized, DType::F32, consts(&[8]));
        let bias = add_value(&mut normalized, DType::F32, consts(&[8]));
        let out = add_value(&mut normalized, DType::F32, Vec::new());
        add_node(
            &mut normalized,
            OpId::LayerNorm,
            &[input, weight, bias],
            &[out],
            OpParams::new().with_axes([-1]).with_epsilon(1e-5),
            0,
        );
        infer_shapes(&mut normalized).unwrap();
        assert_eq!(
            ty(&normalized, out).dims(),
            vec![batch, DimExpr::constant(8)].as_slice()
        );
    }

    #[test]
    fn flatten_axis_range_follows_onnx() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[2, 3, 4]));
        let negative = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Flatten,
            &[input],
            &[negative],
            OpParams::new().with_axes([-1]),
            0,
        );
        let trailing = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Flatten,
            &[input],
            &[trailing],
            OpParams::new().with_axes([3]),
            1,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(ty(&graph, negative).dims(), consts(&[6, 4]).as_slice());
        assert_eq!(ty(&graph, trailing).dims(), consts(&[24, 1]).as_slice());

        let out_of_range = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Flatten,
            &[input],
            &[out_of_range],
            OpParams::new().with_axes([-4]),
            2,
        );
        let error = infer_shapes(&mut graph).unwrap_err();
        assert!(error.to_string().contains("node 2"), "{error}");
    }

    #[test]
    fn matmul_broadcasts_leading_batch_axes() {
        // [2,3,4] x [4,5] is a batched matrix product with a rank-2 right
        // operand: the leading 2 broadcasts against nothing, so the result is
        // [2,3,5].
        let mut graph = Graph::new();
        let lhs = add_value(&mut graph, DType::F32, consts(&[2, 3, 4]));
        let rhs = add_value(&mut graph, DType::F32, consts(&[4, 5]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::MatMul,
            &[lhs, rhs],
            &[out],
            OpParams::new(),
            0,
        );
        infer_shapes(&mut graph).expect("a batched product infers");
        assert_eq!(graph.value(out).unwrap().ty.dims(), consts(&[2, 3, 5]));
    }

    #[test]
    fn matmul_broadcasts_one_batch_axis_against_many() {
        // The attention product: [1,2,400,32] x [1,2,32,400] -> [1,2,400,400],
        // and the same with a rank-3 left operand against a rank-4 right one.
        let mut graph = Graph::new();
        let lhs = add_value(&mut graph, DType::F32, consts(&[1, 2, 400, 32]));
        let rhs = add_value(&mut graph, DType::F32, consts(&[2, 32, 400]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::MatMul,
            &[lhs, rhs],
            &[out],
            OpParams::new(),
            0,
        );
        infer_shapes(&mut graph).expect("the batch axes broadcast");
        assert_eq!(
            graph.value(out).unwrap().ty.dims(),
            consts(&[1, 2, 400, 400])
        );
    }

    #[test]
    fn matmul_rejects_batch_axes_that_do_not_broadcast() {
        let mut graph = Graph::new();
        let lhs = add_value(&mut graph, DType::F32, consts(&[2, 3, 4]));
        let rhs = add_value(&mut graph, DType::F32, consts(&[5, 4, 6]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::MatMul,
            &[lhs, rhs],
            &[out],
            OpParams::new(),
            0,
        );
        let error = infer_shapes(&mut graph).unwrap_err();
        assert!(error.to_string().contains("do not broadcast"), "{error}");
        assert!(error.to_string().contains("node 0"), "{error}");
    }

    #[test]
    fn matmul_rejects_a_rank_one_operand() {
        let mut graph = Graph::new();
        let lhs = add_value(&mut graph, DType::F32, consts(&[4]));
        let rhs = add_value(&mut graph, DType::F32, consts(&[4, 6]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::MatMul,
            &[lhs, rhs],
            &[out],
            OpParams::new(),
            0,
        );
        let error = infer_shapes(&mut graph).unwrap_err();
        assert!(error.to_string().contains("rank >= 2"), "{error}");
    }

    // -----------------------------------------------------------------------
    // CNN operations
    // -----------------------------------------------------------------------

    use crate::inference::backend::{AutoPad, CoordinateTransformationMode, PadMode, ResizeMode};

    fn infer_error(graph: &mut Graph) -> Fn3Error {
        infer_shapes(graph).expect_err("the graph must not infer")
    }

    #[test]
    fn conv_infers_nchw_with_groups_strides_and_pads() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[1, 4, 8, 8]));
        // Two groups, two input channels each, eight outputs.
        let weight = add_value(&mut graph, DType::F32, consts(&[8, 2, 3, 3]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Conv,
            &[input, weight],
            &[out],
            OpParams::new()
                .with_kernel_shape([3, 3])
                .with_strides([2, 2])
                .with_pads([1, 1, 1, 1])
                .with_group(2),
            0,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(ty(&graph, out).dims(), consts(&[1, 8, 4, 4]).as_slice());
    }

    #[test]
    fn conv_auto_pad_same_upper_rounds_the_output_up() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[1, 3, 7, 7]));
        let weight = add_value(&mut graph, DType::F32, consts(&[5, 3, 3, 3]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Conv,
            &[input, weight],
            &[out],
            OpParams::new()
                .with_kernel_shape([3, 3])
                .with_strides([2, 2])
                .with_auto_pad(AutoPad::SameUpper),
            0,
        );
        infer_shapes(&mut graph).unwrap();
        // ceil(7 / 2) = 4 on each spatial axis.
        assert_eq!(ty(&graph, out).dims(), consts(&[1, 5, 4, 4]).as_slice());
    }

    #[test]
    fn conv_errors_name_the_node_and_the_rule() {
        // Missing kernel_shape.
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[1, 3, 8, 8]));
        let weight = add_value(&mut graph, DType::F32, consts(&[4, 3, 3, 3]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Conv,
            &[input, weight],
            &[out],
            OpParams::new(),
            7,
        );
        let error = infer_error(&mut graph);
        assert!(matches!(error, Fn3Error::InvalidShape { .. }), "{error}");
        let text = error.to_string();
        assert!(text.contains("node 7"), "{text}");
        assert!(text.contains("kernel_shape"), "{text}");

        // Group does not divide the input channels.
        let mut mismatch = Graph::new();
        let input = add_value(&mut mismatch, DType::F32, consts(&[1, 3, 8, 8]));
        let weight = add_value(&mut mismatch, DType::F32, consts(&[4, 2, 3, 3]));
        let out = add_value(&mut mismatch, DType::F32, Vec::new());
        add_node(
            &mut mismatch,
            OpId::Conv,
            &[input, weight],
            &[out],
            OpParams::new().with_kernel_shape([3, 3]).with_group(2),
            0,
        );
        let error = infer_error(&mut mismatch);
        assert!(matches!(error, Fn3Error::ShapeMismatch { .. }), "{error}");
        assert!(error.to_string().contains("group"), "{error}");

        // Rank 2 is not a convolution operand.
        let mut rank = Graph::new();
        let input = add_value(&mut rank, DType::F32, consts(&[4, 4]));
        let weight = add_value(&mut rank, DType::F32, consts(&[2, 4]));
        let out = add_value(&mut rank, DType::F32, Vec::new());
        add_node(
            &mut rank,
            OpId::Conv,
            &[input, weight],
            &[out],
            OpParams::new().with_kernel_shape([1, 1]),
            0,
        );
        let error = infer_error(&mut rank);
        assert!(error.to_string().contains("rank 3"), "{error}");
    }

    #[test]
    fn pooling_rules_cover_ceil_mode_and_global_average() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[1, 2, 8, 8]));
        let max_pool = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::MaxPool,
            &[input],
            &[max_pool],
            OpParams::new()
                .with_kernel_shape([2, 2])
                .with_strides([2, 2]),
            0,
        );
        let average = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::AveragePool,
            &[input],
            &[average],
            OpParams::new()
                .with_kernel_shape([3, 3])
                .with_strides([2, 2])
                .with_pads([1, 1, 1, 1])
                .with_count_include_pad(true),
            1,
        );
        let global = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::GlobalAveragePool,
            &[input],
            &[global],
            OpParams::new(),
            2,
        );
        let ceil = add_value(&mut graph, DType::F32, consts(&[1, 1, 5, 5]));
        let ceil_out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::MaxPool,
            &[ceil],
            &[ceil_out],
            OpParams::new()
                .with_kernel_shape([2, 2])
                .with_strides([2, 2])
                .with_ceil_mode(true),
            3,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(
            ty(&graph, max_pool).dims(),
            consts(&[1, 2, 4, 4]).as_slice()
        );
        assert_eq!(ty(&graph, average).dims(), consts(&[1, 2, 4, 4]).as_slice());
        assert_eq!(ty(&graph, global).dims(), consts(&[1, 2, 1, 1]).as_slice());
        // ceil((5 - 2) / 2) + 1 = 3, and the last window still fits.
        assert_eq!(
            ty(&graph, ceil_out).dims(),
            consts(&[1, 1, 3, 3]).as_slice()
        );

        // A pool without kernel_shape is rejected by name.
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::MaxPool,
            &[input],
            &[out],
            OpParams::new(),
            4,
        );
        let error = infer_error(&mut graph);
        assert!(error.to_string().contains("node 4"), "{error}");
        assert!(error.to_string().contains("kernel_shape"), "{error}");
    }

    #[test]
    fn batch_norm_is_the_identity_on_the_data_shape() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[2, 4, 3, 3]));
        let scale = add_value(&mut graph, DType::F32, consts(&[4]));
        let bias = add_value(&mut graph, DType::F32, consts(&[4]));
        let mean = add_value(&mut graph, DType::F32, consts(&[4]));
        let variance = add_value(&mut graph, DType::F32, consts(&[4]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::BatchNorm,
            &[input, scale, bias, mean, variance],
            &[out],
            OpParams::new().with_epsilon(1e-5),
            0,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(ty(&graph, out).dims(), consts(&[2, 4, 3, 3]).as_slice());

        // A parameter that does not match the channel axis is named.
        let wrong = add_value(&mut graph, DType::F32, consts(&[3]));
        let bad = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::BatchNorm,
            &[input, wrong, bias, mean, variance],
            &[bad],
            OpParams::new().with_epsilon(1e-5),
            1,
        );
        let error = infer_error(&mut graph);
        assert!(error.to_string().contains("node 1"), "{error}");
        assert!(error.to_string().contains("scale"), "{error}");
    }

    #[test]
    fn resize_uses_sizes_or_scales_and_ignores_the_roi_placeholder() {
        let mut scales = Graph::new();
        let input = add_value(&mut scales, DType::F32, consts(&[1, 3, 8, 8]));
        let roi = add_value(&mut scales, DType::F32, consts(&[0]));
        let out = add_value(&mut scales, DType::F32, Vec::new());
        add_node(
            &mut scales,
            OpId::Resize,
            &[input, roi],
            &[out],
            OpParams::new()
                .with_resize(
                    ResizeMode::Nearest,
                    CoordinateTransformationMode::Asymmetric,
                    NearestMode::Floor,
                )
                .with_resize_scales([1.0, 1.0, 2.0, 2.0]),
            0,
        );
        infer_shapes(&mut scales).unwrap();
        assert_eq!(ty(&scales, out).dims(), consts(&[1, 3, 16, 16]).as_slice());

        let mut sizes = Graph::new();
        let input = add_value(&mut sizes, DType::F32, consts(&[1, 3, 8, 8]));
        let out = add_value(&mut sizes, DType::F32, Vec::new());
        add_node(
            &mut sizes,
            OpId::Resize,
            &[input],
            &[out],
            OpParams::new().with_resize_sizes([1, 3, 5, 7]),
            0,
        );
        infer_shapes(&mut sizes).unwrap();
        assert_eq!(ty(&sizes, out).dims(), consts(&[1, 3, 5, 7]).as_slice());

        // A roi operand is accepted and ignored: its values are defined only
        // for tf_crop_and_resize, so a four-zero placeholder describes the same
        // resize as the one-operand form.
        let mut placeholder = Graph::new();
        let input = add_value(&mut placeholder, DType::F32, consts(&[1, 3, 8, 8]));
        let roi = add_value(&mut placeholder, DType::F32, consts(&[8]));
        let out = add_value(&mut placeholder, DType::F32, Vec::new());
        add_node(
            &mut placeholder,
            OpId::Resize,
            &[input, roi],
            &[out],
            OpParams::new().with_resize_sizes([1, 3, 4, 4]),
            2,
        );
        infer_shapes(&mut placeholder).expect("a roi placeholder is ignored");
        assert_eq!(
            ty(&placeholder, out).dims(),
            consts(&[1, 3, 4, 4]).as_slice()
        );

        // The one mode that would read the roi values is rejected.
        let mut crop = Graph::new();
        let input = add_value(&mut crop, DType::F32, consts(&[1, 3, 8, 8]));
        let roi = add_value(&mut crop, DType::F32, consts(&[8]));
        let out = add_value(&mut crop, DType::F32, Vec::new());
        add_node(
            &mut crop,
            OpId::Resize,
            &[input, roi],
            &[out],
            OpParams::new()
                .with_resize(
                    ResizeMode::Nearest,
                    CoordinateTransformationMode::TfCropAndResize,
                    NearestMode::Floor,
                )
                .with_resize_sizes([1, 3, 4, 4]),
            2,
        );
        let error = infer_error(&mut crop);
        assert!(matches!(error, Fn3Error::UnsupportedOp { .. }), "{error}");
        assert!(error.to_string().contains("node 2"), "{error}");
        assert!(error.to_string().contains("tf_crop_and_resize"), "{error}");

        // Cubic is parsed but has no rule.
        let mut cubic = Graph::new();
        let input = add_value(&mut cubic, DType::F32, consts(&[1, 3, 8, 8]));
        let out = add_value(&mut cubic, DType::F32, Vec::new());
        add_node(
            &mut cubic,
            OpId::Resize,
            &[input],
            &[out],
            OpParams::new()
                .with_resize(
                    ResizeMode::Cubic,
                    CoordinateTransformationMode::HalfPixel,
                    NearestMode::RoundPreferFloor,
                )
                .with_resize_sizes([1, 3, 4, 4]),
            0,
        );
        let error = infer_error(&mut cubic);
        assert!(error.to_string().contains("cubic"), "{error}");

        // Five scales for a rank-4 operand is a shape error.
        let mut lengths = Graph::new();
        let input = add_value(&mut lengths, DType::F32, consts(&[1, 3, 8, 8]));
        let out = add_value(&mut lengths, DType::F32, Vec::new());
        add_node(
            &mut lengths,
            OpId::Resize,
            &[input],
            &[out],
            OpParams::new().with_resize_scales([1.0, 1.0, 2.0, 2.0, 2.0]),
            0,
        );
        let error = infer_error(&mut lengths);
        assert!(error.to_string().contains("5 extents"), "{error}");
    }

    #[test]
    fn pad_pads_and_crops_and_rejects_bad_extents() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[1, 2, 3, 3]));
        let padded = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Pad,
            &[input],
            &[padded],
            OpParams::new().with_pads([0, 0, 1, 1, 0, 0, 1, 1]),
            0,
        );
        let cropped = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Pad,
            &[input],
            &[cropped],
            OpParams::new()
                .with_pads([0, 0, -1, -1, 0, 0, 0, 0])
                .with_pad(PadMode::Edge, 0.0),
            1,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(ty(&graph, padded).dims(), consts(&[1, 2, 5, 5]).as_slice());
        assert_eq!(ty(&graph, cropped).dims(), consts(&[1, 2, 2, 2]).as_slice());

        // A half-specified pad vector is the classic ONNX mistake.
        let bad = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Pad,
            &[input],
            &[bad],
            OpParams::new().with_pads([1, 1, 1, 1]),
            2,
        );
        let error = infer_error(&mut graph);
        assert!(error.to_string().contains("node 2"), "{error}");
        assert!(error.to_string().contains("begin extents"), "{error}");

        // Cropping past the axis is not an empty tensor, it is an error.
        let mut over = Graph::new();
        let input = add_value(&mut over, DType::F32, consts(&[2, 3]));
        let out = add_value(&mut over, DType::F32, Vec::new());
        add_node(
            &mut over,
            OpId::Pad,
            &[input],
            &[out],
            OpParams::new().with_pads([0, -4, 0, 0]),
            0,
        );
        let error = infer_error(&mut over);
        assert!(matches!(error, Fn3Error::ShapeMismatch { .. }), "{error}");

        // Wrap parses but has no rule.
        let mut wrap = Graph::new();
        let input = add_value(&mut wrap, DType::F32, consts(&[2, 3]));
        let out = add_value(&mut wrap, DType::F32, Vec::new());
        add_node(
            &mut wrap,
            OpId::Pad,
            &[input],
            &[out],
            OpParams::new()
                .with_pads([1, 1])
                .with_pad(PadMode::Wrap, 0.0),
            0,
        );
        let error = infer_error(&mut wrap);
        assert!(matches!(error, Fn3Error::UnsupportedOp { .. }), "{error}");
    }

    #[test]
    fn clip_sigmoid_maximum_and_minimum_are_elementwise() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[2, 3]));
        let clipped = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Clip,
            &[input],
            &[clipped],
            OpParams::new().with_clip(Some(-1.0), Some(1.0)),
            0,
        );
        let sigmoid = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Sigmoid,
            &[input],
            &[sigmoid],
            OpParams::new(),
            1,
        );
        let maximum = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Maximum,
            &[input, clipped],
            &[maximum],
            OpParams::new(),
            2,
        );
        let minimum = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Minimum,
            &[input, sigmoid],
            &[minimum],
            OpParams::new(),
            3,
        );
        infer_shapes(&mut graph).unwrap();
        for id in [clipped, sigmoid, maximum, minimum] {
            assert_eq!(ty(&graph, id).dims(), consts(&[2, 3]).as_slice());
        }

        // A Clip input that did not fold is not guessed.
        let mut unfolded = Graph::new();
        let input = add_value(&mut unfolded, DType::F32, consts(&[2]));
        let bound = add_value(&mut unfolded, DType::F32, consts(&[]));
        let out = add_value(&mut unfolded, DType::F32, Vec::new());
        add_node(
            &mut unfolded,
            OpId::Clip,
            &[input, bound],
            &[out],
            OpParams::new(),
            4,
        );
        let error = infer_error(&mut unfolded);
        assert!(error.to_string().contains("node 4"), "{error}");
        assert!(error.to_string().contains("folded"), "{error}");

        // min above max is a named error, not a reversed clamp.
        let mut reversed = Graph::new();
        let input = add_value(&mut reversed, DType::F32, consts(&[2]));
        let out = add_value(&mut reversed, DType::F32, Vec::new());
        add_node(
            &mut reversed,
            OpId::Clip,
            &[input],
            &[out],
            OpParams::new().with_clip(Some(2.0), Some(-2.0)),
            0,
        );
        let error = infer_error(&mut reversed);
        assert!(error.to_string().contains("above max"), "{error}");
    }

    #[test]
    fn top_k_writes_values_and_indices_on_the_selected_axis() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[2, 5]));
        let values = add_value(&mut graph, DType::F32, Vec::new());
        let indices = add_value(&mut graph, DType::I64, Vec::new());
        add_node(
            &mut graph,
            OpId::TopK,
            &[input],
            &[values, indices],
            OpParams::new().with_topk(1, 3, true, true),
            0,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(ty(&graph, values).dims(), consts(&[2, 3]).as_slice());
        assert_eq!(ty(&graph, indices).dims(), consts(&[2, 3]).as_slice());
        assert_eq!(ty(&graph, indices).dtype, DType::I64);

        // k larger than the axis is a mismatch, named.
        let mut too_large = Graph::new();
        let input = add_value(&mut too_large, DType::F32, consts(&[2, 5]));
        let values = add_value(&mut too_large, DType::F32, Vec::new());
        let indices = add_value(&mut too_large, DType::I64, Vec::new());
        add_node(
            &mut too_large,
            OpId::TopK,
            &[input],
            &[values, indices],
            OpParams::new().with_topk(1, 6, true, true),
            3,
        );
        let error = infer_error(&mut too_large);
        assert!(matches!(error, Fn3Error::ShapeMismatch { .. }), "{error}");
        assert!(error.to_string().contains("node 3"), "{error}");
    }

    #[test]
    fn split_reads_sizes_or_divides_equally() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[2, 6]));
        let equal_a = add_value(&mut graph, DType::F32, Vec::new());
        let equal_b = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Split,
            &[input],
            &[equal_a, equal_b],
            OpParams::new().with_axes([1]),
            0,
        );
        let sized_a = add_value(&mut graph, DType::F32, Vec::new());
        let sized_b = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Split,
            &[input],
            &[sized_a, sized_b],
            OpParams::new().with_axes([1]).with_split_sizes([2, 4]),
            1,
        );
        infer_shapes(&mut graph).unwrap();
        assert_eq!(ty(&graph, equal_a).dims(), consts(&[2, 3]).as_slice());
        assert_eq!(ty(&graph, equal_b).dims(), consts(&[2, 3]).as_slice());
        assert_eq!(ty(&graph, sized_a).dims(), consts(&[2, 2]).as_slice());
        assert_eq!(ty(&graph, sized_b).dims(), consts(&[2, 4]).as_slice());

        // Sizes that do not sum to the axis are a named mismatch.
        let mut bad = Graph::new();
        let input = add_value(&mut bad, DType::F32, consts(&[2, 6]));
        let a = add_value(&mut bad, DType::F32, Vec::new());
        let b = add_value(&mut bad, DType::F32, Vec::new());
        add_node(
            &mut bad,
            OpId::Split,
            &[input],
            &[a, b],
            OpParams::new().with_axes([1]).with_split_sizes([2, 3]),
            2,
        );
        let error = infer_error(&mut bad);
        assert!(matches!(error, Fn3Error::ShapeMismatch { .. }), "{error}");
        assert!(error.to_string().contains("node 2"), "{error}");

        // An equal split that is not divisible is rejected, not rounded.
        let mut indivisible = Graph::new();
        let input = add_value(&mut indivisible, DType::F32, consts(&[2, 5]));
        let a = add_value(&mut indivisible, DType::F32, Vec::new());
        let b = add_value(&mut indivisible, DType::F32, Vec::new());
        add_node(
            &mut indivisible,
            OpId::Split,
            &[input],
            &[a, b],
            OpParams::new().with_axes([1]),
            0,
        );
        let error = infer_error(&mut indivisible);
        assert!(error.to_string().contains("divisible"), "{error}");
    }

    #[test]
    fn slice_uses_the_folded_bounds_and_clamps_them() {
        let mut graph = Graph::new();
        let input = add_value(&mut graph, DType::F32, consts(&[3, 10]));
        let strided = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Slice,
            &[input],
            &[strided],
            OpParams::new().with_axes([1]).with_slice([1], [9], [2]),
            0,
        );
        let reversed = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Slice,
            &[input],
            &[reversed],
            OpParams::new()
                .with_axes([1])
                .with_slice([i64::MAX], [i64::MIN], [-2]),
            1,
        );
        infer_shapes(&mut graph).unwrap();
        // 1, 3, 5, 7 -> four elements.
        assert_eq!(ty(&graph, strided).dims(), consts(&[3, 4]).as_slice());
        // 10, 8, 6, 4, 2, 0 -> six elements, then exclusive end before 0.
        assert_eq!(ty(&graph, reversed).dims(), consts(&[3, 5]).as_slice());

        // Without folded bounds the node is reported, never guessed.
        let mut unfolded = Graph::new();
        let input = add_value(&mut unfolded, DType::F32, consts(&[3]));
        let starts = add_value(&mut unfolded, DType::I64, consts(&[1]));
        let ends = add_value(&mut unfolded, DType::I64, consts(&[2]));
        let out = add_value(&mut unfolded, DType::F32, Vec::new());
        add_node(
            &mut unfolded,
            OpId::Slice,
            &[input, starts, ends],
            &[out],
            OpParams::new(),
            5,
        );
        let error = infer_error(&mut unfolded);
        assert!(matches!(error, Fn3Error::UnsupportedOp { .. }), "{error}");
        assert!(error.to_string().contains("node 5"), "{error}");
    }
}
