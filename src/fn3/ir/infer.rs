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
//! Every output type of the starter set: elementwise binary (broadcasting) and
//! unary, `MatMul`, `Transpose`, `Reshape`, `Squeeze`, `Unsqueeze`, `Flatten`,
//! `Concat`, `Softmax`, `LayerNorm`, `ReduceMean`, `ReduceSum` and `Gather`.
//! Everything else is reported as not implemented, with the source node index.
//! The pass propagates dtype where the op's contract pins it (`MatMul` is f32,
//! `Gather` indices are integer, `LayerNorm` is f32) and otherwise preserves the
//! input dtype; per-kernel dtype support remains a backend question
//! ([`crate::inference::backend::Backend::admits`]), not a shape rule.

use std::fmt;

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
        OpId::Add | OpId::Sub | OpId::Mul | OpId::Div | OpId::Pow => binary(node, inputs),
        OpId::Neg
        | OpId::Sqrt
        | OpId::Erf
        | OpId::Tanh
        | OpId::Gelu
        | OpId::Relu
        | OpId::Sin
        | OpId::Cos => unary(node, inputs),
        OpId::MatMul => matmul(node, inputs),
        OpId::Transpose => transpose(node, inputs),
        OpId::Reshape => reshape(node, inputs),
        OpId::Squeeze => squeeze(node, inputs),
        OpId::Unsqueeze => unsqueeze(node, inputs),
        OpId::Flatten => flatten(node, inputs),
        OpId::Concat => concat(node, inputs),
        OpId::Softmax => softmax(node, inputs),
        OpId::LayerNorm => layer_norm(node, inputs),
        OpId::ReduceMean | OpId::ReduceSum => reduce(node, inputs),
        OpId::Gather => gather(node, inputs),
        _ => Err(unsupported(node)),
    }
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

fn unary(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_attributes(node, "no attributes", false, false, false)?;
    Ok(vec![inputs[0].clone()])
}

fn binary(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 2)?;
    require_attributes(node, "no attributes", false, false, false)?;
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
    require_attributes(node, "no attributes", false, false, false)?;
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
    if lhs.rank() != 2 || rhs.rank() != 2 {
        return Err(invalid(
            node,
            format!(
                "op matmul is a rank-2 contract; operands are {} and {}",
                render(lhs.dims()),
                render(rhs.dims())
            ),
        ));
    }
    if lhs.dims()[1] != rhs.dims()[0] {
        return Err(shape_mismatch(
            node,
            format!("inner extents to match in {}", render(lhs.dims())),
            format!(
                "{} has inner extent {} vs {}",
                render(rhs.dims()),
                rhs.dims()[0],
                lhs.dims()[1]
            ),
        ));
    }
    Ok(vec![lhs.with_shape(vec![lhs.dims()[0], rhs.dims()[1]])])
}

// ---------------------------------------------------------------------------
// Data movement
// ---------------------------------------------------------------------------

fn transpose(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_attributes(node, "only axes (the permutation)", true, false, false)?;
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
    if inputs.len() == 2 {
        return Err(unsupported_variant(
            node,
            "shape inference is not implemented for a runtime reshape target; the IR resolves \
             only the constant form carried in OpParams::axes",
        ));
    }
    expect_inputs(node, inputs, 1)?;
    require_attributes(
        node,
        "only axes (the constant target shape)",
        true,
        false,
        false,
    )?;
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
    require_attributes(node, "only axes", true, false, false)?;
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
    require_attributes(node, "only axes", true, false, false)?;
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
    require_attributes(node, "only axes (the axis)", true, false, false)?;
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
    require_attributes(node, "only axes (the axis)", true, false, false)?;
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

// ---------------------------------------------------------------------------
// Normalization and reduction
// ---------------------------------------------------------------------------

fn softmax(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 1)?;
    require_attributes(node, "only axes (the axis)", true, false, false)?;
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
    require_attributes(node, "axes and epsilon", true, false, true)?;
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
    require_attributes(node, "axes and keep_dims", true, true, false)?;
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
// Gather
// ---------------------------------------------------------------------------

fn gather(node: &Node, inputs: &[TensorType]) -> Result<Vec<TensorType>, Fn3Error> {
    expect_inputs(node, inputs, 2)?;
    require_attributes(node, "only axes (the axis)", true, false, false)?;
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

/// Reject an attribute field the op does not define.
///
/// The crate's rule is that an op rejects a field it does not understand
/// rather than ignoring it, so a planner cannot silently drop a semantic flag.
fn require_attributes(
    node: &Node,
    allowed: &'static str,
    axes: bool,
    keep_dims: bool,
    epsilon: bool,
) -> Result<(), Fn3Error> {
    let params = &node.params;
    // Every attribute vector a handler reads is bounded by the rank cap, so a
    // malformed `OpParams` cannot make the pass allocate without a limit.
    if params.axes.len() > MAX_RANK {
        return Err(invalid(
            node,
            format!(
                "op {} carries {} axes attributes; the IR accepts at most {MAX_RANK}",
                node.op,
                params.axes.len()
            ),
        ));
    }
    if (!axes && !params.axes.is_empty())
        || (!keep_dims && params.keep_dims)
        || (!epsilon && params.epsilon != 0.0)
    {
        return Err(invalid(
            node,
            format!(
                "op {} accepts {allowed}, but the node supplies axes={:?}, keep_dims={}, \
                 epsilon={}",
                node.op, params.axes, params.keep_dims, params.epsilon
            ),
        ));
    }
    Ok(())
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
        let starts = add_value(&mut graph, DType::I64, consts(&[1]));
        let out = add_value(&mut graph, DType::F32, Vec::new());
        add_node(
            &mut graph,
            OpId::Slice,
            &[data, starts],
            &[out],
            OpParams::new(),
            3,
        );
        let error = infer_shapes(&mut graph).unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::UnsupportedOp {
                op: OpId::Slice,
                ..
            }
        ));
        let text = error.to_string();
        assert!(text.contains("node 3"), "{text}");
        assert!(text.contains("slice"), "{text}");
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
    fn matmul_rejects_a_non_rank_two_operand() {
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
        let error = infer_shapes(&mut graph).unwrap_err();
        assert!(error.to_string().contains("rank-2"), "{error}");
        assert!(error.to_string().contains("node 0"), "{error}");
    }
}
