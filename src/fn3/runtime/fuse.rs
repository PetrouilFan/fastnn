//! The plan's fusion pass: fold recognised op chains into one segment.
//!
//! # What this pass rewrites, and what it does not
//!
//! It rewrites the **plan**, never the graph. The typed IR keeps every node and
//! every value; the plan loses the intermediate segments and their arena slots,
//! which is exactly the memory the fused chain no longer moves. A plan built
//! with [`PlanOptions::unfused`](crate::inference::runtime::PlanOptions::unfused)
//! is the same graph one-segment-per-node, and the fusion test compares the two
//! outputs.
//!
//! # When a fusion fires
//!
//! Two rules, both structural:
//!
//! 1. the chain replaces at least two segments (so it always removes one);
//! 2. **every** declared consumer of each intermediate value is inside the
//!    chain. A value that something else reads has to exist in the arena, and a
//!    fused segment writes only its final result, so such a chain is left
//!    alone.
//!
//! # Recognised forms
//!
//! | Graph | Fused segment | Folded |
//! |---|---|---|
//! | `act(conv(x))` | `conv` + activation | `Relu`, `Clip`, `Sigmoid`, `Tanh`, `Gelu` |
//! | `mul(conv(x), sigmoid(conv(x)))` | `conv` + `Swish` | ONNX SiLU, two segments |
//! | `add(conv(x), other)` | `conv` + sum | same shape and dtype only |
//! | `act(add(conv(x), other))` | `conv` + sum + activation | sum first, then the activation |
//!
//! The last row is the only order oneDNN's direct convolution can express: its
//! `sum` post-op must come before any eltwise post-op (and it reads the
//! destination buffer, not a separate argument). A residual add that follows an
//! activation — `add(silu(conv(x)), other)`, which is what the pinned YOLOv8n
//! export writes for every C2f block — would need the opposite order and is
//! therefore **not** fused; those adds stay scalar segments.
//!
//! The pass is deterministic: it walks segments in plan order and every table is
//! a `Vec`. There is no `HashMap`, so two plans over the same graph are equal.

use crate::inference::backend::{Activation, FusedOp};
use crate::inference::ir::{SourceLoc, TensorType};
use crate::inference::ops::OpId;
use crate::inference::runtime::plan::{Fusion, Segment, SegmentOp, SlotId};

/// Fold every recognised chain in `segments`, in plan order.
///
/// `types[i]` is the type of value/slot `i` (slots are one-to-one with values
/// in a freshly built plan). The returned segment list is the plan's, and the
/// returned fusions index it.
pub(crate) fn fuse(segments: Vec<Segment>, types: &[TensorType]) -> (Vec<Segment>, Vec<Fusion>) {
    let count = segments.len();
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); types.len()];
    for (index, segment) in segments.iter().enumerate() {
        for &input in &segment.inputs {
            if let Some(list) = consumers.get_mut(input.index()) {
                list.push(index);
            }
        }
    }

    let mut removed = vec![false; count];
    let mut fused: Vec<Option<(Segment, FusedOp)>> = vec![None; count];
    let mut index = 0usize;
    while index < count {
        if removed[index] {
            index += 1;
            continue;
        }
        if let Some((chain, folded)) = recognise(&segments, index, &consumers, types) {
            let anchor = &segments[index];
            let output = chain.output;
            let inputs = chain.inputs;
            let chain_ops = chain.ops;
            let activation = chain.activation;
            let sum = chain.sum;
            let last = chain.last;
            let loc = SourceLoc {
                node_index: anchor.loc.node_index,
                op_type: format!("Fused({})", chain_ops_name(&chain_ops)),
                output: segments[last].loc.output.clone(),
            };
            let fused_op = FusedOp {
                anchor: anchor.op_id(),
                activation,
                sum,
                chain: chain_ops,
            };
            let segment = Segment {
                op: SegmentOp::Fused(fused_op.clone()),
                engine: anchor.engine,
                inputs,
                outputs: vec![output],
                params: anchor.params.clone(),
                loc,
            };
            for &folded_index in &folded {
                removed[folded_index] = true;
            }
            fused[index] = Some((segment, fused_op));
        }
        index += 1;
    }

    let mut out = Vec::with_capacity(count);
    let mut fusions = Vec::new();
    for (index, segment) in segments.into_iter().enumerate() {
        if removed[index] {
            continue;
        }
        match fused[index].take() {
            Some((fused_segment, fused_op)) => {
                fusions.push(Fusion {
                    segment: out.len(),
                    fused: fused_op,
                });
                out.push(fused_segment);
            }
            None => out.push(segment),
        }
    }
    (out, fusions)
}

/// The parts of a recognised chain the caller turns into a segment.
struct Chain {
    ops: Vec<OpId>,
    activation: Option<Activation>,
    /// Position of the residual operand in the segment's input list.
    sum: Option<usize>,
    inputs: Vec<SlotId>,
    output: SlotId,
    /// Plan index of the chain's last segment (for the source location).
    last: usize,
}

fn chain_ops_name(ops: &[OpId]) -> String {
    ops.iter().map(|op| op.name()).collect::<Vec<_>>().join("+")
}

/// Try to fold a chain that starts at `anchor`.
///
/// Returns the fused chain and the plan indices it absorbs, or `None` when no
/// rule matches or a structural rule fails. Every rule needs the anchor to be a
/// single-output `Conv`, because only a convolution has a oneDNN epilogue.
fn recognise(
    segments: &[Segment],
    anchor: usize,
    consumers: &[Vec<usize>],
    types: &[TensorType],
) -> Option<(Chain, Vec<usize>)> {
    let segment = segments.get(anchor)?;
    if segment.op_id() != OpId::Conv || segment.outputs.len() != 1 {
        return None;
    }
    let conv_out = segment.outputs[0];
    let conv_inputs = segment.inputs.clone();
    let readers = consumers.get(conv_out.index())?;

    // conv -> act
    if let [next] = readers[..] {
        if next <= anchor {
            return None;
        }
        if let Some(activation) = activation_of(&segments[next]) {
            let act_segment = &segments[next];
            if act_segment.inputs.as_slice() != [conv_out].as_slice()
                || act_segment.outputs.len() != 1
            {
                return None;
            }
            return Some((
                Chain {
                    ops: vec![OpId::Conv, activation.op()],
                    activation: Some(activation),
                    sum: None,
                    inputs: conv_inputs,
                    output: act_segment.outputs[0],
                    last: next,
                },
                vec![next],
            ));
        }
        // conv -> add [-> act]
        return recognise_sum(segments, anchor, next, conv_out, conv_inputs, types);
    }

    // conv -> sigmoid -> mul  (the ONNX SiLU)
    if readers.len() == 2 {
        let (sigmoid, mul) = match (segments.get(readers[0]), segments.get(readers[1])) {
            (Some(first), Some(second)) => {
                if first.op_id() == OpId::Sigmoid && second.op_id() == OpId::Mul {
                    (readers[0], readers[1])
                } else if second.op_id() == OpId::Sigmoid && first.op_id() == OpId::Mul {
                    (readers[1], readers[0])
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        if sigmoid <= anchor || mul <= anchor {
            return None;
        }
        let sigmoid_segment = &segments[sigmoid];
        if sigmoid_segment.inputs.as_slice() != [conv_out].as_slice()
            || sigmoid_segment.outputs.len() != 1
        {
            return None;
        }
        let sigmoid_out = sigmoid_segment.outputs[0];
        // The sigmoid must exist only for this multiply.
        if consumers.get(sigmoid_out.index())?.as_slice() != [mul].as_slice() {
            return None;
        }
        let mul_segment = &segments[mul];
        if mul_segment.inputs.len() != 2
            || mul_segment.outputs.len() != 1
            || !mul_segment.inputs.contains(&conv_out)
            || !mul_segment.inputs.contains(&sigmoid_out)
        {
            return None;
        }
        if !same_dtype(types, conv_out, mul_segment.outputs[0]) {
            return None;
        }
        return Some((
            Chain {
                ops: vec![OpId::Conv, OpId::Sigmoid, OpId::Mul],
                activation: Some(Activation::Swish),
                sum: None,
                inputs: conv_inputs,
                output: mul_segment.outputs[0],
                last: mul,
            },
            vec![sigmoid, mul],
        ));
    }
    None
}

/// `add(conv, other)` and `act(add(conv, other))`.
fn recognise_sum(
    segments: &[Segment],
    anchor: usize,
    add: usize,
    conv_out: SlotId,
    mut inputs: Vec<SlotId>,
    types: &[TensorType],
) -> Option<(Chain, Vec<usize>)> {
    let add_segment = &segments[add];
    if add_segment.op_id() != OpId::Add
        || add_segment.inputs.len() != 2
        || add_segment.outputs.len() != 1
    {
        return None;
    }
    let position = add_segment.inputs.iter().position(|&id| id == conv_out)?;
    let other = add_segment.inputs[1 - position];
    if other == conv_out {
        return None;
    }
    let sum_out = add_segment.outputs[0];
    // The type rule: the residual operand and the result have the anchor's own
    // type. No broadcast is fused, because a oneDNN sum has no broadcast and
    // the scalar epilogue would stop being a plain elementwise pass.
    if !types_equal(types, other, sum_out) || !types_equal(types, conv_out, sum_out) {
        return None;
    }
    let sum_index = inputs.len();
    inputs.push(other);
    let mut last = add;
    let mut ops = vec![OpId::Conv, OpId::Add];
    let mut output = sum_out;
    let mut folded = vec![add];
    let mut activation = None;
    // add -> act: the oneDNN post-op order (sum, then eltwise).
    if let [next] = consumers_of_add(segments, add, sum_out)[..] {
        if next > anchor {
            if let Some(act) = activation_of(&segments[next]) {
                let act_segment = &segments[next];
                if act_segment.inputs.as_slice() == [sum_out].as_slice()
                    && act_segment.outputs.len() == 1
                {
                    activation = Some(act);
                    ops.push(act.op());
                    output = act_segment.outputs[0];
                    last = next;
                    folded.push(next);
                }
            }
        }
    }
    Some((
        Chain {
            ops,
            activation,
            sum: Some(sum_index),
            inputs,
            output,
            last,
        },
        folded,
    ))
}

/// The segments reading `value`, for the single-consumer check above.
fn consumers_of_add<'a>(segments: &'a [Segment], add: usize, value: SlotId) -> Vec<usize> {
    let mut readers = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        if index != add && segment.inputs.contains(&value) {
            readers.push(index);
        }
    }
    readers
}

/// The activation a segment is, when it is one the fused epilogue expresses.
fn activation_of(segment: &Segment) -> Option<Activation> {
    if segment.op.fused().is_some() {
        return None;
    }
    let params = &segment.params;
    Some(match segment.op_id() {
        OpId::Relu => Activation::Relu,
        OpId::Sigmoid => Activation::Sigmoid,
        OpId::Tanh => Activation::Tanh,
        OpId::Gelu => Activation::Gelu,
        OpId::Clip => {
            // Both bounds must be folded constants inside a valid interval; a
            // one-sided clip is left as a segment rather than emulated with an
            // infinity oneDNN may not accept.
            let (Some(low), Some(high)) = (params.clip_min, params.clip_max) else {
                return None;
            };
            if !low.is_finite() || !high.is_finite() || low > high {
                return None;
            }
            Activation::Clip { low, high }
        }
        _ => return None,
    })
}

fn same_dtype(types: &[TensorType], left: SlotId, right: SlotId) -> bool {
    match (types.get(left.index()), types.get(right.index())) {
        (Some(left), Some(right)) => left.dtype == right.dtype,
        _ => false,
    }
}

fn types_equal(types: &[TensorType], left: SlotId, right: SlotId) -> bool {
    match (types.get(left.index()), types.get(right.index())) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::backend::OpParams;
    use crate::inference::backend::SegmentEngine;
    use crate::inference::dtype::DType;
    use crate::inference::ir::DimExpr;

    fn ty(extents: &[u64]) -> TensorType {
        TensorType::new(DType::F32, extents.iter().copied().map(DimExpr::constant))
    }

    /// Build a segment list and a slot type table from a small op script.
    struct Script {
        types: Vec<TensorType>,
        segments: Vec<Segment>,
    }

    impl Script {
        fn new(values: &[&[u64]]) -> Script {
            Script {
                types: values.iter().map(|extents| ty(extents)).collect(),
                segments: Vec::new(),
            }
        }

        fn value(&mut self, extents: &[u64]) -> SlotId {
            self.types.push(ty(extents));
            SlotId::new((self.types.len() - 1) as u32)
        }

        fn push(&mut self, op: OpId, inputs: &[SlotId], output: SlotId) {
            self.push_with(op, inputs, output, OpParams::new())
        }

        fn push_with(&mut self, op: OpId, inputs: &[SlotId], output: SlotId, params: OpParams) {
            let index = self.segments.len();
            self.segments.push(Segment {
                op: SegmentOp::Op(op),
                engine: SegmentEngine::Scalar,
                inputs: inputs.to_vec(),
                outputs: vec![output],
                params,
                loc: SourceLoc::new(index, op.name()),
            });
        }
    }

    #[test]
    fn a_convolution_and_a_relu_become_one_segment() {
        let mut script = Script::new(&[]);
        let x = script.value(&[1, 2, 4, 4]);
        let w = script.value(&[4, 2, 3, 3]);
        let conv = script.value(&[1, 4, 4, 4]);
        let out = script.value(&[1, 4, 4, 4]);
        script.push(OpId::Conv, &[x, w], conv);
        script.push(OpId::Relu, &[conv], out);
        let (segments, fusions) = fuse(script.segments, &script.types);
        assert_eq!(segments.len(), 1);
        assert_eq!(fusions.len(), 1);
        assert_eq!(fusions[0].segment, 0);
        assert_eq!(fusions[0].removed(), 1);
        assert_eq!(fusions[0].fused.activation, Some(Activation::Relu));
        assert_eq!(segments[0].outputs, vec![out]);
        assert_eq!(segments[0].inputs, vec![x, w]);
    }

    #[test]
    fn the_onnx_silu_becomes_a_swish_epilogue() {
        let mut script = Script::new(&[]);
        let x = script.value(&[1, 2, 4, 4]);
        let w = script.value(&[4, 2, 3, 3]);
        let conv = script.value(&[1, 4, 4, 4]);
        let sigmoid = script.value(&[1, 4, 4, 4]);
        let out = script.value(&[1, 4, 4, 4]);
        script.push(OpId::Conv, &[x, w], conv);
        script.push(OpId::Sigmoid, &[conv], sigmoid);
        script.push(OpId::Mul, &[conv, sigmoid], out);
        let (segments, fusions) = fuse(script.segments, &script.types);
        assert_eq!(segments.len(), 1);
        assert_eq!(fusions[0].fused.activation, Some(Activation::Swish));
        assert_eq!(fusions[0].chain(), &[OpId::Conv, OpId::Sigmoid, OpId::Mul]);
        assert_eq!(fusions[0].removed(), 2);
        assert_eq!(segments[0].outputs, vec![out]);
    }

    #[test]
    fn a_residual_add_after_the_activation_is_not_fused() {
        let mut script = Script::new(&[]);
        let x = script.value(&[1, 2, 4, 4]);
        let w = script.value(&[4, 2, 3, 3]);
        let other = script.value(&[1, 4, 4, 4]);
        let conv = script.value(&[1, 4, 4, 4]);
        let sigmoid = script.value(&[1, 4, 4, 4]);
        let silu = script.value(&[1, 4, 4, 4]);
        let out = script.value(&[1, 4, 4, 4]);
        script.push(OpId::Conv, &[x, w], conv);
        script.push(OpId::Sigmoid, &[conv], sigmoid);
        script.push(OpId::Mul, &[conv, sigmoid], silu);
        script.push(OpId::Add, &[other, silu], out);
        let (segments, fusions) = fuse(script.segments, &script.types);
        // The SiLU folds; the add needs the opposite arithmetic order and stays.
        assert_eq!(segments.len(), 2);
        assert_eq!(fusions.len(), 1);
        assert_eq!(fusions[0].fused.activation, Some(Activation::Swish));
        assert_eq!(segments[1].op_id(), OpId::Add);
    }

    #[test]
    fn a_residual_add_on_the_convolution_fuses_as_a_sum() {
        let mut script = Script::new(&[]);
        let x = script.value(&[1, 2, 4, 4]);
        let w = script.value(&[4, 2, 3, 3]);
        let other = script.value(&[1, 4, 4, 4]);
        let conv = script.value(&[1, 4, 4, 4]);
        let added = script.value(&[1, 4, 4, 4]);
        let out = script.value(&[1, 4, 4, 4]);
        script.push(OpId::Conv, &[x, w], conv);
        script.push(OpId::Add, &[conv, other], added);
        script.push(OpId::Relu, &[added], out);
        let (segments, fusions) = fuse(script.segments, &script.types);
        assert_eq!(segments.len(), 1);
        assert_eq!(fusions[0].fused.sum, Some(2), "the residual is inputs[2]");
        assert_eq!(fusions[0].fused.activation, Some(Activation::Relu));
        assert_eq!(segments[0].inputs, vec![x, w, other]);
        assert_eq!(segments[0].outputs, vec![out]);
    }

    #[test]
    fn a_value_read_outside_the_chain_stops_the_fusion() {
        let mut script = Script::new(&[]);
        let x = script.value(&[1, 2, 4, 4]);
        let w = script.value(&[4, 2, 3, 3]);
        let conv = script.value(&[1, 4, 4, 4]);
        let sigmoid = script.value(&[1, 4, 4, 4]);
        let silu = script.value(&[1, 4, 4, 4]);
        let sum = script.value(&[1, 4, 4, 4]);
        script.push(OpId::Conv, &[x, w], conv);
        script.push(OpId::Sigmoid, &[conv], sigmoid);
        script.push(OpId::Mul, &[conv, sigmoid], silu);
        // A second reader of the convolution's own output.
        script.push(OpId::Add, &[conv, silu], sum);
        let (segments, _) = fuse(script.segments, &script.types);
        assert_eq!(
            segments.len(),
            4,
            "a third consumer of `conv` blocks the chain"
        );
    }

    #[test]
    fn a_mismatched_residual_type_is_not_fused() {
        let mut script = Script::new(&[]);
        let x = script.value(&[1, 2, 4, 4]);
        let w = script.value(&[4, 2, 3, 3]);
        let other = script.value(&[1, 4, 4]);
        let conv = script.value(&[1, 4, 4, 4]);
        let added = script.value(&[1, 4, 4, 4]);
        script.push(OpId::Conv, &[x, w], conv);
        script.push(OpId::Add, &[conv, other], added);
        let (segments, fusions) = fuse(script.segments, &script.types);
        assert_eq!(segments.len(), 2);
        assert!(fusions.is_empty());
    }
}
