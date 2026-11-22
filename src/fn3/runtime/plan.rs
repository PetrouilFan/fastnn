//! The v3 execution plan: segments over arena slots.
//!
//! # What a plan is
//!
//! A [`Plan`] is the whole decision the runtime makes about memory and order
//! before the first forward:
//!
//! - one [`Segment`] per graph node, in topological order (the IR is already
//!   topological by construction);
//! - one [`Slot`] per graph value. A slot is either a byte range in the
//!   session's single [`Arena`](crate::inference::runtime::Arena) or a read-only
//!   reference to the importer's weight payload;
//! - the graph's input and output slots;
//! - two sizes: [`Plan::arena_bytes`], the one allocation the session makes, and
//!   [`Plan::peak_bytes`], the largest payload that is live at one instant.
//!
//! # One op per segment, for now
//!
//! [`SegmentOp`] has exactly one variant, [`SegmentOp::Op`]. ADR-0001 makes the
//! segment the unit that a provider implements, and phase 3 adds a
//! `SegmentOp::Fused(..)` that a provider may claim. Inventing that variant
//! before a provider exists would create a contract nothing can satisfy, so the
//! planner emits the shape it can execute and nothing else.
//!
//! # Weights are not copied
//!
//! A value that the importer filled from the model file (an initializer or a
//! lifted `Constant`) has no producer. [`Plan::from_graph`] resolves its name in
//! the [`WeightSet`] and records [`SlotSource::Weight`]: the slot points at the
//! importer's payload bytes, and neither the plan, the arena nor the session
//! copies them. The kernel reads them in place, so a session's arena holds
//! activations only.
//!
//! # Liveness-based offsets
//!
//! Every arena slot is assigned a 64-byte-aligned byte range by a deterministic
//! first-fit walk over the segments:
//!
//! 1. the graph inputs are allocated before segment 0 (the caller writes them);
//! 2. each segment's outputs are allocated before its last-read inputs are
//!    released, so a kernel can never write into a slot it is still reading;
//! 3. a slot whose last reader is the segment that just ran is released and can
//!    be reused by a later output;
//! 4. the high-water mark of that walk is [`Plan::arena_bytes`].
//!
//! A forward therefore never grows the arena: the plan already reserved the
//! worst case, and [`Arena::allocation_count`](crate::inference::runtime::Arena::allocation_count)
//! is a constant for the life of a session.
//!
//! # Determinism
//!
//! Every table here is a `Vec` and every walk is in index order. There is no
//! `HashMap` on this path, so two runs over the same graph produce identical
//! plans, offsets and arena sizes.

use std::fmt;

use crate::inference::backend::OpParams;
use crate::inference::error::Fn3Error;
use crate::inference::import::{ImportedGraph, WeightId, WeightSet};
use crate::inference::ir::{Graph, SourceLoc, TensorType, ValueId};
use crate::inference::ops::OpId;
use crate::inference::shape::Shape;

/// Alignment of every arena slot, in bytes.
///
/// Equal to [`crate::inference::storage::STORAGE_ALIGNMENT`]: a slot starts on a
/// cache line, which is also the alignment [`crate::inference::storage::AlignedBuffer`]
/// gives the allocation itself, so a slot view is aligned for every dtype.
pub const SLOT_ALIGNMENT: usize = crate::inference::storage::STORAGE_ALIGNMENT;

/// Position of one slot in a [`Plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(u32);

impl SlotId {
    /// Construct from a zero-based table position.
    pub const fn new(index: u32) -> SlotId {
        SlotId(index)
    }

    /// Zero-based table position.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for SlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot {}", self.0)
    }
}

/// Where a slot's bytes live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotSource {
    /// A byte range of the session arena, at [`Slot::offset`].
    Arena,
    /// The importer's read-only payload for this weight, in place.
    ///
    /// The slot's `offset` is zero and [`Slot::bytes`] is the payload length; the
    /// bytes are [`crate::inference::model::WeightTensor::bytes`] and are never
    /// copied into the arena.
    Weight(WeightId),
}

/// One value of the graph, as the runtime sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    /// The graph value this slot stands for. Always equal to the slot's
    /// position, so `slots[value.index()]` is the slot of that value.
    pub value: ValueId,
    /// Source name, when the model had one. Names are how
    /// [`Session::run`](crate::inference::runtime::Session::run) addresses its
    /// inputs and outputs.
    pub name: Option<String>,
    /// Fully static logical type. A plan is built for one shape environment, so
    /// a symbolic extent is rejected before any byte is reserved.
    pub ty: TensorType,
    /// Where the bytes live.
    pub source: SlotSource,
    /// Byte offset inside the arena. Zero for a weight slot.
    pub offset: usize,
    /// Byte length of the value. Zero for an empty tensor.
    pub bytes: usize,
}

impl Slot {
    /// `true` when the slot's bytes are the importer's payload, not arena memory.
    pub fn is_weight(&self) -> bool {
        matches!(self.source, SlotSource::Weight(_))
    }
}

/// What one segment runs.
///
/// Phase 2 has one variant: one graph op. A fused segment is phase 3, and it will
/// arrive as a new variant rather than as an invented id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentOp {
    /// Execute one graph operation through the provider's kernel for it.
    Op(OpId),
}

impl SegmentOp {
    /// The graph op this segment runs.
    pub const fn op(self) -> OpId {
        match self {
            SegmentOp::Op(op) => op,
        }
    }
}

impl fmt::Display for SegmentOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SegmentOp::Op(op) => write!(f, "{op}"),
        }
    }
}

/// One unit of execution: operands, results, attributes and source location.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// What to run.
    pub op: SegmentOp,
    /// Operand slots, in the op's operand order.
    pub inputs: Vec<SlotId>,
    /// Result slots, in the op's result order.
    pub outputs: Vec<SlotId>,
    /// The op's attributes, read exactly as the IR carries them.
    pub params: OpParams,
    /// Where the segment came from, so a runtime failure names the source node.
    pub loc: SourceLoc,
}

impl Segment {
    /// The op this segment runs.
    pub const fn op_id(&self) -> OpId {
        self.op.op()
    }
}

impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.loc.describe(), self.op)
    }
}

/// A whole executable plan.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Plan {
    /// Segments in execution order.
    pub segments: Vec<Segment>,
    /// Slots indexed by [`SlotId`]; `slots[value.index()]` is that value's slot.
    pub slots: Vec<Slot>,
    /// Graph inputs, in the model's declared order.
    pub inputs: Vec<SlotId>,
    /// Graph outputs, in the model's declared order.
    pub outputs: Vec<SlotId>,
    /// Bytes to allocate once for the session arena. Zero when nothing is live.
    pub arena_bytes: usize,
    /// Largest number of payload bytes live at one instant (`<= arena_bytes`).
    pub peak_bytes: usize,
}

impl Plan {
    /// Build a plan from a typed graph and the importer's weight table.
    ///
    /// The graph's inputs are the values with no producer that are not in the
    /// weight table, in value order; its outputs are the values no node reads,
    /// in value order, which is the shape an ONNX consumer drains. A caller that
    /// knows the declared interface (the importer does) should use
    /// [`Plan::from_imported`], which uses the model's own lists instead.
    ///
    /// A value with no producer whose name is in the weight table is a weight
    /// slot even when the source also lists it as a graph input — ONNX exporters
    /// that repeat their initializers in `graph.input` mean "this is fixed
    /// data", and the plan must not offer it as something to feed.
    ///
    /// Fails with a value-named error when a type is symbolic, when a weight
    /// payload disagrees with its declared shape, or when byte arithmetic
    /// overflows.
    pub fn from_graph(graph: &Graph, weights: &WeightSet) -> Result<Plan, Fn3Error> {
        Plan::build(graph, weights, None, None)
    }

    /// Build a plan for an imported model, using its declared inputs and
    /// outputs.
    pub fn from_imported(imported: &ImportedGraph) -> Result<Plan, Fn3Error> {
        Plan::build(
            imported.graph(),
            imported.weights(),
            Some(imported.inputs()),
            Some(imported.outputs()),
        )
    }

    fn build(
        graph: &Graph,
        weights: &WeightSet,
        declared_inputs: Option<&[ValueId]>,
        declared_outputs: Option<&[ValueId]>,
    ) -> Result<Plan, Fn3Error> {
        let segments = graph
            .nodes()
            .iter()
            .map(|node| Segment {
                op: SegmentOp::Op(node.op),
                inputs: node.inputs.iter().copied().map(slot_of).collect(),
                outputs: node.outputs.iter().copied().map(slot_of).collect(),
                params: node.params.clone(),
                loc: node.loc.clone(),
            })
            .collect::<Vec<_>>();

        // Slots first, without offsets: an offset needs the liveness walk below.
        let mut slots = Vec::with_capacity(graph.value_count());
        for value in graph.values() {
            let shape = value.ty.static_shape().ok_or_else(|| {
                value_error(
                    value.name.as_deref(),
                    value.id,
                    &value.ty,
                    "its type is not fully static",
                )
            })?;
            let elements = shape.numel()?;
            let bytes = elements
                .checked_mul(value.ty.dtype.size_bytes() as u64)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "runtime Plan slot bytes",
                    axis: None,
                    detail: format!(
                        "value `{}` is {} elements of {}",
                        value.name.as_deref().unwrap_or("<unnamed>"),
                        elements,
                        value.ty.dtype
                    ),
                })?;
            let bytes = usize::try_from(bytes).map_err(|_| Fn3Error::ShapeOverflow {
                context: "runtime Plan slot bytes",
                axis: None,
                detail: format!(
                    "value `{}` needs {bytes} bytes, which does not fit in usize",
                    value.name.as_deref().unwrap_or("<unnamed>")
                ),
            })?;

            // A value with no producer is either a graph input or a constant the
            // importer resolved. The weight table decides which, and a weight's
            // payload is authoritative for the byte length.
            let source = if value.producer.is_some() {
                SlotSource::Arena
            } else {
                match value.name.as_deref().and_then(|name| weights.id_of(name)) {
                    Some(id) => {
                        let weight = weights.get(id).ok_or(Fn3Error::OutOfBounds {
                            what: "runtime Plan weight id",
                            index: id.index() as u64,
                            len: weights.len() as u64,
                        })?;
                        let payload = weight.byte_len();
                        if payload != bytes {
                            return Err(value_error(
                                value.name.as_deref(),
                                value.id,
                                &value.ty,
                                &format!(
                                    "the weight payload is {payload} bytes but the value needs \
                                     {bytes}"
                                ),
                            ));
                        }
                        SlotSource::Weight(id)
                    }
                    None => SlotSource::Arena,
                }
            };
            let offset = 0;
            slots.push(Slot {
                value: value.id,
                name: value.name.clone(),
                ty: value.ty.clone(),
                source,
                offset,
                bytes,
            });
        }

        // Inputs and outputs: declared, or inferred from use.
        let inputs: Vec<SlotId> = match declared_inputs {
            Some(values) => values.iter().copied().map(slot_of).collect(),
            None => slots
                .iter()
                .filter(|slot| !slot.is_weight() && graph.producer(slot.value).is_none())
                .map(|slot| slot_of(slot.value))
                .collect(),
        };
        let outputs: Vec<SlotId> = match declared_outputs {
            Some(values) => values.iter().copied().map(slot_of).collect(),
            None => {
                let mut consumed = vec![false; graph.value_count()];
                for node in graph.nodes() {
                    for input in &node.inputs {
                        consumed[input.index()] = true;
                    }
                }
                slots
                    .iter()
                    .filter(|slot| {
                        !slot.is_weight()
                            && graph.producer(slot.value).is_some()
                            && !consumed[slot.value.index()]
                    })
                    .map(|slot| slot_of(slot.value))
                    .collect()
            }
        };
        // Every named slot in these lists must resolve; a declared id that is
        // out of range is a caller bug, reported rather than skipped.
        for &id in inputs.iter().chain(outputs.iter()) {
            slot(&slots, id)?;
        }

        let (arena_bytes, peak_bytes) = assign_offsets(&segments, &mut slots, &inputs, &outputs)?;

        Ok(Plan {
            segments,
            slots,
            inputs,
            outputs,
            arena_bytes,
            peak_bytes,
        })
    }

    /// One slot by id.
    pub fn slot(&self, id: SlotId) -> Result<&Slot, Fn3Error> {
        slot(&self.slots, id)
    }

    /// The slot of a source value.
    pub fn slot_for_value(&self, value: ValueId) -> Result<&Slot, Fn3Error> {
        self.slot(slot_of(value))
    }

    /// The input slot with exactly this source name.
    pub fn input_slot(&self, name: &str) -> Option<SlotId> {
        self.inputs
            .iter()
            .copied()
            .find(|id| self.slots[id.index()].name.as_deref() == Some(name))
    }

    /// The output slot with exactly this source name.
    pub fn output_slot(&self, name: &str) -> Option<SlotId> {
        self.outputs
            .iter()
            .copied()
            .find(|id| self.slots[id.index()].name.as_deref() == Some(name))
    }

    /// The concrete shape of a slot, checked.
    pub fn shape(&self, id: SlotId) -> Result<Shape, Fn3Error> {
        let slot = self.slot(id)?;
        slot.ty
            .static_shape()
            .ok_or_else(|| Fn3Error::InvalidShape {
                context: "runtime Plan::shape",
                detail: format!("slot {id} has the symbolic type {}", slot.ty),
            })
    }

    /// The number of weight slots, which the arena does not hold.
    pub fn weight_bytes(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.is_weight())
            .map(|slot| slot.bytes)
            .sum()
    }
}

/// The slot id of a value: the tables are one-to-one.
const fn slot_of(value: ValueId) -> SlotId {
    SlotId::new(value.index() as u32)
}

fn slot(slots: &[Slot], id: SlotId) -> Result<&Slot, Fn3Error> {
    slots.get(id.index()).ok_or(Fn3Error::OutOfBounds {
        what: "runtime plan slot id",
        index: id.0 as u64,
        len: slots.len() as u64,
    })
}

fn value_error(name: Option<&str>, id: ValueId, ty: &TensorType, detail: &str) -> Fn3Error {
    Fn3Error::InvalidShape {
        context: "runtime Plan::from_graph",
        detail: format!(
            "value `{}` ({id}, {ty}): {detail}",
            name.unwrap_or("<unnamed>")
        ),
    }
}

/// First-fit free list over the arena, in bytes.
///
/// Ranges are kept sorted by offset and merged on release, so the walk is
/// deterministic and a slot reuses the lowest offset that fits.
#[derive(Default)]
struct FreeList {
    ranges: Vec<(usize, usize)>,
}

impl FreeList {
    fn allocate(&mut self, bytes: usize, high_water: &mut usize) -> usize {
        if bytes == 0 {
            return 0;
        }
        for index in 0..self.ranges.len() {
            let (offset, length) = self.ranges[index];
            if length >= bytes {
                if length == bytes {
                    self.ranges.remove(index);
                } else {
                    self.ranges[index] = (offset + bytes, length - bytes);
                }
                return offset;
            }
        }
        let offset = *high_water;
        *high_water = high_water.saturating_add(bytes);
        offset
    }

    fn release(&mut self, offset: usize, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let position = self.ranges.partition_point(|&(start, _)| start < offset);
        self.ranges.insert(position, (offset, bytes));
        // Merge with the next range, then with the previous one.
        if let Some(next) = self.ranges.get(position + 1).copied() {
            let (start, length) = self.ranges[position];
            if start + length == next.0 {
                self.ranges[position] = (start, length + next.1);
                self.ranges.remove(position + 1);
            }
        }
        if position > 0 {
            let (previous_start, previous_length) = self.ranges[position - 1];
            let (start, length) = self.ranges[position];
            if previous_start + previous_length == start {
                self.ranges[position - 1] = (previous_start, previous_length + length);
                self.ranges.remove(position);
            }
        }
    }

    /// Check that two live ranges never overlap; a plan bug, not input.
    #[cfg(debug_assertions)]
    fn check_disjoint(&self, offset: usize, bytes: usize) -> Result<(), Fn3Error> {
        for &(start, length) in &self.ranges {
            if offset < start + length && start < offset + bytes {
                return Err(Fn3Error::InvalidShape {
                    context: "runtime Plan::assign_offsets",
                    detail: format!(
                        "internal: the allocator handed out [{offset}, {}) inside the free range \
                         [{start}, {})",
                        offset + bytes,
                        start + length
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Align `bytes` up to [`SLOT_ALIGNMENT`], checked.
fn aligned(bytes: usize) -> Result<usize, Fn3Error> {
    let mask = SLOT_ALIGNMENT - 1;
    bytes
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(Fn3Error::ShapeOverflow {
            context: "runtime Plan slot alignment",
            axis: None,
            detail: format!("rounding {bytes} bytes up to {SLOT_ALIGNMENT} overflowed usize"),
        })
}

/// Assign every arena slot an offset, then check the arena's invariants.
///
/// Returns `(arena_bytes, peak_bytes)`:
///
/// - `arena_bytes` is the high-water mark of the first-fit walk, i.e. the size
///   of the one allocation a session makes;
/// - `peak_bytes` is the largest sum of *payload* bytes live at one instant.
fn assign_offsets(
    segments: &[Segment],
    slots: &mut [Slot],
    inputs: &[SlotId],
    outputs: &[SlotId],
) -> Result<(usize, usize), Fn3Error> {
    let count = slots.len();
    // `usize::MAX` means "never released": a graph input nothing reads still
    // needs its range (the caller writes it), and a graph output must survive to
    // the read-back.
    let mut last_use = vec![usize::MAX; count];
    for (index, segment) in segments.iter().enumerate() {
        for &input in &segment.inputs {
            last_use[input.index()] = index;
        }
        for &output in &segment.outputs {
            if last_use[output.index()] == usize::MAX {
                last_use[output.index()] = index;
            }
        }
    }
    for &output in outputs {
        last_use[output.index()] = usize::MAX;
    }

    let mut live = vec![false; count];
    let mut sizes = vec![0usize; count];
    let mut free = FreeList::default();
    let mut high_water = 0usize;
    let mut live_bytes = 0usize;
    let mut peak_bytes = 0usize;

    // Graph inputs are live from the start: the caller writes them before
    // segment 0, so their slots must exist even if a segment never reads one.
    for &input in inputs {
        allocate_slot(
            input,
            slots,
            &mut sizes,
            &mut live,
            &mut free,
            &mut high_water,
            &mut live_bytes,
            &mut peak_bytes,
        )?;
    }
    for (index, segment) in segments.iter().enumerate() {
        for &output in &segment.outputs {
            allocate_slot(
                output,
                slots,
                &mut sizes,
                &mut live,
                &mut free,
                &mut high_water,
                &mut live_bytes,
                &mut peak_bytes,
            )?;
        }
        for &id in segment.inputs.iter().chain(segment.outputs.iter()) {
            let slot_index = id.index();
            if live[slot_index] && last_use[slot_index] == index {
                free.release(slots[slot_index].offset, sizes[slot_index]);
                live[slot_index] = false;
                live_bytes = live_bytes.saturating_sub(slots[slot_index].bytes);
            }
        }
    }

    // Two invariants the executor relies on. Both are plan bugs if they fail,
    // never caller input, so they are checked here once rather than per forward.
    for (index, slot) in slots.iter().enumerate() {
        if slot.is_weight() {
            continue;
        }
        if slot.offset % SLOT_ALIGNMENT != 0 {
            return Err(arena_error(format!(
                "internal: {} starts at byte {} which is not {SLOT_ALIGNMENT}-byte aligned",
                slot.value, slot.offset
            )));
        }
        if slot.offset + slot.bytes > high_water {
            return Err(arena_error(format!(
                "internal: {} ends at byte {} past the {high_water}-byte arena",
                slot.value,
                slot.offset + slot.bytes
            )));
        }
        debug_assert_eq!(sizes[index], aligned(slot.bytes).unwrap_or(0));
    }
    for segment in segments {
        for &input in &segment.inputs {
            let input_slot = &slots[input.index()];
            if input_slot.is_weight() {
                continue;
            }
            for &output in &segment.outputs {
                let output_slot = &slots[output.index()];
                if output_slot.is_weight() {
                    continue;
                }
                let overlaps = output_slot.offset < input_slot.offset + input_slot.bytes
                    && input_slot.offset < output_slot.offset + output_slot.bytes;
                if overlaps {
                    return Err(arena_error(format!(
                        "internal: {segment} would write {} into bytes [{}, {}) while reading {} \
                         from [{}, {})",
                        output_slot.value,
                        output_slot.offset,
                        output_slot.offset + output_slot.bytes,
                        input_slot.value,
                        input_slot.offset,
                        input_slot.offset + input_slot.bytes
                    )));
                }
            }
        }
    }
    Ok((high_water, peak_bytes))
}

/// Allocate one arena slot, if it is an arena slot that is not live yet.
#[allow(clippy::too_many_arguments)]
fn allocate_slot(
    id: SlotId,
    slots: &mut [Slot],
    sizes: &mut [usize],
    live: &mut [bool],
    free: &mut FreeList,
    high_water: &mut usize,
    live_bytes: &mut usize,
    peak_bytes: &mut usize,
) -> Result<(), Fn3Error> {
    let index = id.index();
    if live[index] || slots[index].is_weight() {
        return Ok(());
    }
    let bytes = aligned(slots[index].bytes)?;
    let offset = free.allocate(bytes, high_water);
    #[cfg(debug_assertions)]
    free.check_disjoint(offset, bytes)?;
    slots[index].offset = offset;
    sizes[index] = bytes;
    live[index] = true;
    *live_bytes = live_bytes.saturating_add(slots[index].bytes);
    *peak_bytes = (*peak_bytes).max(*live_bytes);
    Ok(())
}

fn arena_error(detail: String) -> Fn3Error {
    Fn3Error::InvalidShape {
        context: "runtime Plan arena",
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::backend::OpParams;
    use crate::inference::dtype::DType;
    use crate::inference::ir::{DimExpr, Graph, SourceLoc, TensorType};
    use crate::inference::model::{WeightSet, WeightSource, WeightTensor};

    fn ty(extents: &[u64]) -> TensorType {
        TensorType::new(DType::F32, extents.iter().copied().map(DimExpr::constant))
    }

    fn shape(extents: &[u64]) -> Shape {
        Shape::from_dims(extents.iter().copied())
    }

    fn weight(name: &str, extents: &[u64]) -> WeightTensor {
        let elements: u64 = extents.iter().product();
        WeightTensor::new(
            name,
            DType::F32,
            shape(extents),
            vec![0u8; (elements * 4) as usize],
            WeightSource::Initializer,
        )
        .unwrap()
    }

    /// `x[2,3] -> relu -> mul(weight[2,3]) -> y[2,3]`.
    fn chain() -> (Graph, WeightSet) {
        let weights = WeightSet::new(vec![weight("w", &[2, 3])]).unwrap();
        let mut graph = Graph::new();
        let x = graph.add_value(Some("x"), ty(&[2, 3])).unwrap();
        let w = graph.add_value(Some("w"), ty(&[2, 3])).unwrap();
        let h = graph.add_value(Some("h"), ty(&[2, 3])).unwrap();
        let y = graph.add_value(Some("y"), ty(&[2, 3])).unwrap();
        graph
            .add_node(
                OpId::Relu,
                vec![x],
                vec![h],
                OpParams::new(),
                SourceLoc::new(0, "Relu"),
            )
            .unwrap();
        graph
            .add_node(
                OpId::Mul,
                vec![h, w],
                vec![y],
                OpParams::new(),
                SourceLoc::new(1, "Mul").with_output("y"),
            )
            .unwrap();
        (graph, weights)
    }

    #[test]
    fn one_segment_per_node_with_weight_slots_out_of_the_arena() {
        let (graph, weights) = chain();
        let plan = Plan::from_graph(&graph, &weights).unwrap();
        assert_eq!(plan.segments.len(), 2);
        assert_eq!(plan.slots.len(), graph.value_count());
        assert_eq!(
            plan.inputs,
            vec![SlotId::new(0)],
            "the weight is not an input"
        );
        assert_eq!(plan.outputs, vec![SlotId::new(3)]);
        assert_eq!(plan.slots[1].source, SlotSource::Weight(WeightId::new(0)));
        assert_eq!(plan.slots[1].bytes, 24);
        assert_eq!(plan.weight_bytes(), 24);
        // x is live until the first segment, h until the second, y is the
        // output: two [2, 3] f32 activations fit in one slot's worth of space.
        assert_eq!(plan.arena_bytes, 128, "two 64-byte-aligned 24-byte slots");
        assert_eq!(plan.peak_bytes, 48);
        // `h` is produced by the first segment and read by the second, so it
        // cannot share `x`'s range; `y` can, because `x` dies first.
        assert_eq!(plan.slots[2].offset, 64);
        assert_eq!(plan.slots[3].offset, 0);
        // Every arena slot is 64-byte aligned and in range.
        for slot in &plan.slots {
            if slot.is_weight() {
                continue;
            }
            assert_eq!(slot.offset % SLOT_ALIGNMENT, 0, "{slot:?}");
            assert!(slot.offset + slot.bytes <= plan.arena_bytes, "{slot:?}");
        }
    }

    #[test]
    fn a_dead_slot_is_reused_by_the_next_segment() {
        // Four [1, 16, 8, 8] values (4 KiB each) chained through three ops. With
        // liveness one previous slot is reusable, so the arena is two slots, not
        // four.
        let mut graph = Graph::new();
        let first = graph.add_value(Some("x"), ty(&[1, 16, 8, 8])).unwrap();
        let mut previous = first;
        for index in 0..3 {
            let next = graph
                .add_value(Some(&format!("v{index}")), ty(&[1, 16, 8, 8]))
                .unwrap();
            graph
                .add_node(
                    OpId::Relu,
                    vec![previous],
                    vec![next],
                    OpParams::new(),
                    SourceLoc::new(index, "Relu"),
                )
                .unwrap();
            previous = next;
        }
        let plan = Plan::from_graph(&graph, &WeightSet::default()).unwrap();
        assert_eq!(plan.arena_bytes, 2 * 4096);
        assert_eq!(plan.peak_bytes, 8192);
        assert_eq!(plan.slots[first.index()].offset, plan.slots[2].offset);
    }

    #[test]
    fn a_symbolic_type_is_rejected_by_name() {
        let mut graph = Graph::new();
        graph
            .add_value(
                Some("tokens"),
                TensorType::new(
                    DType::F32,
                    vec![
                        DimExpr::symbol(crate::inference::shape::Symbol::Batch),
                        DimExpr::constant(4),
                    ],
                ),
            )
            .unwrap();
        let error = Plan::from_graph(&graph, &WeightSet::default()).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("tokens"), "{text}");
        assert!(text.contains("not fully static"), "{text}");
    }

    #[test]
    fn a_variable_rank_zero_value_takes_no_arena_space() {
        let weights = WeightSet::new(vec![weight("eps", &[1])]).unwrap();
        let mut graph = Graph::new();
        let eps = graph.add_value(Some("eps"), ty(&[1])).unwrap();
        let y = graph.add_value(Some("y"), ty(&[1])).unwrap();
        graph
            .add_node(
                OpId::Neg,
                vec![eps],
                vec![y],
                OpParams::new(),
                SourceLoc::new(0, "Neg"),
            )
            .unwrap();
        let plan = Plan::from_graph(&graph, &weights).unwrap();
        // `eps` is a weight (read-only, no arena bytes) and the only arena slot
        // is the output.
        assert!(plan.slots[eps.index()].is_weight());
        assert_eq!(plan.arena_bytes, 64);
        assert_eq!(plan.peak_bytes, 4);
    }

    #[test]
    fn plans_of_the_same_graph_are_identical() {
        let (graph, weights) = chain();
        let first = Plan::from_graph(&graph, &weights).unwrap();
        let second = Plan::from_graph(&graph, &weights).unwrap();
        assert_eq!(first, second);
    }
}
