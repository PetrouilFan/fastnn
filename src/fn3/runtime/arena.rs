//! The one arena a session runs in.
//!
//! # One allocation, zero growth
//!
//! A [`Plan`] already knows the whole lifetime of every activation: it assigns
//! each arena slot a byte range and reports the high-water mark as
//! [`Plan::arena_bytes`]. [`Arena::new`] therefore allocates exactly once — a
//! single 64-byte-aligned, zeroed [`AlignedBuffer`] — and a forward never grows
//! it. [`Arena::allocation_count`] counts those allocations, so a test can run
//! many forwards and prove the count did not move.
//!
//! # Disjoint windows, no unsafe split
//!
//! A segment reads some slots and writes others, and the plan guarantees an
//! output range never overlaps an input range
//! (`Plan::from_graph` checks it). `Arena::frame` turns that promise into
//! Rust's own aliasing rule: it walks the arena once with `split_at_mut`, in
//! offset order, and hands back one disjoint `&[u8]` per input slot and one
//! disjoint `&mut [u8]` per output slot. No `unsafe` is involved in the split,
//! so a plan bug cannot become undefined behaviour — at worst the split returns
//! an error.
//!
//! # Element views
//!
//! [`Arena::slot`] and [`Arena::slot_mut`] are the typed accessors for `f32`
//! slots (the compute dtype). A slot that holds `i64`, `i32` or `bool` is read
//! through [`Arena::slot_bytes`] / [`Arena::slot_bytes_mut`], which return the
//! raw bytes the dtype-generic kernels move.

use bytemuck::Pod;

use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::runtime::plan::{Plan, SlotId, SlotSource};
use crate::inference::storage::AlignedBuffer;

/// The disjoint windows of one segment: shared input windows (aligned to the
/// segment's input list, `None` for a weight operand) and exclusive output
/// windows.
pub(crate) type SegmentWindows<'a> = (Vec<Option<&'a [u8]>>, Vec<&'a mut [u8]>);

/// One slot's byte range inside the arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotSpan {
    /// Byte offset from the start of the allocation.
    offset: usize,
    /// Byte length (the payload, not the aligned reservation).
    bytes: usize,
    /// The dtype the payload holds, so a typed accessor can reject a mismatch.
    dtype: DType,
    /// Arena memory or the importer's read-only payload.
    source: SlotSource,
}

/// The per-session arena: one allocation, sliced into plan slots.
#[derive(Debug)]
pub struct Arena {
    buffer: AlignedBuffer,
    spans: Vec<SlotSpan>,
    allocations: usize,
}

impl Arena {
    /// Allocate the plan's arena once, zeroed, 64-byte aligned.
    ///
    /// `Plan::arena_bytes == 0` performs no allocation at all; every slot is
    /// empty and every accessor returns an empty slice.
    pub fn new(plan: &Plan) -> Result<Arena, Fn3Error> {
        // The slot table already carries the offsets, so the arena is a plain
        // owner of the one buffer; the spans below are a checked copy.
        let buffer = AlignedBuffer::zeroed(plan.arena_bytes)?;
        let allocations = usize::from(plan.arena_bytes != 0);
        let spans = plan
            .slots
            .iter()
            .map(|slot| {
                let span = SlotSpan {
                    offset: slot.offset,
                    bytes: slot.bytes,
                    dtype: slot.ty.dtype,
                    source: slot.source,
                };
                if matches!(slot.source, SlotSource::Arena) {
                    let end =
                        span.offset
                            .checked_add(span.bytes)
                            .ok_or(Fn3Error::ShapeOverflow {
                                context: "runtime Arena slot span",
                                axis: None,
                                detail: format!("slot {} end overflows usize", slot.value),
                            })?;
                    if end > plan.arena_bytes {
                        return Err(Fn3Error::InvalidShape {
                            context: "runtime Arena",
                            detail: format!(
                                "slot {} ends at byte {end}, past the {}-byte arena",
                                slot.value, plan.arena_bytes
                            ),
                        });
                    }
                }
                Ok(span)
            })
            .collect::<Result<Vec<_>, Fn3Error>>()?;
        Ok(Arena {
            buffer,
            spans,
            allocations,
        })
    }

    /// The allocation in bytes.
    pub fn bytes(&self) -> usize {
        self.buffer.len()
    }

    /// How many allocations this arena has made.
    ///
    /// One for a non-empty arena and zero for an empty one, forever: nothing in
    /// [`crate::inference::runtime::Session::execute`] allocates from the arena,
    /// so a test that runs many forwards and asserts this number did not change
    /// is proving memory stability rather than asserting a constant.
    pub fn allocation_count(&self) -> usize {
        self.allocations
    }

    fn span(&self, id: SlotId) -> Result<SlotSpan, Fn3Error> {
        self.spans
            .get(id.index())
            .copied()
            .ok_or(Fn3Error::OutOfBounds {
                what: "runtime arena slot id",
                index: id.index() as u64,
                len: self.spans.len() as u64,
            })
    }

    /// The raw bytes of one arena slot.
    ///
    /// A weight slot is not in the arena: its bytes are
    /// [`crate::inference::model::WeightTensor::bytes`], which the session reads
    /// from the weight table, so asking the arena for them is an error rather
    /// than a read of an unrelated range.
    pub fn slot_bytes(&self, id: SlotId) -> Result<&[u8], Fn3Error> {
        let span = self.span(id)?;
        if span.source != SlotSource::Arena {
            return Err(Fn3Error::InvalidShape {
                context: "runtime arena slot bytes",
                detail: format!(
                    "{id} is a read-only weight slot; read it from the weight table, not the arena"
                ),
            });
        }
        let end = span.offset + span.bytes;
        self.buffer
            .as_bytes()
            .get(span.offset..end)
            .ok_or_else(|| Fn3Error::OutOfBounds {
                what: "runtime arena slot bytes",
                index: end as u64,
                len: self.buffer.len() as u64,
            })
    }

    /// The raw bytes of one arena slot, mutably.
    pub fn slot_bytes_mut(&mut self, id: SlotId) -> Result<&mut [u8], Fn3Error> {
        let span = self.span(id)?;
        if span.source != SlotSource::Arena {
            return Err(Fn3Error::InvalidShape {
                context: "runtime arena slot bytes",
                detail: format!("{id} is a read-only weight slot and cannot be written"),
            });
        }
        let len = self.buffer.len();
        let end = span.offset + span.bytes;
        self.buffer
            .as_bytes_mut()
            .get_mut(span.offset..end)
            .ok_or(Fn3Error::OutOfBounds {
                what: "runtime arena slot bytes",
                index: end as u64,
                len: len as u64,
            })
    }

    /// One `f32` slot.
    ///
    /// A slot of another dtype is reported, not reinterpreted.
    pub fn slot(&self, id: SlotId) -> Result<&[f32], Fn3Error> {
        let span = self.span(id)?;
        require_f32(span, id)?;
        let bytes = self.slot_bytes(id)?;
        bytemuck::try_cast_slice(bytes).map_err(|error| Fn3Error::StorageCast {
            element: "f32",
            byte_len: bytes.len(),
            detail: format!("slot {id}: {error:?}"),
        })
    }

    /// One `f32` slot, mutably.
    pub fn slot_mut(&mut self, id: SlotId) -> Result<&mut [f32], Fn3Error> {
        let span = self.span(id)?;
        require_f32(span, id)?;
        let bytes = self.slot_bytes_mut(id)?;
        let byte_len = bytes.len();
        bytemuck::try_cast_slice_mut(bytes).map_err(|error| Fn3Error::StorageCast {
            element: "f32",
            byte_len,
            detail: format!("slot {id}: {error:?}"),
        })
    }

    /// Write a slot from host elements, which must fill it exactly.
    ///
    /// Used for the session's inputs: the graph has one fixed shape, so an input
    /// that is short or long is a caller error, never a partial write.
    pub fn write_slot<T: Pod>(&mut self, id: SlotId, values: &[T]) -> Result<(), Fn3Error> {
        let bytes = bytemuck::cast_slice(values);
        let target = self.slot_bytes_mut(id)?;
        if target.len() != bytes.len() {
            return Err(Fn3Error::ShapeMismatch {
                op: None,
                expected: format!("{} bytes for slot {id}", target.len()),
                actual: format!("{} bytes supplied", bytes.len()),
            });
        }
        target.copy_from_slice(bytes);
        Ok(())
    }

    /// Split the arena into one disjoint window per requested slot.
    ///
    /// `inputs` get shared windows and `outputs` get exclusive ones, all from a
    /// single `split_at_mut` walk in offset order, so the borrow checker sees the
    /// same disjointness the plan guarantees. A weight slot returns `None` in
    /// `inputs` (it is not in the arena) and is rejected in `outputs`.
    ///
    /// The returned vectors are parallel to the two slice arguments.
    pub(crate) fn frame<'a>(
        &'a mut self,
        plan: &Plan,
        inputs: &[SlotId],
        outputs: &[SlotId],
    ) -> Result<SegmentWindows<'a>, Fn3Error> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Kind {
            Input(usize),
            Output(usize),
        }

        // (offset, bytes, kind) for every arena slot this segment touches.
        let mut wanted: Vec<(usize, usize, Kind)> =
            Vec::with_capacity(inputs.len() + outputs.len());
        for (index, &id) in inputs.iter().enumerate() {
            let slot = plan.slot(id)?;
            match slot.source {
                SlotSource::Arena => wanted.push((slot.offset, slot.bytes, Kind::Input(index))),
                SlotSource::Weight(_) => {}
            }
        }
        for (index, &id) in outputs.iter().enumerate() {
            let slot = plan.slot(id)?;
            if slot.is_weight() {
                return Err(Fn3Error::InvalidShape {
                    context: "runtime Arena::frame",
                    detail: format!("{id} is a read-only weight slot but is named as an output"),
                });
            }
            if slot.bytes == 0 {
                // Zero-length outputs are handled below, after the walk, so the
                // split never has to advance by zero twice at one offset.
                continue;
            }
            wanted.push((slot.offset, slot.bytes, Kind::Output(index)));
        }
        wanted.sort_unstable_by_key(|&(offset, bytes, _)| (offset, bytes));

        let mut shared: Vec<Option<&'a [u8]>> = vec![None; inputs.len()];
        let mut exclusive: Vec<Option<&'a mut [u8]>> = (0..outputs.len()).map(|_| None).collect();

        let mut rest: &'a mut [u8] = self.buffer.as_bytes_mut();
        let mut consumed = 0usize;
        for (offset, bytes, kind) in wanted {
            if offset < consumed {
                return Err(Fn3Error::InvalidShape {
                    context: "runtime Arena::frame",
                    detail: format!(
                        "internal: slot byte range [{offset}, {}) overlaps the previous one at \
                         {consumed}",
                        offset + bytes
                    ),
                });
            }
            let (_, tail) = rest.split_at_mut(offset - consumed);
            let (window, tail) = tail.split_at_mut(bytes);
            rest = tail;
            consumed = offset + bytes;
            match kind {
                // A shared reborrow of a window that no other window overlaps:
                // `split_at_mut` proved the disjointness.
                Kind::Input(index) => shared[index] = Some(window),
                Kind::Output(index) => exclusive[index] = Some(window),
            }
        }

        // Zero-length outputs: `split_at_mut(0)` yields a valid empty slice with
        // the arena's lifetime, and can be repeated at the same cursor.
        for (index, &id) in outputs.iter().enumerate() {
            if exclusive[index].is_none() {
                let slot = plan.slot(id)?;
                if slot.bytes != 0 {
                    return Err(Fn3Error::InvalidShape {
                        context: "runtime Arena::frame",
                        detail: format!("internal: output {id} was not placed by the split"),
                    });
                }
                let (empty, tail) = rest.split_at_mut(0);
                rest = tail;
                exclusive[index] = Some(empty);
            }
        }
        for (index, &id) in inputs.iter().enumerate() {
            if shared[index].is_none() {
                let slot = plan.slot(id)?;
                if slot.source == SlotSource::Arena && slot.bytes != 0 {
                    return Err(Fn3Error::InvalidShape {
                        context: "runtime Arena::frame",
                        detail: format!("internal: input {id} was not placed by the split"),
                    });
                }
                if slot.bytes == 0 {
                    let (empty, tail) = rest.split_at_mut(0);
                    rest = tail;
                    shared[index] = Some(empty);
                }
            }
        }
        Ok((
            shared,
            exclusive
                .into_iter()
                .map(|window| window.expect("every output window was placed"))
                .collect(),
        ))
    }
}

fn require_f32(span: SlotSpan, id: SlotId) -> Result<(), Fn3Error> {
    if span.dtype == DType::F32 {
        Ok(())
    } else {
        Err(Fn3Error::InvalidShape {
            context: "runtime arena typed slot",
            detail: format!(
                "{id} holds {}; the f32 accessor does not reinterpret it",
                span.dtype
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::backend::OpParams;
    use crate::inference::ir::{DimExpr, Graph, SourceLoc, TensorType};
    use crate::inference::model::WeightSet;
    use crate::inference::ops::OpId;

    fn ty(extents: &[u64]) -> TensorType {
        TensorType::new(DType::F32, extents.iter().copied().map(DimExpr::constant))
    }

    /// `a[4] -> relu -> b[4]`, one segment, two slots.
    fn plan() -> Plan {
        let mut graph = Graph::new();
        let a = graph.add_value(Some("a"), ty(&[4])).unwrap();
        let b = graph.add_value(Some("b"), ty(&[4])).unwrap();
        graph
            .add_node(
                OpId::Relu,
                vec![a],
                vec![b],
                OpParams::new(),
                SourceLoc::new(0, "Relu"),
            )
            .unwrap();
        Plan::from_graph(&graph, &WeightSet::default()).unwrap()
    }

    #[test]
    fn the_arena_is_one_zeroed_allocation_of_the_planned_size() {
        let plan = plan();
        let arena = Arena::new(&plan).unwrap();
        assert_eq!(arena.bytes(), plan.arena_bytes);
        assert_eq!(arena.allocation_count(), 1);
        assert_eq!(arena.bytes() % 64, 0);
        assert!(arena
            .slot_bytes(SlotId::new(0))
            .unwrap()
            .iter()
            .all(|&byte| byte == 0));
    }

    #[test]
    fn a_frame_splits_inputs_and_outputs_into_disjoint_windows() {
        let plan = plan();
        let mut arena = Arena::new(&plan).unwrap();
        arena
            .write_slot(SlotId::new(0), &[1.0f32, -2.0, 3.0, -4.0])
            .unwrap();
        let (inputs, mut outputs) = arena
            .frame(&plan, &[SlotId::new(0)], &[SlotId::new(1)])
            .unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);
        assert_eq!(inputs[0].unwrap().len(), 16);
        assert_eq!(outputs[0].len(), 16);
        // The two windows are different bytes: writing the output cannot change
        // the input.
        outputs[0].fill(0xff);
        assert_eq!(arena.slot(SlotId::new(0)).unwrap(), &[1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn a_mismatched_input_length_is_rejected_before_writing() {
        let plan = plan();
        let mut arena = Arena::new(&plan).unwrap();
        let error = arena
            .write_slot(SlotId::new(0), &[1.0f32, 2.0])
            .unwrap_err();
        assert!(error.to_string().contains("16 bytes"), "{error}");
        assert_eq!(arena.slot(SlotId::new(0)).unwrap(), &[0.0; 4]);
    }

    #[test]
    fn an_empty_arena_allocates_nothing_and_can_still_be_framed() {
        // `Shape::from_dims([0])` is legal; its slot is empty.
        let mut graph = Graph::new();
        let a = graph.add_value(Some("a"), ty(&[0])).unwrap();
        let b = graph.add_value(Some("b"), ty(&[0])).unwrap();
        graph
            .add_node(
                OpId::Relu,
                vec![a],
                vec![b],
                OpParams::new(),
                SourceLoc::new(0, "Relu"),
            )
            .unwrap();
        let plan = Plan::from_graph(&graph, &WeightSet::default()).unwrap();
        let mut arena = Arena::new(&plan).unwrap();
        assert_eq!(arena.allocation_count(), 0);
        let (inputs, outputs) = arena
            .frame(
                &plan,
                &[SlotId::new(a.index() as u32)],
                &[SlotId::new(b.index() as u32)],
            )
            .unwrap();
        assert!(inputs[0].unwrap().is_empty());
        assert!(outputs[0].is_empty());
    }
}
