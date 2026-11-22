//! The v3 session: one plan, one arena, one worker pool, one forward at a time.
//!
//! # What a session owns
//!
//! ADR-0001 §5: *a session owns its threads, arena and KV state.* Concretely:
//!
//! | Field | Lifetime | Shared with |
//! |---|---|---|
//! | [`Plan`] | the session | nothing: a plan is plain data and can be cloned |
//! | [`Arena`] | the session | nothing: one allocation per session |
//! | [`SessionThreads`] | the session | nothing: its own `fn3-worker` pool |
//! | weights | the session | read-only, via `Arc<ImportedGraph>`; two sessions on one model share the payload bytes |
//!
//! Two sessions opened on the same `Arc<ImportedGraph>` therefore have isolated
//! activations and isolated pools and share only immutable weight bytes, which is
//! exactly the composability ADR-0001 asks for.
//!
//! # A forward is one walk over the plan
//!
//! [`Session::execute`] does five things and nothing else:
//!
//! 1. validates each named input against its slot (dtype `f32`, exact element
//!    count) and copies it into the arena;
//! 2. for each segment, in plan order, asks the arena for one *disjoint* window
//!    per operand and per result (`Arena::frame` — a `split_at_mut` walk, so
//!    the aliasing rule is the borrow checker's, not a comment);
//! 3. binds each window to the slot's precomputed [`TensorMeta`] and calls the
//!    one dispatch table (`crate::inference::backend::cpu::dispatch_kernels`
//!    for the forty legacy ops, [`crate::inference::runtime::kernels`] for the
//!    twelve CNN ops);
//! 4. reads the graph outputs back into owned [`HostTensor`]s on request.
//!
//! No step allocates from the arena: the plan's offsets are fixed, so
//! [`Arena::allocation_count`] never moves after [`Session::new`]. The forward
//! does build small `Vec`s (one per segment's operand list, and a cloned
//! [`TensorMeta`] per operand), which is why the guarantee is stated as "the
//! arena does not grow" rather than "the forward does not allocate".
//!
//! # Weights
//!
//! A weight slot ([`SlotSource::Weight`]) is never copied into the arena. The
//! segment loop binds the importer's payload bytes in place. A payload whose
//! address is not aligned for its dtype — possible in principle for a `Vec<u8>`
//! — is copied once, at [`Session::new`], into an [`AlignedBuffer`] the session
//! then owns; that path is normalisation, not a per-forward cost.
//!
//! # Determinism
//!
//! The plan fixes the segment order and every kernel fixes its arithmetic order,
//! so the same inputs produce bit-identical outputs whatever
//! [`Threading`] the session was opened
//! with. `tests/fn3_cnn_run.rs` asserts that for one, four and eight workers.

use std::sync::Arc;

use crate::inference::backend::cpu::dispatch_kernels;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::import::ImportedGraph;
use crate::inference::ops::OpId;
use crate::inference::runtime::arena::Arena;
use crate::inference::runtime::kernels;
use crate::inference::runtime::plan::{Plan, SlotId, SlotSource};
use crate::inference::runtime::threads::{SessionThreads, Threading};
use crate::inference::storage::{AlignedBuffer, DTypeAlignment};
use crate::inference::tensor::{TensorMeta, TensorMut, TensorView};

/// One graph output, materialised on the host.
///
/// A model output is not always `f32` (`TopK` writes `i64` indices, `Shape`
/// writes `i64` extents), so the read-back is typed instead of pretending every
/// result is a float vector.
#[derive(Debug, Clone, PartialEq)]
pub enum HostTensor {
    /// `f32` elements.
    F32(Vec<f32>),
    /// `i64` elements.
    I64(Vec<i64>),
    /// `i32` elements.
    I32(Vec<i32>),
    /// `bool` masks.
    Bool(Vec<bool>),
}

impl HostTensor {
    /// The element type, matching the slot's [`DType`].
    pub fn dtype(&self) -> DType {
        match self {
            HostTensor::F32(_) => DType::F32,
            HostTensor::I64(_) => DType::I64,
            HostTensor::I32(_) => DType::I32,
            HostTensor::Bool(_) => DType::Bool,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            HostTensor::F32(values) => values.len(),
            HostTensor::I64(values) => values.len(),
            HostTensor::I32(values) => values.len(),
            HostTensor::Bool(values) => values.len(),
        }
    }

    /// `true` for an empty tensor.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `f32` elements, when this is an `f32` tensor.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            HostTensor::F32(values) => Some(values),
            _ => None,
        }
    }

    /// The `i64` elements, when this is an `i64` tensor.
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        match self {
            HostTensor::I64(values) => Some(values),
            _ => None,
        }
    }
}

/// A session: an immutable plan over one mutable arena.
///
/// `Send` but deliberately not `Sync`: a forward borrows the arena mutably, so
/// sharing one session between threads would be a data race. Open one session
/// per thread instead; they can share the same [`ImportedGraph`] through an
/// [`Arc`].
#[derive(Debug)]
pub struct Session {
    plan: Plan,
    arena: Arena,
    threads: SessionThreads,
    source: Arc<ImportedGraph>,
    /// One contiguous, offset-zero metadata per slot, built once.
    slot_meta: Vec<TensorMeta>,
    /// Per slot: an aligned copy of a weight payload, only when the importer's
    /// bytes were not aligned for their dtype.
    aligned_weights: Vec<Option<AlignedBuffer>>,
}

impl Session {
    /// Open a session on an imported model, building its plan.
    ///
    /// Takes `impl Into<Arc<ImportedGraph>>`, so both `session` ownership and
    /// `Arc::clone` sharing are one call:
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use fastnn::inference::import::ImportedGraph;
    /// # use fastnn::inference::runtime::{Session, Threading};
    /// # fn open(imported: &Arc<ImportedGraph>) -> fastnn::inference::Fn3Result<()> {
    /// let one = Session::new(Arc::clone(imported), Threading::Fixed(1))?;
    /// let many = Session::new(Arc::clone(imported), Threading::Auto)?;
    /// # let _ = (one, many);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        imported: impl Into<Arc<ImportedGraph>>,
        threading: Threading,
    ) -> Result<Session, Fn3Error> {
        let source = imported.into();
        let plan = Plan::from_imported(&source)?;
        Session::from_plan(plan, source, threading)
    }

    /// Open a session on an already-built plan.
    ///
    /// `plan` must have been built from `source`; a mismatch is not detected
    /// here, but the first forward would name a slot the model does not have.
    pub fn from_plan(
        plan: Plan,
        imported: impl Into<Arc<ImportedGraph>>,
        threading: Threading,
    ) -> Result<Session, Fn3Error> {
        let source = imported.into();
        let threads = SessionThreads::new(threading)?;
        let mut slot_meta = Vec::with_capacity(plan.slots.len());
        let mut aligned_weights = Vec::with_capacity(plan.slots.len());
        for slot in &plan.slots {
            let shape = slot
                .ty
                .static_shape()
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "runtime Session::new",
                    detail: format!(
                        "slot {} has the symbolic type {}; a session needs every extent",
                        slot.value, slot.ty
                    ),
                })?;
            slot_meta.push(TensorMeta::contiguous(
                shape,
                slot.ty.dtype,
                crate::inference::device::DeviceId::Cpu,
            )?);
            aligned_weights.push(match slot.source {
                SlotSource::Arena => None,
                SlotSource::Weight(id) => {
                    let weight = source.weights().get(id).ok_or(Fn3Error::OutOfBounds {
                        what: "runtime Session weight id",
                        index: id.index() as u64,
                        len: source.weights().len() as u64,
                    })?;
                    if weight.dtype() != slot.ty.dtype {
                        return Err(Fn3Error::DTypeMismatch {
                            op: None,
                            expected: slot.ty.dtype,
                            actual: weight.dtype(),
                        });
                    }
                    let bytes = weight.bytes();
                    if (bytes.as_ptr() as usize).is_multiple_of(weight.dtype().alignment_hint()) {
                        None
                    } else {
                        // One copy at open time, never per forward.
                        Some(AlignedBuffer::from_bytes(bytes)?)
                    }
                }
            });
        }
        let arena = Arena::new(&plan)?;
        Ok(Session {
            plan,
            arena,
            threads,
            source,
            slot_meta,
            aligned_weights,
        })
    }

    /// The plan this session executes.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The arena, for its size and allocation counter.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// The number of workers this session owns.
    pub fn worker_count(&self) -> usize {
        self.threads.worker_count()
    }

    /// The pool's own thread count; `None` for one worker.
    pub fn pool_num_threads(&self) -> Option<usize> {
        self.threads.pool_num_threads()
    }

    /// The number of times this session's arena has allocated.
    ///
    /// Constant after [`Session::new`]; the test in `tests/fn3_cnn_run.rs` runs
    /// repeated forwards and asserts it did not move.
    pub fn arena_allocation_count(&self) -> usize {
        self.arena.allocation_count()
    }

    /// Run one forward, writing the graph outputs into the arena.
    ///
    /// `inputs` is `(source name, values)`; every graph input must be supplied
    /// exactly once and each value vector must have exactly the slot's element
    /// count. Values are `f32` — the CNN path's compute dtype — and an input slot
    /// of another dtype is reported rather than converted.
    ///
    /// This is the borrowed-IO form: it returns nothing, and
    /// [`Session::outputs`] / [`Session::read_output`] copy a result out when the
    /// caller wants it. Use it for repeated forwards whose result is consumed in
    /// place; [`Session::run`] is the convenience wrapper that copies every
    /// output.
    pub fn execute(&mut self, inputs: &[(&str, &[f32])]) -> Result<(), Fn3Error> {
        let Session {
            plan,
            arena,
            threads,
            source,
            slot_meta,
            aligned_weights,
        } = self;
        if inputs.len() != plan.inputs.len() {
            return Err(Fn3Error::ShapeMismatch {
                op: None,
                expected: format!("{} named inputs", plan.inputs.len()),
                actual: format!("{} supplied", inputs.len()),
            });
        }
        let mut seen = vec![false; plan.inputs.len()];
        for (name, values) in inputs {
            let id = plan
                .input_slot(name)
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "runtime Session::execute",
                    detail: format!(
                        "the model has no input named {name:?}; inputs are {:?}",
                        plan.inputs
                            .iter()
                            .filter_map(|&id| plan.slots[id.index()].name.as_deref())
                            .collect::<Vec<_>>()
                    ),
                })?;
            let position = plan
                .inputs
                .iter()
                .position(|&candidate| candidate == id)
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "runtime Session::execute",
                    detail: format!("internal: input slot {id} is not in the plan's input list"),
                })?;
            if seen[position] {
                return Err(Fn3Error::InvalidShape {
                    context: "runtime Session::execute",
                    detail: format!("the input {name:?} was supplied twice"),
                });
            }
            seen[position] = true;
            let slot = plan.slot(id)?;
            if slot.ty.dtype != DType::F32 {
                return Err(Fn3Error::UnsupportedDType {
                    op: None,
                    dtype: slot.ty.dtype,
                    context: "runtime Session::execute takes f32 inputs",
                });
            }
            arena.write_slot(id, values)?;
        }

        for segment in &plan.segments {
            let (input_windows, output_windows) =
                arena.frame(plan, &segment.inputs, &segment.outputs)?;
            let mut views = Vec::with_capacity(segment.inputs.len());
            for (position, &id) in segment.inputs.iter().enumerate() {
                let bytes = match input_windows[position] {
                    Some(bytes) => bytes,
                    None => weight_bytes(source, aligned_weights, plan, id)?,
                };
                views.push(TensorView::from_host_bytes(
                    bytes,
                    slot_meta[id.index()].clone(),
                )?);
            }
            let mut results = Vec::with_capacity(segment.outputs.len());
            for (&id, window) in segment.outputs.iter().zip(output_windows) {
                results.push(TensorMut::from_host_bytes(
                    window,
                    slot_meta[id.index()].clone(),
                )?);
            }
            let op = segment.op_id();
            if kernels::is_cnn_op(op) {
                kernels::dispatch(op, &views, &mut results, &segment.params, Some(threads))?;
            } else {
                // The legacy kernels take a thread budget and would otherwise ask
                // Rayon's *global* pool. Running them inside this session's pool
                // with a budget of one keeps them on one `fn3-worker` thread and
                // makes the backend's nested-region guard (`in_worker`) report
                // true, so no library call can reach the global pool.
                threads
                    .install(|| dispatch_kernels(op, &views, &mut results, &segment.params, 1))?;
            }
        }
        Ok(())
    }

    /// Run one forward and copy every graph output out.
    ///
    /// Fails with [`Fn3Error::UnsupportedDType`] for a non-`f32` graph output;
    /// use [`Session::run_typed`] when the model's interface includes indices or
    /// masks.
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Result<Vec<(String, Vec<f32>)>, Fn3Error> {
        self.execute(inputs)?;
        let mut outputs = Vec::with_capacity(self.plan.outputs.len());
        for &id in &self.plan.outputs {
            let slot = self.plan.slot(id)?;
            let name = slot.name.clone().ok_or_else(|| Fn3Error::InvalidShape {
                context: "runtime Session::run",
                detail: format!("output slot {id} has no source name"),
            })?;
            let tensor = self.read_output_slot(id)?;
            match tensor {
                HostTensor::F32(values) => outputs.push((name, values)),
                other => {
                    return Err(Fn3Error::UnsupportedDType {
                        op: None,
                        dtype: other.dtype(),
                        context: "runtime Session::run returns f32 outputs; use run_typed",
                    })
                }
            }
        }
        Ok(outputs)
    }

    /// Run one forward and copy every graph output out, typed.
    pub fn run_typed(
        &mut self,
        inputs: &[(&str, &[f32])],
    ) -> Result<Vec<(String, HostTensor)>, Fn3Error> {
        self.execute(inputs)?;
        self.outputs()
    }

    /// Copy every graph output out of the arena, in the model's declared order.
    pub fn outputs(&self) -> Result<Vec<(String, HostTensor)>, Fn3Error> {
        let mut outputs = Vec::with_capacity(self.plan.outputs.len());
        for &id in &self.plan.outputs {
            let name = self
                .plan
                .slot(id)?
                .name
                .clone()
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "runtime Session::outputs",
                    detail: format!("output slot {id} has no source name"),
                })?;
            outputs.push((name, self.read_output_slot(id)?));
        }
        Ok(outputs)
    }

    /// Copy the named output out of the arena.
    pub fn read_output(&self, name: &str) -> Result<HostTensor, Fn3Error> {
        let id = self
            .plan
            .output_slot(name)
            .ok_or_else(|| Fn3Error::InvalidShape {
                context: "runtime Session::read_output",
                detail: format!("the model has no output named {name:?}"),
            })?;
        self.read_output_slot(id)
    }

    /// Copy one output slot out of the arena.
    pub fn read_output_slot(&self, id: SlotId) -> Result<HostTensor, Fn3Error> {
        let slot = self.plan.slot(id)?;
        let bytes = self.arena.slot_bytes(id)?;
        match slot.ty.dtype {
            DType::F32 => Ok(HostTensor::F32(
                bytemuck::try_cast_slice::<u8, f32>(bytes)
                    .map_err(|error| Fn3Error::StorageCast {
                        element: "f32",
                        byte_len: bytes.len(),
                        detail: format!("output slot {id}: {error:?}"),
                    })?
                    .to_vec(),
            )),
            DType::I64 => Ok(HostTensor::I64(
                bytemuck::try_cast_slice::<u8, i64>(bytes)
                    .map_err(|error| Fn3Error::StorageCast {
                        element: "i64",
                        byte_len: bytes.len(),
                        detail: format!("output slot {id}: {error:?}"),
                    })?
                    .to_vec(),
            )),
            DType::I32 => Ok(HostTensor::I32(
                bytemuck::try_cast_slice::<u8, i32>(bytes)
                    .map_err(|error| Fn3Error::StorageCast {
                        element: "i32",
                        byte_len: bytes.len(),
                        detail: format!("output slot {id}: {error:?}"),
                    })?
                    .to_vec(),
            )),
            DType::Bool => Ok(HostTensor::Bool(
                bytes.iter().map(|&byte| byte != 0).collect(),
            )),
        }
    }
}

/// The bytes of a weight slot: the importer's payload, or its aligned copy.
fn weight_bytes<'a>(
    source: &'a ImportedGraph,
    aligned: &'a [Option<AlignedBuffer>],
    plan: &Plan,
    id: SlotId,
) -> Result<&'a [u8], Fn3Error> {
    let slot = plan.slot(id)?;
    let SlotSource::Weight(weight_id) = slot.source else {
        return Err(Fn3Error::InvalidShape {
            context: "runtime Session::execute",
            detail: format!("internal: {id} has no arena window and is not a weight"),
        });
    };
    if let Some(buffer) = aligned.get(id.index()).and_then(Option::as_ref) {
        return Ok(buffer.as_bytes());
    }
    let weight = source
        .weights()
        .get(weight_id)
        .ok_or(Fn3Error::OutOfBounds {
            what: "runtime Session weight id",
            index: weight_id.index() as u64,
            len: source.weights().len() as u64,
        })?;
    Ok(weight.bytes())
}

/// The op ids the CNN kernel table owns, for a caller that wants to explain a
/// plan without importing the private table.
pub fn is_cnn_op(op: OpId) -> bool {
    kernels::is_cnn_op(op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::backend::OpParams;
    use crate::inference::ir::{DimExpr, Graph, SourceLoc, TensorType, ValueId};
    use crate::inference::model::{WeightSet, WeightSource, WeightTensor};
    use crate::inference::shape::Shape;

    /// A model with two inputs, one `Add`, one weight `Mul` and two segments:
    /// the smallest session that exercises two named inputs, a weight slot and
    /// two dispatches.
    fn imported() -> ImportedGraph {
        let weight = WeightTensor::new(
            "w",
            DType::F32,
            Shape::from_dims([4]),
            bytemuck::cast_slice(&[2.0f32, 2.0, 2.0, 2.0]).to_vec(),
            WeightSource::Initializer,
        )
        .unwrap();
        let weights = WeightSet::new(vec![weight]).unwrap();
        let mut graph = Graph::new();
        let ty = || {
            TensorType::new(
                DType::F32,
                [DimExpr::constant(1), DimExpr::constant(4)].into_iter(),
            )
        };
        let x = graph.add_value(Some("x"), ty()).unwrap();
        let x2 = graph.add_value(Some("x2"), ty()).unwrap();
        let w = graph.add_value(Some("w"), ty()).unwrap();
        let h = graph.add_value(Some("h"), ty()).unwrap();
        let y = graph.add_value(Some("y"), ty()).unwrap();
        graph
            .add_node(
                OpId::Add,
                vec![x, x2],
                vec![h],
                OpParams::new(),
                SourceLoc::new(0, "Add"),
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
        crate::inference::import::ImportedGraph::for_tests(graph, weights, vec![x, x2], vec![y])
            .unwrap()
    }

    fn plan_and_graph() -> (Plan, ImportedGraph) {
        let imported = imported();
        let plan = Plan::from_imported(&imported).unwrap();
        (plan, imported)
    }

    #[test]
    fn a_forward_reads_weights_in_place_and_leaves_the_arena_unchanged() {
        let (plan, imported) = plan_and_graph();
        assert_eq!(plan.slots[2].bytes, 16, "the weight slot payload");
        let mut session = Session::from_plan(plan, imported, Threading::Fixed(1)).unwrap();
        let before = session.arena_allocation_count();
        for _ in 0..3 {
            let outputs = session
                .run(&[
                    ("x", &[-1.0, 2.0, -3.0, 4.0]),
                    ("x2", &[1.0, 2.0, 3.0, 4.0]),
                ])
                .unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].0, "y");
            assert_eq!(outputs[0].1, [0.0, 8.0, 0.0, 16.0]);
        }
        assert_eq!(
            session.arena_allocation_count(),
            before,
            "a forward must not allocate from the arena"
        );
    }

    #[test]
    fn a_missing_duplicate_or_short_input_is_rejected() {
        let mut session = Session::new(imported(), Threading::Fixed(1)).unwrap();
        let error = session.run(&[]).unwrap_err();
        assert!(error.to_string().contains("named inputs"), "{error}");
        let error = session
            .run(&[("nope", &[0.0; 4]), ("x2", &[0.0; 4])])
            .unwrap_err();
        assert!(error.to_string().contains("no input named"), "{error}");
        let error = session
            .run(&[("x", &[0.0; 3]), ("x2", &[0.0; 4])])
            .unwrap_err();
        assert!(error.to_string().contains("16 bytes"), "{error}");
        session
            .run(&[("x", &[0.0; 4]), ("x2", &[0.0; 4])])
            .expect("a valid input still runs after a rejected one");
        let error = session
            .run(&[("x", &[0.0; 4]), ("x", &[0.0; 4])])
            .unwrap_err();
        assert!(error.to_string().contains("supplied twice"), "{error}");
    }

    #[test]
    fn a_session_sends_and_is_not_sync() {
        fn is_send<T: Send>() {}
        is_send::<Session>();
    }

    #[test]
    fn the_declared_output_slot_is_the_one_read_back() {
        let (plan, imported) = plan_and_graph();
        let session = Session::from_plan(plan, imported, Threading::Fixed(1)).unwrap();
        let output = plan_output(&session);
        assert_eq!(output, vec![ValueId::new(4)]);
    }

    fn plan_output(session: &Session) -> Vec<ValueId> {
        session
            .plan()
            .outputs
            .iter()
            .map(|slot| session.plan().slots[slot.index()].value)
            .collect()
    }
}
