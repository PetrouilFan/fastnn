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
//! | the oneDNN engine (`cpu-onednn`) | the session | nothing: one engine, one prep stream, its prepacked weights and blocks |
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
//! 3. dispatches on the segment's engine ([`SegmentEngine::Scalar`] binds each
//!    window to the slot's precomputed [`TensorMeta`] and calls the one
//!    dispatch table — `backend::cpu::dispatch_kernels` for the forty legacy
//!    ops, [`crate::inference::runtime::kernels`] for the twelve CNN ops;
//!    [`SegmentEngine::OneDnn`] hands the segment's arena windows to the
//!    prepared blocks and runs them on this pool);
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
use crate::inference::backend::{ScalarOnly, SegmentEngine, SegmentProvider};
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::import::ImportedGraph;
use crate::inference::model::format::Artifact;
use crate::inference::model::weights::WeightId;
use crate::inference::ops::OpId;
use crate::inference::runtime::arena::Arena;
use crate::inference::runtime::kernels;
use crate::inference::runtime::plan::{
    EngineCounts, Fusion, LayoutStrategy, Plan, PlanOptions, SlotId, SlotSource,
};
use crate::inference::runtime::threads::{SessionThreads, Threading};
use crate::inference::storage::{AlignedBuffer, DTypeAlignment};
use crate::inference::tensor::{TensorMeta, TensorMut, TensorView};

#[cfg(feature = "cpu-onednn")]
use crate::inference::backend::cpu::onednn::geometry::ConvSplitAxis;
#[cfg(feature = "cpu-onednn")]
use crate::inference::backend::cpu::onednn::{OneDnnCapability, OneDnnEngine, SegmentSpec};

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

/// Which provider a new session should plan with.
///
/// The choice is made **once, at plan build time**, and recorded per segment in
/// [`Plan::segments`]. It is not re-decided per forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineChoice {
    /// The best engine this build has: the optional oneDNN CPU engine for the
    /// segments it claims (`cpu-onednn`), scalar for everything else. Without
    /// the feature this is exactly [`EngineChoice::Scalar`].
    #[default]
    Auto,
    /// The portable scalar kernels for every segment.
    ///
    /// This is the reference a oneDNN session is compared against, and the only
    /// choice in a build without `cpu-onednn`.
    Scalar,
}

/// What the oneDNN engine covers in one session.
///
/// A report, not a control surface: it says how many segments the engine
/// received, how much weight data it prepacked, and whether its blocks run in
/// parallel. Empty (`None` from [`Session::one_dnn_report`]) when the session
/// has no oneDNN segment or the feature is off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OneDnnReport {
    /// Linked oneDNN version, `"3.14.1"`.
    pub version: String,
    /// CPU threading runtime the library was built with (`"seq"` expected).
    pub cpu_runtime: &'static str,
    /// Segments prepared on the engine.
    pub segments: usize,
    /// Convolution segments prepared.
    pub conv_segments: usize,
    /// Convolution segments divided along output channels.
    pub conv_channel_segments: usize,
    /// Convolution segments divided along output rows with a halo.
    pub conv_row_segments: usize,
    /// Pooling segments prepared.
    pub pool_segments: usize,
    /// Matmul segments prepared.
    pub matmul_segments: usize,
    /// Segment prepared with a fused epilogue (a `sum` and/or an activation).
    pub fused_segments: usize,
    /// Segments whose result stays in the engine's channels-last layout.
    pub blocked_segments: usize,
    /// Bytes of engine-owned channels-last buffers.
    pub layout_bytes: usize,
    /// Layout conversions run inside segments, per forward.
    pub reorders: usize,
    /// Bytes those conversions move per forward.
    pub reorder_bytes: usize,
    /// Engine-owned prepacked weight and bias bytes.
    pub prepacked_bytes: usize,
    /// Weight blocks oneDNN wanted in a layout other than the ONNX payload's,
    /// i.e. blocks that needed a real reorder at session open.
    pub reordered_blocks: usize,
    /// Bytes of dense source/destination staging owned by row-split blocks.
    pub staging_bytes: usize,
    /// `true` when blocks are executed on the session's pool; `false` when a
    /// non-sequential oneDNN build forced serial block execution.
    pub parallel_blocks: bool,
}

/// Which dimension a convolution segment's blocks divide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvBlockAxis {
    /// Contiguous output-channel ranges.
    Channels,
    /// Contiguous output-row ranges with an input halo.
    Rows,
}

/// One convolution's block plan, recorded at session open.
///
/// A forward never re-decides this. The split is a pure function of the
/// segment's shape and attributes and the session's worker count, and it is
/// stored in the prepared segment, so this report is what the engine will
/// execute rather than a second calculation that could disagree with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvSplitReport {
    /// Position of the convolution in [`Plan::segments`].
    pub segment: usize,
    /// The dimension the blocks divide.
    pub axis: ConvBlockAxis,
    /// Number of blocks.
    pub blocks: usize,
    /// `(start, count)` per block, in the split dimension: output channels for
    /// [`ConvBlockAxis::Channels`], output rows for [`ConvBlockAxis::Rows`].
    pub ranges: Vec<(usize, usize)>,
    /// Input rows read across all blocks: the halo traffic proxy.
    pub source_rows: usize,
    /// The split dimension's total extent in the parent output.
    pub out_extent: usize,
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
    threads: SessionThreads,
    /// The optional oneDNN engine, present only when the plan has a segment it
    /// prepared. Declared before the arena and the weights so it is dropped
    /// first: its memory objects borrow those buffers for the session's life.
    #[cfg(feature = "cpu-onednn")]
    onednn: Option<OneDnnEngine>,
    arena: Arena,
    source: WeightStore,
    /// One contiguous, offset-zero metadata per slot, built once.
    slot_meta: Vec<TensorMeta>,
    /// Per slot: an aligned copy of a weight payload, only when the importer's
    /// bytes were not aligned for their dtype.
    aligned_weights: Vec<Option<AlignedBuffer>>,
    /// Per-segment wall times, only in a `segment-timing` build.
    ///
    /// Plain per-session data, updated after a segment returns by the thread
    /// that called [`Session::execute`]; see `runtime::timing`.
    #[cfg(feature = "segment-timing")]
    timings: crate::inference::runtime::timing::SegmentTimings,
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
        Session::new_with(imported, threading, EngineChoice::Auto)
    }

    /// Open a session with an explicit engine choice.
    ///
    /// [`EngineChoice::Auto`] is what [`Session::new`] uses. The choice is
    /// applied when the plan is built, so it decides which segments a provider
    /// may claim; it cannot change afterwards.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use fastnn::inference::import::ImportedGraph;
    /// # use fastnn::inference::runtime::{EngineChoice, Session, Threading};
    /// # fn open(imported: &Arc<ImportedGraph>) -> fastnn::inference::Fn3Result<()> {
    /// let reference = Session::new_with(
    ///     Arc::clone(imported),
    ///     Threading::Fixed(1),
    ///     EngineChoice::Scalar,
    /// )?;
    /// # let _ = reference;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with(
        imported: impl Into<Arc<ImportedGraph>>,
        threading: Threading,
        choice: EngineChoice,
    ) -> Result<Session, Fn3Error> {
        Session::new_with_options(imported, threading, choice, PlanOptions::new())
    }

    /// Open a session with an explicit engine choice and plan options.
    ///
    /// [`PlanOptions::unfused`] builds one segment per graph node, which is the
    /// oracle the fusion test compares a fused session against: same graph,
    /// same provider, same tolerance.
    pub fn new_with_options(
        imported: impl Into<Arc<ImportedGraph>>,
        threading: Threading,
        choice: EngineChoice,
        options: PlanOptions,
    ) -> Result<Session, Fn3Error> {
        let source: Arc<ImportedGraph> = imported.into();
        let plan = Plan::from_imported_with_options(&source, provider_for(choice), options)?;
        Session::from_plan_and_store(plan, WeightStore::Imported(source), threading)
    }

    /// Open a session on an already-built plan.
    ///
    /// `plan` must have been built from `source`; a mismatch is not detected
    /// here, but the first forward would name a slot the model does not have.
    /// A plan whose segments name [`SegmentEngine::OneDnn`] is rejected by a
    /// build without the `cpu-onednn` feature instead of being run on the
    /// scalar kernels behind the caller's back.
    pub fn from_plan(
        plan: Plan,
        imported: impl Into<Arc<ImportedGraph>>,
        threading: Threading,
    ) -> Result<Session, Fn3Error> {
        Session::from_plan_and_store(plan, WeightStore::Imported(imported.into()), threading)
    }

    /// Open a session on a mapped artifact and its stored plan.
    ///
    /// The plan is the one `prepare` serialized for the artifact's default
    /// target; it is decoded, not rebuilt. Weight slots point into the mapping
    /// and are never copied.
    pub fn from_artifact(artifact: &Artifact, threading: Threading) -> Result<Session, Fn3Error> {
        Session::from_shared_artifact(Arc::new(artifact.clone()), threading)
    }

    /// Open a session on an already shared artifact, without cloning its
    /// metadata. This is the deployment path: `load` maps the file once and
    /// every session shares the one `Arc`.
    pub fn from_shared_artifact(
        artifact: Arc<Artifact>,
        threading: Threading,
    ) -> Result<Session, Fn3Error> {
        let plan = artifact.default_plan().clone();
        Session::from_plan_and_store(plan, WeightStore::Artifact(artifact), threading)
    }

    /// Open a session on a named target plan from an artifact.
    ///
    /// The target must be in the file; an unknown name is a typed error rather
    /// than a silent fallback to the default plan.
    pub fn from_artifact_for(
        artifact: &Artifact,
        target: &str,
        threading: Threading,
    ) -> Result<Session, Fn3Error> {
        let plan = artifact
            .plan_for(target)
            .ok_or_else(|| Fn3Error::InvalidShape {
                context: "runtime Session::from_artifact_for",
                detail: format!(
                    "the artifact has no target `{target}`; targets are {}",
                    artifact
                        .plans()
                        .iter()
                        .map(|plan| plan.target())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            })?;
        Session::from_plan_and_store(
            plan.clone(),
            WeightStore::Artifact(Arc::new(artifact.clone())),
            threading,
        )
    }

    fn from_plan_and_store(
        plan: Plan,
        source: WeightStore,
        threading: Threading,
    ) -> Result<Session, Fn3Error> {
        // Mutable only when the optional engine records its layout decisions on
        // the plan; a build without it stores the plan unchanged.
        #[allow(unused_mut)]
        let mut plan = plan;
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
                    let (dtype, bytes) = source.bytes(id)?;
                    if dtype != slot.ty.dtype {
                        return Err(Fn3Error::DTypeMismatch {
                            op: None,
                            expected: slot.ty.dtype,
                            actual: dtype,
                        });
                    }
                    if (bytes.as_ptr() as usize).is_multiple_of(dtype.alignment_hint()) {
                        None
                    } else {
                        // One copy at open time, never per forward. An artifact
                        // weight is 64-byte aligned, so this is only reachable
                        // for an in-memory import's `Vec<u8>` payload.
                        Some(AlignedBuffer::from_bytes(bytes)?)
                    }
                }
            });
        }
        if plan.engine_counts().one_dnn > 0 {
            #[cfg(not(feature = "cpu-onednn"))]
            return Err(Fn3Error::NotImplemented {
                what: "a plan with a segment assigned to the oneDNN engine, in a build without \
                       the `cpu-onednn` feature",
                owner: "phase 3a (oneDNN CPU engine)",
            });
        }
        #[cfg(feature = "cpu-onednn")]
        let onednn = prepare_onednn(&plan, &source, &aligned_weights, threads.worker_count())?;
        // The engine's layout pass is a plan decision taken here, at session
        // open, because it needs oneDNN's descriptor choice. Record it on the
        // plan so a caller can see what the layout policy cost before running
        // a single forward.
        #[cfg(feature = "cpu-onednn")]
        if let Some(engine) = onednn.as_ref() {
            plan.reorders = engine
                .reorders()
                .iter()
                .map(|reorder| crate::inference::runtime::plan::Reorder {
                    segment: reorder.segment,
                    slot: SlotId::new(reorder.slot as u32),
                    direction: if reorder.to_blocked {
                        crate::inference::runtime::plan::ReorderDirection::NchwToBlocked
                    } else {
                        crate::inference::runtime::plan::ReorderDirection::BlockedToNchw
                    },
                    bytes: reorder.bytes,
                })
                .collect();
            plan.reorder_bytes = engine.reorder_bytes();
        }
        let arena = Arena::new(&plan)?;
        Ok(Session {
            plan,
            threads,
            #[cfg(feature = "cpu-onednn")]
            onednn,
            arena,
            source,
            slot_meta,
            aligned_weights,
            #[cfg(feature = "segment-timing")]
            timings: crate::inference::runtime::timing::SegmentTimings::default(),
        })
    }

    /// The plan this session executes.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The artifact this session was opened from, when it was opened from one.
    ///
    /// The `Arc<Artifact>` is what keeps the mmap alive for the weight slices
    /// below.
    pub fn artifact(&self) -> Option<&Artifact> {
        match &self.source {
            WeightStore::Artifact(artifact) => Some(artifact),
            WeightStore::Imported(_) => None,
        }
    }

    /// The bytes of a weight slot, borrowed from the model.
    ///
    /// For an artifact session this is a slice of the mapping: no copy, and the
    /// same address for every session opened on the same artifact.
    pub fn weight_slot_bytes(&self, id: SlotId) -> Result<&[u8], Fn3Error> {
        weight_bytes(&self.source, &self.aligned_weights, &self.plan, id)
    }

    /// The `f32` elements of a weight slot, borrowed from the model.
    pub fn weight_slot_f32(&self, id: SlotId) -> Result<&[f32], Fn3Error> {
        let bytes = self.weight_slot_bytes(id)?;
        bytemuck::try_cast_slice(bytes).map_err(|error| Fn3Error::StorageCast {
            element: "f32",
            byte_len: bytes.len(),
            detail: format!("weight slot {id}: {error:?}"),
        })
    }

    /// The number of weights the model carries.
    pub fn weight_count(&self) -> usize {
        self.source.len()
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
            threads,
            #[cfg(feature = "cpu-onednn")]
            onednn,
            arena,
            source,
            slot_meta,
            aligned_weights,
            #[cfg(feature = "segment-timing")]
            timings,
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

        for (segment_index, segment) in plan.segments.iter().enumerate() {
            // One clock pair per segment, compiled out entirely without the
            // `segment-timing` feature. The record happens on this thread after
            // the segment (and any pool work it started) has finished, so the
            // table is never shared mutably.
            #[cfg(feature = "segment-timing")]
            let started = std::time::Instant::now();
            match segment.engine {
                SegmentEngine::Scalar => {
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
                    match &segment.op {
                        crate::inference::runtime::plan::SegmentOp::Fused(fused) => {
                            kernels::fused::run(
                                fused,
                                &views,
                                &mut results,
                                &segment.params,
                                Some(threads),
                            )?;
                        }
                        crate::inference::runtime::plan::SegmentOp::Op(op) => {
                            let op = *op;
                            if kernels::is_cnn_op(op) {
                                kernels::dispatch(
                                    op,
                                    &views,
                                    &mut results,
                                    &segment.params,
                                    Some(threads),
                                )?;
                            } else {
                                // The legacy kernels take a thread budget of one here, so they
                                // run serially on this thread and cannot submit a parallel
                                // region (`par_slices_mut`/`par_map` refuse a budget of 1
                                // whatever `in_worker` says). The executor previously wrapped
                                // this in `threads.install`, which made `in_worker` true by
                                // parking the calling thread on a pool worker; at eight
                                // workers that cost tens of microseconds per tail segment
                                // (measured: the same serial ops were 1.5 ms slower at eight
                                // workers than at one). No legacy kernel reaches an external
                                // library: the GEMM is in-repo and takes the same budget.
                                dispatch_kernels(op, &views, &mut results, &segment.params, 1)?;
                            }
                        }
                    }
                }
                SegmentEngine::OneDnn => {
                    #[cfg(feature = "cpu-onednn")]
                    {
                        let engine = onednn.as_ref().ok_or_else(|| {
                            engine_unavailable(segment_index, "the session has no oneDNN engine")
                        })?;
                        let prepared = engine.prepared(segment_index).ok_or_else(|| {
                            engine_unavailable(
                                segment_index,
                                "the session prepared no blocks for this segment",
                            )
                        })?;
                        // The arena windows are the live source/destination
                        // pointers. A weight operand has no window (its bytes
                        // are the block's prepacked copy); its address is zero
                        // and the block never reads it from the window list.
                        let (input_windows, mut output_windows) =
                            arena.frame(plan, &segment.inputs, &segment.outputs)?;
                        // A fused `sum` is oneDNN's sum post-op, which adds the
                        // destination's previous contents: the residual operand
                        // is copied into the result slot first. Both slots are
                        // contiguous NCHW with the same byte length, so this is
                        // one memcpy and it happens before any block writes.
                        if let Some(position) = segment.fused().and_then(|fused| fused.sum) {
                            let residual =
                                input_windows.get(position).copied().flatten().ok_or_else(
                                    || {
                                        engine_unavailable(
                                            segment_index,
                                            "the fused residual operand has no arena window",
                                        )
                                    },
                                )?;
                            let destination: &mut [u8] = match output_windows.first_mut() {
                                Some(window) => window,
                                None => {
                                    return Err(engine_unavailable(
                                        segment_index,
                                        "the fused result has no arena window",
                                    ))
                                }
                            };
                            if residual.len() != destination.len() {
                                return Err(engine_unavailable(
                                    segment_index,
                                    "the fused residual and result differ in byte length",
                                ));
                            }
                            destination.copy_from_slice(residual);
                        }
                        let mut addresses = Vec::with_capacity(input_windows.len());
                        for window in &input_windows {
                            addresses.push(match window {
                                Some(bytes) => bytes.as_ptr() as usize,
                                None => 0,
                            });
                        }
                        let mut results = Vec::with_capacity(output_windows.len());
                        for window in &mut output_windows {
                            results.push(window.as_mut_ptr() as usize);
                        }
                        prepared.run_prelude(&addresses)?;
                        if engine.blocks_run_in_parallel() {
                            threads.run_blocks(prepared.block_count(), |block| {
                                prepared.run_block(block, &addresses, &results)
                            })?;
                        } else {
                            // A non-sequential oneDNN build owns the threads
                            // inside a call; nesting the pool would
                            // oversubscribe. Same blocks, same results, serial.
                            for block in 0..prepared.block_count() {
                                prepared.run_block(block, &addresses, &results)?;
                            }
                        }
                    }
                    #[cfg(not(feature = "cpu-onednn"))]
                    {
                        return Err(engine_unavailable(
                            segment_index,
                            "this build has no `cpu-onednn` feature",
                        ));
                    }
                }
            }
            #[cfg(feature = "segment-timing")]
            {
                // The label names the whole segment: a fused `Conv+Swish`
                // must not be reported as plain `Conv`, or the epilogue looks
                // free.
                let label = segment.op.to_string();
                timings.record(segment.op_id(), &label, started.elapsed());
            }
        }
        Ok(())
    }

    /// How many segments each engine runs in this session's plan.
    pub fn engine_counts(&self) -> EngineCounts {
        self.plan.engine_counts()
    }

    /// Every chain the planner folded, in segment order.
    ///
    /// The same report is reachable as [`Plan::fusions`]; this accessor exists
    /// so a caller holding only a session does not have to go through
    /// [`Session::plan`].
    pub fn fusions(&self) -> &[Fusion] {
        self.plan.fusions()
    }

    /// The layout policy this session's plan asked for.
    pub fn layout_strategy(&self) -> LayoutStrategy {
        self.plan.layout
    }

    /// The per-op wall times recorded since the last reset.
    ///
    /// Present only in a `segment-timing` build; a normal build has no timing
    /// field and no clock read. The table is per session and is updated by the
    /// thread running [`Session::execute`], so one session's numbers cannot be
    /// observed or changed by another session or by a pool worker.
    #[cfg(feature = "segment-timing")]
    pub fn segment_timings(&self) -> &crate::inference::runtime::timing::SegmentTimings {
        &self.timings
    }

    /// Clear the per-op timing table, keeping its allocation.
    ///
    /// Opening a session starts with an empty table; a caller that wants warm
    /// timings only resets after its warm-up forwards.
    #[cfg(feature = "segment-timing")]
    pub fn reset_segment_timings(&mut self) {
        self.timings.reset();
    }

    /// What the optional oneDNN engine covers, or `None`.
    ///
    /// `None` means the session has no oneDNN segment: either this build has no
    /// `cpu-onednn` feature, the plan chose `EngineChoice::Scalar`, or no
    /// segment was claimed (for example a graph of elementwise ops only).
    pub fn one_dnn_report(&self) -> Option<OneDnnReport> {
        #[cfg(feature = "cpu-onednn")]
        {
            let engine = self.onednn.as_ref()?;
            let (conv, pool, matmul) = engine.segment_kinds();
            let splits = self.one_dnn_splits();
            let conv_channel_segments = splits
                .iter()
                .filter(|split| split.axis == ConvBlockAxis::Channels)
                .count();
            Some(OneDnnReport {
                version: engine.version().number(),
                cpu_runtime: engine.version().cpu_runtime_name(),
                segments: self.plan.engine_counts().one_dnn,
                conv_segments: conv,
                conv_channel_segments,
                conv_row_segments: splits.len() - conv_channel_segments,
                pool_segments: pool,
                matmul_segments: matmul,
                fused_segments: engine.fused_segments(),
                blocked_segments: engine.blocked_segments(),
                layout_bytes: engine.layout_bytes(),
                reorders: engine.reorders().len(),
                reorder_bytes: engine.reorder_bytes(),
                prepacked_bytes: engine.prepacked_bytes(),
                reordered_blocks: engine.reordered_blocks(),
                staging_bytes: engine.staging_bytes(),
                parallel_blocks: engine.blocks_run_in_parallel(),
            })
        }
        #[cfg(not(feature = "cpu-onednn"))]
        {
            None
        }
    }

    /// The block plan of every prepared convolution, in plan order.
    ///
    /// Empty without the `cpu-onednn` feature or when the plan assigned no
    /// convolution to the engine. This is the recorded decision, not a
    /// recomputation: see [`ConvSplitReport`].
    pub fn one_dnn_splits(&self) -> Vec<ConvSplitReport> {
        #[cfg(feature = "cpu-onednn")]
        {
            let Some(engine) = self.onednn.as_ref() else {
                return Vec::new();
            };
            let mut reports = Vec::new();
            for index in 0..self.plan.segments.len() {
                let Some(split) = engine.prepared_split(index) else {
                    continue;
                };
                reports.push(ConvSplitReport {
                    segment: index,
                    axis: match split.axis {
                        ConvSplitAxis::Channels => ConvBlockAxis::Channels,
                        ConvSplitAxis::Rows => ConvBlockAxis::Rows,
                    },
                    blocks: split.ranges.len(),
                    ranges: split.ranges.clone(),
                    source_rows: split.source_rows,
                    out_extent: split.out_extent,
                });
            }
            reports
        }
        #[cfg(not(feature = "cpu-onednn"))]
        {
            Vec::new()
        }
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

/// The provider a new plan is built with.
///
/// The feature test lives here, at the session boundary, not in the planner: a
/// plan built in either build is the same type with the same rules, and the
/// segment engines are decided by the provider's answers.
fn provider_for(choice: EngineChoice) -> &'static dyn SegmentProvider {
    match choice {
        EngineChoice::Scalar => &ScalarOnly,
        EngineChoice::Auto => {
            #[cfg(feature = "cpu-onednn")]
            {
                &OneDnnCapability
            }
            #[cfg(not(feature = "cpu-onednn"))]
            {
                &ScalarOnly
            }
        }
    }
}

fn engine_unavailable(segment: usize, detail: &str) -> Fn3Error {
    Fn3Error::ProviderFailure {
        provider: "oneDNN",
        call: "runtime Session::execute",
        status: 0,
        detail: format!("plan segment {segment} is assigned to the oneDNN engine, but {detail}"),
    }
}

/// Prepare every oneDNN segment at session open.
///
/// All prepacking (weight layout, bias copy, scratchpads, primitives) happens
/// here, once. A forward only binds pointers and calls `execute`.
#[cfg(feature = "cpu-onednn")]
fn prepare_onednn(
    plan: &Plan,
    source: &WeightStore,
    aligned: &[Option<AlignedBuffer>],
    workers: usize,
) -> Result<Option<OneDnnEngine>, Fn3Error> {
    if plan.engine_counts().one_dnn == 0 {
        return Ok(None);
    }
    let mut engine = OneDnnEngine::new(plan.segments.len())?;
    for (index, segment) in plan.segments.iter().enumerate() {
        if segment.engine != SegmentEngine::OneDnn {
            continue;
        }
        let blocked = blocked_destination(plan, index);
        let spec = segment_spec(plan, source, aligned, segment, workers, blocked)?;
        engine
            .prepare_segment(index, &spec)
            .map_err(|error| match error {
                Fn3Error::ProviderFailure {
                    provider,
                    call,
                    status,
                    detail,
                } => Fn3Error::ProviderFailure {
                    provider,
                    call,
                    status,
                    detail: format!("segment {index} ({}): {detail}", segment.loc.describe()),
                },
                other => other,
            })?;
    }
    // One shared staging pool per session, sized after every segment has
    // declared its row blocks (#115).
    engine.finish_staging(workers)?;
    Ok(Some(engine))
}

/// Whether the engine may keep this segment's result in its channels-last
/// layout.
///
/// The rule is structural and decided here, where the whole plan is visible:
///
/// - the plan asked for [`LayoutStrategy::Chained`];
/// - the segment is a oneDNN convolution with a single output;
/// - the output is not a graph output, so the session's contract stays NCHW;
/// - **every** consumer is a oneDNN convolution that reads this value as its
///   source (a pool, whose channel blocks take strided source submemories, or a
///   scalar kernel would need the NCHW bytes);
/// - the segment has no fused residual: the `sum` post-op reads the destination
///   buffer back, which a channels-last destination would not hold in NCHW;
/// - the consumer is not grouped: a group's channel slice is strided in a
///   channels-last tensor.
///
/// The engine still decides whether the block split can produce it (a row split
/// with at least two blocks); when it cannot, it prepares the plain layout and
/// records no reorder.
#[cfg(feature = "cpu-onednn")]
fn blocked_destination(plan: &Plan, index: usize) -> bool {
    if plan.layout != LayoutStrategy::Chained {
        return false;
    }
    let segment = &plan.segments[index];
    if segment.engine != SegmentEngine::OneDnn
        || segment.op_id() != OpId::Conv
        || segment.outputs.len() != 1
        || segment.fused().is_some_and(|fused| fused.has_sum())
    {
        return false;
    }
    let produced = segment.outputs[0];
    if plan.outputs.contains(&produced) {
        return false;
    }
    let mut consumed = false;
    for consumer in plan.segments.iter() {
        if !consumer.inputs.contains(&produced) {
            continue;
        }
        consumed = true;
        if consumer.engine != SegmentEngine::OneDnn
            || consumer.op_id() != OpId::Conv
            || consumer.params.group > 1
            || consumer.inputs.first() != Some(&produced)
        {
            return false;
        }
    }
    consumed
}

/// Describe one segment to the engine in the terms it can use.
#[cfg(feature = "cpu-onednn")]
fn segment_spec<'a>(
    plan: &'a Plan,
    source: &'a WeightStore,
    aligned: &'a [Option<AlignedBuffer>],
    segment: &'a crate::inference::runtime::plan::Segment,
    workers: usize,
    blocked_destination: bool,
) -> Result<SegmentSpec<'a>, Fn3Error> {
    let mut inputs = Vec::with_capacity(segment.inputs.len());
    let mut input_dtypes = Vec::with_capacity(segment.inputs.len());
    let mut input_constants = Vec::with_capacity(segment.inputs.len());
    for &id in &segment.inputs {
        let slot = plan.slot(id)?;
        inputs.push(
            slot.ty
                .static_shape()
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "runtime Session oneDNN preparation",
                    detail: format!("input slot {id} has the symbolic type {}", slot.ty),
                })?,
        );
        input_dtypes.push(slot.ty.dtype);
        input_constants.push(slot.is_weight());
    }
    let mut outputs = Vec::with_capacity(segment.outputs.len());
    let mut output_dtypes = Vec::with_capacity(segment.outputs.len());
    for &id in &segment.outputs {
        let slot = plan.slot(id)?;
        outputs.push(
            slot.ty
                .static_shape()
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "runtime Session oneDNN preparation",
                    detail: format!("output slot {id} has the symbolic type {}", slot.ty),
                })?,
        );
        output_dtypes.push(slot.ty.dtype);
    }
    let operand_bytes = |position: usize| -> Result<Option<&'a [u8]>, Fn3Error> {
        match segment.inputs.get(position) {
            Some(&id) if plan.slot(id)?.is_weight() => {
                Ok(Some(weight_bytes(source, aligned, plan, id)?))
            }
            _ => Ok(None),
        }
    };
    let weights = operand_bytes(1)?;
    let bias = operand_bytes(2)?;
    Ok(SegmentSpec {
        op: segment.op_id(),
        inputs,
        input_dtypes,
        input_constants,
        outputs,
        output_dtypes,
        params: &segment.params,
        weights,
        bias,
        workers,
        fused: segment.fused(),
        input_slots: segment.inputs.iter().map(|id| id.index()).collect(),
        output_slots: segment.outputs.iter().map(|id| id.index()).collect(),
        dst_blocked: blocked_destination,
    })
}

/// The bytes of a weight slot: the importer's payload, or its aligned copy.
fn weight_bytes<'a>(
    source: &'a WeightStore,
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
    Ok(source.bytes(weight_id)?.1)
}

/// Where a session's read-only weight bytes come from.
#[derive(Debug, Clone)]
enum WeightStore {
    /// An importer's owned weight table (the in-memory path).
    Imported(Arc<ImportedGraph>),
    /// A mapped artifact. The `Arc<Artifact>` keeps the mapping alive for every
    /// slice this session hands to a kernel.
    Artifact(Arc<Artifact>),
}

impl WeightStore {
    fn len(&self) -> usize {
        match self {
            WeightStore::Imported(imported) => imported.weights().len(),
            WeightStore::Artifact(artifact) => artifact.weights().len(),
        }
    }

    fn bytes(&self, id: WeightId) -> Result<(DType, &[u8]), Fn3Error> {
        match self {
            WeightStore::Imported(imported) => {
                let weight = imported.weights().get(id).ok_or(Fn3Error::OutOfBounds {
                    what: "runtime Session weight id",
                    index: id.index() as u64,
                    len: imported.weights().len() as u64,
                })?;
                Ok((weight.dtype(), weight.bytes()))
            }
            WeightStore::Artifact(artifact) => {
                let entry = artifact
                    .weights()
                    .get(id.index())
                    .ok_or(Fn3Error::OutOfBounds {
                        what: "runtime Session weight id",
                        index: id.index() as u64,
                        len: artifact.weights().len() as u64,
                    })?;
                let bytes = artifact.weight_bytes_at(id).ok_or(Fn3Error::OutOfBounds {
                    what: "runtime Session weight bytes",
                    index: id.index() as u64,
                    len: artifact.weights().len() as u64,
                })?;
                Ok((entry.dtype(), bytes))
            }
        }
    }
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
        let ty = || TensorType::new(DType::F32, [DimExpr::constant(1), DimExpr::constant(4)]);
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
