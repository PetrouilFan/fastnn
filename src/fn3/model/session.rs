//! The inference session: one conversation's mutable state.
//!
//! # Ownership
//!
//! ```text
//! PreparedModel   immutable, Send + Sync, shared by every session
//!   ├── Arc<ExecutionPlan>   the model's own plan (fixed-shape models)
//!   ├── Arc<ModelWeights>    immutable weight tensors, Arc-shared storage
//!   ├── DeviceId             the device the model and its KV cache live on
//!   └── ShapeEnvelope        the declared dynamic bounds
//!
//! Session         mutable, Send but NOT Sync, one per concurrent stream
//!   ├── &PreparedModel       borrowed, never owned
//!   ├── StepRuntime          planner + backend + plan cache + slot pool
//!   ├── Vec<KvCacheState>    capacity + live extent + keys/values buffers
//!   ├── step: u64            successful `decode` calls
//!   ├── tokens: u64          tokens consumed (prefill + decode)
//!   └── phase                Empty | Prefilling | Ready
//! ```
//!
//! A session never shares a mutable byte with another session: the caches, the
//! plan cache, the slot pool and every counter are per-session. Weights are
//! `Arc`-shared and plans are immutable, so two sessions over one
//! [`PreparedModel`] see the same weights and build content-identical plans
//! (same [`PlanId`]) while stepping independently.
//!
//! # What "step" counts
//!
//! [`Session::step`] counts **successful `decode` calls**, not tokens and not
//! prefills. `prefill` moves the phase to [`SessionPhase::Ready`] and does not
//! increment it, so a 13-token prompt followed by four one-token decode calls
//! reports `step() == 4` with `cache_len() == 17`. [`Session::tokens_consumed`]
//! is the token total (17 in the same example).
//!
//! # Step atomicity
//!
//! A step is a transaction over the caches and the counters:
//!
//! ```text
//! 1. plan (PlanCache)            ─┐
//! 2. capacity check, all layers   │  fallible, mutates nothing
//! 3. materialise slots            │
//! 4. bind inputs                  │
//! 5. execute the plan             │
//! 6. extract and copy the logits  │
//! 7. validate every `present`    ─┘
//! 8. append, all layers          ── commit: checks already passed
//! 9. counters + phase
//! ```
//!
//! A failure in 1–7 leaves every live extent, `step()`, `tokens_consumed()` and
//! the phase exactly as they were. Step 8 only writes after
//! [`KvCacheState::check_append`] and [`KvCacheState::check_present`] passed for
//! **every** layer, so a multi-layer commit cannot be half-applied.
//!
//! The work counters (`plan_hits`, `plan_misses`, `slot_allocations`,
//! `slot_reuses`, `slot_bytes_allocated`) are diagnostics of work actually
//! performed, not session state: a step that fails after building a plan did
//! build that plan, and the counter says so. Everything that describes *state*
//! — extents, `step`, `tokens_consumed`, phase — is transactional.
//!
//! # Device execution
//!
//! A session holds exactly one backend and executes every instruction through
//! it ([`Plan::execute`]). It never owns or borrows a second backend, so a
//! session over a Vulkan [`PreparedModel`] cannot dispatch a CPU kernel: the
//! constructor rejects a backend whose device differs from the model's
//! ([`Fn3Error::DeviceMismatch`]). On a Vulkan device the cache and every value
//! slot are host-staged (see [`super::kv`]); the backend records
//! `host_staged = true` for each op and proves the arithmetic ran on the device.
//!
//! [`PlanId`]: super::PlanId

use std::cell::Cell;
use std::collections::BTreeMap;
use std::marker::PhantomData;

use crate::fn3::backend::Backend;
use crate::fn3::device::DeviceId;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::import::{ModelImport, PositionEmbedding};
use crate::fn3::shape::{LiveDims, Shape, Strides, Symbol};
use crate::fn3::tensor::{Tensor, TensorMeta, TensorView};

use super::kv::{allocate, KvCacheConfig, KvCacheState};
use super::plan::{Plan, PlanId, SlotPlan, SlotRole};
use super::specialize::Planner;
use super::{ModelWeights, PreparedModel};

/// How many plans a session keeps at most.
///
/// The default capacity is derived from the KV capacity: one plan per prefill
/// length plus one per decode position. Past that, [`super::PlanCache`] evicts
/// the smallest key deterministically, and a rebuild is correct but not free.
const MAX_SESSION_PLAN_CACHE: usize = 64;

/// Session lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionPhase {
    /// No prompt has been consumed. `prefill` is allowed.
    Empty,
    /// A prompt is being consumed. No call is allowed.
    Prefilling,
    /// A prompt has been consumed. `decode` is allowed.
    Ready,
}

impl SessionPhase {
    /// Stable name used in diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            SessionPhase::Empty => "empty",
            SessionPhase::Prefilling => "prefilling",
            SessionPhase::Ready => "ready",
        }
    }
}

/// Geometry a step needs, fixed by the weights and the cache config.
#[derive(Debug, Clone, Copy)]
struct SessionGeometry {
    batch: u64,
    query_heads: u64,
    kv_heads: u64,
    head_dim: u64,
    max_positions: u64,
}

/// Graph-spelled KV value names of one layer, resolved from the import.
#[derive(Debug, Clone)]
struct KvNames {
    past_key: String,
    past_value: String,
    present_key: String,
    present_value: String,
}

/// Cache key of one reusable slot buffer: name, dtype and exact shape.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SlotKey {
    name: String,
    dtype: u8,
    shape: Vec<u64>,
}

impl SlotKey {
    fn of(slot: &SlotPlan) -> SlotKey {
        SlotKey {
            name: slot.name().to_string(),
            dtype: slot.dtype().code(),
            shape: slot.shape().dims().to_vec(),
        }
    }
}

/// Everything a step needs besides the model: who plans, who executes, and what
/// has been allocated so far.
struct StepRuntime<'model> {
    planner: Planner<'model>,
    backend: &'model dyn Backend,
    plans: super::PlanCache,
    kv: Vec<KvNames>,
    /// Instruction-output buffers from the previous step, keyed by
    /// name/dtype/shape.
    pool: BTreeMap<SlotKey, Tensor>,
    allocations: u64,
    reuses: u64,
    allocated_bytes: u64,
    last_plan: Option<PlanId>,
    last_dims: Option<LiveDims>,
}

impl StepRuntime<'_> {
    /// Allocate (or reuse) every value slot a plan addresses.
    ///
    /// Weight slots clone `Arc` storage; folded constants share their host
    /// allocation but are tagged for the model's device; instruction outputs
    /// come from the previous step's pool when name, dtype and shape match.
    fn materialise(
        &mut self,
        plan: &Plan,
        weights: &ModelWeights,
        device: DeviceId,
    ) -> Result<Vec<Option<Tensor>>, Fn3Error> {
        let mut slots = Vec::with_capacity(plan.slots().len());
        for slot in plan.slots() {
            let tensor = match slot.role() {
                SlotRole::GraphInput => None,
                SlotRole::Weight { id } => {
                    let tensor = weights
                        .get(id.index())
                        .ok_or_else(|| Fn3Error::OutOfBounds {
                            what: "Session::materialise (weight index)",
                            index: id.index() as u64,
                            len: weights.len() as u64,
                        })?;
                    if tensor.dtype() != slot.dtype() || tensor.shape() != slot.shape() {
                        return Err(Fn3Error::InvalidShape {
                            context: "Session::materialise",
                            detail: format!(
                                "weight `{}` is {} {} but slot expects {} {}",
                                slot.name(),
                                tensor.dtype(),
                                tensor.shape(),
                                slot.dtype(),
                                slot.shape()
                            ),
                        });
                    }
                    Some(tensor.clone())
                }
                SlotRole::Constant { index } => {
                    let constant = plan.constants().get(*index as usize).ok_or_else(|| {
                        Fn3Error::OutOfBounds {
                            what: "Session::materialise (constant index)",
                            index: u64::from(*index),
                            len: plan.constants().len() as u64,
                        }
                    })?;
                    let tensor = constant.to_tensor()?;
                    // `to_tensor` always builds CPU metadata; re-tag the same
                    // allocation for the model's device so a Vulkan kernel sees
                    // an operand on its own device. No bytes are copied.
                    let meta =
                        TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), device)?;
                    Some(Tensor::from_storage(tensor.storage().clone(), meta)?)
                }
                SlotRole::InstructionOutput { .. } => {
                    let key = SlotKey::of(slot);
                    match self.pool.remove(&key) {
                        Some(tensor) => {
                            self.reuses += 1;
                            Some(tensor)
                        }
                        None => {
                            let tensor = allocate(slot.shape().clone(), slot.dtype(), device)?;
                            self.allocations += 1;
                            self.allocated_bytes += tensor.storage().len() as u64;
                            Some(tensor)
                        }
                    }
                }
            };
            slots.push(tensor);
        }
        Ok(slots)
    }

    /// Keep the instruction outputs of a finished step for the next one.
    ///
    /// Only one generation is retained, so the pool is bounded by one plan's
    /// output slots (28.5 MB for the pinned prefill at `seq = 13`, a few hundred
    /// kilobytes for a decode step).
    fn recycle(&mut self, plan: &Plan, slots: Vec<Option<Tensor>>) {
        for (slot, tensor) in plan.slots().iter().zip(slots) {
            if matches!(slot.role(), SlotRole::InstructionOutput { .. }) {
                if let Some(tensor) = tensor {
                    self.pool.insert(SlotKey::of(slot), tensor);
                }
            }
        }
    }
}

/// Mutable inference state for one conversation.
///
/// # Concurrency
///
/// A session is the unit of exclusive mutable state. One session must be
/// stepped by one thread at a time: it is `Send` (it can move between threads)
/// but deliberately not `Sync` (it cannot be shared by reference). Use one
/// session per concurrent request; they share the [`PreparedModel`] and nothing
/// else. `Mutex<Session>` is still usable because `Session: Send`.
///
/// # Atomicity
///
/// A failed step leaves the KV cache live extents, [`Session::step`],
/// [`Session::tokens_consumed`] and [`Session::phase`] unchanged (see the module
/// docs for the commit ordering).
pub struct Session<'model> {
    model: &'model PreparedModel,
    runtime: Option<StepRuntime<'model>>,
    geometry: Option<SessionGeometry>,
    caches: Vec<KvCacheState>,
    step: u64,
    tokens: u64,
    phase: SessionPhase,
    /// Makes `Session` `!Sync` on purpose. See the type-level docs.
    _not_sync: PhantomData<Cell<()>>,
}

impl<'model> Session<'model> {
    /// Create a session over `model`, allocating its KV cache buffers.
    ///
    /// This constructor binds no planner and no backend: it is enough to own
    /// cache state, phase and counters, and it is what a model with a single
    /// fixed-shape plan uses. [`Session::prefill`] and [`Session::decode`] need
    /// to specialise a plan per `(seq, past)` pair, so they require
    /// [`Session::with_runtime`] and report a typed [`Fn3Error::SessionState`]
    /// here.
    ///
    /// Fails if the model's cache geometry is invalid or a buffer does not fit
    /// the device. Two sessions created from the same model never share a byte
    /// of mutable state.
    pub fn new(model: &'model PreparedModel) -> Result<Session<'model>, Fn3Error> {
        let caches = Session::allocate_caches(model)?;
        Ok(Session {
            model,
            runtime: None,
            geometry: None,
            caches,
            step: 0,
            tokens: 0,
            phase: SessionPhase::Empty,
            _not_sync: PhantomData,
        })
    }

    /// Create a session that can `prefill` and `decode`.
    ///
    /// `import` supplies the decoder geometry and the graph-spelled KV value
    /// names; `backend` is the one and only executor this session will ever
    /// use. The backend's device must equal the model's device, so a model
    /// prepared for a Vulkan device can never be stepped by the CPU backend.
    ///
    /// This is the entry point #62 (end-to-end), #63 (CLI) and #64 (Python)
    /// call:
    ///
    /// ```ignore
    /// let session = Session::with_runtime(&model, &import, &backend)?;
    /// let logits = session.prefill(&tokens)?;   // [1, 1, vocab]
    /// let next   = session.decode(&[token])?;   // [1, 1, vocab]
    /// ```
    pub fn with_runtime(
        model: &'model PreparedModel,
        import: &'model ModelImport,
        backend: &'model dyn Backend,
    ) -> Result<Session<'model>, Fn3Error> {
        if backend.device() != model.device() {
            return Err(Fn3Error::DeviceMismatch {
                op: None,
                expected: model.device(),
                actual: backend.device(),
            });
        }
        let config = *model.kv_config().ok_or_else(|| Fn3Error::SessionState {
            action: "Session::with_runtime",
            phase: "empty",
            detail: format!(
                "the model on {} declares no KV cache geometry, so it cannot prefill or decode; \
                 build it with DecoderGeometry::kv_cache_config",
                model.device()
            ),
        })?;
        config.validate(&model.envelope())?;
        if config.batch != 1 {
            return Err(Fn3Error::InvalidShape {
                context: "Session::with_runtime",
                detail: format!(
                    "the alpha session serves one sequence at a time, but the cache declares batch \
                     {}; a batch > 1 needs per-row cache bookkeeping",
                    config.batch
                ),
            });
        }
        let geometry = import.geometry();
        let expected = [
            ("layers", u64::from(geometry.layers), config.layers),
            (
                "n_kv_heads",
                u64::from(geometry.kv_heads),
                config.n_kv_heads,
            ),
            ("head_dim", u64::from(geometry.head_dim), config.head_dim),
        ];
        for (name, from_graph, from_config) in expected {
            if from_graph != from_config {
                return Err(Fn3Error::InvalidShape {
                    context: "Session::with_runtime",
                    detail: format!(
                        "the cache geometry disagrees with the imported graph: {name} is \
                         {from_graph} in the graph and {from_config} in the KV cache config"
                    ),
                });
            }
        }
        let max_positions = match geometry.positions {
            PositionEmbedding::Learned { max_positions, .. } => u64::from(max_positions),
        };
        let graph = import.graph();
        let mut kv = Vec::with_capacity(config.layers as usize);
        for interface in geometry.kv.iter().take(config.layers as usize) {
            kv.push(KvNames {
                past_key: graph.value(interface.past_key).name.clone(),
                past_value: graph.value(interface.past_value).name.clone(),
                present_key: graph.value(interface.present_key).name.clone(),
                present_value: graph.value(interface.present_value).name.clone(),
            });
        }
        if kv.len() != config.layers as usize {
            return Err(Fn3Error::InvalidShape {
                context: "Session::with_runtime",
                detail: format!(
                    "the cache declares {} layers but the graph exposes {} KV interfaces",
                    config.layers,
                    geometry.kv.len()
                ),
            });
        }
        // Keep every immutable weight on the device once per backend (FN3-26).
        // The backend decides what that means: the CPU backend has no-op
        // persistence, the Vulkan backend uploads each weight exactly once and
        // resolves later binds by handle. The host tensors are untouched, so the
        // CPU/fallback path keeps its bytes.
        let weights = model.weights();
        let mut persistent: Vec<&Tensor> = Vec::with_capacity(weights.len());
        for index in 0..weights.len() {
            if let Some(tensor) = weights.get(index) {
                persistent.push(tensor);
            }
        }
        backend.prepare_persistent(&persistent)?;
        let caches = Session::allocate_caches(model)?;
        let capacity =
            (config.capacity_tokens.saturating_mul(2) as usize + 1).min(MAX_SESSION_PLAN_CACHE);
        let plans = super::PlanCache::new(model.envelope(), backend.name(), capacity)?;
        Ok(Session {
            model,
            runtime: Some(StepRuntime {
                planner: Planner::new(import, backend),
                backend,
                plans,
                kv,
                pool: BTreeMap::new(),
                allocations: 0,
                reuses: 0,
                allocated_bytes: 0,
                last_plan: None,
                last_dims: None,
            }),
            geometry: Some(SessionGeometry {
                batch: config.batch,
                query_heads: u64::from(geometry.query_heads),
                kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                max_positions,
            }),
            caches,
            step: 0,
            tokens: 0,
            phase: SessionPhase::Empty,
            _not_sync: PhantomData,
        })
    }

    fn allocate_caches(model: &PreparedModel) -> Result<Vec<KvCacheState>, Fn3Error> {
        let mut caches = Vec::new();
        if let Some(config) = model.kv_config() {
            config.validate(&model.envelope())?;
            caches.reserve(config.layers as usize);
            for _ in 0..config.layers {
                caches.push(KvCacheState::new(*config, model.device())?);
            }
        }
        Ok(caches)
    }

    /// The immutable model this session runs.
    pub fn model(&self) -> &'model PreparedModel {
        self.model
    }

    /// The backend every instruction of this session is executed by, when the
    /// session was built with [`Session::with_runtime`].
    pub fn backend(&self) -> Option<&'model dyn Backend> {
        self.runtime.as_ref().map(|runtime| runtime.backend)
    }

    /// Current lifecycle phase.
    pub fn phase(&self) -> SessionPhase {
        self.phase
    }

    /// Number of successfully completed `decode` calls.
    ///
    /// Incremented only after a step's outputs and cache updates are committed;
    /// a failed step never changes it.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Tokens consumed by successful `prefill` and `decode` calls.
    pub fn tokens_consumed(&self) -> u64 {
        self.tokens
    }

    /// Live KV extent of the first layer, or 0 when the model has no cache.
    pub fn cache_len(&self) -> u64 {
        self.caches.first().map(KvCacheState::live_len).unwrap_or(0)
    }

    /// Per-layer cache state. Read-only: mutation happens inside steps.
    pub fn caches(&self) -> &[KvCacheState] {
        &self.caches
    }

    /// Bytes the session holds for its KV cache (keys and values, all layers).
    pub fn kv_bytes(&self) -> u64 {
        self.caches.iter().map(KvCacheState::allocated_bytes).sum()
    }

    /// Plans served from this session's cache (no rebuild).
    pub fn plan_hits(&self) -> u64 {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.plans.hits())
            .unwrap_or(0)
    }

    /// Plans this session had to build.
    pub fn plan_misses(&self) -> u64 {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.plans.misses())
            .unwrap_or(0)
    }

    /// Plans currently cached.
    pub fn plan_cache_len(&self) -> usize {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.plans.len())
            .unwrap_or(0)
    }

    /// Plans dropped because the session's cache was full.
    pub fn plan_evictions(&self) -> u64 {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.plans.evictions())
            .unwrap_or(0)
    }

    /// Instruction-output buffers allocated by this session.
    pub fn slot_allocations(&self) -> u64 {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.allocations)
            .unwrap_or(0)
    }

    /// Instruction-output buffers reused from the previous step.
    pub fn slot_reuses(&self) -> u64 {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.reuses)
            .unwrap_or(0)
    }

    /// Bytes allocated for instruction outputs over this session's lifetime.
    pub fn slot_bytes_allocated(&self) -> u64 {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.allocated_bytes)
            .unwrap_or(0)
    }

    /// Identity of the plan the last successful step executed.
    pub fn last_plan(&self) -> Option<PlanId> {
        self.runtime.as_ref().and_then(|runtime| runtime.last_plan)
    }

    /// Dynamic environment of the plan the last successful step executed.
    pub fn last_plan_dims(&self) -> Option<LiveDims> {
        self.runtime.as_ref().and_then(|runtime| runtime.last_dims)
    }

    /// Consume a prompt and populate the KV cache.
    ///
    /// Builds the plan for `past = 0`, `seq = tokens.len()`, executes it, appends
    /// the `present` K/V of every layer at the live extent (0), sets
    /// `cache_len() == tokens.len()` and the phase to [`SessionPhase::Ready`],
    /// and returns **the logits of the final prompt position**, shape
    /// `[1, 1, vocab]` for the pinned model (`[batch, 1, vocab]` generally) as a
    /// fresh, session-independent tensor.
    ///
    /// Requires [`SessionPhase::Empty`] and a session built by
    /// [`Session::with_runtime`]. On any error the cache, the counters and the
    /// phase are unchanged.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, Fn3Error> {
        if self.phase != SessionPhase::Empty {
            return Err(Fn3Error::SessionState {
                action: "prefill",
                phase: self.phase.name(),
                detail: "prefill requires a fresh session; call reset() first".to_string(),
            });
        }
        if tokens.is_empty() {
            return Err(Fn3Error::SessionState {
                action: "prefill",
                phase: self.phase.name(),
                detail: "prefill requires at least one token".to_string(),
            });
        }
        self.require_runtime("prefill")?;
        // `Prefilling` is the in-step phase: no call is allowed while a step is
        // running, and a failure returns the session to `Empty`.
        self.phase = SessionPhase::Prefilling;
        match self.run_step(tokens, 0) {
            Ok(logits) => {
                self.phase = SessionPhase::Ready;
                Ok(logits)
            }
            Err(error) => {
                self.phase = SessionPhase::Empty;
                Err(error)
            }
        }
    }

    /// Consume one or more new tokens, appending to the KV cache.
    ///
    /// Builds the plan for `past = cache_len()` and `seq = tokens.len()`,
    /// executes it, appends the `present` K/V and advances the extent by
    /// `tokens.len()`. Returns the logits of the last new position, shape
    /// `[1, 1, vocab]` for the pinned model, as a fresh tensor.
    ///
    /// Requires [`SessionPhase::Ready`] and increments [`Session::step`] exactly
    /// once on success. On any error the cache live extents, the counters and
    /// the phase are unchanged.
    pub fn decode(&mut self, tokens: &[u32]) -> Result<Tensor, Fn3Error> {
        if self.phase != SessionPhase::Ready {
            return Err(Fn3Error::SessionState {
                action: "decode",
                phase: self.phase.name(),
                detail: "decode requires a prefilled session; call prefill() first".to_string(),
            });
        }
        if tokens.is_empty() {
            return Err(Fn3Error::SessionState {
                action: "decode",
                phase: self.phase.name(),
                detail: "decode requires at least one token".to_string(),
            });
        }
        self.require_runtime("decode")?;
        let past = self.check_extents()?;
        self.phase = SessionPhase::Prefilling;
        match self.run_step(tokens, past) {
            Ok(logits) => {
                self.phase = SessionPhase::Ready;
                self.step += 1;
                Ok(logits)
            }
            Err(error) => {
                self.phase = SessionPhase::Ready;
                Err(error)
            }
        }
    }

    /// Resume a session from an externally supplied KV prefix.
    ///
    /// `keys[l]` and `values[l]` are layer `l`'s cache for the first `tokens`
    /// positions, shaped `[batch, n_kv_heads, tokens, head_dim]` on the model's
    /// device; they are copied into the cache buffers and the live extent
    /// becomes `tokens`, so the next [`Session::decode`] runs at
    /// `past = tokens`. Nothing is computed: this is a data-in path for a
    /// persisted prompt cache, and the session's counters start from zero
    /// (`step() == 0`, `tokens_consumed() == tokens`).
    ///
    /// Requires [`SessionPhase::Empty`]. All layers are validated before any
    /// buffer is written.
    pub fn resume(
        &mut self,
        keys: &[Tensor],
        values: &[Tensor],
        tokens: u64,
    ) -> Result<(), Fn3Error> {
        if self.phase != SessionPhase::Empty {
            return Err(Fn3Error::SessionState {
                action: "resume",
                phase: self.phase.name(),
                detail: "resume requires a fresh session; call reset() first".to_string(),
            });
        }
        if self.caches.is_empty() {
            return Err(Fn3Error::SessionState {
                action: "resume",
                phase: self.phase.name(),
                detail: "the model declares no KV cache to resume".to_string(),
            });
        }
        if keys.len() != self.caches.len() || values.len() != self.caches.len() {
            return Err(Fn3Error::ShapeMismatch {
                op: None,
                expected: format!("{} layers of keys and values", self.caches.len()),
                actual: format!("{} keys and {} values", keys.len(), values.len()),
            });
        }
        self.model
            .envelope()
            .get(Symbol::Past)
            .check(Symbol::Past, tokens)?;
        // Validate every layer before writing any of them.
        for (layer, (cache, (keys, values))) in
            self.caches.iter().zip(keys.iter().zip(values)).enumerate()
        {
            cache.check_append(tokens)?;
            let expected = cache.config().shape_for(tokens);
            for (half, tensor) in [("keys", keys), ("values", values)] {
                if tensor.shape() != &expected {
                    return Err(Fn3Error::ShapeMismatch {
                        op: None,
                        expected: format!(
                            "{expected} (layer {layer} {half} of a {tokens}-token prefix)"
                        ),
                        actual: format!("{}", tensor.shape()),
                    });
                }
            }
            let keys_view = keys.as_view()?;
            let values_view = values.as_view()?;
            cache.check_present(&keys_view, &values_view)?;
        }
        for (cache, (keys, values)) in self.caches.iter_mut().zip(keys.iter().zip(values)) {
            cache.append(&keys.as_view()?, &values.as_view()?)?;
        }
        self.phase = SessionPhase::Ready;
        self.step = 0;
        self.tokens = tokens;
        Ok(())
    }

    /// Clear all cache live extents and return to [`SessionPhase::Empty`].
    ///
    /// Buffers are kept, not reallocated; the step and token counters return to
    /// 0. Plans stay cached: a reset session runs the same shapes again.
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
        self.step = 0;
        self.tokens = 0;
        if let Some(runtime) = self.runtime.as_mut() {
            runtime.pool.clear();
            runtime.last_plan = None;
            runtime.last_dims = None;
        }
        self.phase = SessionPhase::Empty;
    }

    fn require_runtime(&self, action: &'static str) -> Result<(), Fn3Error> {
        if self.runtime.is_some() {
            Ok(())
        } else {
            Err(Fn3Error::SessionState {
                action,
                phase: self.phase.name(),
                detail: "this session was built by Session::new, which binds no shape planner and \
                         no backend; use Session::with_runtime(&model, &import, &backend) to \
                         prefill and decode"
                    .to_string(),
            })
        }
    }

    /// The live extent, after checking that every layer agrees.
    fn check_extents(&self) -> Result<u64, Fn3Error> {
        let mut live: Option<u64> = None;
        for (layer, cache) in self.caches.iter().enumerate() {
            match live {
                None => live = Some(cache.live_len()),
                Some(expected) if expected == cache.live_len() => {}
                Some(expected) => {
                    return Err(Fn3Error::SessionState {
                        action: "decode",
                        phase: self.phase.name(),
                        detail: format!(
                            "layer 0 has a live KV extent of {expected} tokens but layer {layer} \
                             has {}; the layers must advance together",
                            cache.live_len()
                        ),
                    })
                }
            }
        }
        Ok(live.unwrap_or(0))
    }

    /// Execute one step: plan, run, extract, append, commit.
    ///
    /// Mutates the caches and the token counter only on success. The caller owns
    /// the phase and the `step` counter, because their failure rules differ
    /// between prefill and decode.
    fn run_step(&mut self, tokens: &[u32], past: u64) -> Result<Tensor, Fn3Error> {
        let geometry = self.geometry.ok_or_else(|| Fn3Error::SessionState {
            action: "step",
            phase: self.phase.name(),
            detail: "no planner is bound to this session".to_string(),
        })?;
        let config = *self
            .model
            .kv_config()
            .ok_or_else(|| Fn3Error::SessionState {
                action: "step",
                phase: self.phase.name(),
                detail: "the model declares no KV cache geometry".to_string(),
            })?;
        let seq = tokens.len() as u64;
        // `LiveDims::new` checks `past + seq`; the position-table bound below
        // uses the checked total, never an unchecked sum.
        let dims = LiveDims::new(
            geometry.batch,
            seq,
            past,
            geometry.query_heads,
            geometry.kv_heads,
            geometry.head_dim,
        )?;
        let last_position = dims.total;
        if last_position > geometry.max_positions {
            return Err(Fn3Error::CapacityExceeded {
                what: "position embedding rows",
                requested: last_position,
                capacity: geometry.max_positions,
            });
        }
        // Capacity is checked for every layer before any work happens.
        for cache in &self.caches {
            cache.check_capacity(seq)?;
        }
        let envelope = self.model.envelope();
        let caches = &mut self.caches;
        let runtime = self
            .runtime
            .as_mut()
            .ok_or_else(|| Fn3Error::SessionState {
                action: "step",
                phase: self.phase.name(),
                detail: "no planner is bound to this session".to_string(),
            })?;
        let plan = runtime.plans.plan(&runtime.planner, &envelope, dims)?;
        let mut slots = runtime.materialise(&plan, self.model.weights(), self.model.device())?;

        // Bind the decoder interface. The plan's own input table is the
        // authority on shapes and dtypes: a disagreement is a named error, never
        // a silent reshape.
        bind_ids(&plan, &mut slots, "input_ids", tokens, self.model.device())?;
        let positions: Vec<i64> = (past..last_position).map(|value| value as i64).collect();
        bind_i64(
            &plan,
            &mut slots,
            "position_ids",
            &positions,
            self.model.device(),
        )?;
        let mask = vec![1i64; last_position as usize];
        bind_i64(
            &plan,
            &mut slots,
            "attention_mask",
            &mask,
            self.model.device(),
        )?;
        for (layer, cache) in caches.iter().enumerate() {
            let names = &runtime.kv[layer];
            bind_tensor(&plan, &mut slots, &names.past_key, cache.live_keys()?)?;
            bind_tensor(&plan, &mut slots, &names.past_value, cache.live_values()?)?;
        }

        // The only execution path: the session's backend. `begin_step` /
        // `end_step` give a batching backend one submission scope per step; its
        // default bodies are no-ops, so the CPU path is unchanged. The scope is
        // always closed, including on failure, so a failed step cannot leak a
        // recording context into the next one.
        let label = if past == 0 { "prefill" } else { "decode" };
        runtime.backend.begin_step(label)?;
        let executed = plan.execute(runtime.backend, &mut slots);
        let closed = runtime.backend.end_step();
        executed?;
        closed?;

        // Extract the logits before touching the caches, so a malformed output
        // is still an untouched session.
        let logits = last_position_logits(&plan, &slots, seq, geometry.batch)?;

        // Validate every layer's `present` before appending to any layer.
        let mut presents: Vec<(TensorView<'_>, TensorView<'_>)> = Vec::with_capacity(caches.len());
        for (layer, cache) in caches.iter().enumerate() {
            let names = &runtime.kv[layer];
            let keys = present_slice(&plan, &slots, &names.present_key, past, seq, &config)?;
            let values = present_slice(&plan, &slots, &names.present_value, past, seq, &config)?;
            cache.check_append(seq)?;
            cache.check_present(&keys, &values)?;
            presents.push((keys, values));
        }
        // Commit. Every check above passed for every layer, so these appends
        // cannot fail; `?` is kept so a refactor that breaks that invariant is
        // reported rather than ignored.
        for (cache, (keys, values)) in caches.iter_mut().zip(presents.iter()) {
            cache.append(keys, values)?;
        }
        drop(presents);

        runtime.recycle(&plan, slots);
        runtime.last_plan = Some(plan.identity());
        runtime.last_dims = Some(dims);
        self.tokens += seq;
        Ok(logits)
    }
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("phase", &self.phase)
            .field("step", &self.step)
            .field("tokens", &self.tokens)
            .field("caches", &self.caches.len())
            .field("cache_len", &self.cache_len())
            .field("plans", &self.plan_cache_len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Input binding and output extraction
// ---------------------------------------------------------------------------

/// Look up a graph input the session must supply.
fn input<'a>(plan: &'a Plan, name: &str) -> Result<&'a super::BoundValue, Fn3Error> {
    plan.input(name).ok_or_else(|| Fn3Error::SessionState {
        action: "step",
        phase: "running",
        detail: format!(
            "the plan has no graph input `{name}`; the imported decoder interface and the \
             specialised plan disagree"
        ),
    })
}

/// Look up a graph output the session must read.
fn output<'a>(plan: &'a Plan, name: &str) -> Result<&'a super::BoundValue, Fn3Error> {
    plan.output(name).ok_or_else(|| Fn3Error::SessionState {
        action: "step",
        phase: "running",
        detail: format!(
            "the plan has no graph output `{name}`; the imported decoder interface and the \
             specialised plan disagree"
        ),
    })
}

/// Bind a token-id input (`[batch, seq]` `i64`).
fn bind_ids(
    plan: &Plan,
    slots: &mut [Option<Tensor>],
    name: &str,
    tokens: &[u32],
    device: DeviceId,
) -> Result<(), Fn3Error> {
    let values: Vec<i64> = tokens.iter().map(|&token| i64::from(token)).collect();
    bind_i64(plan, slots, name, &values, device)
}

/// Allocate a host-staged `i64` input on `device` and bind it.
fn bind_i64(
    plan: &Plan,
    slots: &mut [Option<Tensor>],
    name: &str,
    values: &[i64],
    device: DeviceId,
) -> Result<(), Fn3Error> {
    let input = input(plan, name)?;
    if input.dtype() != DType::I64 {
        return Err(Fn3Error::DTypeMismatch {
            op: None,
            expected: DType::I64,
            actual: input.dtype(),
        });
    }
    let expected = input.shape().numel()?;
    if expected != values.len() as u64 {
        return Err(Fn3Error::ShapeMismatch {
            op: None,
            expected: format!(
                "{expected} elements for graph input `{name}` {}",
                input.shape()
            ),
            actual: format!("{} values", values.len()),
        });
    }
    let mut tensor = allocate(input.shape().clone(), DType::I64, device)?;
    {
        let mut view = tensor.as_mut_view()?;
        let destination = view.storage_slice_mut::<i64>()?;
        destination.copy_from_slice(values);
    }
    slots[input.slot() as usize] = Some(tensor);
    Ok(())
}

/// Bind an already-built input, checking it against the plan's declaration.
fn bind_tensor(
    plan: &Plan,
    slots: &mut [Option<Tensor>],
    name: &str,
    tensor: Tensor,
) -> Result<(), Fn3Error> {
    let input = input(plan, name)?;
    if input.dtype() != tensor.dtype() {
        return Err(Fn3Error::DTypeMismatch {
            op: None,
            expected: input.dtype(),
            actual: tensor.dtype(),
        });
    }
    if input.shape() != tensor.shape() {
        return Err(Fn3Error::ShapeMismatch {
            op: None,
            expected: format!("{} for graph input `{name}`", input.shape()),
            actual: format!("{}", tensor.shape()),
        });
    }
    slots[input.slot() as usize] = Some(tensor);
    Ok(())
}

/// Copy the logits of the last position of every batch row into `[batch, 1,
/// vocab]`.
fn last_position_logits(
    plan: &Plan,
    slots: &[Option<Tensor>],
    seq: u64,
    batch: u64,
) -> Result<Tensor, Fn3Error> {
    let output = output(plan, "logits")?;
    let tensor = slots[output.slot() as usize]
        .as_ref()
        .ok_or_else(|| Fn3Error::SessionState {
            action: "step",
            phase: "running",
            detail: "the plan wrote no logits tensor".to_string(),
        })?;
    let dims = tensor.shape().dims();
    if dims.len() != 3 || dims[0] != batch || dims[1] != seq {
        return Err(Fn3Error::ShapeMismatch {
            op: None,
            expected: format!("[{batch}, {seq}, vocab] logits"),
            actual: format!("{}", tensor.shape()),
        });
    }
    let vocab = dims[2] as usize;
    if vocab == 0 || seq == 0 {
        return Err(Fn3Error::InvalidShape {
            context: "Session::logits",
            detail: format!(
                "a step needs at least one position and one logit; logits are {}",
                tensor.shape()
            ),
        });
    }
    let source = tensor.as_f32_slice()?;
    let mut last = allocate(
        Shape::from_dims([batch, 1, vocab as u64]),
        DType::F32,
        tensor.device(),
    )?;
    {
        let mut view = last.as_mut_view()?;
        let destination = view.storage_slice_mut::<f32>()?;
        for row in 0..batch as usize {
            let start = (row * seq as usize + seq as usize - 1) * vocab;
            destination[row * vocab..(row + 1) * vocab]
                .copy_from_slice(&source[start..start + vocab]);
        }
    }
    Ok(last)
}

/// A zero-copy `[batch, kv_heads, seq, head_dim]` view of a `present` output
/// starting at `past`.
fn present_slice<'a>(
    plan: &Plan,
    slots: &'a [Option<Tensor>],
    name: &str,
    past: u64,
    seq: u64,
    config: &KvCacheConfig,
) -> Result<TensorView<'a>, Fn3Error> {
    let output = output(plan, name)?;
    let tensor = slots[output.slot() as usize]
        .as_ref()
        .ok_or_else(|| Fn3Error::SessionState {
            action: "step",
            phase: "running",
            detail: format!("the plan wrote no `{name}` tensor"),
        })?;
    let total = past + seq;
    let declared = Shape::from_dims([config.batch, config.n_kv_heads, total, config.head_dim]);
    if tensor.shape() != &declared {
        return Err(Fn3Error::ShapeMismatch {
            op: None,
            expected: format!("{declared} for graph output `{name}`"),
            actual: format!("{}", tensor.shape()),
        });
    }
    if tensor.dtype() != config.dtype {
        return Err(Fn3Error::DTypeMismatch {
            op: None,
            expected: config.dtype,
            actual: tensor.dtype(),
        });
    }
    let strides = Strides::from_elements(vec![
        (config.n_kv_heads * total * config.head_dim) as i64,
        (total * config.head_dim) as i64,
        config.head_dim as i64,
        1,
    ]);
    let meta = TensorMeta::strided(
        config.shape_for(seq),
        strides,
        (past * config.head_dim) as usize,
        config.dtype,
        tensor.device(),
    )?;
    TensorView::new(tensor.storage(), meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fn3::dtype::DType as TestDType;
    use crate::fn3::model::{ExecutionPlan, Instruction};
    use crate::fn3::ops::OpId;
    use crate::fn3::shape::{Shape, ShapeEnvelope};
    use crate::fn3::{DeviceId, ModelWeights};

    fn small_model() -> PreparedModel {
        let plan =
            ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![2])], 3).unwrap();
        let weights = ModelWeights::new(vec![
            Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap()
        ])
        .unwrap();
        PreparedModel::new(
            plan,
            weights,
            DeviceId::Cpu,
            ShapeEnvelope::uniform(4).unwrap(),
            Some(KvCacheConfig {
                layers: 1,
                capacity_tokens: 4,
                batch: 1,
                n_kv_heads: 2,
                head_dim: 2,
                dtype: TestDType::F32,
                layout: super::super::KvCacheLayout::BatchHeadsCapacityHeadDim,
            }),
        )
        .unwrap()
    }

    #[test]
    fn sessions_are_independent_and_state_rules_hold() {
        let model = small_model();
        let mut first = Session::new(&model).unwrap();
        let second = Session::new(&model).unwrap();
        assert_eq!(first.cache_len(), 0);
        assert_eq!(first.step(), 0);
        assert_eq!(first.tokens_consumed(), 0);
        assert_eq!(first.phase(), SessionPhase::Empty);
        let error = first.decode(&[1]).unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::SessionState {
                action: "decode",
                phase: "empty",
                ..
            }
        ));
        // A session without a planner cannot step, and says why.
        let error = first.prefill(&[1, 2]).unwrap_err();
        assert!(matches!(error, Fn3Error::SessionState { .. }), "{error}");
        assert!(error.to_string().contains("with_runtime"), "{error}");
        assert_eq!(first.step(), 0);
        assert_eq!(first.cache_len(), 0);
        assert_eq!(first.phase(), SessionPhase::Empty);
        assert_eq!(first.caches().len(), 1);
        assert_ne!(
            first.caches()[0].keys().storage().as_bytes().as_ptr(),
            second.caches()[0].keys().storage().as_bytes().as_ptr(),
            "sessions must not share KV buffers"
        );
        first.reset();
        assert_eq!(first.phase(), SessionPhase::Empty);
    }

    #[test]
    fn resume_checks_the_prefix_before_writing_anything() {
        let model = small_model();
        let mut session = Session::new(&model).unwrap();
        let error = session.resume(&[], &[], 1).unwrap_err();
        assert!(matches!(error, Fn3Error::ShapeMismatch { .. }), "{error}");
        assert_eq!(session.cache_len(), 0);
        assert_eq!(session.phase(), SessionPhase::Empty);

        let keys = Tensor::from_f32(Shape::from_dims([1, 2, 1, 2]), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let wrong = Tensor::from_f32(Shape::from_dims([1, 2, 1, 2]), &[1.0; 4]).unwrap();
        let cached = [keys.clone()];
        let error = session.resume(&cached, &[wrong], 2).unwrap_err();
        assert!(matches!(error, Fn3Error::ShapeMismatch { .. }), "{error}");
        assert_eq!(session.cache_len(), 0, "a rejected resume changes nothing");

        let cached = [keys.clone()];
        session
            .resume(&cached, std::slice::from_ref(&keys), 1)
            .unwrap();
        assert_eq!(session.phase(), SessionPhase::Ready);
        assert_eq!(session.cache_len(), 1);
        assert_eq!(session.tokens_consumed(), 1);
        assert_eq!(session.step(), 0);
        let error = session.resume(&[], &[], 1).unwrap_err();
        assert!(matches!(error, Fn3Error::SessionState { .. }), "{error}");
    }
}
