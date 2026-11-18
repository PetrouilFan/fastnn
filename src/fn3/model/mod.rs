//! Prepared models, execution plans and inference sessions.
//!
//! # Ownership split (the rule downstream workers must preserve)
//!
//! ```text
//! PreparedModel  (immutable, Send + Sync, cheap to clone via Arc)
//!   ├── Arc<ExecutionPlan>   instruction list, immutable
//!   ├── Arc<ModelWeights>    immutable weight tensors, Arc-shared storage
//!   ├── DeviceId             the device every weight lives on
//!   └── ShapeEnvelope        the declared dynamic bounds
//!
//! Session        (mutable, Send but NOT Sync, one per concurrent step)
//!   ├── &PreparedModel       borrowed, never owned
//!   ├── Vec<KvCacheState>    capacity + live extent + buffers
//!   ├── step: u64            successful step counter
//!   └── phase                Empty | Ready
//! ```
//!
//! A `PreparedModel` never changes after construction, so it may be shared by
//! any number of threads (`Arc<PreparedModel>`). A `Session` owns every mutable
//! byte of inference state: two sessions over one model are independent, and
//! one session must not be stepped from two threads. `Session` is deliberately
//! `!Sync` (see the `PhantomData<Cell<()>>` field): `Arc<Session>` cannot be
//! sent across threads, while `Mutex<Session>` still works because `Session`
//! is `Send`.
//!
//! # Step atomicity
//!
//! A failed step leaves the KV cache live extents and the step counter
//! unchanged. Implementations must validate every output extent and capacity
//! before mutating a cache buffer, and commit cache appends and the counter
//! only after all instructions succeeded.
//!
//! # Who implements what
//!
//! | Area | Owner |
//! |---|---|
//! | checked tensor metadata / storage hardening | FN3-06 (#52) |
//! | portable CPU reference kernels | FN3-07 (#53) |
//! | strict ONNX import and typed weights | FN3-08 (#54) |
//! | versioned prepared-model format | FN3-09 (#55) |
//! | bounded specialisation and plan building | FN3-10 (#56) |
//! | Vulkan backend | FN3-11 (#57) |
//! | real KV append, `prefill`/`decode` | FN3-12 (#58) |
//! | end-to-end pinned model | FN3-16 (#62) |
//!
//! The signatures and ownership rules below are the contract those issues
//! code against. Entry points that are not implemented yet return
//! [`Fn3Error::NotImplemented`] naming the owner; they never panic and never
//! approximate.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::fn3::backend::Backend;
use crate::fn3::device::DeviceId;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::shape::{DimBound, LiveDims, Shape, ShapeEnvelope, Symbol};
use crate::fn3::tensor::Tensor;

/// One op invocation inside an [`ExecutionPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    /// Operation to execute.
    pub op: OpId,
    /// Value slots read by the op, in operand order.
    pub inputs: Vec<u32>,
    /// Value slots written by the op, in result order.
    pub outputs: Vec<u32>,
    /// Op attributes.
    pub params: crate::fn3::backend::OpParams,
}

impl Instruction {
    /// Construct with default attributes.
    pub fn new(op: OpId, inputs: Vec<u32>, outputs: Vec<u32>) -> Instruction {
        Instruction {
            op,
            inputs,
            outputs,
            params: crate::fn3::backend::OpParams::default(),
        }
    }

    /// Builder: replace the attributes.
    pub fn with_params(mut self, params: crate::fn3::backend::OpParams) -> Instruction {
        self.params = params;
        self
    }
}

/// An ordered list of instructions over a flat table of value slots.
///
/// Slots are indices into the caller's `&mut [Option<Tensor>]`. Inputs must be
/// present; outputs must be pre-allocated (this keeps allocation policy in the
/// planner, FN3-10 / #56). An instruction may not read and write the same slot,
/// and may not write one slot twice: that is how the alpha plan rules out
/// hidden aliasing.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionPlan {
    instructions: Vec<Instruction>,
    value_count: u32,
}

impl ExecutionPlan {
    /// Validate and construct a plan.
    pub fn new(
        instructions: Vec<Instruction>,
        value_count: u32,
    ) -> Result<ExecutionPlan, Fn3Error> {
        if value_count == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "ExecutionPlan::new",
                detail: "a plan needs at least one value slot".to_string(),
            });
        }
        for (index, instruction) in instructions.iter().enumerate() {
            if instruction.outputs.is_empty() {
                return Err(Fn3Error::InvalidShape {
                    context: "ExecutionPlan::new",
                    detail: format!(
                        "instruction {index} ({}) has no output slot",
                        instruction.op
                    ),
                });
            }
            for &slot in instruction.inputs.iter().chain(instruction.outputs.iter()) {
                if slot >= value_count {
                    return Err(Fn3Error::OutOfBounds {
                        what: "ExecutionPlan::new (value slot)",
                        index: slot as u64,
                        len: value_count as u64,
                    });
                }
            }
            for (position, &slot) in instruction.outputs.iter().enumerate() {
                if instruction.inputs.contains(&slot) {
                    return Err(Fn3Error::InvalidShape {
                        context: "ExecutionPlan::new",
                        detail: format!(
                            "instruction {index} ({}) reads and writes value slot {slot}",
                            instruction.op
                        ),
                    });
                }
                if instruction.outputs[..position].contains(&slot) {
                    return Err(Fn3Error::InvalidShape {
                        context: "ExecutionPlan::new",
                        detail: format!(
                            "instruction {index} ({}) writes value slot {slot} twice",
                            instruction.op
                        ),
                    });
                }
            }
        }
        Ok(ExecutionPlan {
            instructions,
            value_count,
        })
    }

    /// The instructions, in execution order.
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Number of value slots the plan addresses.
    pub fn value_count(&self) -> u32 {
        self.value_count
    }

    /// Reject plans that contain an op no backend can run.
    ///
    /// `Partial` support is accepted here; the op itself must reject the shapes
    /// it does not implement. Use [`crate::fn3::backend::Support::require_full`]
    /// when a plan must be guaranteed to run without falling back to a
    /// decomposed path.
    pub fn check_supported(&self, backend: &dyn Backend) -> Result<(), Fn3Error> {
        for (index, instruction) in self.instructions.iter().enumerate() {
            backend
                .supports(instruction.op)
                .require_supported(instruction.op, backend.name())
                .map_err(|error| context_note(error, instruction.op, index))?;
        }
        Ok(())
    }

    /// Execute every instruction against `slots`.
    ///
    /// `slots` must have exactly [`ExecutionPlan::value_count`] entries and
    /// every slot read by the plan must be `Some`. Output slots are taken out
    /// of the table, written through [`Tensor::as_mut_view`] and put back
    /// afterwards, including on failure, so a failed instruction does not
    /// destroy the caller's tensors.
    pub fn execute(
        &self,
        backend: &dyn Backend,
        slots: &mut [Option<Tensor>],
    ) -> Result<(), Fn3Error> {
        if slots.len() != self.value_count as usize {
            return Err(Fn3Error::CapacityExceeded {
                what: "execution plan value slots",
                requested: slots.len() as u64,
                capacity: self.value_count as u64,
            });
        }
        for (index, instruction) in self.instructions.iter().enumerate() {
            execute_instruction(backend, slots, instruction)
                .map_err(|error| context_note(error, instruction.op, index))?;
        }
        Ok(())
    }
}

fn execute_instruction(
    backend: &dyn Backend,
    slots: &mut [Option<Tensor>],
    instruction: &Instruction,
) -> Result<(), Fn3Error> {
    for &slot in &instruction.outputs {
        if slots[slot as usize].is_none() {
            return Err(Fn3Error::SessionState {
                action: "ExecutionPlan::execute",
                phase: "running",
                detail: format!(
                    "output value slot {slot} for op {} is empty; allocate it before running the plan",
                    instruction.op
                ),
            });
        }
    }
    let mut taken: Vec<Tensor> = instruction
        .outputs
        .iter()
        .map(|&slot| {
            slots[slot as usize]
                .take()
                .expect("output slots were checked to be present")
        })
        .collect();

    let result = (|| {
        let mut input_views = Vec::with_capacity(instruction.inputs.len());
        for &slot in &instruction.inputs {
            let tensor = slots[slot as usize]
                .as_ref()
                .ok_or_else(|| Fn3Error::SessionState {
                    action: "ExecutionPlan::execute",
                    phase: "running",
                    detail: format!("input value slot {slot} for op {} is empty", instruction.op),
                })?;
            input_views.push(tensor.as_view()?);
        }
        let mut output_views = Vec::with_capacity(taken.len());
        for tensor in taken.iter_mut() {
            output_views.push(tensor.as_mut_view()?);
        }
        backend.execute_op(
            instruction.op,
            &input_views,
            &mut output_views,
            &instruction.params,
        )
    })();

    for (&slot, tensor) in instruction.outputs.iter().zip(taken) {
        slots[slot as usize] = Some(tensor);
    }
    result
}

/// Add the plan position to an unsupported-op diagnostic.
fn context_note(error: Fn3Error, op: OpId, index: usize) -> Fn3Error {
    match error {
        Fn3Error::UnsupportedOp {
            op,
            backend,
            reason,
        } => Fn3Error::UnsupportedOp {
            op,
            backend,
            reason: format!("{reason} (at instruction {index}, op {op})"),
        },
        other => {
            let _ = op;
            other
        }
    }
}

/// Immutable, `Arc`-shared weight tensors.
///
/// Weights are stored behind `Arc<Tensor>` so a `PreparedModel` and every
/// session clone share one copy of every weight buffer. There is no API that
/// mutates a weight: a plan reads weights through [`TensorView`] only.
///
/// [`TensorView`]: crate::fn3::tensor::TensorView
#[derive(Debug, Clone, Default)]
pub struct ModelWeights {
    tensors: Vec<Arc<Tensor>>,
    names: Vec<String>,
}

impl ModelWeights {
    /// Construct from unnamed tensors. All weights must share one device.
    pub fn new(tensors: Vec<Tensor>) -> Result<ModelWeights, Fn3Error> {
        let named: Vec<(String, Tensor)> = tensors
            .into_iter()
            .enumerate()
            .map(|(index, tensor)| (format!("weight_{index}"), tensor))
            .collect();
        ModelWeights::with_names(named)
    }

    /// Construct from named tensors. All weights must share one device.
    pub fn with_names(tensors: Vec<(String, Tensor)>) -> Result<ModelWeights, Fn3Error> {
        let mut expected: Option<DeviceId> = None;
        for (_name, tensor) in &tensors {
            match expected {
                None => expected = Some(tensor.device()),
                Some(device) if device == tensor.device() => {}
                Some(device) => {
                    return Err(Fn3Error::DeviceMismatch {
                        op: None,
                        expected: device,
                        actual: tensor.device(),
                    })
                }
            }
        }
        let mut names = Vec::with_capacity(tensors.len());
        let mut handles = Vec::with_capacity(tensors.len());
        for (name, tensor) in tensors {
            names.push(name);
            handles.push(Arc::new(tensor));
        }
        Ok(ModelWeights {
            tensors: handles,
            names,
        })
    }

    /// Number of weights.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// `true` if there are no weights.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// A weight by index.
    pub fn get(&self, index: usize) -> Option<&Tensor> {
        self.tensors.get(index).map(|tensor| tensor.as_ref())
    }

    /// A weight by name.
    pub fn named(&self, name: &str) -> Option<&Tensor> {
        self.names
            .iter()
            .position(|candidate| candidate == name)
            .and_then(|index| self.get(index))
    }

    /// The name of a weight, if it has one.
    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    /// The shared handle of a weight by index.
    pub fn handle(&self, index: usize) -> Option<&Arc<Tensor>> {
        self.tensors.get(index)
    }

    /// The device all weights live on, if any.
    pub fn device(&self) -> Option<DeviceId> {
        self.tensors.first().map(|tensor| tensor.device())
    }
}

/// Layout of the key and value cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheLayout {
    /// `[batch, n_kv_heads, capacity_tokens, head_dim]`, keys and values
    /// identical. The innermost axis is `head_dim`, which keeps an appended
    /// token's vector contiguous.
    BatchHeadsCapacityHeadDim,
}

/// Largest layer count a KV cache configuration may declare.
///
/// This is a sanity bound: it keeps a corrupt or hostile model header from
/// asking a session to build an unbounded number of cache buffers before the
/// allocator reports a failure.
pub const MAX_KV_LAYERS: u64 = 1024;

/// Geometry and dtype of one layer's key/value cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    /// Number of transformer layers that each own a key and value cache.
    pub layers: u64,
    /// Cache capacity in tokens (the safety bound, not the live extent).
    pub capacity_tokens: u64,
    /// Batch size.
    pub batch: u64,
    /// Key/value head count.
    pub n_kv_heads: u64,
    /// Per-head width.
    pub head_dim: u64,
    /// Element type of the cache.
    pub dtype: DType,
    /// Layout of keys and values.
    pub layout: KvCacheLayout,
}

impl KvCacheConfig {
    /// Validate against the model envelope and compute nothing.
    pub fn validate(&self, envelope: &ShapeEnvelope) -> Result<(), Fn3Error> {
        if self.layers == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "KvCacheConfig::validate",
                detail: "a model with a KV cache needs at least one layer".to_string(),
            });
        }
        if self.layers > MAX_KV_LAYERS {
            return Err(Fn3Error::CapacityExceeded {
                what: "KV cache layer count",
                requested: self.layers,
                capacity: MAX_KV_LAYERS,
            });
        }
        if self.batch == 0 || self.n_kv_heads == 0 || self.head_dim == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "KvCacheConfig::validate",
                detail: format!(
                    "batch, n_kv_heads and head_dim must be positive, found {}, {}, {}",
                    self.batch, self.n_kv_heads, self.head_dim
                ),
            });
        }
        let total = envelope.get(Symbol::Total);
        if self.capacity_tokens > total.max() {
            return Err(Fn3Error::CapacityExceeded {
                what: "KV cache capacity vs envelope total",
                requested: self.capacity_tokens,
                capacity: total.max(),
            });
        }
        DimBound::new(self.capacity_tokens, self.capacity_tokens)?;
        Ok(())
    }

    /// Shape of one cache buffer.
    pub fn shape(&self) -> Shape {
        Shape::from_dims([
            self.batch,
            self.n_kv_heads,
            self.capacity_tokens,
            self.head_dim,
        ])
    }

    /// Bytes of one cache buffer (keys or values).
    pub fn buffer_bytes(&self) -> Result<u64, Fn3Error> {
        let elements = self.shape().numel()?;
        elements
            .checked_mul(self.dtype.size_bytes() as u64)
            .ok_or_else(|| Fn3Error::ShapeOverflow {
                context: "KvCacheConfig::buffer_bytes",
                axis: None,
                detail: format!("{} elements of {}", self.shape(), self.dtype),
            })
    }
}

/// Mutable per-layer cache state owned by a [`Session`].
///
/// `capacity_tokens` is the allocation bound; `live_len` is the semantic
/// extent. Reads must never address beyond `live_len`. FN3-12 (#58) implements
/// the append-and-commit protocol; this skeleton allocates and resets the
/// buffers so ownership and isolation are already real.
#[derive(Debug)]
pub struct KvCacheState {
    config: KvCacheConfig,
    keys: Tensor,
    values: Tensor,
    live_len: u64,
}

impl KvCacheState {
    /// Allocate zero-filled key and value buffers at full capacity.
    pub fn new(config: KvCacheConfig, device: DeviceId) -> Result<KvCacheState, Fn3Error> {
        let shape = config.shape();
        let keys = Tensor::zeros(shape.clone(), config.dtype, device)?;
        let values = Tensor::zeros(shape, config.dtype, device)?;
        Ok(KvCacheState {
            config,
            keys,
            values,
            live_len: 0,
        })
    }

    /// Configuration.
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Key buffer, shape `[batch, n_kv_heads, capacity, head_dim]`.
    pub fn keys(&self) -> &Tensor {
        &self.keys
    }

    /// Value buffer, shape `[batch, n_kv_heads, capacity, head_dim]`.
    pub fn values(&self) -> &Tensor {
        &self.values
    }

    /// Allocated capacity in tokens.
    pub fn capacity(&self) -> u64 {
        self.config.capacity_tokens
    }

    /// Live semantic extent in tokens.
    pub fn live_len(&self) -> u64 {
        self.live_len
    }

    /// Clear the live extent without reallocating or zeroing bytes.
    ///
    /// Stale bytes beyond `live_len` are never readable through the session
    /// contract; a reader uses the live extent, not the buffer length.
    pub fn reset(&mut self) {
        self.live_len = 0;
    }

    /// Check that `additional` more tokens fit, without mutating anything.
    pub fn check_capacity(&self, additional: u64) -> Result<(), Fn3Error> {
        let requested =
            self.live_len
                .checked_add(additional)
                .ok_or_else(|| Fn3Error::ShapeOverflow {
                    context: "KvCacheState::check_capacity",
                    axis: None,
                    detail: format!("{} + {additional} exceeds u64::MAX", self.live_len),
                })?;
        if requested > self.config.capacity_tokens {
            return Err(Fn3Error::CapacityExceeded {
                what: "KV cache live extent",
                requested,
                capacity: self.config.capacity_tokens,
            });
        }
        Ok(())
    }
}

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

/// Mutable inference state for one conversation.
///
/// # Concurrency
///
/// A session is the unit of exclusive mutable state. One session must be
/// stepped by one thread at a time: it is `Send` (it can move between threads)
/// but deliberately not `Sync` (it cannot be shared by reference). Use one
/// session per concurrent request; they share the [`PreparedModel`] and nothing
/// else.
///
/// # Atomicity
///
/// A failed step leaves the KV cache live extents and the step counter
/// unchanged. `reset` and `cache_len` are always consistent with the last
/// successful step.
pub struct Session<'model> {
    model: &'model PreparedModel,
    caches: Vec<KvCacheState>,
    step: u64,
    phase: SessionPhase,
    /// Makes `Session` `!Sync` on purpose. See the type-level docs.
    _not_sync: PhantomData<Cell<()>>,
}

impl<'model> Session<'model> {
    /// Create a session over `model`, allocating its KV cache buffers.
    ///
    /// Fails if the model's cache geometry is invalid or a buffer does not fit
    /// the device. Two sessions created from the same model never share a byte
    /// of mutable state.
    pub fn new(model: &'model PreparedModel) -> Result<Session<'model>, Fn3Error> {
        let mut caches = Vec::new();
        if let Some(config) = model.kv_config() {
            config.validate(&model.envelope)?;
            caches.reserve(config.layers as usize);
            for _ in 0..config.layers {
                caches.push(KvCacheState::new(*config, model.device())?);
            }
        }
        Ok(Session {
            model,
            caches,
            step: 0,
            phase: SessionPhase::Empty,
            _not_sync: PhantomData,
        })
    }

    /// The immutable model this session runs.
    pub fn model(&self) -> &'model PreparedModel {
        self.model
    }

    /// The execution plan being run.
    pub fn plan(&self) -> &ExecutionPlan {
        self.model.plan()
    }

    /// Current lifecycle phase.
    pub fn phase(&self) -> SessionPhase {
        self.phase
    }

    /// Number of successfully completed steps.
    ///
    /// Incremented only after a step's outputs and cache updates are committed;
    /// a failed step must not change it.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Live KV extent of the first layer, or 0 when the model has no cache.
    pub fn cache_len(&self) -> u64 {
        self.caches.first().map(KvCacheState::live_len).unwrap_or(0)
    }

    /// Per-layer cache state. Read-only: mutation happens inside steps.
    pub fn caches(&self) -> &[KvCacheState] {
        &self.caches
    }

    /// Consume a prompt and populate the KV cache.
    ///
    /// Contract: requires [`SessionPhase::Empty`]; on success returns the
    /// logits for the final prompt position and sets
    /// `cache_len() == tokens.len()` with phase [`SessionPhase::Ready`]. On any
    /// error the cache and counter are unchanged and the phase stays `Empty`.
    ///
    /// Owner: FN3-12 (#58) with the CPU kernels from FN3-07 (#53).
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
        for cache in &self.caches {
            cache.check_capacity(tokens.len() as u64)?;
        }
        Err(Fn3Error::NotImplemented {
            what: "Session::prefill",
            owner: "FN3-12 (#58)",
        })
    }

    /// Consume one or more new tokens, appending to the KV cache.
    ///
    /// Contract: requires [`SessionPhase::Ready`]; on success advances
    /// `cache_len()` by `tokens.len()` and increments [`Session::step`] exactly
    /// once. On any error the cache live extents and the counter are unchanged.
    ///
    /// Owner: FN3-12 (#58).
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
        for cache in &self.caches {
            cache.check_capacity(tokens.len() as u64)?;
        }
        Err(Fn3Error::NotImplemented {
            what: "Session::decode",
            owner: "FN3-12 (#58)",
        })
    }

    /// Clear all cache live extents and return to [`SessionPhase::Empty`].
    ///
    /// Buffers are kept, not reallocated; the step counter returns to 0.
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
        self.step = 0;
        self.phase = SessionPhase::Empty;
    }
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("phase", &self.phase)
            .field("step", &self.step)
            .field("caches", &self.caches.len())
            .field("cache_len", &self.cache_len())
            .finish_non_exhaustive()
    }
}

/// An immutable, shareable prepared model.
///
/// Holds the plan, the weights and the declared dynamic bounds. Construct once,
/// share with `Arc`, create one [`Session`] per concurrent inference stream.
#[derive(Debug, Clone)]
pub struct PreparedModel {
    plan: Arc<ExecutionPlan>,
    weights: Arc<ModelWeights>,
    device: DeviceId,
    envelope: ShapeEnvelope,
    kv_config: Option<KvCacheConfig>,
}

impl PreparedModel {
    /// Assemble a model, validating device agreement and the envelope.
    ///
    /// Does not consult a backend: call [`ExecutionPlan::check_supported`] with
    /// the backend that will run it (a `Session` does this before its first
    /// step, FN3-12 / #58).
    pub fn new(
        plan: ExecutionPlan,
        weights: ModelWeights,
        device: DeviceId,
        envelope: ShapeEnvelope,
        kv_config: Option<KvCacheConfig>,
    ) -> Result<PreparedModel, Fn3Error> {
        envelope.validate()?;
        if let Some(weights_device) = weights.device() {
            if weights_device != device {
                return Err(Fn3Error::DeviceMismatch {
                    op: None,
                    expected: device,
                    actual: weights_device,
                });
            }
        }
        if let Some(config) = kv_config {
            config.validate(&envelope)?;
            if !device.is_cpu() {
                // The alpha build can only allocate host KV buffers. Report it
                // instead of silently allocating on the CPU.
                return Err(
                    crate::fn3::device::DeviceResolveError::unavailable_device(device).into(),
                );
            }
        }
        Ok(PreparedModel {
            plan: Arc::new(plan),
            weights: Arc::new(weights),
            device,
            envelope,
            kv_config,
        })
    }

    /// The execution plan.
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// The immutable weights.
    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// The device the model lives on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The declared dynamic bounds.
    pub fn envelope(&self) -> ShapeEnvelope {
        self.envelope
    }

    /// The KV cache geometry, for models that have one.
    pub fn kv_config(&self) -> Option<&KvCacheConfig> {
        self.kv_config.as_ref()
    }

    /// Run the plan against pre-allocated value slots.
    ///
    /// See [`ExecutionPlan::execute`]. This is the model-level entry point
    /// FN3-12 (#58) wraps with cache append and counter commit.
    pub fn execute(
        &self,
        backend: &dyn Backend,
        slots: &mut [Option<Tensor>],
    ) -> Result<(), Fn3Error> {
        if backend.device() != self.device {
            return Err(Fn3Error::DeviceMismatch {
                op: None,
                expected: self.device,
                actual: backend.device(),
            });
        }
        self.plan.execute(backend, slots)
    }

    /// Check a concrete dynamic shape against the envelope.
    pub fn check_dims(&self, dims: &LiveDims) -> Result<(), Fn3Error> {
        self.envelope.check(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fn3::backend::cpu::CpuBackend;
    use crate::fn3::shape::Shape;

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
                dtype: DType::F32,
                layout: KvCacheLayout::BatchHeadsCapacityHeadDim,
            }),
        )
        .unwrap()
    }

    #[test]
    fn plan_runs_add_and_keeps_slots_on_failure() {
        let model = small_model();
        let backend = CpuBackend::new();
        let lhs = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
        let rhs = Tensor::from_f32(Shape::from_dims([2]), &[10.0, 20.0]).unwrap();
        let out = Tensor::zeros(Shape::from_dims([2]), DType::F32, DeviceId::Cpu).unwrap();
        let mut slots = vec![Some(lhs), Some(rhs), Some(out)];
        model.execute(&backend, &mut slots).unwrap();
        assert_eq!(
            slots[2].as_ref().unwrap().as_f32_slice().unwrap(),
            &[11.0, 22.0]
        );
    }

    #[test]
    fn plan_rejects_read_write_aliasing() {
        let error = ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![0])], 2)
            .unwrap_err();
        assert!(error.to_string().contains("reads and writes"), "{error}");
    }

    #[test]
    fn session_isolation_and_phase_rules() {
        let model = small_model();
        let mut first = Session::new(&model).unwrap();
        let second = Session::new(&model).unwrap();
        assert_eq!(first.cache_len(), 0);
        assert_eq!(first.phase(), SessionPhase::Empty);
        let error = first.decode(&[1]).unwrap_err();
        assert!(matches!(error, Fn3Error::SessionState { .. }));
        let error = first.prefill(&[1, 2]).unwrap_err();
        assert!(matches!(error, Fn3Error::NotImplemented { .. }));
        assert_eq!(first.step(), 0);
        assert_eq!(first.caches().len(), 1);
        assert_ne!(
            first.caches()[0].keys().storage().as_bytes().as_ptr(),
            second.caches()[0].keys().storage().as_bytes().as_ptr()
        );
        first.reset();
        assert_eq!(first.phase(), SessionPhase::Empty);
    }
}
