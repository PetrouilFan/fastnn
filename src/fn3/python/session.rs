//! Sessions and logits for Python.
//!
//! # Ownership and the one `unsafe` block in this binding
//!
//! [`Session`] borrows its model, its import and its backend for its whole
//! life, because that is what the Rust API guarantees: a session can never
//! outlive the thing it executes on. A Python object must *own* its state, and
//! here the three referents are already heap allocations behind `Arc`s. So
//! [`OwnedSession`] stores the three `Arc`s next to the session and extends the
//! three borrows to `'static`; the `Arc`s are declared after the session, so
//! they are dropped after it, and nothing in this file ever hands a `'static`
//! reference out. The alternative — leaking the model per session — would cost
//! 28 MB per `Model.session()` and is not acceptable.
//!
//! # Concurrency
//!
//! A session is `Send` but deliberately not `Sync`: it is the unit of exclusive
//! mutable state. The Python object holds a `parking_lot::Mutex`, so two Python
//! threads that call the same session **serialise** (the second waits without
//! the GIL) instead of corrupting state or raising "already borrowed". Two
//! *different* sessions run in parallel: they share no mutable byte.
//!
//! # GIL
//!
//! `prefill` and `decode` release the GIL (`Python::detach`) around planning,
//! execution, the logits copy and the cache append. Everything that touches
//! Python objects — token extraction, the logits `bytes` copy, numpy conversion
//! — happens with the GIL held. The work the GIL is released for is exactly the
//! work that borrows no Python object: the closure captures `&Mutex<…>` and a
//! `Vec<u32>`, both `Send`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use crate::inference::backend::Backend;
use crate::inference::device::DeviceId;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error as CoreError;
use crate::inference::import::ModelImport;
use crate::inference::model::{PreparedModel, Session};
use crate::inference::tensor::Tensor;

use super::device::{describe as describe_device, PyDevice};
use super::errors;

/// Process-local session identity, for reprs and diagnostics.
static SESSION_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// Extend a borrow to `'static`.
///
/// # Safety
///
/// The caller must prove that the referent outlives every use of the returned
/// reference. This is the only place in the binding that extends a lifetime and
/// it is used for exactly one structure ([`OwnedSession`]), whose field order
/// makes the proof local.
unsafe fn extend_lifetime<T: ?Sized>(value: &T) -> &'static T {
    // SAFETY: `value` is a valid reference; the caller's contract is that the
    // referent outlives every use of the result. Reborrowing through a raw
    // pointer keeps the metadata of an unsized type (`dyn Backend`) intact.
    unsafe { &*(value as *const T) }
}

/// A session that owns everything it borrows.
///
/// Field order is load-bearing: `session` is dropped before the `Arc`s it
/// borrows from.
struct OwnedSession {
    session: Session<'static>,
    _model: Arc<PreparedModel>,
    _import: Arc<ModelImport>,
    _backend: Arc<dyn Backend>,
}

impl OwnedSession {
    fn open(
        model: Arc<PreparedModel>,
        import: Arc<ModelImport>,
        backend: Arc<dyn Backend>,
    ) -> Result<OwnedSession, CoreError> {
        // SAFETY: the three referents are heap allocations owned by the `Arc`s
        // moved into the returned `OwnedSession`. `Arc` pointees do not move
        // when the handles move, the struct is never cloned, and `session` is
        // declared (and therefore dropped) before the three handles, so every
        // borrow ends before the referent is freed. No `'static` reference is
        // returned, stored elsewhere, or exposed to Python.
        let session = unsafe {
            Session::with_runtime(
                extend_lifetime(model.as_ref()),
                extend_lifetime(import.as_ref()),
                extend_lifetime(backend.as_ref()),
            )?
        };
        Ok(OwnedSession {
            session,
            _model: model,
            _import: import,
            _backend: backend,
        })
    }
}

// ---------------------------------------------------------------------------
// Logits
// ---------------------------------------------------------------------------

/// Logits of the last position of a step, as an owned byte buffer.
///
/// The values are copied once out of the session's fresh output tensor into an
/// immutable `bytes` object, so the logits stay valid after the session is
/// reset or dropped, and `numpy.frombuffer` can view them without a second
/// copy. `np.asarray(logits)` and `logits.to_numpy()` both give that
/// zero-extra-copy, read-only `float32` view; `np.array(logits)` copies if a
/// writable array is wanted.
#[pyclass(module = "fastnn._core.inference", name = "Logits", frozen)]
pub struct PyLogits {
    bytes: Py<PyBytes>,
    shape: Vec<u64>,
    numel: u64,
}

impl PyLogits {
    fn of(py: Python<'_>, tensor: &Tensor) -> PyResult<PyLogits> {
        if tensor.dtype() != DType::F32 {
            return Err(errors::from_core(
                py,
                CoreError::UnsupportedDType {
                    op: None,
                    dtype: tensor.dtype(),
                    context: "Python logits conversion",
                },
            ));
        }
        let values = tensor
            .as_f32_slice()
            .map_err(|error| errors::from_core(py, error))?;
        // `f32` storage is 4-byte aligned and `u8` needs no alignment, so this
        // is the one cast that cannot fail; the copy into `bytes` is explicit.
        let raw: &[u8] = bytemuck::cast_slice(values);
        let shape: Vec<u64> = tensor.shape().dims().to_vec();
        Ok(PyLogits {
            bytes: PyBytes::new(py, raw).unbind(),
            shape,
            numel: values.len() as u64,
        })
    }
}

#[pymethods]
impl PyLogits {
    /// Shape of the logits as a tuple, `(batch, 1, vocab)` for the pinned model.
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        Ok(pyo3::types::PyTuple::new(py, self.shape.iter())?.unbind())
    }

    /// Element type, always `"float32"`.
    #[getter]
    fn dtype(&self) -> &'static str {
        "float32"
    }

    /// Number of elements.
    #[getter]
    fn numel(&self) -> u64 {
        self.numel
    }

    /// Payload size in bytes (`numel * 4`).
    #[getter]
    fn nbytes(&self) -> u64 {
        self.numel * 4
    }

    /// The raw little-endian `float32` payload.
    ///
    /// This is the one copy; `numpy.frombuffer(logits.data, "<f4")` views it.
    #[getter]
    fn data(&self, py: Python<'_>) -> Py<PyBytes> {
        Py::clone_ref(&self.bytes, py)
    }

    /// A read-only `float32` NumPy view over [`PyLogits::data`].
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let numpy = py.import("numpy")?;
        let dtype = numpy.getattr("float32")?;
        let array = numpy.call_method1("frombuffer", (self.bytes.bind(py), dtype))?;
        let shape: Vec<u64> = self.shape.clone();
        array.call_method1("reshape", (shape,))
    }

    /// NumPy's `__array__` protocol, so `np.asarray(logits)` works.
    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut array = self.to_numpy(py)?;
        if let Some(dtype) = dtype {
            array = array.call_method1("astype", (dtype,))?;
        }
        if copy == Some(true) {
            array = array.call_method0("copy")?;
        }
        Ok(array)
    }

    /// Index of the largest logit, i.e. the greedy next token.
    ///
    /// NaN logits are never selected; if every logit is NaN this raises
    /// `Fn3ExecutionError` rather than returning a misleading zero.
    fn argmax(&self, py: Python<'_>) -> PyResult<u64> {
        let bytes = self.bytes.bind(py).as_bytes();
        let mut best: Option<(u64, f32)> = None;
        for (index, chunk) in bytes.chunks_exact(4).enumerate() {
            let value = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if value.is_nan() {
                continue;
            }
            match best {
                Some((_, current)) if value <= current => {}
                _ => best = Some((index as u64, value)),
            }
        }
        best.map(|(index, _)| index).ok_or_else(|| {
            errors::from_core(
                py,
                CoreError::InvalidFormat {
                    format: "logits",
                    detail: "every logit is NaN; the model did not produce a prediction"
                        .to_string(),
                },
            )
        })
    }

    /// The logits as a nested Python list (copies), nested like `shape`.
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let bytes = self.bytes.bind(py).as_bytes();
        let flat: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        nest(py, &flat, &self.shape)
    }

    fn __repr__(&self) -> String {
        format!(
            "Logits(shape={:?}, dtype='float32', nbytes={})",
            self.shape, self.numel
        )
    }
}

// ---------------------------------------------------------------------------
// Session statistics
// ---------------------------------------------------------------------------

/// A snapshot of everything a session reports about itself.
#[pyclass(module = "fastnn._core.inference", name = "SessionStats", frozen)]
pub struct PySessionStats {
    phase: String,
    step: u64,
    tokens_consumed: u64,
    cache_len: u64,
    capacity: u64,
    kv_bytes: u64,
    plan_hits: u64,
    plan_misses: u64,
    plan_evictions: u64,
    plan_cache_len: usize,
    slot_allocations: u64,
    slot_reuses: u64,
    slot_bytes_allocated: u64,
    last_plan: Option<String>,
    last_plan_dims: Option<(u64, u64, u64, u64, u64, u64, u64)>,
    cache_addresses: Vec<u64>,
}

#[pymethods]
impl PySessionStats {
    /// `"empty"`, `"prefilling"` or `"ready"`.
    #[getter]
    fn phase(&self) -> &str {
        &self.phase
    }

    /// Successful `decode` calls (prefills do not count).
    #[getter]
    fn step(&self) -> u64 {
        self.step
    }

    /// Tokens consumed by successful `prefill` and `decode` calls.
    #[getter]
    fn tokens_consumed(&self) -> u64 {
        self.tokens_consumed
    }

    /// Live KV extent in tokens.
    #[getter]
    fn cache_len(&self) -> u64 {
        self.cache_len
    }

    /// KV capacity in tokens.
    #[getter]
    fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes the KV cache occupies.
    #[getter]
    fn kv_bytes(&self) -> u64 {
        self.kv_bytes
    }

    /// Plans served from this session's plan cache.
    #[getter]
    fn plan_hits(&self) -> u64 {
        self.plan_hits
    }

    /// Plans this session had to build.
    #[getter]
    fn plan_misses(&self) -> u64 {
        self.plan_misses
    }

    /// Plans evicted because the session's plan cache was full.
    #[getter]
    fn plan_evictions(&self) -> u64 {
        self.plan_evictions
    }

    /// Plans currently cached.
    #[getter]
    fn plan_cache_len(&self) -> usize {
        self.plan_cache_len
    }

    /// Instruction-output buffers allocated.
    #[getter]
    fn slot_allocations(&self) -> u64 {
        self.slot_allocations
    }

    /// Instruction-output buffers reused from the previous step.
    #[getter]
    fn slot_reuses(&self) -> u64 {
        self.slot_reuses
    }

    /// Bytes allocated for instruction outputs over this session's life.
    #[getter]
    fn slot_bytes_allocated(&self) -> u64 {
        self.slot_bytes_allocated
    }

    /// Hex identity of the plan the last successful step executed.
    #[getter]
    fn last_plan(&self) -> Option<&str> {
        self.last_plan.as_deref()
    }

    /// `(batch, seq, past, total, n_heads, n_kv_heads, head_dim)` of the last plan.
    #[getter]
    fn last_plan_dims(&self) -> Option<(u64, u64, u64, u64, u64, u64, u64)> {
        self.last_plan_dims
    }

    /// Addresses of this session's per-layer key buffers.
    ///
    /// Diagnostic only, and the direct proof that two sessions never share a
    /// KV allocation: the buffers are allocated once per session and never
    /// reallocated, so the addresses are stable for its life.
    #[getter]
    fn cache_addresses(&self) -> Vec<u64> {
        self.cache_addresses.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SessionStats(phase={:?}, cache_len={}/{}, step={}, tokens={}, plans={} hits/{} \
             misses/{} evicted)",
            self.phase,
            self.cache_len,
            self.capacity,
            self.step,
            self.tokens_consumed,
            self.plan_hits,
            self.plan_misses,
            self.plan_evictions
        )
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// One conversation's inference state.
///
/// Every session owns its KV cache, plan cache and counters. Two sessions over
/// the same model share the immutable weights and nothing else; the Python
/// object serialises concurrent calls on *one* session and lets different
/// sessions run in parallel.
#[pyclass(module = "fastnn._core.inference", name = "Session")]
pub struct PySession {
    inner: parking_lot::Mutex<OwnedSession>,
    device: DeviceId,
    capacity: u64,
    vocab_size: u32,
    instance: u64,
    model_instance: u64,
}

impl PySession {
    /// Open a session over an already-loaded model.
    ///
    /// The caller owns the three `Arc`s; they are moved into the session, which
    /// keeps them alive for its whole life (see [`OwnedSession`]).
    pub(crate) fn open(
        py: Python<'_>,
        model: Arc<PreparedModel>,
        import: Arc<ModelImport>,
        backend: Arc<dyn Backend>,
        device: DeviceId,
        capacity: u64,
        model_instance: u64,
    ) -> PyResult<Py<PySession>> {
        let vocab_size = import.geometry().vocab_size;
        let session = OwnedSession::open(model, import, backend)
            .map_err(|error| errors::from_core(py, error))?;
        Py::new(
            py,
            PySession {
                inner: parking_lot::Mutex::new(session),
                device,
                capacity,
                vocab_size,
                instance: SESSION_IDENTITY.fetch_add(1, Ordering::Relaxed),
                model_instance,
            },
        )
    }

    /// Read session state, waiting without the GIL if another thread is stepping.
    ///
    /// The fast path (no step running) takes the lock immediately. The slow path
    /// waits on the mutex with the GIL released, so a caller that polls
    /// `cache_len` from another thread does not freeze the interpreter for the
    /// duration of a step.
    fn read<R: Send>(&self, py: Python<'_>, f: impl FnOnce(&Session<'static>) -> R + Send) -> R {
        if let Some(guard) = self.inner.try_lock() {
            return f(&guard.session);
        }
        py.detach(move || {
            let guard = self.inner.lock();
            f(&guard.session)
        })
    }

    fn snapshot(&self, py: Python<'_>) -> PySessionStats {
        self.read(py, |session| PySessionStats {
            phase: session.phase().name().to_string(),
            step: session.step(),
            tokens_consumed: session.tokens_consumed(),
            cache_len: session.cache_len(),
            capacity: self.capacity,
            kv_bytes: session.kv_bytes(),
            plan_hits: session.plan_hits(),
            plan_misses: session.plan_misses(),
            plan_evictions: session.plan_evictions(),
            plan_cache_len: session.plan_cache_len(),
            slot_allocations: session.slot_allocations(),
            slot_reuses: session.slot_reuses(),
            slot_bytes_allocated: session.slot_bytes_allocated(),
            last_plan: session.last_plan().map(|plan| plan.to_hex()),
            last_plan_dims: session.last_plan_dims().map(|dims| {
                (
                    dims.batch,
                    dims.seq,
                    dims.past,
                    dims.total,
                    dims.n_heads,
                    dims.n_kv_heads,
                    dims.head_dim,
                )
            }),
            cache_addresses: session
                .caches()
                .iter()
                .map(|cache| cache.keys().storage().as_bytes().as_ptr() as u64)
                .collect(),
        })
    }
}

/// Nest a flat row-major slice into Python lists that follow `shape`.
fn nest(py: Python<'_>, values: &[f32], shape: &[u64]) -> PyResult<Py<PyAny>> {
    use pyo3::types::PyList;

    if shape.len() <= 1 {
        return Ok(PyList::new(py, values)?.into_any().unbind());
    }
    let stride = shape[1..]
        .iter()
        .map(|extent| *extent as usize)
        .product::<usize>()
        .max(1);
    let mut items = Vec::with_capacity(shape[0] as usize);
    for chunk in values.chunks(stride) {
        items.push(nest(py, chunk, &shape[1..])?);
    }
    Ok(PyList::new(py, items)?.into_any().unbind())
}

/// The largest token count one call may present.
///
/// Matches `DimBound::MAX_EXTENT`, the core's own policy cap on a dynamic
/// extent: no legal envelope is larger, so a longer list cannot be a prompt and
/// an untrusted `__len__` cannot make this loop unbounded.
const MAX_TOKENS: usize = 1 << 24;

/// Extract and validate token ids.
///
/// Any sequence of integers works: a list, a tuple, a `range`, and a
/// one-dimensional NumPy integer array. An array is not a `PySequence` (it
/// indexes through the mapping protocol), so the iterator protocol is the
/// fallback. A value outside `0..vocab_size` is `Fn3TokenError`, naming the
/// position, the value and the vocabulary; it is never truncated or wrapped into
/// range. A non-integer element is reported at its own position rather than as a
/// whole-list failure.
fn parse_tokens(py: Python<'_>, tokens: &Bound<'_, PyAny>, vocab_size: u32) -> PyResult<Vec<u32>> {
    use pyo3::types::PySequence;

    let type_name = tokens
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    let not_a_sequence = || {
        errors::token_error(
            py,
            format!(
                "tokens must be a sequence of integers (list, tuple or 1-D integer array), found \
                 {type_name}; each element is a token id in 0..{vocab_size}"
            ),
            -1,
            vocab_size,
        )
    };

    let items: Vec<Bound<'_, PyAny>> = if let Ok(sequence) = tokens.cast::<PySequence>() {
        let length = sequence.len().map_err(|_| not_a_sequence())?;
        if length > MAX_TOKENS {
            return Err(errors::token_error(
                py,
                format!(
                    "tokens has {length} elements, more than the {MAX_TOKENS}-token policy cap of \
                     one call; split the prompt across prefill and decode calls"
                ),
                -1,
                vocab_size,
            ));
        }
        // A custom `__len__` is caller-controlled: do not size a Vec from it.
        let mut collected = Vec::with_capacity(length.min(4096));
        for index in 0..length {
            collected.push(sequence.get_item(index).map_err(|_| not_a_sequence())?);
        }
        collected
    } else if let Ok(iterator) = tokens.try_iter() {
        let mut collected = Vec::new();
        for item in iterator {
            if collected.len() >= MAX_TOKENS {
                return Err(errors::token_error(
                    py,
                    format!(
                        "tokens has more than the {MAX_TOKENS}-token policy cap of one call; split \
                         the prompt across prefill and decode calls"
                    ),
                    -1,
                    vocab_size,
                ));
            }
            let item = item.map_err(|_| not_a_sequence())?;
            collected.push(item);
        }
        collected
    } else {
        return Err(not_a_sequence());
    };

    let mut parsed = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        let value: i64 = item.extract().map_err(|_| {
            errors::token_error(
                py,
                format!(
                    "token at position {index} is {item}, which is not an integer token id in \
                     0..{vocab_size}"
                ),
                -1,
                vocab_size,
            )
        })?;
        if value < 0 || value >= i64::from(vocab_size) {
            return Err(errors::token_error(
                py,
                format!(
                    "token id {value} at position {index} is outside 0..{vocab_size}; the model's \
                     token embedding has {vocab_size} rows"
                ),
                value,
                vocab_size,
            ));
        }
        parsed.push(value as u32);
    }
    Ok(parsed)
}

#[pymethods]
impl PySession {
    /// Consume a prompt, populate the cache and return the last position's logits.
    ///
    /// The GIL is released for planning, execution and the cache append.
    fn prefill(&self, py: Python<'_>, tokens: &Bound<'_, PyAny>) -> PyResult<Py<PyLogits>> {
        let tokens = parse_tokens(py, tokens, self.vocab_size)?;
        let inner = &self.inner;
        let tensor = py
            .detach(move || -> Result<Tensor, CoreError> {
                let mut guard = inner.lock();
                guard.session.prefill(&tokens)
            })
            .map_err(|error| errors::from_core(py, error))?;
        Py::new(py, PyLogits::of(py, &tensor)?)
    }

    /// Consume one or more new tokens and return the last position's logits.
    ///
    /// The GIL is released for planning, execution and the cache append.
    fn decode(&self, py: Python<'_>, tokens: &Bound<'_, PyAny>) -> PyResult<Py<PyLogits>> {
        let tokens = parse_tokens(py, tokens, self.vocab_size)?;
        let inner = &self.inner;
        let tensor = py
            .detach(move || -> Result<Tensor, CoreError> {
                let mut guard = inner.lock();
                guard.session.decode(&tokens)
            })
            .map_err(|error| errors::from_core(py, error))?;
        Py::new(py, PyLogits::of(py, &tensor)?)
    }

    /// Clear the cache extent and the counters; keep the buffers and the plans.
    fn reset(&self, py: Python<'_>) {
        py.detach(|| {
            let mut guard = self.inner.lock();
            guard.session.reset();
        })
    }

    /// Lifecycle phase: `"empty"`, `"prefilling"` or `"ready"`.
    #[getter]
    fn phase(&self, py: Python<'_>) -> String {
        self.read(py, |session| session.phase().name().to_string())
    }

    /// Successful `decode` calls.
    #[getter]
    fn step(&self, py: Python<'_>) -> u64 {
        self.read(py, |session| session.step())
    }

    /// Tokens consumed by successful `prefill` and `decode` calls.
    #[getter]
    fn tokens_consumed(&self, py: Python<'_>) -> u64 {
        self.read(py, |session| session.tokens_consumed())
    }

    /// Live KV extent in tokens.
    #[getter]
    fn cache_len(&self, py: Python<'_>) -> u64 {
        self.read(py, |session| session.cache_len())
    }

    /// KV capacity in tokens.
    #[getter]
    fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes the KV cache occupies.
    #[getter]
    fn kv_bytes(&self, py: Python<'_>) -> u64 {
        self.read(py, |session| session.kv_bytes())
    }

    /// Device this session executes on.
    #[getter]
    fn device(&self, py: Python<'_>) -> PyResult<Py<PyDevice>> {
        describe_device(py, self.device)
    }

    /// Process-local session identity.
    #[getter]
    fn instance(&self) -> u64 {
        self.instance
    }

    /// Process-local identity of the model this session runs.
    #[getter]
    fn model_instance(&self) -> u64 {
        self.model_instance
    }

    /// A consistent snapshot of every observable counter.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PySessionStats>> {
        Py::new(py, self.snapshot(py))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let stats = self.snapshot(py);
        format!(
            "Session(#{} model=#{} device={} phase={:?} cache_len={}/{})",
            self.instance,
            self.model_instance,
            self.device,
            stats.phase,
            stats.cache_len,
            self.capacity
        )
    }
}

/// Register the session surface.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLogits>()?;
    module.add_class::<PySessionStats>()?;
    module.add_class::<PySession>()?;
    Ok(())
}
