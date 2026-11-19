//! Prepared-model artifacts and loaded models for Python.
//!
//! # What a Python caller sees
//!
//! ```text
//! prepare(onnx, artifact)   ONNX ──► versioned artifact        (writes a file)
//! inspect(artifact)         bytes ──► geometry/size/checksum   (no device)
//! load(artifact, source=None) artifact ► Model ──► Session     (needs a device)
//! ```
//!
//! `prepare` is idempotent in the useful sense: if the destination already
//! holds exactly the artifact those arguments produce, it reports
//! `status="already_valid"` and writes nothing. "I wrote it" and "it was
//! already there and valid" are therefore different, observable outcomes
//! ([`PyPrepareReport::wrote`]).
//!
//! # The artifact is the input
//!
//! `load` reads one file. The prepared artifact carries the graph, the weights,
//! the geometry and the declared window, so the planner and the session are fed
//! from it and no ONNX file is needed. `source=<ONNX path>` is optional and
//! only adds the proof that the two files still describe the same model:
//! [`check_source_agrees`] compares provenance, geometry, every value, every
//! node and every weight byte before anything executes, and a disagreement is
//! refused. The artifact's bytes are what run either way — with no source they
//! are used directly, with a source they are first proved byte-identical — so
//! the two paths cannot drift numerically.
//!
//! # Device selection
//!
//! The device is resolved *before* any file is read, so asking for a Vulkan
//! adapter on a build without the feature fails immediately with the request,
//! the reason and the list of available devices — never with a CPU result.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::fn3::backend::cpu::CpuBackend;
#[cfg(feature = "vulkan")]
use crate::fn3::backend::vulkan::VulkanBackend;
use crate::fn3::backend::Backend;
use crate::fn3::device::DeviceId;
#[cfg(feature = "vulkan")]
use crate::fn3::device::DeviceRequest;
use crate::fn3::error::Fn3Error as CoreError;
use crate::fn3::import::{import_onnx_file, DecoderGeometry, ModelImport, PositionEmbedding};
use crate::fn3::model::format::{ArtifactMetadata, PreparedArtifact, PreparedHeader};
use crate::fn3::model::weights::WeightTensor;
use crate::fn3::model::{ModelWeights, Planner, PreparedModel};
use crate::fn3::shape::{LiveDims, ShapeEnvelope, Symbol};
use crate::fn3::tensor::TensorMeta;

use super::device::{describe as describe_device, resolve as resolve_device, PyDevice};
use super::errors;
use super::session::PySession;
use super::sha256;

/// Process-local identity, so two loaded models are distinguishable in a repr.
static MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Reporting types
// ---------------------------------------------------------------------------

/// Architecture geometry detected by the importer and carried by the artifact.
#[pyclass(module = "fastnn._core.fn3", name = "Geometry", frozen)]
pub struct PyGeometry {
    layers: u32,
    hidden_size: u32,
    mlp_hidden_size: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    max_positions: u32,
    layer_norm_epsilon: Option<f32>,
}

impl PyGeometry {
    fn of(geometry: &DecoderGeometry) -> PyGeometry {
        // `Learned` is the only position-embedding kind this build defines.
        let PositionEmbedding::Learned { max_positions, .. } = geometry.positions;
        PyGeometry {
            layers: geometry.layers,
            hidden_size: geometry.hidden_size,
            mlp_hidden_size: geometry.mlp_hidden_size,
            query_heads: geometry.query_heads,
            kv_heads: geometry.kv_heads,
            head_dim: geometry.head_dim,
            vocab_size: geometry.vocab_size,
            max_positions,
            layer_norm_epsilon: geometry.layer_norm_epsilon,
        }
    }
}

#[pymethods]
impl PyGeometry {
    /// Transformer blocks.
    #[getter]
    fn layers(&self) -> u32 {
        self.layers
    }

    /// Model width.
    #[getter]
    fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    /// MLP intermediate width.
    #[getter]
    fn mlp_hidden_size(&self) -> u32 {
        self.mlp_hidden_size
    }

    /// Query heads.
    #[getter]
    fn query_heads(&self) -> u32 {
        self.query_heads
    }

    /// Key/value heads.
    #[getter]
    fn kv_heads(&self) -> u32 {
        self.kv_heads
    }

    /// Per-head width.
    #[getter]
    fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Vocabulary size, from the token embedding table.
    #[getter]
    fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Rows in the learned position table.
    #[getter]
    fn max_positions(&self) -> u32 {
        self.max_positions
    }

    /// LayerNorm epsilon when the importer detected it structurally.
    #[getter]
    fn layer_norm_epsilon(&self) -> Option<f32> {
        self.layer_norm_epsilon
    }

    /// KV cache bytes one session allocates per token (both buffers, all layers).
    #[getter]
    fn kv_bytes_per_token(&self) -> u64 {
        u64::from(self.layers) * 2 * u64::from(self.kv_heads) * u64::from(self.head_dim) * 4
    }

    fn __repr__(&self) -> String {
        self.render()
    }
}

impl PyGeometry {
    fn render(&self) -> String {
        format!(
            "Geometry(layers={}, hidden={}, heads={}/{}, head_dim={}, vocab={}, max_positions={})",
            self.layers,
            self.hidden_size,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.vocab_size,
            self.max_positions
        )
    }
}

/// The artifact's declared dynamic window, per symbol.
#[pyclass(module = "fastnn._core.fn3", name = "Envelope", frozen)]
pub struct PyEnvelope {
    bounds: [(u64, u64); Symbol::COUNT],
}

impl PyEnvelope {
    fn of(envelope: &ShapeEnvelope) -> PyEnvelope {
        let mut bounds = [(0, 0); Symbol::COUNT];
        for symbol in Symbol::ALL {
            let bound = envelope.get(symbol);
            bounds[symbol as usize] = (bound.min(), bound.max());
        }
        PyEnvelope { bounds }
    }

    fn bound(&self, symbol: Symbol) -> (u64, u64) {
        self.bounds[symbol as usize]
    }
}

#[pymethods]
impl PyEnvelope {
    /// `batch` bound, inclusive.
    #[getter]
    fn batch(&self) -> (u64, u64) {
        self.bound(Symbol::Batch)
    }

    /// `seq` bound: tokens in one call.
    #[getter]
    fn seq(&self) -> (u64, u64) {
        self.bound(Symbol::Seq)
    }

    /// `past` bound: tokens already cached.
    #[getter]
    fn past(&self) -> (u64, u64) {
        self.bound(Symbol::Past)
    }

    /// `total` bound: `past + seq`.
    #[getter]
    fn total(&self) -> (u64, u64) {
        self.bound(Symbol::Total)
    }

    /// Query-head count (fixed by the weights).
    #[getter]
    fn n_heads(&self) -> (u64, u64) {
        self.bound(Symbol::NHeads)
    }

    /// Key/value-head count (fixed by the weights).
    #[getter]
    fn n_kv_heads(&self) -> (u64, u64) {
        self.bound(Symbol::NKvHeads)
    }

    /// Per-head width (fixed by the weights).
    #[getter]
    fn head_dim(&self) -> (u64, u64) {
        self.bound(Symbol::HeadDim)
    }

    /// Every bound as `{symbol: (min, max)}`.
    fn as_dict(&self) -> std::collections::BTreeMap<&'static str, (u64, u64)> {
        Symbol::ALL
            .into_iter()
            .map(|symbol| (symbol.name(), self.bound(symbol)))
            .collect()
    }

    fn __repr__(&self) -> String {
        self.render()
    }
}

impl PyEnvelope {
    fn render(&self) -> String {
        let rendered: Vec<String> = Symbol::ALL
            .into_iter()
            .map(|symbol| {
                let (min, max) = self.bound(symbol);
                format!("{}={min}..={max}", symbol.name())
            })
            .collect();
        format!("Envelope({})", rendered.join(", "))
    }
}

/// Provenance recorded in the artifact.
#[pyclass(module = "fastnn._core.fn3", name = "Metadata", frozen)]
pub struct PyMetadata {
    ir_version: i64,
    opset_version: i64,
    opset_domain: String,
    producer: String,
    producer_version: String,
    graph_name: String,
}

impl PyMetadata {
    fn of(metadata: &ArtifactMetadata) -> PyMetadata {
        PyMetadata {
            ir_version: metadata.ir_version,
            opset_version: metadata.opset.version,
            opset_domain: metadata.opset.domain.clone(),
            producer: metadata.producer.clone(),
            producer_version: metadata.producer_version.clone(),
            graph_name: metadata.graph_name.clone(),
        }
    }
}

#[pymethods]
impl PyMetadata {
    /// ONNX IR version of the source model.
    #[getter]
    fn ir_version(&self) -> i64 {
        self.ir_version
    }

    /// Opset version of the source model's default domain.
    #[getter]
    fn opset_version(&self) -> i64 {
        self.opset_version
    }

    /// Opset domain (empty for the default domain).
    #[getter]
    fn opset_domain(&self) -> &str {
        &self.opset_domain
    }

    /// Exporter name.
    #[getter]
    fn producer(&self) -> &str {
        &self.producer
    }

    /// Exporter version.
    #[getter]
    fn producer_version(&self) -> &str {
        &self.producer_version
    }

    /// Source graph name.
    #[getter]
    fn graph_name(&self) -> &str {
        &self.graph_name
    }

    fn __repr__(&self) -> String {
        self.render()
    }
}

impl PyMetadata {
    fn render(&self) -> String {
        format!(
            "Metadata(producer={:?}, opset={}, ir_version={})",
            self.producer, self.opset_version, self.ir_version
        )
    }
}

/// The outcome of [`prepare`].
#[pyclass(module = "fastnn._core.fn3", name = "PrepareReport", frozen)]
pub struct PyPrepareReport {
    path: String,
    status: String,
    size_bytes: u64,
    sha256: String,
    payload_crc32: u32,
    weight_bytes: u64,
    geometry: Py<PyGeometry>,
    envelope: Py<PyEnvelope>,
    metadata: Py<PyMetadata>,
    elapsed_ms: f64,
    verified: bool,
    /// Differences from the file that was already there, when it differed and
    /// `overwrite=True` replaced it.
    replaced_differences: Vec<String>,
}

#[pymethods]
impl PyPrepareReport {
    /// Path that was written, or that already held this artifact.
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    /// `"created"`, `"replaced"` or `"already_valid"`.
    #[getter]
    fn status(&self) -> &str {
        &self.status
    }

    /// `True` when this call created or replaced the file.
    #[getter]
    fn wrote(&self) -> bool {
        self.status != "already_valid"
    }

    /// File size in bytes.
    #[getter]
    fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// SHA-256 of the file on disk, comparable with `sha256sum`.
    #[getter]
    fn sha256(&self) -> &str {
        &self.sha256
    }

    /// The artifact's own payload CRC-32, as stored in its header.
    #[getter]
    fn payload_crc32(&self) -> u32 {
        self.payload_crc32
    }

    /// Weight payload bytes the artifact carries.
    #[getter]
    fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Detected architecture geometry.
    #[getter]
    fn geometry(&self, py: Python<'_>) -> Py<PyGeometry> {
        Py::clone_ref(&self.geometry, py)
    }

    /// Declared dynamic window.
    #[getter]
    fn envelope(&self, py: Python<'_>) -> Py<PyEnvelope> {
        Py::clone_ref(&self.envelope, py)
    }

    /// Provenance.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> Py<PyMetadata> {
        Py::clone_ref(&self.metadata, py)
    }

    /// Wall time of the call in milliseconds.
    #[getter]
    fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
    }

    /// `True` when the file on disk is byte-identical to the artifact this call
    /// built (compared by SHA-256), for all three statuses.
    #[getter]
    fn verified(&self) -> bool {
        self.verified
    }

    /// What differed from the file that `overwrite=True` replaced.
    #[getter]
    fn replaced_differences(&self) -> Vec<String> {
        self.replaced_differences.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PrepareReport(status={:?}, path={:?}, size_bytes={}, sha256={:?})",
            self.status,
            self.path,
            self.size_bytes,
            &self.sha256[..16.min(self.sha256.len())]
        )
    }
}

/// What [`inspect`] learned about an artifact file.
#[pyclass(module = "fastnn._core.fn3", name = "ArtifactInfo", frozen)]
pub struct PyArtifactInfo {
    path: String,
    size_bytes: u64,
    sha256: String,
    format_version: u16,
    payload_bytes: u64,
    payload_crc32: u32,
    weight_bytes: u64,
    value_count: usize,
    node_count: usize,
    weight_count: usize,
    geometry: Py<PyGeometry>,
    envelope: Py<PyEnvelope>,
    metadata: Py<PyMetadata>,
    elapsed_ms: f64,
}

#[pymethods]
impl PyArtifactInfo {
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// SHA-256 of the file, comparable with `sha256sum`.
    #[getter]
    fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Prepared-model format version in the header.
    #[getter]
    fn format_version(&self) -> u16 {
        self.format_version
    }

    #[getter]
    fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// The artifact's own payload CRC-32.
    #[getter]
    fn payload_crc32(&self) -> u32 {
        self.payload_crc32
    }

    #[getter]
    fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    #[getter]
    fn value_count(&self) -> usize {
        self.value_count
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.node_count
    }

    #[getter]
    fn weight_count(&self) -> usize {
        self.weight_count
    }

    #[getter]
    fn geometry(&self, py: Python<'_>) -> Py<PyGeometry> {
        Py::clone_ref(&self.geometry, py)
    }

    #[getter]
    fn envelope(&self, py: Python<'_>) -> Py<PyEnvelope> {
        Py::clone_ref(&self.envelope, py)
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> Py<PyMetadata> {
        Py::clone_ref(&self.metadata, py)
    }

    #[getter]
    fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "ArtifactInfo(path={:?}, size_bytes={}, format_version={}, sha256={:?}, {})",
            self.path,
            self.size_bytes,
            self.format_version,
            &self.sha256[..16.min(self.sha256.len())],
            self.geometry.bind(py).borrow().render()
        )
    }
}

// ---------------------------------------------------------------------------
// Loaded model
// ---------------------------------------------------------------------------

/// A loaded, validated prepared model bound to one device.
///
/// Immutable and shareable by Python threads: it holds `Arc`s, so every
/// [`PySession`] created from it shares the weights and the backend and owns its
/// own KV cache.
#[pyclass(module = "fastnn._core.fn3", name = "Model", frozen)]
pub struct PyModel {
    model: Arc<PreparedModel>,
    import: Arc<ModelImport>,
    backend: Arc<dyn Backend>,
    device: DeviceId,
    capacity: u64,
    instance: u64,
    path: String,
    source: Option<String>,
    sha256: String,
    size_bytes: u64,
    geometry: Py<PyGeometry>,
    envelope: Py<PyEnvelope>,
    metadata: Py<PyMetadata>,
    weight_bytes: u64,
    weight_count: usize,
    value_count: usize,
    node_count: usize,
    kv_bytes: u64,
}

#[pymethods]
impl PyModel {
    /// Path of the artifact this model was loaded from.
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    /// Path of the ONNX source this model was verified against, when one was
    /// given; `None` for an artifact-only load.
    #[getter]
    fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// SHA-256 of the artifact file.
    #[getter]
    fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Artifact size in bytes.
    #[getter]
    fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// Resolved device, as a `Device`.
    #[getter]
    fn device(&self, py: Python<'_>) -> PyResult<Py<PyDevice>> {
        describe_device(py, self.device)
    }

    /// Architecture geometry.
    #[getter]
    fn geometry(&self, py: Python<'_>) -> Py<PyGeometry> {
        Py::clone_ref(&self.geometry, py)
    }

    /// Declared dynamic window.
    #[getter]
    fn envelope(&self, py: Python<'_>) -> Py<PyEnvelope> {
        Py::clone_ref(&self.envelope, py)
    }

    /// Provenance.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> Py<PyMetadata> {
        Py::clone_ref(&self.metadata, py)
    }

    /// KV cache capacity in tokens that each session allocates.
    #[getter]
    fn capacity(&self) -> u64 {
        self.capacity
    }

    /// KV cache bytes each session allocates (keys and values, all layers).
    #[getter]
    fn kv_bytes(&self) -> u64 {
        self.kv_bytes
    }

    /// Weight tensors the artifact carried.
    #[getter]
    fn weight_count(&self) -> usize {
        self.weight_count
    }

    /// Weight payload bytes.
    #[getter]
    fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Values in the imported graph.
    #[getter]
    fn value_count(&self) -> usize {
        self.value_count
    }

    /// Operations in the imported graph.
    #[getter]
    fn node_count(&self) -> usize {
        self.node_count
    }

    /// Open an independent session over this model.
    ///
    /// Each session owns its KV cache, plan cache and counters; two sessions
    /// share only the immutable weights and the backend.
    fn session(&self, py: Python<'_>) -> PyResult<Py<PySession>> {
        PySession::open(
            py,
            Arc::clone(&self.model),
            Arc::clone(&self.import),
            Arc::clone(&self.backend),
            self.device,
            self.capacity,
            self.instance,
        )
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Model(#{} path={:?} device={} capacity={} {})",
            self.instance,
            self.path,
            self.device,
            self.capacity,
            self.geometry.bind(py).borrow().render()
        )
    }
}

// ---------------------------------------------------------------------------
// prepare / inspect / load
// ---------------------------------------------------------------------------

/// Read a file, mapping the operating-system error to a typed exception.
fn read_file_py(py: Python<'_>, path: &str) -> PyResult<Vec<u8>> {
    let display = path.to_string();
    let owned = path.to_string();
    py.detach(move || std::fs::read(&owned)).map_err(|error| {
        errors::file_error(
            py,
            format!("cannot read {display:?}: {error}; check the path and its permissions"),
            display,
        )
    })
}

/// Import an ONNX model, releasing the GIL.
fn import_onnx(py: Python<'_>, path: &str) -> PyResult<ModelImport> {
    if !Path::new(path).exists() {
        return Err(errors::file_error(
            py,
            format!(
                "ONNX model {path:?} does not exist; pass the path of an .onnx file that was \
                 exported from PyTorch (opset 13..=17)"
            ),
            path.to_string(),
        ));
    }
    let owned = path.to_string();
    py.detach(move || import_onnx_file(&owned))
        .map_err(|error| errors::from_import(py, error))
}

/// The declared window, with defaults derived from the model's position table.
struct Window {
    envelope: ShapeEnvelope,
}

fn window_for(
    py: Python<'_>,
    import: &ModelImport,
    max_total: Option<u64>,
    max_seq: Option<u64>,
    max_past: Option<u64>,
) -> PyResult<Window> {
    // `Learned` is the only position-embedding kind this build defines; a new
    // kind must decide its own default window here.
    let PositionEmbedding::Learned { max_positions, .. } = import.geometry().positions;
    let max_positions = u64::from(max_positions);
    let total = max_total.unwrap_or(max_positions.min(1024));
    if total == 0 {
        return Err(errors::model_error(
            py,
            "max_total must be at least 2: a decode step needs a cached token and a new one"
                .to_string(),
        ));
    }
    let seq = max_seq.unwrap_or(total);
    let past = max_past.unwrap_or(total.saturating_sub(1));
    // The CPU backend is used only as the envelope's bound check: the artifact
    // is device-free and `Planner::envelope` reads nothing but the geometry.
    let backend = CpuBackend::new();
    let planner = Planner::new(import, &backend);
    let envelope = planner
        .envelope(1, seq, past, total)
        .map_err(|error| errors::from_core(py, error))?;
    Ok(Window { envelope })
}

/// Import an ONNX model and write (or confirm) a versioned artifact.
///
/// The destination is only rewritten when the bytes it holds would differ:
///
/// * nothing there → write it, `status="created"`;
/// * the same artifact → write nothing, `status="already_valid"`;
/// * a different artifact → `Fn3ModelExistsError` unless `overwrite=True`, in
///   which case `status="replaced"` and `replaced_differences` says what
///   changed.
///
/// The written file is re-read, re-parsed, re-validated and compared with the
/// SHA-256 of the bytes that were written, so a reported success is a verified
/// one.
#[pyfunction]
#[pyo3(signature = (onnx, artifact, *, max_total = None, max_seq = None, max_past = None, overwrite = false))]
pub fn prepare(
    py: Python<'_>,
    onnx: &str,
    artifact: &str,
    max_total: Option<u64>,
    max_seq: Option<u64>,
    max_past: Option<u64>,
    overwrite: bool,
) -> PyResult<Py<PyPrepareReport>> {
    let started = Instant::now();
    let import = import_onnx(py, onnx)?;
    let window = window_for(py, &import, max_total, max_seq, max_past)?;
    let built = py
        .detach(|| PreparedArtifact::from_import(&import, window.envelope))
        .map_err(|error| errors::from_format(py, error))?;
    let bytes = py
        .detach(|| built.to_bytes())
        .map_err(|error| errors::from_format(py, error))?;
    let digest = py.detach(|| sha256::hex(&bytes));

    let destination = artifact.to_string();
    let existing_path = destination.clone();
    let existing = match py.detach(move || std::fs::read(&existing_path)) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(errors::file_error(
                py,
                format!("cannot read {artifact:?}: {error}"),
                destination,
            ))
        }
    };

    let mut status = "created";
    let mut differences = Vec::new();
    if let Some(existing) = &existing {
        let existing_digest = sha256::hex(existing);
        if existing_digest == digest {
            status = "already_valid";
        } else {
            match PreparedArtifact::from_bytes(existing) {
                Ok(previous) => {
                    differences = describe_difference(&previous, &built);
                    if !overwrite {
                        return Err(errors::model_exists(
                            py,
                            format!(
                                "{artifact:?} already holds a valid prepared model that differs \
                                 from what this ONNX and these bounds produce ({}); pass \
                                 overwrite=True to replace it, or choose another path",
                                differences.join("; ")
                            ),
                            destination,
                            differences.clone(),
                        ));
                    }
                    status = "replaced";
                }
                Err(error) => {
                    if !overwrite {
                        return Err(errors::from_format(py, error));
                    }
                    differences = vec![format!(
                        "the previous file was not a valid artifact: {error}"
                    )];
                    status = "replaced";
                }
            }
        }
    }

    // Both paths prove the file on disk is this artifact: `already_valid` by the
    // digest comparison above, a write by reading it back and hashing it.
    let verified = true;
    if status != "already_valid" {
        py.detach(|| built.write_file(artifact))
            .map_err(|error| errors::from_format(py, error))?;
        // Read back what is actually on disk. A short or torn write must not be
        // reported as success.
        let on_disk = read_file_py(py, artifact)?;
        let on_disk_digest = py.detach(|| sha256::hex(&on_disk));
        if on_disk_digest != digest {
            return Err(errors::model_error(
                py,
                format!(
                    "{artifact:?} was written but its SHA-256 is {on_disk_digest}, not {digest}; \
                     the file is not the artifact this call built (a torn or replaced write)"
                ),
            ));
        }
        let reparsed = PreparedArtifact::from_bytes(&on_disk)
            .map_err(|error| errors::from_format(py, error))?;
        reparsed
            .validate()
            .map_err(|error| errors::from_format(py, error))?;
    }

    let header = PreparedHeader::parse(&bytes).map_err(|error| errors::from_format(py, error))?;
    let report = PyPrepareReport {
        path: destination,
        status: status.to_string(),
        size_bytes: bytes.len() as u64,
        sha256: digest,
        payload_crc32: header.payload_crc32,
        weight_bytes: built.weight_bytes(),
        geometry: Py::new(py, PyGeometry::of(built.geometry()))?,
        envelope: Py::new(py, PyEnvelope::of(&window.envelope))?,
        metadata: Py::new(py, PyMetadata::of(built.metadata()))?,
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
        verified,
        replaced_differences: differences,
    };
    Py::new(py, report)
}

/// What differs between two artifacts, for the replace/refuse message.
fn describe_difference(previous: &PreparedArtifact, wanted: &PreparedArtifact) -> Vec<String> {
    let mut differences = Vec::new();
    if previous.envelope() != wanted.envelope() {
        differences.push(format!(
            "envelope: existing {}, this call {}",
            PyEnvelope::of(&previous.envelope()).render(),
            PyEnvelope::of(&wanted.envelope()).render()
        ));
    }
    if previous.geometry() != wanted.geometry() {
        differences.push(format!(
            "geometry: existing {}, this call {}",
            PyGeometry::of(previous.geometry()).render(),
            PyGeometry::of(wanted.geometry()).render()
        ));
    }
    if previous.metadata() != wanted.metadata() {
        differences.push(format!(
            "provenance: existing {}, this call {}",
            PyMetadata::of(previous.metadata()).render(),
            PyMetadata::of(wanted.metadata()).render()
        ));
    }
    if previous.weights().len() != wanted.weights().len() {
        differences.push(format!(
            "{} weights vs {} weights",
            previous.weights().len(),
            wanted.weights().len()
        ));
    } else {
        for (index, (left, right)) in previous
            .weights()
            .iter()
            .zip(wanted.weights().iter())
            .enumerate()
        {
            if let Some(difference) = weight_difference(index, left, right) {
                differences.push(difference);
                break;
            }
        }
    }
    if previous.graph().value_count() != wanted.graph().value_count()
        || previous.graph().nodes().len() != wanted.graph().nodes().len()
    {
        differences.push(format!(
            "graph {} values/{} nodes vs {} values/{} nodes",
            previous.graph().value_count(),
            previous.graph().nodes().len(),
            wanted.graph().value_count(),
            wanted.graph().nodes().len()
        ));
    }
    if differences.is_empty() {
        differences.push("the artifacts differ in bytes but agree structurally".to_string());
    }
    differences
}

/// Report what an artifact file contains without opening a device or a session.
#[pyfunction]
#[pyo3(signature = (artifact,))]
pub fn inspect(py: Python<'_>, artifact: &str) -> PyResult<Py<PyArtifactInfo>> {
    let started = Instant::now();
    let path = artifact.to_string();
    let bytes = read_file_py(py, artifact)?;
    let parsed = py
        .detach(|| {
            PreparedArtifact::from_bytes(&bytes).and_then(|parsed| {
                parsed.validate()?;
                Ok(parsed)
            })
        })
        .map_err(|error| errors::from_format(py, error))?;
    let header = py
        .detach(|| PreparedHeader::parse(&bytes))
        .map_err(|error| errors::from_format(py, error))?;
    let info = PyArtifactInfo {
        path,
        size_bytes: bytes.len() as u64,
        sha256: py.detach(|| sha256::hex(&bytes)),
        format_version: header.version,
        payload_bytes: header.payload_bytes,
        payload_crc32: header.payload_crc32,
        weight_bytes: parsed.weight_bytes(),
        value_count: parsed.graph().value_count(),
        node_count: parsed.graph().nodes().len(),
        weight_count: parsed.weights().len(),
        geometry: Py::new(py, PyGeometry::of(parsed.geometry()))?,
        envelope: Py::new(py, PyEnvelope::of(&parsed.envelope()))?,
        metadata: Py::new(py, PyMetadata::of(parsed.metadata()))?,
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
    };
    Py::new(py, info)
}

/// What differs between two weights, phrased for a message.
fn weight_difference(index: usize, left: &WeightTensor, right: &WeightTensor) -> Option<String> {
    if left.name() != right.name() {
        return Some(format!(
            "weight {index}: the artifact holds `{}`, the source holds `{}`",
            left.name(),
            right.name()
        ));
    }
    if left.dtype() != right.dtype() {
        return Some(format!(
            "weight {index} `{}`: dtype {} vs {}",
            left.name(),
            left.dtype(),
            right.dtype()
        ));
    }
    if left.shape() != right.shape() {
        return Some(format!(
            "weight {index} `{}`: shape {} vs {}",
            left.name(),
            left.shape(),
            right.shape()
        ));
    }
    if left.bytes() != right.bytes() {
        return Some(format!(
            "weight {index} `{}`: the payload bytes differ ({} bytes each), so the source is a \
             different export or a modified file",
            left.name(),
            left.byte_len()
        ));
    }
    None
}

/// Check that an ONNX source describes the same model as the artifact.
///
/// This is the optional `source=` proof, not the input path: the artifact alone
/// plans and runs. A source that disagrees in provenance, geometry, values,
/// nodes or weight bytes is a mistake worth refusing rather than accepting as a
/// verification of a model that is half one file and half another.
fn check_source_agrees(
    py: Python<'_>,
    artifact: &PreparedArtifact,
    import: &ModelImport,
) -> PyResult<()> {
    let mut problems = Vec::new();
    if artifact.geometry() != import.geometry() {
        problems.push(format!(
            "geometry: artifact {}, source {}",
            PyGeometry::of(artifact.geometry()).render(),
            PyGeometry::of(import.geometry()).render()
        ));
    }
    let artifact_values = artifact.graph().values();
    let source_values = import.graph().values();
    if artifact_values != source_values {
        let first = artifact_values
            .iter()
            .zip(source_values.iter())
            .position(|(left, right)| left != right);
        problems.push(match first {
            Some(index) => {
                let left = &artifact_values[index];
                let right = &source_values[index];
                format!(
                    "value {index}: artifact `{}` {} {}, source `{}` {} {}",
                    left.name,
                    left.dtype,
                    left.shape.render(),
                    right.name,
                    right.dtype,
                    right.shape.render()
                )
            }
            None => format!(
                "values: artifact has {}, source has {}",
                artifact_values.len(),
                source_values.len()
            ),
        });
    }
    let artifact_nodes = artifact.graph().nodes();
    let source_nodes = import.graph().nodes();
    if artifact_nodes != source_nodes {
        let first = artifact_nodes
            .iter()
            .zip(source_nodes.iter())
            .position(|(left, right)| left != right);
        problems.push(match first {
            Some(index) => {
                let left = &artifact_nodes[index];
                let right = &source_nodes[index];
                format!(
                    "operation {index}: artifact {} ({}) reads {} writes {}, source {} ({}) reads \
                     {} writes {}",
                    left.op,
                    left.onnx_op,
                    left.inputs.len(),
                    left.outputs.len(),
                    right.op,
                    right.onnx_op,
                    right.inputs.len(),
                    right.outputs.len()
                )
            }
            None => format!(
                "operations: artifact has {}, source has {}",
                artifact_nodes.len(),
                source_nodes.len()
            ),
        });
    }
    let artifact_weights = artifact.weights();
    let source_weights = import.weights();
    if artifact_weights.len() != source_weights.len() {
        problems.push(format!(
            "weights: artifact has {}, source has {}",
            artifact_weights.len(),
            source_weights.len()
        ));
    } else {
        for (index, (left, right)) in artifact_weights
            .iter()
            .zip(source_weights.iter())
            .enumerate()
        {
            if let Some(difference) = weight_difference(index, left, right) {
                problems.push(difference);
                break;
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(errors::model_error(
            py,
            format!(
                "the ONNX source and the prepared artifact disagree ({}); the artifact is what \
                 would run, so loading both would execute a model that is half of each file — \
                 regenerate the artifact from this source with fastnn.fn3.prepare",
                problems.join("; ")
            ),
        ))
    }
}

/// Create the one backend a model and all of its sessions execute on.
fn open_backend(py: Python<'_>, device: DeviceId) -> PyResult<Arc<dyn Backend>> {
    match device {
        DeviceId::Cpu => Ok(Arc::new(CpuBackend::new())),
        #[cfg(feature = "vulkan")]
        DeviceId::Vulkan(index) => {
            let request = DeviceRequest::Vulkan {
                index: Some(index.index()),
            };
            let backend =
                VulkanBackend::open(&request).map_err(|error| errors::from_core(py, error))?;
            Ok(backend)
        }
        #[cfg(not(feature = "vulkan"))]
        DeviceId::Vulkan(_) => Err(errors::from_core(
            py,
            CoreError::DeviceUnavailable(
                crate::fn3::device::DeviceResolveError::unavailable_device(device),
            ),
        )),
    }
}

/// Re-tag weights for a device without copying bytes.
///
/// `PreparedArtifact::to_model_weights` builds host tensors tagged `cpu`; a
/// session on a Vulkan device needs the same bytes tagged for that device. This
/// is the alpha's host-staging contract, not a second compute path.
fn retag_weights(weights: ModelWeights, device: DeviceId) -> Result<ModelWeights, CoreError> {
    if device == DeviceId::Cpu {
        return Ok(weights);
    }
    let mut named = Vec::with_capacity(weights.len());
    for index in 0..weights.len() {
        let tensor = weights.get(index).ok_or(CoreError::OutOfBounds {
            what: "load (weight index)",
            index: index as u64,
            len: weights.len() as u64,
        })?;
        let meta = TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), device)?;
        let retagged = crate::fn3::tensor::Tensor::from_storage(tensor.storage().clone(), meta)?;
        named.push((
            weights
                .name(index)
                .ok_or(CoreError::OutOfBounds {
                    what: "load (weight name)",
                    index: index as u64,
                    len: weights.len() as u64,
                })?
                .to_string(),
            retagged,
        ));
    }
    ModelWeights::with_names(named)
}

/// Load a prepared artifact, bind it to an explicit device and return a model.
#[pyfunction]
#[pyo3(signature = (artifact, *, source = None, device = None, capacity = None))]
pub fn load(
    py: Python<'_>,
    artifact: &str,
    source: Option<String>,
    device: Option<Bound<'_, PyAny>>,
    capacity: Option<u64>,
) -> PyResult<Py<PyModel>> {
    // 1. The device first: a wrong request must fail before 28 MB of I/O.
    let device = resolve_device(py, device.as_ref())?;
    let backend = open_backend(py, device)?;

    // 2. The artifact bytes: one read, one hash, one parse.
    let bytes = read_file_py(py, artifact)?;
    let digest = py.detach(|| sha256::hex(&bytes));
    let size_bytes = bytes.len() as u64;
    let parsed = py
        .detach(|| {
            PreparedArtifact::from_bytes(&bytes).and_then(|parsed| {
                parsed.validate()?;
                Ok(parsed)
            })
        })
        .map_err(|error| errors::from_format(py, error))?;
    drop(bytes);

    // 3. The planner and session input. The artifact is the input: with no
    // `source`, the graph, weights, geometry and window it carries are exactly
    // what plans and runs. A `source` only adds the verification that the two
    // files still describe the same model.
    let geometry = parsed.geometry().clone();
    let envelope = parsed.envelope();
    let provenance = Py::new(py, PyMetadata::of(parsed.metadata()))?;
    let weight_bytes = parsed.weight_bytes();
    let weight_count = parsed.weights().len();
    let value_count = parsed.graph().value_count();
    let node_count = parsed.graph().nodes().len();

    let (import, source) = match source {
        Some(source) => {
            let import = import_onnx(py, &source)?;
            check_source_agrees(py, &parsed, &import)?;
            (import, Some(source))
        }
        None => (parsed.into_model_import(), None),
    };

    // 4. Capacity, the one session-scoped bound.
    let capacity = capacity.unwrap_or(envelope.get(Symbol::Total).max());
    if capacity == 0 {
        return Err(errors::capacity(
            py,
            format!(
                "capacity must be at least 1 token (the declared window is 1..={}); pass \
                 capacity= to choose a smaller cache than the artifact's window",
                envelope.get(Symbol::Total).max()
            ),
            "kv cache capacity",
            0,
            envelope.get(Symbol::Total).max(),
        ));
    }
    let kv_config = geometry.kv_cache_config(capacity, 1);
    kv_config
        .validate(&envelope)
        .map_err(|error| errors::from_core(py, error))?;

    // 5. Plan, weights, model.
    let import_arc = Arc::new(import);
    let backend_arc = backend;
    let planner = Planner::new(import_arc.as_ref(), backend_arc.as_ref());
    let dims = LiveDims::new(
        1,
        1,
        0,
        u64::from(geometry.query_heads),
        u64::from(geometry.kv_heads),
        u64::from(geometry.head_dim),
    )
    .map_err(|error| errors::from_core(py, error))?;
    let plan = planner
        .plan(&envelope, dims)
        .map_err(|error| errors::from_core(py, error))?
        .execution()
        .clone();
    // Weights come from the import that will run the plan. Without a source
    // that import *is* the artifact; with one, `check_source_agrees` proved
    // every weight byte identical, so the artifact is still what executes.
    let weights = py
        .detach(|| import_arc.to_model_weights())
        .map_err(|error| errors::from_core(py, error))?;
    let weights = retag_weights(weights, device).map_err(|error| errors::from_core(py, error))?;
    let model = PreparedModel::new(plan, weights, device, envelope, Some(kv_config))
        .map_err(|error| errors::from_core(py, error))?;

    let report = PyModel {
        model: Arc::new(model),
        import: import_arc,
        backend: backend_arc,
        device,
        capacity,
        instance: MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
        path: artifact.to_string(),
        source,
        sha256: digest,
        size_bytes,
        geometry: Py::new(py, PyGeometry::of(&geometry))?,
        envelope: Py::new(py, PyEnvelope::of(&envelope))?,
        metadata: provenance,
        weight_bytes,
        weight_count,
        value_count,
        node_count,
        kv_bytes: kv_bytes_for(&geometry, capacity),
    };
    Py::new(py, report)
}

/// Bytes one session's KV cache occupies for `capacity` tokens.
fn kv_bytes_for(geometry: &DecoderGeometry, capacity: u64) -> u64 {
    u64::from(geometry.layers)
        * 2
        * capacity
        * u64::from(geometry.kv_heads)
        * u64::from(geometry.head_dim)
        * 4
}

/// Register the model surface.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGeometry>()?;
    module.add_class::<PyEnvelope>()?;
    module.add_class::<PyMetadata>()?;
    module.add_class::<PyPrepareReport>()?;
    module.add_class::<PyArtifactInfo>()?;
    module.add_class::<PyModel>()?;
    module.add_function(wrap_pyfunction!(prepare, module)?)?;
    module.add_function(wrap_pyfunction!(inspect, module)?)?;
    module.add_function(wrap_pyfunction!(load, module)?)?;
    Ok(())
}
