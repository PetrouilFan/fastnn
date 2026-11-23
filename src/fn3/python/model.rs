//! Prepared-model artifacts, loaded models and diagnostics for Python.
//!
//! ```text
//! prepare(onnx, artifact, target=..., precision=...)  ONNX -> one .fnn3 file
//! inspect(artifact)                                   describe without mapping
//! load(artifact, target=...)                          mmap -> Model
//! Model.session()                                     one independent Session
//! Session.run({name: values})                         one forward
//! doctor(onnx)                                        diagnose before prepare
//! ```
//!
//! The artifact is the input. `load` maps it, validates every section hash and
//! decodes the stored plan; no ONNX file is read and nothing is planned again.
//! `Model` is immutable and shareable; each `Session` owns its arena, worker
//! pool and mutable state.

use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::inference::device::DeviceId;
use crate::inference::digest::{sha256_hex, Sha256};
use crate::inference::import::import_onnx_graph_with_shapes;
use crate::inference::model::format::{
    inspect_file, resolve_targets, Artifact, ArtifactMetadata, Precision, PreparedArtifact, TARGETS,
};
use crate::inference::runtime::{Session as RuntimeSession, Threading};

use super::device::{describe as describe_device, resolve as resolve_device};
use super::errors;
use super::session::PySession;

/// Process-local identity, so two loaded models are distinguishable in a repr.
static MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Reporting types
// ---------------------------------------------------------------------------

/// Provenance recorded in the artifact.
#[pyclass(module = "fastnn._core.inference", name = "Metadata", frozen)]
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
        format!(
            "Metadata(producer={:?}, opset={}, ir_version={})",
            self.producer, self.opset_version, self.ir_version
        )
    }
}

/// One target plan, as a Python report.
#[pyclass(module = "fastnn._core.inference", name = "PlanInfo", frozen)]
pub struct PyPlanInfo {
    target: String,
    precision: String,
    segments: usize,
    unfused_segments: usize,
    fusions: usize,
    reorders: usize,
    reorder_bytes: u64,
    arena_bytes: u64,
    peak_bytes: u64,
    scalar_segments: usize,
    one_dnn_segments: usize,
    layout: String,
}

impl PyPlanInfo {
    fn of(plan: &crate::inference::model::format::PlanSummary) -> PyPlanInfo {
        PyPlanInfo {
            target: plan.target.clone(),
            precision: plan.precision.name().to_string(),
            segments: plan.segments,
            unfused_segments: plan.unfused_segments,
            fusions: plan.fusions,
            reorders: plan.reorders,
            reorder_bytes: plan.reorder_bytes,
            arena_bytes: plan.arena_bytes,
            peak_bytes: plan.peak_bytes,
            scalar_segments: plan.engines.scalar,
            one_dnn_segments: plan.engines.one_dnn,
            layout: plan.layout.name().to_string(),
        }
    }
}

#[pymethods]
impl PyPlanInfo {
    #[getter]
    fn target(&self) -> &str {
        &self.target
    }

    #[getter]
    fn precision(&self) -> &str {
        &self.precision
    }

    #[getter]
    fn segments(&self) -> usize {
        self.segments
    }

    #[getter]
    fn unfused_segments(&self) -> usize {
        self.unfused_segments
    }

    #[getter]
    fn fusions(&self) -> usize {
        self.fusions
    }

    #[getter]
    fn reorders(&self) -> usize {
        self.reorders
    }

    #[getter]
    fn reorder_bytes(&self) -> u64 {
        self.reorder_bytes
    }

    #[getter]
    fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    #[getter]
    fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    #[getter]
    fn scalar_segments(&self) -> usize {
        self.scalar_segments
    }

    #[getter]
    fn one_dnn_segments(&self) -> usize {
        self.one_dnn_segments
    }

    #[getter]
    fn layout(&self) -> &str {
        &self.layout
    }

    /// Every field as a dictionary.
    fn as_dict(&self) -> std::collections::BTreeMap<&'static str, String> {
        let mut map = std::collections::BTreeMap::new();
        map.insert("target", self.target.clone());
        map.insert("precision", self.precision.clone());
        map.insert("segments", self.segments.to_string());
        map.insert("unfused_segments", self.unfused_segments.to_string());
        map.insert("fusions", self.fusions.to_string());
        map.insert("reorders", self.reorders.to_string());
        map.insert("reorder_bytes", self.reorder_bytes.to_string());
        map.insert("arena_bytes", self.arena_bytes.to_string());
        map.insert("peak_bytes", self.peak_bytes.to_string());
        map.insert("scalar_segments", self.scalar_segments.to_string());
        map.insert("one_dnn_segments", self.one_dnn_segments.to_string());
        map.insert("layout", self.layout.clone());
        map
    }

    fn __repr__(&self) -> String {
        format!(
            "PlanInfo(target={:?}, segments={}, fusions={}, arena_bytes={})",
            self.target, self.segments, self.fusions, self.arena_bytes
        )
    }
}

fn plan_infos(plans: &[crate::inference::model::format::PlanSummary]) -> Vec<PyPlanInfo> {
    plans.iter().map(PyPlanInfo::of).collect()
}

/// One section, as Python sees it.
#[pyclass(module = "fastnn._core.inference", name = "Section", frozen)]
pub struct PySection {
    name: String,
    offset: u64,
    length: u64,
    sha256: String,
    verified: bool,
}

#[pymethods]
impl PySection {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn offset(&self) -> u64 {
        self.offset
    }

    #[getter]
    fn length(&self) -> u64 {
        self.length
    }

    #[getter]
    fn sha256(&self) -> &str {
        &self.sha256
    }

    #[getter]
    fn verified(&self) -> bool {
        self.verified
    }

    fn __repr__(&self) -> String {
        format!(
            "Section(name={:?}, offset={}, length={}, verified={})",
            self.name, self.offset, self.length, self.verified
        )
    }
}

/// The outcome of [`prepare`].
#[pyclass(module = "fastnn._core.inference", name = "PrepareReport", frozen)]
pub struct PyPrepareReport {
    path: String,
    status: String,
    size_bytes: u64,
    sha256: String,
    source_sha256: String,
    nodes: usize,
    values: usize,
    weights: usize,
    weight_bytes: u64,
    plans: Vec<PyPlanInfo>,
    metadata: Py<PyMetadata>,
    elapsed_ms: f64,
}

#[pymethods]
impl PyPrepareReport {
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    /// `"created"`, `"replaced"` or `"already_valid"`.
    #[getter]
    fn status(&self) -> &str {
        &self.status
    }

    #[getter]
    fn wrote(&self) -> bool {
        self.status != "already_valid"
    }

    #[getter]
    fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// SHA-256 of the artifact file.
    #[getter]
    fn sha256(&self) -> &str {
        &self.sha256
    }

    /// SHA-256 of the source ONNX file.
    #[getter]
    fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    #[getter]
    fn nodes(&self) -> usize {
        self.nodes
    }

    #[getter]
    fn values(&self) -> usize {
        self.values
    }

    #[getter]
    fn weights(&self) -> usize {
        self.weights
    }

    #[getter]
    fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    #[getter]
    fn plans(&self) -> Vec<PyPlanInfo> {
        self.plans
            .iter()
            .map(|plan| PyPlanInfo {
                target: plan.target.clone(),
                precision: plan.precision.clone(),
                segments: plan.segments,
                unfused_segments: plan.unfused_segments,
                fusions: plan.fusions,
                reorders: plan.reorders,
                reorder_bytes: plan.reorder_bytes,
                arena_bytes: plan.arena_bytes,
                peak_bytes: plan.peak_bytes,
                scalar_segments: plan.scalar_segments,
                one_dnn_segments: plan.one_dnn_segments,
                layout: plan.layout.clone(),
            })
            .collect()
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> Py<PyMetadata> {
        Py::clone_ref(&self.metadata, py)
    }

    #[getter]
    fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
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
#[pyclass(module = "fastnn._core.inference", name = "ArtifactInfo", frozen)]
pub struct PyArtifactInfo {
    path: String,
    size_bytes: u64,
    format_version: u16,
    source_sha256: String,
    nodes: usize,
    values: usize,
    inputs: usize,
    outputs: usize,
    weights: usize,
    weight_bytes: u64,
    plans: Vec<PyPlanInfo>,
    quant_precision: String,
    quant_entries: usize,
    sections: Vec<PySection>,
    mapped_bytes: u64,
    bytes_read: u64,
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

    #[getter]
    fn format_version(&self) -> u16 {
        self.format_version
    }

    #[getter]
    fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    #[getter]
    fn nodes(&self) -> usize {
        self.nodes
    }

    #[getter]
    fn values(&self) -> usize {
        self.values
    }

    #[getter]
    fn inputs(&self) -> usize {
        self.inputs
    }

    #[getter]
    fn outputs(&self) -> usize {
        self.outputs
    }

    #[getter]
    fn weights(&self) -> usize {
        self.weights
    }

    #[getter]
    fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    #[getter]
    fn plans(&self) -> Vec<PyPlanInfo> {
        self.plans
            .iter()
            .map(|plan| PyPlanInfo {
                target: plan.target.clone(),
                precision: plan.precision.clone(),
                segments: plan.segments,
                unfused_segments: plan.unfused_segments,
                fusions: plan.fusions,
                reorders: plan.reorders,
                reorder_bytes: plan.reorder_bytes,
                arena_bytes: plan.arena_bytes,
                peak_bytes: plan.peak_bytes,
                scalar_segments: plan.scalar_segments,
                one_dnn_segments: plan.one_dnn_segments,
                layout: plan.layout.clone(),
            })
            .collect()
    }

    #[getter]
    fn quant_precision(&self) -> &str {
        &self.quant_precision
    }

    #[getter]
    fn quant_entries(&self) -> usize {
        self.quant_entries
    }

    #[getter]
    fn sections(&self) -> Vec<PySection> {
        self.sections
            .iter()
            .map(|section| PySection {
                name: section.name.clone(),
                offset: section.offset,
                length: section.length,
                sha256: section.sha256.clone(),
                verified: section.verified,
            })
            .collect()
    }

    /// Always 0: `inspect` never maps the file.
    #[getter]
    fn mapped_bytes(&self) -> u64 {
        self.mapped_bytes
    }

    #[getter]
    fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    #[getter]
    fn metadata(&self, py: Python<'_>) -> Py<PyMetadata> {
        Py::clone_ref(&self.metadata, py)
    }

    #[getter]
    fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "ArtifactInfo(path={:?}, size_bytes={}, nodes={}, values={}, plans={})",
            self.path,
            self.size_bytes,
            self.nodes,
            self.values,
            self.plans.len()
        )
    }
}

/// One doctor finding.
#[pyclass(module = "fastnn._core.inference", name = "DoctorIssue", frozen)]
pub struct PyDoctorIssue {
    node: Option<String>,
    category: String,
    message: String,
}

#[pymethods]
impl PyDoctorIssue {
    #[getter]
    fn node(&self) -> Option<&str> {
        self.node.as_deref()
    }

    #[getter]
    fn category(&self) -> &str {
        &self.category
    }

    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    fn __repr__(&self) -> String {
        format!(
            "DoctorIssue(category={:?}, node={:?})",
            self.category, self.node
        )
    }
}

/// The outcome of [`doctor`].
#[pyclass(module = "fastnn._core.inference", name = "DoctorReport", frozen)]
pub struct PyDoctorReport {
    path: String,
    ok: bool,
    nodes: usize,
    values: usize,
    weights: usize,
    weight_bytes: u64,
    arena_bytes: u64,
    peak_bytes: u64,
    expected_memory_bytes: u64,
    problems: Vec<PyDoctorIssue>,
    inputs: Vec<String>,
    dynamic_symbols: Vec<(String, usize)>,
    dtypes: Vec<(String, usize)>,
    elapsed_ms: f64,
}

#[pymethods]
impl PyDoctorReport {
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    fn ok(&self) -> bool {
        self.ok
    }

    #[getter]
    fn nodes(&self) -> usize {
        self.nodes
    }

    #[getter]
    fn values(&self) -> usize {
        self.values
    }

    #[getter]
    fn weights(&self) -> usize {
        self.weights
    }

    #[getter]
    fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    #[getter]
    fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    #[getter]
    fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    #[getter]
    fn expected_memory_bytes(&self) -> u64 {
        self.expected_memory_bytes
    }

    #[getter]
    fn problems(&self) -> Vec<PyDoctorIssue> {
        self.problems
            .iter()
            .map(|issue| PyDoctorIssue {
                node: issue.node.clone(),
                category: issue.category.clone(),
                message: issue.message.clone(),
            })
            .collect()
    }

    /// The graph inputs with their declared shapes.
    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.inputs.clone()
    }

    #[getter]
    fn dynamic_symbols(&self) -> std::collections::BTreeMap<String, usize> {
        self.dynamic_symbols.iter().cloned().collect()
    }

    #[getter]
    fn dtypes(&self) -> std::collections::BTreeMap<String, usize> {
        self.dtypes.iter().cloned().collect()
    }

    #[getter]
    fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "DoctorReport(ok={}, nodes={}, problems={})",
            self.ok,
            self.nodes,
            self.problems.len()
        )
    }
}

// ---------------------------------------------------------------------------
// The loaded model
// ---------------------------------------------------------------------------

/// A loaded, validated prepared model. Immutable and shareable by threads.
#[pyclass(module = "fastnn._core.inference", name = "Model", frozen)]
pub struct PyModel {
    artifact: Arc<Artifact>,
    instance: u64,
    path: String,
    sha256: String,
    size_bytes: u64,
    target: Option<String>,
    metadata: Py<PyMetadata>,
    plans: Vec<PyPlanInfo>,
    elapsed_ms: f64,
}

#[pymethods]
impl PyModel {
    /// Path of the artifact this model was loaded from.
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    /// SHA-256 of the artifact file, computed by streaming it on load.
    #[getter]
    fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Artifact size in bytes.
    #[getter]
    fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    /// The selected target plan, when one was requested.
    #[getter]
    fn target(&self) -> Option<&str> {
        self.target.as_deref()
    }

    /// The default target name carried by the artifact.
    #[getter]
    fn default_target(&self) -> &str {
        self.artifact.default_target()
    }

    /// Provenance.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> Py<PyMetadata> {
        Py::clone_ref(&self.metadata, py)
    }

    #[getter]
    fn plans(&self) -> Vec<PyPlanInfo> {
        self.plans
            .iter()
            .map(|plan| PyPlanInfo {
                target: plan.target.clone(),
                precision: plan.precision.clone(),
                segments: plan.segments,
                unfused_segments: plan.unfused_segments,
                fusions: plan.fusions,
                reorders: plan.reorders,
                reorder_bytes: plan.reorder_bytes,
                arena_bytes: plan.arena_bytes,
                peak_bytes: plan.peak_bytes,
                scalar_segments: plan.scalar_segments,
                one_dnn_segments: plan.one_dnn_segments,
                layout: plan.layout.clone(),
            })
            .collect()
    }

    /// Typed graph nodes.
    #[getter]
    fn nodes(&self) -> usize {
        self.artifact.graph().node_count()
    }

    /// Typed graph values.
    #[getter]
    fn values(&self) -> usize {
        self.artifact.graph().value_count()
    }

    /// Weight table entries.
    #[getter]
    fn weights(&self) -> usize {
        self.artifact.weights().len()
    }

    /// Weight payload bytes.
    #[getter]
    fn weight_bytes(&self) -> u64 {
        self.artifact.weight_bytes()
    }

    /// Graph input names, in declaration order.
    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.artifact.inputs().to_vec()
    }

    /// Graph output names, in declaration order.
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.artifact.outputs().to_vec()
    }

    /// The device every session executes on (`cpu` for this runtime).
    #[getter]
    fn device(&self, py: Python<'_>) -> PyResult<Py<super::device::PyDevice>> {
        describe_device(py, DeviceId::Cpu)
    }

    /// Wall time the load took, in milliseconds.
    #[getter]
    fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
    }

    /// Open an independent session over this model.
    ///
    /// An explicit `threads` is a whole number of workers; `None` means the
    /// physical core count. `device` accepts `None`/`"cpu"`; the typed-graph
    /// runtime is CPU-only today and an explicit other device is refused rather
    /// than silently executed on the CPU.
    #[pyo3(signature = (threads = None, device = None))]
    fn session(
        &self,
        py: Python<'_>,
        threads: Option<usize>,
        device: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PySession>> {
        let resolved = resolve_device(py, device.as_ref())?;
        if resolved != DeviceId::Cpu {
            return Err(errors::from_core(
                py,
                crate::inference::error::Fn3Error::UnsupportedOp {
                    op: crate::inference::OpId::Conv,
                    backend: "typed-graph runtime",
                    reason: format!(
                        "the typed-graph artifact runtime executes on the CPU only; {resolved} \
                         was requested"
                    ),
                },
            ));
        }
        let threading = match threads {
            None => Threading::Auto,
            Some(0) => {
                return Err(errors::from_core(
                    py,
                    crate::inference::error::Fn3Error::InvalidShape {
                        context: "Model.session",
                        detail: "threads must be at least 1".to_string(),
                    },
                ))
            }
            Some(count) => Threading::Fixed(count),
        };
        let session = py.detach(|| {
            if let Some(target) = &self.target {
                RuntimeSession::from_artifact_for(&self.artifact, target, threading)
            } else {
                RuntimeSession::from_shared_artifact(Arc::clone(&self.artifact), threading)
            }
        });
        let session = session.map_err(|error| errors::from_core(py, error))?;
        PySession::open(py, session, self.instance)
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(#{} path={:?} target={:?} nodes={} values={})",
            self.instance,
            self.path,
            self.target,
            self.artifact.graph().node_count(),
            self.artifact.graph().value_count()
        )
    }
}

// ---------------------------------------------------------------------------
// prepare / inspect / load / doctor
// ---------------------------------------------------------------------------

/// Parse a lowercase SHA-256 digest into bytes.
fn digest_bytes(hex: &str) -> [u8; 32] {
    let mut digest = [0u8; 32];
    for (index, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
        if let Ok(chunk) = std::str::from_utf8(chunk) {
            digest[index] = u8::from_str_radix(chunk, 16).unwrap_or(0);
        }
    }
    digest
}

/// Stream a file through SHA-256 without loading it into memory.
fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(crate::inference::digest::hex(&hasher.finish()))
}

/// Import an ONNX model and write (or confirm) a versioned artifact.
#[pyfunction]
#[pyo3(signature = (onnx, artifact, *, target = "cpu", precision = "f32", overwrite = false, shapes = None))]
pub fn prepare(
    py: Python<'_>,
    onnx: &str,
    artifact: &str,
    target: &str,
    precision: &str,
    overwrite: bool,
    shapes: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyPrepareReport>> {
    let started = Instant::now();
    let source_bytes = std::fs::read(onnx).map_err(|error| {
        errors::file_error(
            py,
            format!("cannot read ONNX model {onnx:?}: {error}"),
            onnx.to_string(),
        )
    })?;
    let bindings = shape_bindings(shapes)?;
    let imported = py
        .detach(|| import_onnx_graph_with_shapes(&source_bytes, &bindings))
        .map_err(|error| errors::from_import(py, error))?;
    let precision = Precision::parse(precision).ok_or_else(|| {
        errors::model_error(
            py,
            format!("precision {precision:?} is not implemented; this build writes f32 only"),
        )
    })?;
    let targets = resolve_targets(target).map_err(|error| errors::from_format(py, error))?;
    let source_sha256 = sha256_hex(&source_bytes);
    let built = py
        .detach(|| {
            PreparedArtifact::from_imported(
                &imported,
                &targets,
                precision,
                digest_bytes(&source_sha256),
            )
        })
        .map_err(|error| errors::from_format(py, error))?;
    let bytes = py
        .detach(|| built.to_bytes())
        .map_err(|error| errors::from_format(py, error))?;
    let digest = sha256_hex(&bytes);

    let destination = artifact.to_string();
    let existing = match std::fs::read(artifact) {
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
    let status = match existing {
        Some(existing) if sha256_hex(&existing) == digest => "already_valid",
        Some(_) if !overwrite => {
            return Err(errors::model_exists(
                py,
                format!(
                    "{artifact:?} already holds a different prepared model; pass overwrite=True \
                     to replace it"
                ),
                destination,
                vec!["the file contents differ".to_string()],
            ))
        }
        Some(_) => {
            std::fs::write(artifact, &bytes).map_err(|error| {
                errors::file_error(
                    py,
                    format!("cannot write {artifact:?}: {error}"),
                    destination.clone(),
                )
            })?;
            "replaced"
        }
        None => {
            std::fs::write(artifact, &bytes).map_err(|error| {
                errors::file_error(
                    py,
                    format!("cannot write {artifact:?}: {error}"),
                    destination.clone(),
                )
            })?;
            "created"
        }
    };
    if status != "already_valid" {
        let on_disk = std::fs::read(artifact).map_err(|error| {
            errors::file_error(
                py,
                format!("cannot read back {artifact:?}: {error}"),
                destination.clone(),
            )
        })?;
        if sha256_hex(&on_disk) != digest {
            return Err(errors::model_error(
                py,
                format!("{artifact:?} was written but its SHA-256 changed; a torn write"),
            ));
        }
    }

    let loaded = py
        .detach(|| Artifact::open(artifact))
        .map_err(|error| errors::from_format(py, error))?;
    let report = PyPrepareReport {
        path: destination,
        status: status.to_string(),
        size_bytes: bytes.len() as u64,
        sha256: digest,
        source_sha256,
        nodes: loaded.graph().node_count(),
        values: loaded.graph().value_count(),
        weights: loaded.weights().len(),
        weight_bytes: loaded.weight_bytes(),
        plans: plan_infos(&loaded.plan_summaries()),
        metadata: Py::new(py, PyMetadata::of(loaded.metadata()))?,
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
    };
    Py::new(py, report)
}

/// Describe an artifact without mapping its weight payload.
#[pyfunction]
pub fn inspect(py: Python<'_>, artifact: &str) -> PyResult<Py<PyArtifactInfo>> {
    let started = Instant::now();
    let report = py
        .detach(|| inspect_file(artifact))
        .map_err(|error| errors::from_format(py, error))?;
    let source_sha256 = report
        .source_onnx_sha256
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let info = PyArtifactInfo {
        path: artifact.to_string(),
        size_bytes: report.file_bytes,
        format_version: report.format_version,
        source_sha256,
        nodes: report.nodes,
        values: report.values,
        inputs: report.inputs,
        outputs: report.outputs,
        weights: report.weights,
        weight_bytes: report.weight_bytes,
        plans: plan_infos(&report.plans),
        quant_precision: report.quant.precision().name().to_string(),
        quant_entries: report.quant.tensors().len(),
        sections: report
            .sections
            .iter()
            .map(|section| PySection {
                name: section.name.to_string(),
                offset: section.offset,
                length: section.length,
                sha256: section.sha256_hex(),
                verified: section.verified,
            })
            .collect(),
        mapped_bytes: report.mapped_bytes,
        bytes_read: report.bytes_read,
        metadata: Py::new(py, PyMetadata::of(&report.metadata))?,
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
    };
    Py::new(py, info)
}

/// Load a prepared artifact and return an immutable model.
#[pyfunction]
#[pyo3(signature = (artifact, *, target = None, device = None))]
pub fn load(
    py: Python<'_>,
    artifact: &str,
    target: Option<String>,
    device: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyModel>> {
    let started = Instant::now();
    let resolved = resolve_device(py, device.as_ref())?;
    if resolved != DeviceId::Cpu {
        return Err(errors::from_core(
            py,
            crate::inference::error::Fn3Error::UnsupportedOp {
                op: crate::inference::OpId::Conv,
                backend: "typed-graph runtime",
                reason: format!(
                    "the typed-graph artifact runtime executes on the CPU only; {resolved} was \
                     requested"
                ),
            },
        ));
    }
    let size_bytes = std::fs::metadata(artifact)
        .map(|metadata| metadata.len())
        .map_err(|error| {
            errors::file_error(
                py,
                format!("cannot stat {artifact:?}: {error}"),
                artifact.to_string(),
            )
        })?;
    let digest = sha256_file(Path::new(artifact)).map_err(|error| {
        errors::file_error(
            py,
            format!("cannot hash {artifact:?}: {error}"),
            artifact.to_string(),
        )
    })?;
    let loaded = py
        .detach(|| Artifact::open(artifact))
        .map_err(|error| errors::from_format(py, error))?;
    if let Some(target) = &target {
        if loaded.plan_for(target).is_none() {
            let available: Vec<&str> = loaded.plans().iter().map(|plan| plan.target()).collect();
            return Err(errors::model_error(
                py,
                format!(
                    "{artifact:?} has no target {target:?}; the artifact carries {}",
                    available.join(", ")
                ),
            ));
        }
    }
    let metadata = Py::new(py, PyMetadata::of(loaded.metadata()))?;
    let plans = plan_infos(&loaded.plan_summaries());
    let model = PyModel {
        artifact: Arc::new(loaded),
        instance: MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
        path: artifact.to_string(),
        sha256: digest,
        size_bytes,
        target,
        metadata,
        plans,
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
    };
    Py::new(py, model)
}

/// Convert `{name: [d0, d1, ...]}` into input-shape bindings.
///
/// A dynamic export has no numeric extents for the plan to use, so the caller
/// supplies the shape the model will be run at. The conversion happens while
/// the GIL is held; the import itself runs detached.
fn shape_bindings(shapes: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, Vec<u64>)>> {
    let Some(shapes) = shapes else {
        return Ok(Vec::new());
    };
    let mut bindings: Vec<(String, Vec<u64>)> = Vec::with_capacity(shapes.len());
    for (key, value) in shapes.iter() {
        let name: String = key.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("shapes keys must be str graph-input names")
        })?;
        if bindings.iter().any(|(existing, _)| existing == &name) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shapes names the input {name:?} twice"
            )));
        }
        let extents: Vec<u64> = value.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "shapes[{name:?}] must be a sequence of positive ints"
            ))
        })?;
        if extents.is_empty() || extents.iter().any(|&extent| extent == 0) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shapes[{name:?}] needs at least one extent and none may be 0"
            )));
        }
        bindings.push((name, extents));
    }
    Ok(bindings)
}

/// Diagnose an ONNX model before preparing it.
#[pyfunction]
#[pyo3(signature = (onnx, *, shapes = None))]
pub fn doctor(
    py: Python<'_>,
    onnx: &str,
    shapes: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyDoctorReport>> {
    let started = Instant::now();
    let bindings = shape_bindings(shapes)?;
    let report = py
        .detach(|| crate::inference::doctor_with_shapes(onnx, &bindings))
        .map_err(|error| errors::from_core(py, error))?;
    let doctor = PyDoctorReport {
        path: onnx.to_string(),
        ok: report.ok,
        nodes: report.nodes,
        values: report.values,
        weights: report.weights,
        weight_bytes: report.weight_bytes,
        arena_bytes: report.arena_bytes,
        peak_bytes: report.peak_bytes,
        expected_memory_bytes: report.expected_memory_bytes,
        problems: report
            .problems
            .iter()
            .map(|issue| PyDoctorIssue {
                node: issue.node.clone(),
                category: issue.category.to_string(),
                message: issue.message.clone(),
            })
            .collect(),
        inputs: report.inputs.clone(),
        dynamic_symbols: report
            .dynamic_symbols
            .iter()
            .map(|entry| (entry.symbol.to_string(), entry.values))
            .collect(),
        dtypes: report
            .dtypes
            .iter()
            .map(|entry| (entry.dtype.to_string(), entry.values))
            .collect(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
    };
    Py::new(py, doctor)
}

/// `TARGETS` as a Python tuple.
#[pyfunction]
pub fn targets() -> Vec<&'static str> {
    TARGETS.to_vec()
}

/// Register the model surface.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMetadata>()?;
    module.add_class::<PyPlanInfo>()?;
    module.add_class::<PySection>()?;
    module.add_class::<PyPrepareReport>()?;
    module.add_class::<PyArtifactInfo>()?;
    module.add_class::<PyDoctorIssue>()?;
    module.add_class::<PyDoctorReport>()?;
    module.add_class::<PyModel>()?;
    module.add_function(wrap_pyfunction!(prepare, module)?)?;
    module.add_function(wrap_pyfunction!(inspect, module)?)?;
    module.add_function(wrap_pyfunction!(load, module)?)?;
    module.add_function(wrap_pyfunction!(doctor, module)?)?;
    module.add_function(wrap_pyfunction!(targets, module)?)?;
    Ok(())
}
