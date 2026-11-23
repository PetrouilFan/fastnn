//! The high-level v3 API: `prepare`, `load`, `inspect` and `doctor`.
//!
//! This is the surface the README, the CLI and the Python facade describe:
//!
//! ```text
//! prepare(onnx, artifact, target, precision)  ONNX -> one versioned file
//! load(artifact, threading)                   mmap + stored plan -> Session
//! run(&mut session, inputs)                   one forward (Session::run)
//! inspect(artifact)                           describe without mapping weights
//! doctor(onnx)                                diagnose before prepare
//! ```
//!
//! The low-level readers stay public in [`crate::inference::model::format`]:
//! they return [`FormatError`](crate::inference::model::format::FormatError),
//! which distinguishes a corrupt section from an obsolete version. These
//! convenience functions flatten the failure into [`Fn3Error`] so a caller that
//! only wants one error type does not have to match two.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::inference::backend::cpu::CpuBackend;
use crate::inference::backend::{Backend, ScalarOnly, Support};
use crate::inference::digest::sha256_hex;
use crate::inference::error::Fn3Error;
use crate::inference::import::{declared_inputs, import_onnx_graph_with_shapes, ImportedGraph};
use crate::inference::ir::DimExpr;
use crate::inference::model::format::{
    inspect_file, resolve_targets, Artifact, ArtifactInspection, PlanSummary, Precision,
    PreparedArtifact,
};
use crate::inference::runtime::{Plan, PlanOptions, Session, Threading};
use crate::inference::shape::Symbol;

/// One target's plan, as the prepare report describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetPlanReport {
    /// Target name.
    pub target: String,
    /// Execution precision.
    pub precision: Precision,
    /// Segments in the plan.
    pub segments: usize,
    /// Recognised fusions.
    pub fusions: usize,
    /// Layout conversions.
    pub reorders: usize,
    /// Bytes those conversions move per forward.
    pub reorder_bytes: u64,
    /// Arena bytes.
    pub arena_bytes: u64,
    /// Peak live bytes.
    pub peak_bytes: u64,
    /// Scalar segments.
    pub scalar_segments: usize,
    /// oneDNN segments.
    pub one_dnn_segments: usize,
}

impl TargetPlanReport {
    fn of(summary: &PlanSummary) -> TargetPlanReport {
        TargetPlanReport {
            target: summary.target.clone(),
            precision: summary.precision,
            segments: summary.segments,
            fusions: summary.fusions,
            reorders: summary.reorders,
            reorder_bytes: summary.reorder_bytes,
            arena_bytes: summary.arena_bytes,
            peak_bytes: summary.peak_bytes,
            scalar_segments: summary.engines.scalar,
            one_dnn_segments: summary.engines.one_dnn,
        }
    }
}

/// The outcome of [`prepare`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareReport {
    /// Source ONNX path.
    pub onnx_path: PathBuf,
    /// Destination artifact path.
    pub artifact_path: PathBuf,
    /// Source ONNX SHA-256, lowercase hexadecimal.
    pub source_sha256: String,
    /// Artifact SHA-256, lowercase hexadecimal.
    pub artifact_sha256: String,
    /// Typed graph nodes.
    pub nodes: usize,
    /// Typed graph values.
    pub values: usize,
    /// Weights in the table.
    pub weights: usize,
    /// One entry per target profile written.
    pub targets: Vec<TargetPlanReport>,
    /// Segments in the default plan.
    pub segments: usize,
    /// Fusions in the default plan.
    pub fusions: usize,
    /// Reorders in the default plan.
    pub reorders: usize,
    /// Arena bytes of the default plan.
    pub arena_bytes: u64,
    /// Peak bytes of the default plan.
    pub peak_bytes: u64,
    /// Weight payload bytes.
    pub weight_bytes: u64,
    /// Artifact file size.
    pub artifact_bytes: u64,
}

/// One doctor finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoctorIssue {
    /// Source node, when the finding belongs to one, rendered exactly as
    /// `node 7 (onnx "Conv", output "y")`.
    pub node: Option<String>,
    /// Stable category: `unsupported-operator`, `dynamic-shape`, `dtype`,
    /// `shape`, `limits`, `wire` or `import`.
    pub category: &'static str,
    /// Full message.
    pub message: String,
}

/// How many values use one dynamic symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicSymbolUse {
    /// Symbol name (`seq`).
    pub symbol: &'static str,
    /// Values that use it.
    pub values: usize,
}

/// How many values carry one dtype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtypeUse {
    /// Dtype name (`f32`).
    pub dtype: &'static str,
    /// Values of that dtype.
    pub values: usize,
}

/// The outcome of [`doctor`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DoctorReport {
    /// Source ONNX path.
    pub onnx_path: PathBuf,
    /// `true` when the model would prepare as-is.
    pub ok: bool,
    /// Typed graph nodes (0 when import failed).
    pub nodes: usize,
    /// Typed graph values.
    pub values: usize,
    /// Weight table entries.
    pub weights: usize,
    /// Weight payload bytes.
    pub weight_bytes: u64,
    /// Arena bytes the default plan would allocate.
    pub arena_bytes: u64,
    /// Peak live bytes.
    pub peak_bytes: u64,
    /// Weights plus arena: the expected resident cost of one session.
    pub expected_memory_bytes: u64,
    /// Findings, in file order.
    pub problems: Vec<DoctorIssue>,
    /// Graph inputs with their declared shapes (`images: [batch, 3, height, width]`),
    /// read before any binding is applied.
    pub inputs: Vec<String>,
    /// Dynamic symbols the graph uses.
    pub dynamic_symbols: Vec<DynamicSymbolUse>,
    /// Value dtype census.
    pub dtypes: Vec<DtypeUse>,
}

impl DoctorReport {
    /// The first problem, when there is one.
    pub fn first_problem(&self) -> Option<&DoctorIssue> {
        self.problems.first()
    }
}

/// Import an ONNX file, plan it and write one versioned artifact.
///
/// `target` is `cpu`, `cpu-reference` or `all`; `precision` is `f32`.
pub fn prepare(
    onnx_path: impl AsRef<Path>,
    artifact_path: impl AsRef<Path>,
    target: &str,
    precision: &str,
) -> Result<PrepareReport, Fn3Error> {
    prepare_with_shapes(onnx_path, artifact_path, target, precision, &[])
}

/// Refuse a precision this build cannot execute, naming why.
///
/// `w8a8` is a plan decision, but a decision is only honest if something can
/// run it. The oneDNN int8 convolution is a reference kernel on a CPU without
/// VNNI: measured on this host at **1332 ms** for a 32x80x80x3x3 convolution
/// against **1.57 ms** for the same f32 convolution, so writing a W8A8 artifact
/// here would be an 848x trap. The check is in `prepare`, before the ONNX is
/// read, so a refused request writes nothing (#103).
pub(crate) fn refuse_unexecutable_precision(precision: Precision) -> Result<(), Fn3Error> {
    if precision != Precision::W8A8 {
        return Ok(());
    }
    #[cfg(feature = "cpu-onednn")]
    {
        let (_, isa, available) = crate::inference::backend::cpu::onednn::int8_execution_support();
        let reason = if available {
            "int8 kernels exist on that ISA, but no W8A8 executor is implemented yet"
        } else {
            "without VNNI oneDNN serves an int8 convolution from its reference kernel, \
             measured at 1332 ms against 1.57 ms for the same 32x80x80x3x3 f32 convolution \
             on this host"
        };
        Err(Fn3Error::InvalidFormat {
            format: "precision",
            detail: format!(
                "`w8a8` was requested and nothing was written: this build has no W8A8 \
                 executor. oneDNN reports ISA `{isa}`, and {reason} (issue #103)"
            ),
        })
    }
    #[cfg(not(feature = "cpu-onednn"))]
    {
        Err(Fn3Error::InvalidFormat {
            format: "precision",
            detail: "`w8a8` was requested and nothing was written: this build has no \
                     quantized engine, because it was compiled without `cpu-onednn` \
                     (issue #103)"
                .to_string(),
        })
    }
}

#[cfg(all(test, feature = "cpu-onednn"))]
mod w8a8_tests {
    use super::*;

    #[test]
    fn w8a8_is_refused_with_the_measured_isa() {
        let error = refuse_unexecutable_precision(Precision::W8A8).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("w8a8"), "{text}");
        assert!(text.contains("nothing was written"), "{text}");
        assert!(text.contains("ISA"), "the refusal names the ISA:\n{text}");
        // f32 is never refused.
        assert!(refuse_unexecutable_precision(Precision::F32).is_ok());
    }
}

/// [`prepare`] with concrete extents for the model's graph inputs.
///
/// A dynamic ONNX export declares its input as `[batch, 3, height, width]` and
/// leaves a `Reshape` target symbolic. Binding the extents the model will be run
/// at is what makes the plan static, so this is part of preparing rather than a
/// runtime concern: the artifact has one shape per input.
pub fn prepare_with_shapes(
    onnx_path: impl AsRef<Path>,
    artifact_path: impl AsRef<Path>,
    target: &str,
    precision: &str,
    shapes: &[(String, Vec<u64>)],
) -> Result<PrepareReport, Fn3Error> {
    let onnx_path = onnx_path.as_ref();
    let artifact_path = artifact_path.as_ref();
    if onnx_path == artifact_path
        || (onnx_path.exists()
            && artifact_path.exists()
            && std::fs::canonicalize(onnx_path).ok() == std::fs::canonicalize(artifact_path).ok())
    {
        return Err(Fn3Error::InvalidFormat {
            format: "prepare",
            detail: "the ONNX source and the artifact destination resolve to the same file"
                .to_string(),
        });
    }
    let source_bytes = std::fs::read(onnx_path).map_err(|error| Fn3Error::InvalidFormat {
        format: "onnx model",
        detail: format!("cannot read `{}`: {error}", onnx_path.display()),
    })?;
    let source_digest = digest(&source_bytes);
    let imported = import_onnx_graph_with_shapes(&source_bytes, shapes)?;
    let precision = Precision::parse(precision).ok_or_else(|| Fn3Error::InvalidFormat {
        format: "precision",
        detail: format!(
            "`{precision}` is not a precision this build knows; accepted names are `f32` and              `w8a8`"
        ),
    })?;
    refuse_unexecutable_precision(precision)?;
    let targets = resolve_targets(target)?;
    let prepared = PreparedArtifact::from_imported(&imported, &targets, precision, source_digest)?;
    let bytes = prepared.to_bytes()?;
    std::fs::write(artifact_path, &bytes).map_err(|error| Fn3Error::InvalidFormat {
        format: "prepared-model artifact",
        detail: format!("cannot write `{}`: {error}", artifact_path.display()),
    })?;
    // The report describes what is on disk, read through the strict reader.
    let artifact = Artifact::open(artifact_path)?;
    let summaries = artifact.plan_summaries();
    let default = summaries
        .first()
        .cloned()
        .ok_or_else(|| Fn3Error::InvalidFormat {
            format: "prepared-model artifact",
            detail: "the written artifact carries no target plan".to_string(),
        })?;
    Ok(PrepareReport {
        onnx_path: onnx_path.to_path_buf(),
        artifact_path: artifact_path.to_path_buf(),
        source_sha256: sha256_hex(&source_bytes),
        artifact_sha256: sha256_hex(&bytes),
        nodes: artifact.graph().node_count(),
        values: artifact.graph().value_count(),
        weights: artifact.weights().len(),
        targets: summaries.iter().map(TargetPlanReport::of).collect(),
        segments: default.segments,
        fusions: default.fusions,
        reorders: default.reorders,
        arena_bytes: default.arena_bytes,
        peak_bytes: default.peak_bytes,
        weight_bytes: artifact.weight_bytes(),
        artifact_bytes: artifact.header().file_bytes,
    })
}

/// Map an artifact and open a session on its stored default plan.
///
/// No ONNX file is read and no plan is rebuilt.
pub fn load(artifact_path: impl AsRef<Path>, threading: Threading) -> Result<Session, Fn3Error> {
    let artifact = Artifact::open(artifact_path)?;
    Session::from_shared_artifact(Arc::new(artifact), threading)
}

/// Describe an artifact without mapping its weight payload.
pub fn inspect(artifact_path: impl AsRef<Path>) -> Result<ArtifactInspection, Fn3Error> {
    Ok(inspect_file(artifact_path)?)
}

/// Diagnose an ONNX model before preparing it.
///
/// An import failure is a report, not an `Err`: the point of `doctor` is to list
/// every unsupported node, dynamic limit and dtype problem it can find, with the
/// node that owns each one. Only "cannot read the file" is returned as an error.
pub fn doctor(onnx_path: impl AsRef<Path>) -> Result<DoctorReport, Fn3Error> {
    doctor_with_shapes(onnx_path, &[])
}

/// [`doctor`] with the input-shape bindings that would be used to prepare.
///
/// Pass the same `shapes` you would pass to [`prepare_with_shapes`] to see the
/// verdict the model will actually get; pass none to see it as declared.
pub fn doctor_with_shapes(
    onnx_path: impl AsRef<Path>,
    shapes: &[(String, Vec<u64>)],
) -> Result<DoctorReport, Fn3Error> {
    let onnx_path = onnx_path.as_ref();
    let bytes = std::fs::read(onnx_path).map_err(|error| Fn3Error::InvalidFormat {
        format: "onnx model",
        detail: format!("cannot read `{}`: {error}", onnx_path.display()),
    })?;
    let mut report = DoctorReport {
        onnx_path: onnx_path.to_path_buf(),
        ..DoctorReport::default()
    };
    // The declared inputs are readable even when the typed pass refuses the
    // model, which is exactly the case where a caller needs to know what to
    // bind.
    if let Ok(inputs) = declared_inputs(&bytes) {
        report.inputs = inputs
            .into_iter()
            .map(|input| format!("{}: {}", input.name, input.shape))
            .collect();
    }
    let imported = match import_onnx_graph_with_shapes(&bytes, shapes) {
        Ok(imported) => imported,
        Err(error) => {
            for problem in error.problems() {
                report.problems.push(DoctorIssue {
                    node: problem.node.as_ref().map(ToString::to_string),
                    category: classify_import_problem(&problem.kind),
                    message: problem.to_string(),
                });
            }
            report.ok = false;
            return Ok(report);
        }
    };

    report.nodes = imported.graph().node_count();
    report.values = imported.graph().value_count();
    report.weights = imported.weights().len();
    report.weight_bytes = imported.weights().total_bytes();
    report.dtypes = dtype_census(&imported);
    report.dynamic_symbols = dynamic_census(&imported);
    for use_count in &report.dynamic_symbols {
        let (node, example) = dynamic_example(&imported, use_count.symbol);
        report.problems.push(DoctorIssue {
            node,
            category: "dynamic-shape",
            message: format!(
                "{} values use the dynamic symbol `{}` (first: {example}); the artifact plan \
                 requires every extent to be static, so prepare would reject them",
                use_count.values, use_count.symbol
            ),
        });
    }

    // A graph input the runtime cannot feed is a dtype problem even when the
    // importer accepts it.
    for &id in imported.inputs() {
        if let Some(value) = imported.graph().value(id) {
            if value.ty.dtype != crate::inference::DType::F32 {
                report.problems.push(DoctorIssue {
                    node: None,
                    category: "dtype",
                    message: format!(
                        "graph input `{}` is {}; a session feeds f32 activations only",
                        value.name.as_deref().unwrap_or("<unnamed>"),
                        value.ty.dtype
                    ),
                });
            }
        }
    }

    // Operator support: the planner cannot paper over a CPU op the backend
    // reports unsupported.
    let backend = CpuBackend::new();
    for node in imported.graph().nodes() {
        match backend.supports(node.op) {
            Support::Full => {}
            Support::Partial { reason } => report.problems.push(DoctorIssue {
                node: Some(node.loc.describe()),
                category: "unsupported-operator",
                message: format!(
                    "{} is only partially supported on the CPU: {reason}",
                    node.op
                ),
            }),
            Support::Unsupported { reason } => report.problems.push(DoctorIssue {
                node: Some(node.loc.describe()),
                category: "unsupported-operator",
                message: format!("{} is unsupported on the CPU: {reason}", node.op),
            }),
        }
    }

    // Memory: build the scalar plan. Planning is not execution; nothing runs.
    match Plan::from_imported_with_options(&imported, &ScalarOnly, PlanOptions::new()) {
        Ok(plan) => {
            report.arena_bytes = plan.arena_bytes as u64;
            report.peak_bytes = plan.peak_bytes as u64;
            report.expected_memory_bytes = report.weight_bytes.saturating_add(report.arena_bytes);
        }
        Err(error) => {
            report.problems.push(DoctorIssue {
                node: None,
                category: "shape",
                message: format!("the graph could not be planned: {error}"),
            });
        }
    }

    report.ok = report.problems.is_empty();
    Ok(report)
}

fn dtype_census(imported: &ImportedGraph) -> Vec<DtypeUse> {
    let mut counts = [0usize; 4];
    for value in imported.graph().values() {
        let index = match value.ty.dtype {
            crate::inference::DType::F32 => 0,
            crate::inference::DType::I64 => 1,
            crate::inference::DType::I32 => 2,
            crate::inference::DType::Bool => 3,
        };
        counts[index] += 1;
    }
    crate::inference::DType::ALL
        .into_iter()
        .enumerate()
        .filter(|(index, _)| counts[*index] > 0)
        .map(|(index, dtype)| DtypeUse {
            dtype: dtype.name(),
            values: counts[index],
        })
        .collect()
}

/// The first value that uses `symbol`, with its producer's source location.
fn dynamic_example(imported: &ImportedGraph, symbol: &'static str) -> (Option<String>, String) {
    for value in imported.graph().values() {
        if !value
            .ty
            .dims()
            .iter()
            .any(|dim| matches!(dim, DimExpr::Symbol(found) if found.name() == symbol))
        {
            continue;
        }
        let name = value.name.as_deref().unwrap_or("<unnamed>");
        let node = value
            .producer
            .and_then(|producer| imported.graph().node(producer))
            .map(|node| node.loc.describe());
        return (node, format!("value `{name}` is {}", value.ty));
    }
    (None, "no value found".to_string())
}

fn dynamic_census(imported: &ImportedGraph) -> Vec<DynamicSymbolUse> {
    let mut counts = [0usize; Symbol::COUNT];
    for value in imported.graph().values() {
        for dim in value.ty.dims() {
            if let DimExpr::Symbol(symbol) = dim {
                counts[*symbol as usize] += 1;
            }
        }
    }
    Symbol::ALL
        .into_iter()
        .filter(|symbol| counts[*symbol as usize] > 0)
        .map(|symbol| DynamicSymbolUse {
            symbol: symbol.name(),
            values: counts[symbol as usize],
        })
        .collect()
}

fn classify_import_problem(kind: &crate::inference::import::ImportProblemKind) -> &'static str {
    use crate::inference::import::ImportProblemKind as Kind;
    match kind {
        Kind::UnmappedOp { .. }
        | Kind::UnimplementedOpId { .. }
        | Kind::ControlFlow { .. }
        | Kind::UnsupportedDomain { .. }
        | Kind::Subgraph { .. }
        | Kind::LocalFunctions { .. } => "unsupported-operator",
        Kind::UnsupportedDType { .. } | Kind::DTypeMismatch { .. } | Kind::IndexDType { .. } => {
            "dtype"
        }
        Kind::LimitExceeded { .. }
        | Kind::NegativeExtent { .. }
        | Kind::TensorLength { .. }
        | Kind::TooManyProblems { .. } => "limits",
        Kind::Proto(_) => "wire",
        Kind::ShapeMismatch { .. }
        | Kind::UndefinedValue { .. }
        | Kind::DuplicateValue { .. }
        | Kind::NotTopological { .. }
        | Kind::MissingName { .. } => "shape",
        _ => "import",
    }
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    let text = sha256_hex(bytes);
    let mut digest = [0u8; 32];
    for (index, chunk) in text.as_bytes().chunks_exact(2).enumerate() {
        if let Ok(chunk) = std::str::from_utf8(chunk) {
            digest[index] = u8::from_str_radix(chunk, 16).unwrap_or(0);
        }
    }
    digest
}
