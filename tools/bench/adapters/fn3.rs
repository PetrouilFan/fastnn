//! FN3 3.0 runtime adapter for the benchmark harness (FN3-04 / #50, seam
//! completed by FN3-21 / #67).
//!
//! The `baseline` adapter measures a synthetic 2.x decoder. This adapter
//! measures the real prepared model: a manifest names the ONNX source and the
//! `.fnn3` artifact, `load` reads and cross-checks both, `prepare` builds the
//! planner, plan and weight tensors once, and one [`RunSession`] reuses the
//! prepared model for prefill and decode. The session's plan cache is the
//! reason repeated measurements do not recompile: a second prefill at the same
//! `(seq, past)` is a cache hit, and every measured decode step runs at the
//! same `past` because the untimed reset/replay restores it.
//!
//! # Manifest
//!
//! ```json
//! {
//!   "schema": 1,
//!   "name": "tiny-random-gpt2",
//!   "revision": "90f61e71d6fa8e571d0ab0f95a637a5d7d8ed52f",
//!   "source": "/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx",
//!   "artifact": "/tmp/opencode/fn3-bench/tiny-gpt2.fnn3",
//!   "source_bytes": 27897974,
//!   "artifact_bytes": 27879928,
//!   "device": "cpu",
//!   "capacity": 1024,
//!   "prompt_tokens": [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13],
//!   "decode_tokens": [640, 11, 612]
//! }
//! ```
//!
//! Relative `source` and `artifact` paths resolve against the manifest's own
//! directory. `source_bytes` and `artifact_bytes` are optional but, when
//! present, are enforced before any work: a benchmark of the wrong file is
//! worse than no benchmark.
//!
//! # Measurement honesty
//!
//! * `load` and `prepare` are separate measured series. `load` parses the
//!   artifact, imports the ONNX graph and proves they describe the same model;
//!   `prepare` plans and materialises weights.
//! * `prefill` and `decode` are timed around [`fastnn::fn3::Session`] calls,
//!   which return only after every backend instruction completed (the Vulkan
//!   submission path waits on its fence). The host stopwatch is therefore a
//!   synchronized wall time, not a submission time.
//! * `last_device_time_ms` and `peak_device_memory_bytes` return `None` when the
//!   backend does not report them. They never return a fabricated zero; the
//!   schema records an unavailable metric with a reason.

use super::{BenchConfig, Suite, SuiteOutcome};

/// Whether this build contains the linked fn3 adapter.
pub fn is_available() -> bool {
    cfg!(feature = "fn3")
}

pub fn run(suite: Suite, config: &BenchConfig) -> Result<SuiteOutcome, String> {
    #[cfg(feature = "fn3")]
    {
        real::run(suite, config).map_err(|error| error.to_string())
    }
    #[cfg(not(feature = "fn3"))]
    {
        let _ = (suite, &config.fn3_manifest);
        Err("fn3 adapter is not built into this binary.\n\
             Rebuild with `--features fn3` (and `--no-default-features` when building the \
             `fn3-bench` binary, because the default `python` feature is an extension-module):\n\
             \x20 CARGO_BUILD_JOBS=12 cargo build --release --no-default-features \\\n\
             \x20   --features fn3,simd,parallel,prepared-plan,fusion-forward --bin fn3-bench\n\
             This failure is intentional: the harness refuses to emit baseline numbers under the \
             fn3 adapter name."
            .to_string())
    }
}

#[cfg(feature = "fn3")]
mod real {
    use super::{BenchConfig, Suite, SuiteOutcome};
    use crate::schema::{Checksum, DeviceTime, Geometry, Metric, ModelInfo, Precision};
    use fastnn::fn3::backend::cpu::CpuBackend;
    use fastnn::fn3::import::{import_onnx_model, ModelImport};
    use fastnn::fn3::model::format::PreparedArtifact;
    use fastnn::fn3::model::{ModelWeights, Planner, PreparedModel as CorePreparedModel};
    use fastnn::fn3::shape::{LiveDims, Symbol};
    use fastnn::fn3::tensor::Tensor;
    use fastnn::fn3::{Backend, DeviceId, Fn3Error, Session};
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::fmt;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    /// Schema version of the bench manifest, not of the prepared artifact.
    pub const MANIFEST_SCHEMA: u32 = 1;

    // -----------------------------------------------------------------------
    // Errors
    // -----------------------------------------------------------------------

    /// A benchmark-adapter failure with the stage that produced it.
    ///
    /// `Display` is the operator-facing line; `run` records it instead of
    /// emitting placeholder numbers.
    #[derive(Debug)]
    pub enum BenchError {
        Manifest { path: PathBuf, detail: String },
        Field { field: &'static str, detail: String },
        Io { path: PathBuf, detail: String },
        Format { path: PathBuf, detail: String },
        Import { path: PathBuf, detail: String },
        Disagree { detail: String },
        Device { detail: String },
        Plan { detail: String },
        Session { detail: String },
    }

    impl fmt::Display for BenchError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Manifest { path, detail } => {
                    write!(f, "fn3 manifest {}: {detail}", path.display())
                }
                Self::Field { field, detail } => {
                    write!(f, "fn3 manifest field `{field}`: {detail}")
                }
                Self::Io { path, detail } => write!(f, "fn3 I/O {}: {detail}", path.display()),
                Self::Format { path, detail } => {
                    write!(f, "fn3 artifact {}: {detail}", path.display())
                }
                Self::Import { path, detail } => {
                    write!(f, "fn3 ONNX source {}: {detail}", path.display())
                }
                Self::Disagree { detail } => write!(f, "fn3 source/artifact mismatch: {detail}"),
                Self::Device { detail } => write!(f, "fn3 device: {detail}"),
                Self::Plan { detail } => write!(f, "fn3 plan: {detail}"),
                Self::Session { detail } => write!(f, "fn3 session: {detail}"),
            }
        }
    }

    impl std::error::Error for BenchError {}

    impl From<Fn3Error> for BenchError {
        fn from(error: Fn3Error) -> Self {
            Self::Session {
                detail: error.to_string(),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Manifest
    // -----------------------------------------------------------------------

    /// One model the fn3 adapter is allowed to measure.
    #[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
    pub struct ModelManifest {
        pub schema: u32,
        pub name: String,
        pub revision: String,
        pub source: PathBuf,
        pub artifact: PathBuf,
        #[serde(default)]
        pub source_bytes: Option<u64>,
        #[serde(default)]
        pub artifact_bytes: Option<u64>,
        #[serde(default = "default_device")]
        pub device: String,
        #[serde(default)]
        pub capacity: Option<u64>,
        #[serde(default)]
        pub prompt_tokens: Vec<u32>,
        #[serde(default)]
        pub decode_tokens: Vec<u32>,
    }

    fn default_device() -> String {
        "cpu".to_string()
    }

    impl ModelManifest {
        /// Read and validate a manifest, resolving relative paths against its
        /// directory.
        pub fn from_path(path: impl AsRef<Path>) -> Result<Self, BenchError> {
            let path = path.as_ref();
            let text = std::fs::read_to_string(path).map_err(|error| BenchError::Io {
                path: path.to_path_buf(),
                detail: error.to_string(),
            })?;
            let mut manifest: ModelManifest =
                serde_json::from_str(&text).map_err(|error| BenchError::Manifest {
                    path: path.to_path_buf(),
                    detail: format!("not valid JSON: {error}"),
                })?;
            let base = path.parent().unwrap_or_else(|| Path::new("."));
            if manifest.source.is_relative() {
                manifest.source = base.join(&manifest.source);
            }
            if manifest.artifact.is_relative() {
                manifest.artifact = base.join(&manifest.artifact);
            }
            manifest.validate()?;
            Ok(manifest)
        }

        /// Check the fields a measurement depends on.
        pub fn validate(&self) -> Result<(), BenchError> {
            if self.schema != MANIFEST_SCHEMA {
                return Err(BenchError::Field {
                    field: "schema",
                    detail: format!(
                        "this build reads manifest schema {MANIFEST_SCHEMA}, found {}",
                        self.schema
                    ),
                });
            }
            if self.name.trim().is_empty() {
                return Err(BenchError::Field {
                    field: "name",
                    detail: "must name the model".to_string(),
                });
            }
            if self.revision.trim().is_empty() {
                return Err(BenchError::Field {
                    field: "revision",
                    detail: "must pin the model revision (or a local digest)".to_string(),
                });
            }
            if self.prompt_tokens.is_empty() {
                return Err(BenchError::Field {
                    field: "prompt_tokens",
                    detail: "needs at least one token ID for the prefill".to_string(),
                });
            }
            if self.decode_tokens.is_empty() {
                return Err(BenchError::Field {
                    field: "decode_tokens",
                    detail: "needs at least one token ID for the decode series".to_string(),
                });
            }
            if self.device.trim().is_empty() {
                return Err(BenchError::Field {
                    field: "device",
                    detail: "use `cpu` or `vulkan`".to_string(),
                });
            }
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Loaded and prepared models
    // -----------------------------------------------------------------------

    /// Everything `load` read from disk, before any plan or weight tensor is
    /// built. `prepare` consumes it.
    #[derive(Clone, Debug)]
    pub struct LoadedModel {
        manifest: ModelManifest,
        artifact: PreparedArtifact,
        import: ModelImport,
        artifact_bytes: u64,
        artifact_checksum: u64,
        source_bytes: u64,
        source_checksum: u64,
        weight_bytes: u64,
    }

    /// One immutable prepared model, the handle every session reuses.
    pub struct PreparedModel {
        manifest: ModelManifest,
        import: Arc<ModelImport>,
        model: CorePreparedModel,
        backend: Arc<dyn Backend>,
        artifact_bytes: u64,
        artifact_checksum: u64,
        source_bytes: u64,
        source_checksum: u64,
        weight_bytes: u64,
    }

    impl fmt::Debug for PreparedModel {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("PreparedModel")
                .field("name", &self.manifest.name)
                .field("revision", &self.manifest.revision)
                .field("device", &self.model.device())
                .field("artifact_bytes", &self.artifact_bytes)
                .field("checksum", &format_args!("{:016x}", self.artifact_checksum))
                .finish()
        }
    }

    /// FNV-1a 64-bit, the harness schema's non-cryptographic identity.
    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    fn read_bytes(path: &Path, what: &str) -> Result<Vec<u8>, BenchError> {
        std::fs::read(path).map_err(|error| BenchError::Io {
            path: path.to_path_buf(),
            detail: format!("could not read the {what}: {error}"),
        })
    }

    /// `load`: parse the manifest, read the artifact and the ONNX source, prove
    /// they describe one model. No planner and no weight tensor is built here.
    pub fn load(path: impl AsRef<Path>) -> Result<LoadedModel, BenchError> {
        let manifest = ModelManifest::from_path(path)?;
        let artifact_bytes = read_bytes(&manifest.artifact, "prepared artifact")?;
        if let Some(expected) = manifest.artifact_bytes {
            if artifact_bytes.len() as u64 != expected {
                return Err(BenchError::Field {
                    field: "artifact_bytes",
                    detail: format!(
                        "manifest says {expected}, {} has {} bytes",
                        manifest.artifact.display(),
                        artifact_bytes.len()
                    ),
                });
            }
        }
        let artifact = PreparedArtifact::from_bytes(&artifact_bytes)
            .and_then(|artifact| artifact.validate().map(|()| artifact))
            .map_err(|error| BenchError::Format {
                path: manifest.artifact.clone(),
                detail: error.to_string(),
            })?;

        let source_bytes = read_bytes(&manifest.source, "ONNX source")?;
        if let Some(expected) = manifest.source_bytes {
            if source_bytes.len() as u64 != expected {
                return Err(BenchError::Field {
                    field: "source_bytes",
                    detail: format!(
                        "manifest says {expected}, {} has {} bytes",
                        manifest.source.display(),
                        source_bytes.len()
                    ),
                });
            }
        }
        let import = import_onnx_model(&source_bytes).map_err(|error| BenchError::Import {
            path: manifest.source.clone(),
            detail: error.to_string(),
        })?;
        check_source_agrees(&artifact, &import)?;

        Ok(LoadedModel {
            artifact_checksum: fnv1a64(&artifact_bytes),
            artifact_bytes: artifact_bytes.len() as u64,
            source_checksum: fnv1a64(&source_bytes),
            source_bytes: source_bytes.len() as u64,
            weight_bytes: artifact.weight_bytes(),
            manifest,
            artifact,
            import,
        })
    }

    /// `prepare`: open the declared backend, build the initial plan and
    /// materialise the weights. One call, reused by every measured session.
    pub fn prepare(loaded: LoadedModel) -> Result<PreparedModel, BenchError> {
        let backend = open_backend(&loaded.manifest.device)?;
        let device = backend.device();
        let envelope = loaded.artifact.envelope();
        let capacity = loaded
            .manifest
            .capacity
            .unwrap_or_else(|| envelope.get(Symbol::Total).max());
        let geometry = loaded.import.geometry().clone();
        let kv_config = geometry.kv_cache_config(capacity, 1);
        kv_config
            .validate(&envelope)
            .map_err(|error| BenchError::Plan {
                detail: format!("capacity {capacity} does not fit the artifact window: {error}"),
            })?;

        let import = Arc::new(loaded.import);
        let planner = Planner::new(import.as_ref(), backend.as_ref());
        let dims = LiveDims::new(
            1,
            1,
            0,
            u64::from(geometry.query_heads),
            u64::from(geometry.kv_heads),
            u64::from(geometry.head_dim),
        )
        .map_err(|error| BenchError::Plan {
            detail: error.to_string(),
        })?;
        let plan = planner
            .plan(&envelope, dims)
            .map_err(|error| BenchError::Plan {
                detail: error.to_string(),
            })?
            .execution()
            .clone();
        let weights = loaded
            .artifact
            .to_model_weights()
            .map_err(|error| BenchError::Plan {
                detail: format!("weights: {error}"),
            })?;
        let weights = retag_weights(weights, device)?;
        let model = CorePreparedModel::new(plan, weights, device, envelope, Some(kv_config))
            .map_err(|error| BenchError::Plan {
                detail: error.to_string(),
            })?;

        Ok(PreparedModel {
            manifest: loaded.manifest,
            import,
            model,
            backend,
            artifact_bytes: loaded.artifact_bytes,
            artifact_checksum: loaded.artifact_checksum,
            source_bytes: loaded.source_bytes,
            source_checksum: loaded.source_checksum,
            weight_bytes: loaded.weight_bytes,
        })
    }

    fn open_backend(device: &str) -> Result<Arc<dyn Backend>, BenchError> {
        let device = device.trim();
        if device == "cpu" {
            return Ok(Arc::new(CpuBackend::new()));
        }
        #[cfg(feature = "vulkan")]
        {
            use fastnn::fn3::backend::vulkan::{
                adapters as vulkan_adapters, VulkanBackend, VulkanConfig,
            };
            use fastnn::fn3::DeviceRequest;
            let request = if device == "vulkan" {
                DeviceRequest::Vulkan { index: None }
            } else if let Some(rest) = device.strip_prefix("vulkan:") {
                if let Ok(index) = rest.parse::<u32>() {
                    DeviceRequest::Vulkan { index: Some(index) }
                } else {
                    let uuid = normalise_uuid(rest).ok_or_else(|| BenchError::Device {
                        detail: format!(
                            "`{device}` is not `vulkan`, `vulkan:<index>` or `vulkan:<uuid>`"
                        ),
                    })?;
                    let adapter = vulkan_adapters()
                        .iter()
                        .find(|adapter| {
                            normalise_uuid(&adapter.uuid_string()).as_deref() == Some(&uuid)
                        })
                        .ok_or_else(|| BenchError::Device {
                            detail: format!(
                                "no Vulkan adapter has UUID `{uuid}`; `fn3-infer diagnose` lists \
                                 the adapters this loader reports"
                            ),
                        })?;
                    DeviceRequest::Vulkan {
                        index: Some(adapter.index),
                    }
                }
            } else {
                return Err(BenchError::Device {
                    detail: format!("`{device}` is not `cpu` or `vulkan[:index|uuid]`"),
                });
            };
            let backend =
                VulkanBackend::open_with(&request, VulkanConfig::default()).map_err(|error| {
                    BenchError::Device {
                        detail: error.to_string(),
                    }
                })?;
            Ok(backend)
        }
        #[cfg(not(feature = "vulkan"))]
        Err(BenchError::Device {
            detail: format!(
                "`{device}` needs a build with the `vulkan` feature; this binary only has `cpu`"
            ),
        })
    }

    #[cfg(feature = "vulkan")]
    fn normalise_uuid(text: &str) -> Option<String> {
        let hex: String = text
            .chars()
            .filter(|character| *character != '-')
            .map(|character| character.to_ascii_lowercase())
            .collect();
        (hex.len() == 32 && hex.chars().all(|character| character.is_ascii_hexdigit()))
            .then_some(hex)
    }

    /// Compare the imported graph and weights with the artifact, the same rule
    /// the Python facade applies before it runs anything.
    fn check_source_agrees(
        artifact: &PreparedArtifact,
        import: &ModelImport,
    ) -> Result<(), BenchError> {
        let mut problems = Vec::new();
        if artifact.geometry() != import.geometry() {
            problems.push("geometry differs".to_string());
        }
        if artifact.graph().values() != import.graph().values() {
            problems.push("value table differs".to_string());
        }
        if artifact.graph().nodes() != import.graph().nodes() {
            problems.push("operation list differs".to_string());
        }
        let artifact_weights = artifact.weights();
        let source_weights = import.weights();
        if artifact_weights.len() != source_weights.len() {
            problems.push(format!(
                "artifact has {} weights, source has {}",
                artifact_weights.len(),
                source_weights.len()
            ));
        } else {
            for (index, (left, right)) in artifact_weights
                .iter()
                .zip(source_weights.iter())
                .enumerate()
            {
                if left.name() != right.name()
                    || left.dtype() != right.dtype()
                    || left.shape() != right.shape()
                    || left.bytes() != right.bytes()
                {
                    problems.push(format!("weight {index} `{}` differs", left.name()));
                    break;
                }
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(BenchError::Disagree {
                detail: format!(
                    "{}; regenerate the artifact from this source with `fn3-infer prepare`",
                    problems.join("; ")
                ),
            })
        }
    }

    fn retag_weights(weights: ModelWeights, device: DeviceId) -> Result<ModelWeights, BenchError> {
        if device == DeviceId::Cpu {
            return Ok(weights);
        }
        let mut named = Vec::with_capacity(weights.len());
        for index in 0..weights.len() {
            let tensor = weights.get(index).ok_or_else(|| BenchError::Plan {
                detail: format!("weight index {index} is out of range"),
            })?;
            let meta = fastnn::fn3::tensor::TensorMeta::contiguous(
                tensor.shape().clone(),
                tensor.dtype(),
                device,
            )
            .map_err(|error| BenchError::Plan {
                detail: error.to_string(),
            })?;
            let retagged =
                Tensor::from_storage(tensor.storage().clone(), meta).map_err(|error| {
                    BenchError::Plan {
                        detail: error.to_string(),
                    }
                })?;
            let name = weights.name(index).ok_or_else(|| BenchError::Plan {
                detail: format!("weight index {index} has no name"),
            })?;
            named.push((name.to_string(), retagged));
        }
        ModelWeights::with_names(named).map_err(|error| BenchError::Plan {
            detail: error.to_string(),
        })
    }

    // -----------------------------------------------------------------------
    // Running
    // -----------------------------------------------------------------------

    /// The observable result of one synchronized prefill or decode call.
    #[derive(Clone, Debug)]
    pub struct StepReport {
        /// Host wall time around the call, in milliseconds.
        pub host_ms: f64,
        /// Device time when the backend reports it.
        pub device_time_ms: Option<f64>,
        /// Logits returned for the final position.
        pub logits: Vec<f32>,
        /// Operation counters after the call.
        pub counters: BTreeMap<String, u64>,
    }

    impl StepReport {
        /// Greedy top-1 token ID, or `None` when no finite logit exists.
        pub fn top1(&self) -> Option<u32> {
            let mut best = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for (index, value) in self.logits.iter().enumerate() {
                if value.is_finite() && *value > best_value {
                    best_value = *value;
                    best = index;
                }
            }
            best_value.is_finite().then_some(best as u32)
        }
    }

    /// One session over one prepared model. The session, and therefore the KV
    /// cache and plan cache, lives across every `prefill` and `decode` call
    /// until the caller drops or resets it.
    pub struct RunSession<'a> {
        session: Session<'a>,
    }

    impl RunSession<'_> {
        /// Time a prefill. The session must be reset first.
        pub fn prefill(&mut self, tokens: &[u32]) -> Result<StepReport, BenchError> {
            let start = Instant::now();
            let logits = self.session.prefill(tokens)?;
            let host_ms = start.elapsed().as_secs_f64() * 1000.0;
            self.report(host_ms, logits)
        }

        /// Time one cached decode step.
        pub fn decode(&mut self, token: u32) -> Result<StepReport, BenchError> {
            let start = Instant::now();
            let logits = self.session.decode(&[token])?;
            let host_ms = start.elapsed().as_secs_f64() * 1000.0;
            self.report(host_ms, logits)
        }

        /// Drop the live cache extents and return to `Empty`. Buffers and plans
        /// are kept, so the next prefill at the same shape is a plan-cache hit.
        pub fn reset(&mut self) {
            self.session.reset();
        }

        /// Tokens currently in the cache.
        pub fn cache_len(&self) -> u64 {
            self.session.cache_len()
        }

        /// Plans built so far by this session's plan cache.
        pub fn plan_misses(&self) -> u64 {
            self.session.plan_misses()
        }

        fn report(&self, host_ms: f64, logits: Tensor) -> Result<StepReport, BenchError> {
            let values = logits.as_f32_slice().map_err(|error| BenchError::Session {
                detail: format!("logits are not a readable f32 tensor: {error}"),
            })?;
            if !values.iter().all(|value| value.is_finite()) {
                return Err(BenchError::Session {
                    detail: "logits contain a non-finite value; a benchmark of NaN is not a \
                             measurement"
                        .to_string(),
                });
            }
            Ok(StepReport {
                host_ms,
                device_time_ms: None,
                logits: values.to_vec(),
                counters: self.counters(),
            })
        }

        fn counters(&self) -> BTreeMap<String, u64> {
            let mut counters = BTreeMap::new();
            counters.insert("plan.hits".to_string(), self.session.plan_hits());
            counters.insert("plan.misses".to_string(), self.session.plan_misses());
            counters.insert("plan.evictions".to_string(), self.session.plan_evictions());
            counters.insert(
                "slot.allocations".to_string(),
                self.session.slot_allocations(),
            );
            counters.insert("slot.reuses".to_string(), self.session.slot_reuses());
            counters.insert(
                "slot.bytes_allocated".to_string(),
                self.session.slot_bytes_allocated(),
            );
            counters.insert("kv.bytes".to_string(), self.session.kv_bytes());
            counters.insert("session.step".to_string(), self.session.step());
            counters.insert("session.cache_len".to_string(), self.session.cache_len());
            counters.insert(
                "session.tokens_consumed".to_string(),
                self.session.tokens_consumed(),
            );
            counters
        }
    }

    impl PreparedModel {
        /// The manifest this model was loaded from.
        pub fn manifest(&self) -> &ModelManifest {
            &self.manifest
        }

        /// The backend device the model and every session execute on.
        pub fn device(&self) -> DeviceId {
            self.model.device()
        }

        /// Decoder geometry in the benchmark schema's terms.
        pub fn geometry(&self) -> Geometry {
            let geometry = self.import.geometry();
            let envelope = self.model.envelope();
            Geometry {
                batch: 1,
                prompt_tokens: self.manifest.prompt_tokens.len() as u64,
                decode_tokens: self.manifest.decode_tokens.len() as u64,
                context_capacity: envelope.get(Symbol::Total).max(),
                hidden_size: u64::from(geometry.hidden_size),
                num_heads: u64::from(geometry.query_heads),
                num_kv_heads: (geometry.kv_heads != geometry.query_heads)
                    .then_some(u64::from(geometry.kv_heads)),
                head_dim: u64::from(geometry.head_dim),
                num_layers: u64::from(geometry.layers),
                intermediate_size: u64::from(geometry.mlp_hidden_size),
                vocab_size: u64::from(geometry.vocab_size),
                attention: "causal_kv_cache".to_string(),
                kv_cache: true,
            }
        }

        /// Model identity recorded with every result.
        pub fn identity(&self) -> ModelInfo {
            ModelInfo {
                name: self.manifest.name.clone(),
                revision: self.manifest.revision.clone(),
                source: "local-artifact".to_string(),
                parameter_count: None,
                weights_bytes: Some(self.weight_bytes),
                seed: None,
                checksums: vec![
                    Checksum {
                        subject: "artifact".to_string(),
                        algorithm: "fnv1a64".to_string(),
                        value: format!("{:016x}", self.artifact_checksum),
                    },
                    Checksum {
                        subject: "source".to_string(),
                        algorithm: "fnv1a64".to_string(),
                        value: format!("{:016x}", self.source_checksum),
                    },
                    Checksum {
                        subject: "artifact-bytes".to_string(),
                        algorithm: "count".to_string(),
                        value: self.artifact_bytes.to_string(),
                    },
                    Checksum {
                        subject: "source-bytes".to_string(),
                        algorithm: "count".to_string(),
                        value: self.source_bytes.to_string(),
                    },
                ],
            }
        }

        /// Open one session that reuses this prepared model. The caller decides
        /// how long it lives; a whole prefill/decode chain shares one session.
        pub fn session(&self) -> Result<RunSession<'_>, BenchError> {
            let session =
                Session::with_runtime(&self.model, self.import.as_ref(), self.backend.as_ref())?;
            Ok(RunSession { session })
        }
    }

    /// The adapter runtime: owns the selected device identity and the most
    /// recent optional observations.
    pub struct Runtime {
        last_device_time_ms: Option<f64>,
        peak_device_memory_bytes: Option<u64>,
        last_operation_counters: BTreeMap<String, u64>,
    }

    impl Runtime {
        /// Select the backend named by the manifest and initialise the optional
        /// observations. `None`, never `0`, means "not reported".
        pub fn new(device: impl Into<String>) -> Result<Self, BenchError> {
            let device = device.into();
            // Opening once here fails before 28 MB of I/O if the device is
            // unavailable or the feature is off.
            let _ = open_backend(&device)?;
            Ok(Self {
                last_device_time_ms: None,
                peak_device_memory_bytes: None,
                last_operation_counters: BTreeMap::new(),
            })
        }

        pub fn load(&self, manifest: impl AsRef<Path>) -> Result<LoadedModel, BenchError> {
            load(manifest)
        }

        pub fn prepare(&self, loaded: LoadedModel) -> Result<PreparedModel, BenchError> {
            prepare(loaded)
        }

        /// Record one step's counters and optional device observations.
        pub fn observe(&mut self, report: &StepReport) {
            self.last_device_time_ms = report.device_time_ms;
            self.last_operation_counters = report.counters.clone();
        }

        /// Device time for the most recent operation, or `None` when the
        /// backend has no timestamp query.
        pub fn last_device_time_ms(&self) -> Option<f64> {
            self.last_device_time_ms
        }

        /// Peak device memory in bytes, or `None` when the backend cannot
        /// report it. Never `Some(0)` as a stand-in for "unknown".
        pub fn peak_device_memory_bytes(&self) -> Option<u64> {
            self.peak_device_memory_bytes
        }

        /// Operation counters for the most recent step.
        pub fn last_operation_counters(&self) -> BTreeMap<String, u64> {
            self.last_operation_counters.clone()
        }
    }

    // -----------------------------------------------------------------------
    // Suite
    // -----------------------------------------------------------------------

    fn device_time(device: &str, samples: &[Option<f64>]) -> DeviceTime {
        let measured: Vec<f64> = samples.iter().flatten().copied().collect();
        if measured.is_empty() {
            DeviceTime::unavailable(format!(
                "the {device} backend reports no device timestamps or per-operation device time"
            ))
        } else {
            DeviceTime::measured(format!("{device}-device-time"), measured)
        }
    }

    /// Run the pinned model through the fn3 adapter.
    pub fn run(suite: Suite, config: &BenchConfig) -> Result<SuiteOutcome, BenchError> {
        if suite != Suite::Model {
            return Err(BenchError::Field {
                field: "suite",
                detail: format!(
                    "the fn3 adapter measures the pinned prepared model; use `--suite model` \
                     (got `{}`)",
                    suite.as_str()
                ),
            });
        }
        let manifest_path = config
            .fn3_manifest
            .as_ref()
            .ok_or_else(|| BenchError::Field {
                field: "--fn3-manifest",
                detail: "the fn3 adapter needs a model manifest; see tools/bench/README.md"
                    .to_string(),
            })?;
        let manifest = ModelManifest::from_path(manifest_path)?;
        let mut runtime = Runtime::new(manifest.device.clone())?;

        // 1. Load: manifest + artifact parse + ONNX import + agreement check.
        let load_reps = config.load_repetitions();
        let mut load_samples = Vec::with_capacity(load_reps as usize);
        for _ in 0..load_reps {
            let start = Instant::now();
            let _loaded = runtime.load(manifest_path)?;
            load_samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // 2. Prepare: each sample gets a loaded model read outside the timed
        //    region, so the series measures planning and weight materialisation.
        let prepare_reps = config.quick_samples(2, 3);
        let mut prepare_inputs = Vec::with_capacity(prepare_reps as usize);
        for _ in 0..prepare_reps {
            prepare_inputs.push(runtime.load(manifest_path)?);
        }
        let mut prepare_samples = Vec::with_capacity(prepare_reps as usize);
        let mut prepared = None;
        for input in prepare_inputs {
            let start = Instant::now();
            let model = runtime.prepare(input)?;
            prepare_samples.push(start.elapsed().as_secs_f64() * 1000.0);
            prepared = Some(model);
        }
        let prepared = prepared.ok_or_else(|| BenchError::Field {
            field: "prepare",
            detail: "no prepare repetition produced a model".to_string(),
        })?;

        let identity = prepared.identity();
        let geometry = prepared.geometry();
        let device = prepared.device().diagnostic();
        let prompt = prepared.manifest().prompt_tokens.clone();
        let decode_token = prepared.manifest().decode_tokens[0];
        let prefill_warmup = config.quick_warmup(1, 2);
        let prefill_samples_count = config.quick_samples(3, 10);
        let decode_warmup = config.quick_warmup(2, 5);
        let decode_steps = config.samples.unwrap_or(config.decode_tokens.max(1)).max(1);

        let mut prefill_samples = Vec::with_capacity(prefill_samples_count as usize);
        let mut prefill_device = Vec::with_capacity(prefill_samples_count as usize);
        let mut prefill_misses = Vec::with_capacity(prefill_samples_count as usize);
        let mut decode_samples = Vec::with_capacity(decode_steps as usize);
        let mut decode_device = Vec::with_capacity(decode_steps as usize);
        let mut decode_misses = Vec::with_capacity(decode_steps as usize);
        let mut last_top1 = None;
        let mut cache_len_after_prefill = 0;

        {
            let mut session = prepared.session()?;

            // Prime both shapes once: the prefill plan and the one-token
            // decode plan at past = prompt length.
            for _ in 0..prefill_warmup {
                session.reset();
                let report = session.prefill(&prompt)?;
                runtime.observe(&report);
            }
            for _ in 0..decode_warmup {
                let report = session.decode(decode_token)?;
                runtime.observe(&report);
                session.reset();
                let replay = session.prefill(&prompt)?;
                runtime.observe(&replay);
            }

            // Measured prefill: reset between samples, same shape, cached plan.
            for _ in 0..prefill_samples_count {
                session.reset();
                let before = session.plan_misses();
                let report = session.prefill(&prompt)?;
                prefill_misses.push(report.counters["plan.misses"].saturating_sub(before) as f64);
                cache_len_after_prefill = session.cache_len();
                runtime.observe(&report);
                prefill_samples.push(report.host_ms);
                prefill_device.push(report.device_time_ms);
            }

            // Measured decode: one token at the same `past` every time. The
            // untimed reset/replay between samples restores that `past`, so a
            // plan miss here would be a real per-call recompilation.
            for index in 0..decode_steps {
                let before = session.plan_misses();
                let report = session.decode(decode_token)?;
                decode_misses.push(report.counters["plan.misses"].saturating_sub(before) as f64);
                last_top1 = report.top1();
                runtime.observe(&report);
                decode_samples.push(report.host_ms);
                decode_device.push(report.device_time_ms);
                if index + 1 < decode_steps {
                    session.reset();
                    let replay = session.prefill(&prompt)?;
                    runtime.observe(&replay);
                }
            }
        }

        let mut metrics = Vec::new();
        metrics.push(
            Metric::measured_time("model.load", "load", load_samples, 0, load_reps)
                .with_meta(
                    "includes",
                    "artifact parse + ONNX import + source/artifact check",
                )
                .with_meta(
                    "artifact_bytes",
                    manifest_artifact_bytes(&manifest).to_string(),
                )
                .with_meta("source_bytes", manifest_source_bytes(&manifest).to_string()),
        );
        metrics.push(
            Metric::measured_time("model.prepare", "prepare", prepare_samples, 0, prepare_reps)
                .with_meta(
                    "includes",
                    "planner + initial plan + weight materialisation",
                )
                .with_meta("cached_plan_shapes", "prefill (seq,0), decode (1,prompt)")
                .with_meta("device", device.clone()),
        );
        metrics.push(
            Metric::measured_time(
                "prefill.time_to_first_token",
                "prefill",
                prefill_samples.clone(),
                prefill_warmup,
                prefill_samples_count,
            )
            .with_meta("prompt_tokens", prompt.len().to_string())
            .with_meta("cache_after", cache_len_after_prefill.to_string())
            .with_meta("includes", "one cached-prefill forward pass")
            .with_device_time(device_time(&device, &prefill_device)),
        );
        if !prefill_samples.is_empty() {
            metrics.push(
                Metric::measured_value(
                    "prefill.throughput",
                    "throughput",
                    "tokens/s",
                    crate::stats::throughput_samples(&prefill_samples, prompt.len() as f64),
                    0,
                    1,
                )
                .with_meta("clock", "derived from host_monotonic prefill samples"),
            );
        }
        let mut decode_metric = Metric::measured_time(
            "decode.per_token",
            "decode",
            decode_samples.clone(),
            decode_warmup,
            1,
        )
        .with_meta("decode_steps", decode_steps.to_string())
        .with_meta("includes", "one-token cached decode at a fixed past")
        .with_meta(
            "top1",
            last_top1
                .map(|token| token.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
        )
        .with_device_time(device_time(&device, &decode_device))
        .with_meta(
            "last_device_time_ms",
            runtime
                .last_device_time_ms()
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "unavailable".to_string()),
        );
        decode_metric.counters = runtime.last_operation_counters();
        metrics.push(decode_metric);
        if !decode_samples.is_empty() {
            metrics.push(
                Metric::measured_value(
                    "decode.throughput",
                    "throughput",
                    "tokens/s",
                    crate::stats::throughput_samples(&decode_samples, 1.0),
                    0,
                    1,
                )
                .with_meta("clock", "derived from host_monotonic decode samples"),
            );
        }
        metrics.push(match crate::env::peak_rss_bytes() {
            Some(bytes) => Metric::measured_value(
                "memory.peak_rss",
                "memory",
                "MiB",
                vec![bytes as f64 / (1024.0 * 1024.0)],
                0,
                1,
            )
            .with_meta("scope", "whole process, VmHWM after all measurements")
            .with_meta("source", "/proc/self/status"),
            None => Metric::unavailable(
                "memory.peak_rss",
                "memory",
                "MiB",
                "/proc/self/status VmHWM is unavailable on this host",
            ),
        });
        metrics.push(match runtime.peak_device_memory_bytes() {
            Some(bytes) => Metric::measured_value(
                "memory.device_peak",
                "memory",
                "MiB",
                vec![bytes as f64 / (1024.0 * 1024.0)],
                0,
                1,
            )
            .with_meta("source", "backend memory stats"),
            None => Metric::unavailable(
                "memory.device_peak",
                "memory",
                "MiB",
                format!(
                    "the {device} backend reports no peak device memory, so the value is \
                     unknown rather than zero"
                ),
            )
            .with_meta("backend", device.clone()),
        });
        metrics.push(
            Metric::measured_value(
                "allocation.plan_misses_per_prefill",
                "allocation",
                "count",
                prefill_misses.clone(),
                0,
                1,
            )
            .with_meta("expectation", "0 after the warmup plan is cached"),
        );
        metrics.push(
            Metric::measured_value(
                "allocation.plan_misses_per_decode",
                "allocation",
                "count",
                decode_misses.clone(),
                0,
                1,
            )
            .with_meta(
                "expectation",
                "0; a miss is per-call recompilation at a fixed past",
            ),
        );

        let mut outcome = SuiteOutcome::new(metrics);
        outcome.model = identity;
        outcome.geometry = geometry;
        outcome.precision = Precision {
            weights_dtype: "f32".to_string(),
            activations_dtype: "f32".to_string(),
            accumulation_dtype: "f32".to_string(),
            quantization: None,
        };
        outcome.notes.push(
            "Real pinned model, not a synthetic stack: the manifest names the ONNX source and the \
             prepared `.fnn3` artifact, and load refuses a pair that disagrees."
                .to_string(),
        );
        outcome.notes.push(format!(
            "One prepared model and one session are reused: prefill plan misses \
             [{}], decode plan misses [{}]. A non-zero decode miss at a fixed past would be \
             per-call recompilation.",
            join_counts(&prefill_misses),
            join_counts(&decode_misses)
        ));
        if device != "cpu" {
            outcome.notes.push(format!(
                "Backend `{device}`: device time and device memory are reported only when the \
                 backend exposes them; unavailable values are `None`, never zero."
            ));
        }
        Ok(outcome)
    }

    fn manifest_artifact_bytes(manifest: &ModelManifest) -> String {
        manifest
            .artifact_bytes
            .map(|bytes| bytes.to_string())
            .unwrap_or_else(|| "unchecked".to_string())
    }

    fn manifest_source_bytes(manifest: &ModelManifest) -> String {
        manifest
            .source_bytes
            .map(|bytes| bytes.to_string())
            .unwrap_or_else(|| "unchecked".to_string())
    }

    fn join_counts(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.0}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn manifest_json() -> String {
            serde_json::json!({
                "schema": 1,
                "name": "tiny",
                "revision": "test",
                "source": "model.onnx",
                "artifact": "model.fnn3",
                "device": "cpu",
                "prompt_tokens": [1, 2],
                "decode_tokens": [3]
            })
            .to_string()
        }

        #[test]
        fn manifest_rejects_a_future_schema_and_names_the_field() {
            let mut value: serde_json::Value =
                serde_json::from_str(&manifest_json()).expect("json");
            value["schema"] = serde_json::json!(2);
            let path = std::env::temp_dir().join("fn3-bench-manifest-schema.json");
            std::fs::write(&path, value.to_string()).expect("write");
            let error = ModelManifest::from_path(&path).expect_err("schema must be refused");
            std::fs::remove_file(&path).ok();
            assert!(error.to_string().contains("schema"), "{error}");
            assert!(error.to_string().contains("found 2"), "{error}");
        }

        #[test]
        fn manifest_resolves_paths_against_its_own_directory() {
            let directory = std::env::temp_dir().join("fn3-bench-manifest-relative");
            std::fs::create_dir_all(&directory).expect("mkdir");
            let path = directory.join("manifest.json");
            std::fs::write(&path, manifest_json()).expect("write");
            let manifest = ModelManifest::from_path(&path).expect("manifest");
            std::fs::remove_file(&path).ok();
            std::fs::remove_dir(&directory).ok();
            assert_eq!(manifest.source, directory.join("model.onnx"));
            assert_eq!(manifest.artifact, directory.join("model.fnn3"));
            assert_eq!(manifest.device, "cpu");
        }

        #[test]
        fn manifest_requires_tokens_for_both_phases() {
            let mut value: serde_json::Value =
                serde_json::from_str(&manifest_json()).expect("json");
            value["decode_tokens"] = serde_json::json!([]);
            let json: ModelManifest = serde_json::from_value(value).expect("decode");
            let error = json.validate().expect_err("empty decode must be refused");
            assert!(error.to_string().contains("decode_tokens"), "{error}");
        }

        #[test]
        fn missing_manifest_is_an_error_not_a_zero_measurement() {
            let config = BenchConfig {
                quick: true,
                seed: 1,
                prompt_tokens: 2,
                layers: 1,
                decode_tokens: 1,
                samples: Some(1),
                warmup: Some(0),
                manifest_dir: std::path::PathBuf::from("."),
                reuse_criterion: false,
                fn3_manifest: None,
            };
            let error = run(Suite::Model, &config).expect_err("manifest is required");
            assert!(error.to_string().contains("--fn3-manifest"), "{error}");
            assert!(
                error.to_string().contains("tools/bench/README.md"),
                "{error}"
            );
        }

        #[test]
        fn the_adapter_is_available_only_when_the_feature_is_linked() {
            assert!(super::super::is_available());
        }

        #[test]
        fn errors_display_the_stage_and_cause() {
            let error = BenchError::Field {
                field: "prompt_tokens",
                detail: "needs at least one token ID".to_string(),
            };
            assert_eq!(
                error.to_string(),
                "fn3 manifest field `prompt_tokens`: needs at least one token ID"
            );
        }
    }
}
