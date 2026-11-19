//! Trained-model INT8-weight gate (#74): Vulkan resident W8A16 against ORT.
//!
//! This is the automated gate the 3.0 scope was missing: the trained GPT-2
//! artifact runs with `SessionOptions::weight_int8`, the default resident
//! session and validation on, on the pinned physical GTX 1080 Ti. It reuses the
//! committed ORT reference rows that `fn3_trained_parity.rs` already checks
//! (length and SHA-256 from the fixture manifest), so it adds no model or
//! numeric fixture.
//!
//! The gate is `#[ignore]` and opt-in through `FN3_TRAINED_INT8=1`. Once
//! enabled it never skips: a missing artifact, missing reference, a build
//! without `--features vulkan`, a non-physical or unmatched device, or a
//! missing validation layer is a named failure. It is independent of
//! `FN3_TRAINED_VULKAN`.
//!
//! ```bash
//! env -u VK_INSTANCE_LAYERS -u FN3_VK_STAGED \
//!   FN3_TRAINED_INT8=1 FN3_VK_REQUIRE_PHYSICAL=1 FN3_VK_VALIDATION=1 \
//!   FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//!   flock /tmp/opencode/fn3-measure.lock \
//!   cargo test --release --features vulkan --test fn3_trained_int8 \
//!     -- trained_gpt2_int8_matches_ort_on_physical_vulkan --ignored --nocapture
//! ```
//!
//! What it asserts, on the 13-token prompt and decode tokens 640 and 11:
//!
//! * top-1 exact against ORT on all three phases;
//! * FP32 top-5 is 5/5 against ORT on all three phases (the reference control);
//! * INT8 top-5 follows #68 §6: 5/5 when the ORT 5th-place gap is decisive
//!   (> 2 * centered max), 4/5 otherwise. The trained decode boundaries are
//!   near-ties (7.9e-4 and 1.0e-2), so decode measures 4/5 and prefill 5/5;
//!   both sets and the raw overlap are printed. A literal 5/5 there would be a
//!   tolerance exception, not a stronger gate;
//! * the #68 logits gate: prefill KL <= 1e-3, decode KL <= 0.02, centered max
//!   <= 0.8 and centered RMSE <= 0.25;
//! * the live plan packs exactly 48 MatMuls and leaves exactly the 24 attention
//!   products plus the tied LM head on FP32, by RHS name;
//! * packed prepare traffic is exactly the 90,243,072-byte payload, prepare
//!   staging is not that payload, there are zero input readbacks, and
//!   validation reports 0 errors / 0 warnings;
//! * CPU INT8 and Vulkan INT8 logits agree within 1e-3 max abs on every phase.

#![allow(clippy::chunks_exact_to_as_chunks)]

#[path = "fn3_e2e/harness.rs"]
mod harness;

fn flag(name: &str) -> bool {
    match std::env::var(name).as_deref() {
        Ok("1") => true,
        Ok("0") | Err(std::env::VarError::NotPresent) => false,
        Ok(other) => panic!("[fn3-trained-int8][invalid-env] {name}=`{other}`; use 1 or 0"),
        Err(error) => panic!("[fn3-trained-int8][invalid-env] {name}: {error}"),
    }
}

fn named_fail(code: &str, detail: impl std::fmt::Display) -> ! {
    panic!("[fn3-trained-int8][{code}] {detail}")
}

#[test]
#[ignore = "opt-in: set FN3_TRAINED_INT8=1 with the trained artifact and the physical Vulkan device"]
fn trained_gpt2_int8_matches_ort_on_physical_vulkan() {
    if !flag("FN3_TRAINED_INT8") {
        println!(
            "[fn3-trained-int8][skip] FN3_TRAINED_INT8 is not 1; the gate is opt-in and this \
             run did not execute it"
        );
        return;
    }

    #[cfg(feature = "vulkan")]
    gate::run();

    #[cfg(not(feature = "vulkan"))]
    named_fail(
        "missing-vulkan-support",
        "FN3_TRAINED_INT8=1 needs a `--features vulkan` build; this binary has no Vulkan backend",
    );
}

#[cfg(feature = "vulkan")]
mod gate {
    use std::path::{Path, PathBuf};

    use fastnn::inference::backend::vulkan::instance::validation_counts;
    use fastnn::inference::backend::vulkan::resident::ResidencyPhase;
    use fastnn::inference::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
    use fastnn::inference::device::DeviceRequest;
    use fastnn::inference::import::ModelImport;
    use fastnn::inference::model::format::PreparedArtifact;
    use fastnn::inference::model::{ModelWeights, Planner, PreparedModel, Session, SessionOptions};
    use fastnn::inference::shape::ShapeEnvelope;
    use fastnn::inference::{Backend, CpuBackend, DeviceId, LiveDims, Tensor, TensorMeta};

    use super::{harness, named_fail};

    const PROMPT: [u32; 13] = [
        464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257,
    ];
    const DECODE: [u32; 2] = [640, 11];
    const VOCAB: usize = 50_257;
    /// Cache window used by the FP32 gate; the three phases only need 15 tokens.
    const CONTEXT: u64 = 32;
    const PHASES: [&str; 3] = ["prefill", "decode.step1", "decode.step2"];
    const DEFAULT_ARTIFACT: &str = "/tmp/opencode/fn3-68/trained-gpt2.fnn3";
    const PACKED_MATMULS: usize = 48;
    const FP32_MATMULS: usize = 25;
    const MATMULS: usize = PACKED_MATMULS + FP32_MATMULS;
    const PACKED_PAYLOAD_BYTES: u64 = 90_243_072;
    const CPU_VS_VULKAN_INT8_MAX_ABS: f64 = 1e-3;
    const PREFILL_KL_BOUND: f64 = 1e-3;
    const DECODE_KL_BOUND: f64 = 0.02;
    const CENTERED_MAX_BOUND: f64 = 0.8;
    const CENTERED_RMSE_BOUND: f64 = 0.25;
    const MAX_PREPARE_STAGING_BYTES: u64 = 1 << 20;

    pub(super) fn run() {
        reject_external_overrides();

        let artifact_path = artifact_path();
        let (import, artifact_bytes) = load_artifact(&artifact_path);
        let references = load_references();
        println!(
            "[fn3-trained-int8][artifact] path={} bytes={} vocab={} layers={} heads={}/{} \
             head_dim={}",
            artifact_path.display(),
            artifact_bytes,
            import.geometry().vocab_size,
            import.geometry().layers,
            import.geometry().query_heads,
            import.geometry().kv_heads,
            import.geometry().head_dim,
        );

        let backend = open_pinned_physical();
        let validation_before = validation_counts();
        println!(
            "[fn3-trained-int8][device] name=\"{}\" uuid={} software={} validation={}",
            backend.adapter().name,
            backend.adapter().uuid_string(),
            backend.is_software(),
            backend.validation_enabled(),
        );

        // ---- CPU INT8 arm --------------------------------------------------
        let cpu_backend = CpuBackend::new();
        let (cpu_model, _) = prepare_model(&import, DeviceId::Cpu, &cpu_backend);
        let cpu_logits = {
            let mut session = Session::with_runtime_options(
                &cpu_model,
                &import,
                &cpu_backend,
                SessionOptions {
                    weight_int8: true,
                    ..SessionOptions::default()
                },
            )
            .unwrap_or_else(|error| named_fail("cpu-session", error));
            run_phases(&mut session, "cpu-int8", true)
        };
        drop(cpu_model);

        // ---- Vulkan INT8 arm (the gate) -----------------------------------
        let device = backend.device_id();
        let (vulkan_model, _) = prepare_model(&import, device, &*backend);
        backend.residency().clear();
        backend.reset_residency_stats();
        backend.set_phase(ResidencyPhase::Prepare);
        let vulkan_int8 = {
            let mut session = Session::with_runtime_options(
                &vulkan_model,
                &import,
                &*backend,
                SessionOptions {
                    weight_int8: true,
                    ..SessionOptions::default()
                },
            )
            .unwrap_or_else(|error| named_fail("vulkan-int8-session", error));
            assert!(
                !session.model().device().is_cpu(),
                "the INT8 gate must run on the Vulkan device"
            );
            run_phases(&mut session, "vulkan-int8", true)
        };
        assert_int8_traffic(&backend);

        // ---- Vulkan FP32 arm (the FP32-vs-INT8 reference) -----------------
        let vulkan_fp32 = {
            let mut session = Session::with_runtime_options(
                &vulkan_model,
                &import,
                &*backend,
                SessionOptions::default(),
            )
            .unwrap_or_else(|error| named_fail("vulkan-fp32-session", error));
            assert!(
                session.weight_int8_report().is_none(),
                "a default session must not carry an INT8 report"
            );
            run_phases(&mut session, "vulkan-fp32", false)
        };

        let validation_after = validation_counts();
        assert_eq!(
            validation_after,
            (0, 0),
            "validation reported {validation_after:?} (errors, warnings)"
        );
        assert_eq!(
            validation_after, validation_before,
            "validation messages changed during the trained INT8 gate"
        );
        println!(
            "[fn3-trained-int8][validation] errors={} warnings={}",
            validation_after.0, validation_after.1
        );

        let mut failures = Vec::new();
        for (index, phase) in PHASES.iter().enumerate() {
            assert_phase(
                phase,
                index,
                &vulkan_int8[index],
                &vulkan_fp32[index],
                &cpu_logits[index],
                &references[index],
                &mut failures,
            );
        }
        if !failures.is_empty() {
            named_fail(
                "phase-check",
                format!(
                    "{} phase check(s) failed:\n{}",
                    failures.len(),
                    failures.join("\n")
                ),
            );
        }
    }

    fn reject_external_overrides() {
        if let Ok(layers) = std::env::var("VK_INSTANCE_LAYERS") {
            if !layers.trim().is_empty() {
                named_fail(
                    "external-vulkan-layers",
                    format!(
                        "VK_INSTANCE_LAYERS={layers:?}; the gate needs the default instance layers"
                    ),
                );
            }
        }
        match std::env::var("FN3_VK_STAGED").as_deref() {
            Ok(value)
                if matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                ) =>
            {
                named_fail(
                    "staged-override",
                    "FN3_VK_STAGED is set; the gate requires the default resident session",
                )
            }
            _ => {}
        }
    }

    fn artifact_path() -> PathBuf {
        std::env::var_os("FN3_TRAINED_ARTIFACT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_ARTIFACT))
    }

    fn load_artifact(path: &Path) -> (ModelImport, u64) {
        if !path.is_file() {
            named_fail(
                "missing-artifact",
                format!(
                    "{} does not exist; set FN3_TRAINED_ARTIFACT to the prepared trained .fnn3",
                    path.display()
                ),
            );
        }
        let bytes = std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or_else(|error| {
                named_fail("missing-artifact", format!("{}: {error}", path.display()))
            });
        let artifact = PreparedArtifact::read_file(path).unwrap_or_else(|error| {
            named_fail(
                "unreadable-artifact",
                format!("{}: {error}", path.display()),
            )
        });
        let geometry = artifact.geometry();
        let expected = (50_257, 12, 12, 12, 64);
        let actual = (
            geometry.vocab_size,
            geometry.layers,
            geometry.query_heads,
            geometry.kv_heads,
            geometry.head_dim,
        );
        if actual != expected {
            named_fail(
                "wrong-artifact",
                format!(
                    "{} is not the pinned trained GPT-2 geometry: vocab/layers/heads/kv/head_dim \
                     = {actual:?}, expected {expected:?}",
                    path.display()
                ),
            );
        }
        (artifact.into_model_import(), bytes)
    }

    fn load_references() -> [Vec<f32>; 3] {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/trained-gpt2");
        let manifest_path = root.join("manifest.json");
        let manifest_bytes = std::fs::read(&manifest_path).unwrap_or_else(|error| {
            named_fail(
                "missing-reference",
                format!("{}: {error}", manifest_path.display()),
            )
        });
        let manifest: serde_json::Value =
            serde_json::from_slice(&manifest_bytes).unwrap_or_else(|error| {
                named_fail(
                    "malformed-reference",
                    format!("{}: {error}", manifest_path.display()),
                )
            });
        [
            "prefill.p13.last_logits",
            "decode.step1.logits",
            "decode.step2.logits",
        ]
        .map(|name| {
            let entry = manifest["tensors"]
                .as_array()
                .and_then(|tensors| tensors.iter().find(|tensor| tensor["name"] == name))
                .unwrap_or_else(|| {
                    named_fail("missing-reference", format!("manifest entry {name}"))
                });
            let path =
                if name == "decode.step2.logits" {
                    root.join("gate-reference/decode.step2.logits.f32")
                } else {
                    root.join(entry["path"].as_str().unwrap_or_else(|| {
                        named_fail("missing-reference", format!("{name}: path"))
                    }))
                };
            let bytes = std::fs::read(&path).unwrap_or_else(|error| {
                named_fail("missing-reference", format!("{}: {error}", path.display()))
            });
            if entry["dtype"] != "f32" {
                named_fail("reference-dtype", format!("{}: not f32", path.display()));
            }
            if entry["bytes"].as_u64() != Some((VOCAB * 4) as u64) {
                named_fail(
                    "reference-length",
                    format!(
                        "{}: manifest byte length is not {}",
                        path.display(),
                        VOCAB * 4
                    ),
                );
            }
            if bytes.len() != VOCAB * 4 {
                named_fail(
                    "reference-length",
                    format!(
                        "{}: {} bytes, expected {}",
                        path.display(),
                        bytes.len(),
                        VOCAB * 4
                    ),
                );
            }
            if Some(harness::sha256(&bytes).as_str()) != entry["sha256"].as_str() {
                named_fail(
                    "reference-digest",
                    format!(
                        "{}: payload SHA-256 disagrees with the manifest",
                        path.display()
                    ),
                );
            }
            let logits: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let recorded = entry["summary"]["argmax_flat"].as_u64().unwrap_or_else(|| {
                named_fail("missing-reference", format!("{name}: recorded argmax"))
            });
            if argmax(&logits) as u64 != recorded {
                named_fail(
                    "reference-argmax",
                    format!("{name}: payload argmax disagrees with the manifest"),
                );
            }
            logits
        })
    }

    fn open_pinned_physical() -> std::sync::Arc<VulkanBackend> {
        let pinned = std::env::var("FN3_VK_DEVICE_UUID").unwrap_or_else(|error| {
            named_fail(
                "missing-device-uuid",
                format!("FN3_VK_DEVICE_UUID: {error}; the gate needs the pinned GPU UUID"),
            )
        });
        let pinned = pinned.trim().to_ascii_lowercase();
        if pinned.is_empty() {
            named_fail(
                "missing-device-uuid",
                "FN3_VK_DEVICE_UUID is empty; the gate needs the pinned GPU UUID",
            );
        }
        let adapters = adapters();
        if adapters.is_empty() {
            named_fail(
                "missing-vulkan-support",
                "no Vulkan adapters were probed; install an ICD and run under --features vulkan",
            );
        }
        let mut saw_software = false;
        let mut index = None;
        for adapter in adapters {
            if !adapter.uuid_string().eq_ignore_ascii_case(&pinned) {
                continue;
            }
            if !adapter.has_compute_queue() {
                named_fail(
                    "unmatched-device",
                    format!("adapter {} has no compute queue", adapter.uuid_string()),
                );
            }
            if adapter.is_software() {
                saw_software = true;
                continue;
            }
            index = Some(adapter.index);
            break;
        }
        let Some(index) = index else {
            if saw_software {
                named_fail(
                    "non-physical-device",
                    format!(
                        "the adapter matching UUID {pinned} is software; the gate requires the \
                         physical GTX 1080 Ti"
                    ),
                );
            }
            named_fail(
                "unmatched-device",
                format!("no physical Vulkan compute device matches UUID {pinned}"),
            );
        };
        let backend = VulkanBackend::open_with(
            &DeviceRequest::Vulkan { index: Some(index) },
            VulkanConfig::validated(),
        )
        .unwrap_or_else(|error| named_fail("vulkan-open-failed", error));
        if backend.is_software() {
            named_fail(
                "non-physical-device",
                "the opened Vulkan adapter is software, not the pinned GTX 1080 Ti",
            );
        }
        if !backend.validation_enabled() {
            named_fail(
                "validation-unavailable",
                "VulkanConfig::validated() did not enable VK_LAYER_KHRONOS_validation",
            );
        }
        backend
    }

    fn prepare_model(
        import: &ModelImport,
        device: DeviceId,
        backend: &dyn Backend,
    ) -> (PreparedModel, ShapeEnvelope) {
        let planner = Planner::new(import, backend);
        let envelope = harness::envelope(&planner, CONTEXT);
        let config = import.geometry().kv_cache_config(CONTEXT, 1);
        let source = import.to_model_weights().unwrap_or_else(|error| {
            named_fail("model-weights", format!("copying host weights: {error}"))
        });
        let mut named = Vec::with_capacity(source.len());
        for index in 0..source.len() {
            let tensor = source
                .get(index)
                .unwrap_or_else(|| named_fail("model-weights", format!("weight {index}")));
            let name = source
                .name(index)
                .unwrap_or_else(|| named_fail("model-weights", format!("weight {index} name")));
            let meta = TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), device)
                .unwrap_or_else(|error| named_fail("model-weights", format!("{name}: {error}")));
            let retagged = Tensor::from_storage(tensor.storage().clone(), meta)
                .unwrap_or_else(|error| named_fail("model-weights", format!("{name}: {error}")));
            named.push((name.to_string(), retagged));
        }
        let weights = ModelWeights::with_names(named)
            .unwrap_or_else(|error| named_fail("model-weights", error));
        let geometry = import.geometry();
        let live = LiveDims::new(
            1,
            4,
            0,
            u64::from(geometry.query_heads),
            u64::from(geometry.kv_heads),
            u64::from(geometry.head_dim),
        )
        .unwrap_or_else(|error| named_fail("model-plan", error));
        let plan = planner
            .plan(&envelope, live)
            .unwrap_or_else(|error| named_fail("model-plan", error))
            .execution()
            .clone();
        let model = PreparedModel::new(plan, weights, device, envelope, Some(config))
            .unwrap_or_else(|error| named_fail("model-plan", error));
        (model, envelope)
    }

    fn run_phases(session: &mut Session<'_>, label: &str, expect_int8: bool) -> [Vec<f32>; 3] {
        let mut logits: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        logits[0] = capture(label, PHASES[0], session.prefill(&PROMPT));
        if expect_int8 {
            assert_int8_coverage(label, PHASES[0], session);
        }
        for (index, token) in DECODE.iter().enumerate() {
            let phase = PHASES[index + 1];
            logits[index + 1] = capture(label, phase, session.decode(&[*token]));
            if expect_int8 {
                assert_int8_coverage(label, phase, session);
            }
        }
        logits
    }

    fn capture(
        label: &str,
        phase: &str,
        result: Result<Tensor, fastnn::inference::Fn3Error>,
    ) -> Vec<f32> {
        let logits = result
            .unwrap_or_else(|error| named_fail("run-step", format!("{label} {phase}: {error}")));
        let host = logits.to_host().unwrap_or_else(|error| {
            named_fail("run-step", format!("{label} {phase} logits: {error}"))
        });
        let values = host
            .as_f32_slice()
            .unwrap_or_else(|error| {
                named_fail("run-step", format!("{label} {phase} logits: {error}"))
            })
            .to_vec();
        if values.len() != VOCAB {
            named_fail(
                "run-step",
                format!("{label} {phase}: {} logits, expected {VOCAB}", values.len()),
            );
        }
        values
    }

    fn assert_int8_coverage(label: &str, phase: &str, session: &Session<'_>) {
        let report = session.weight_int8_report().unwrap_or_else(|| {
            named_fail(
                "missing-int8-report",
                format!("{label} {phase}: weight_int8=true but the session has no report"),
            )
        });
        let mut packed = 0usize;
        let mut fallbacks = Vec::new();
        for route in &report.routes {
            if route.fallback.is_none() {
                packed += 1;
            } else {
                fallbacks.push(route);
            }
        }
        if report.routes.len() != MATMULS {
            named_fail(
                "coverage-count",
                format!(
                    "{label} {phase}: {} MatMuls in the plan, expected {MATMULS}",
                    report.routes.len()
                ),
            );
        }
        if packed != PACKED_MATMULS || fallbacks.len() != FP32_MATMULS {
            named_fail(
                "coverage-count",
                format!(
                    "{label} {phase}: packed={packed} fp32={} expected {PACKED_MATMULS}/\
                     {FP32_MATMULS}",
                    fallbacks.len()
                ),
            );
        }
        let attention_scores = fallbacks
            .iter()
            .filter(|route| route.rhs.contains("/attn/Mul_1_output_0"))
            .count();
        let attention_values = fallbacks
            .iter()
            .filter(|route| route.rhs.starts_with("present.") && route.rhs.ends_with(".value"))
            .count();
        let tied_head = fallbacks
            .iter()
            .filter(|route| route.rhs.contains("weight_transposed"))
            .count();
        if attention_scores != 12 || attention_values != 12 || tied_head != 1 {
            let names: Vec<&str> = fallbacks.iter().map(|route| route.rhs.as_str()).collect();
            named_fail(
                "fallback-shape",
                format!(
                    "{label} {phase}: expected 12 attention scores, 12 attention values and the \
                     tied LM head; got {attention_scores}/{attention_values}/{tied_head} from \
                     {names:?}"
                ),
            );
        }
        for route in &fallbacks {
            let code = route
                .fallback
                .as_ref()
                .map(|fallback| fallback.code())
                .unwrap_or("packed");
            if code != "computed-rhs" {
                named_fail(
                    "fallback-route",
                    format!(
                        "{label} {phase}: {} fell back with `{code}`, expected computed-rhs",
                        route.rhs
                    ),
                );
            }
        }
        if report.packed_payload_bytes != PACKED_PAYLOAD_BYTES {
            named_fail(
                "packed-payload",
                format!(
                    "{label} {phase}: packed payload is {} B, expected {PACKED_PAYLOAD_BYTES}",
                    report.packed_payload_bytes
                ),
            );
        }
        println!(
            "[fn3-trained-int8][coverage] arm={label} phase={phase} packed={packed}/{MATMULS} \
             fp32={} attention_scores={attention_scores} attention_values={attention_values} \
             tied_lm_head={tied_head} packed_payload={}B",
            fallbacks.len(),
            report.packed_payload_bytes,
        );
    }

    fn assert_int8_traffic(backend: &VulkanBackend) {
        let stats = backend.residency_stats();
        if stats.packed_prepare_bytes != PACKED_PAYLOAD_BYTES {
            named_fail(
                "packed-prepare-bytes",
                format!(
                    "packed prepare moved {} B, expected the {PACKED_PAYLOAD_BYTES}-byte payload",
                    stats.packed_prepare_bytes
                ),
            );
        }
        if stats.staged_upload_bytes[ResidencyPhase::Prepare.index()] >= MAX_PREPARE_STAGING_BYTES {
            named_fail(
                "prepare-staging",
                format!(
                    "prepare staged {} B of host bytes; packed weights must be prepared, not \
                     staged (payload {PACKED_PAYLOAD_BYTES} B)",
                    stats.staged_upload_bytes[ResidencyPhase::Prepare.index()]
                ),
            );
        }
        if stats.late_persistent_uploads != 0 || stats.late_persistent_bytes != 0 {
            named_fail(
                "late-persistent",
                format!(
                    "{} late persistent upload(s), {} B",
                    stats.late_persistent_uploads, stats.late_persistent_bytes
                ),
            );
        }
        let summary = backend.residency().summary();
        if summary.input_readbacks != 0 {
            named_fail(
                "input-readbacks",
                format!(
                    "{} resident inputs were read back on the host",
                    summary.input_readbacks
                ),
            );
        }
        println!(
            "[fn3-trained-int8][traffic] packed_prepare={}B staged_prepare={}B staged_prefill={}B \
             staged_decode={}B late_persistent={}B readback_calls={} readback_bytes={}B \
             input_readbacks={}",
            stats.packed_prepare_bytes,
            stats.staged_upload_bytes[ResidencyPhase::Prepare.index()],
            stats.staged_upload_bytes[ResidencyPhase::Prefill.index()],
            stats.staged_upload_bytes[ResidencyPhase::Decode.index()],
            stats.late_persistent_bytes,
            stats.readback_calls,
            stats.readback_bytes,
            summary.input_readbacks,
        );
    }

    struct Metrics {
        max_abs: f64,
        centered_max: f64,
        centered_rmse: f64,
        kl: f64,
        top1: usize,
        reference_top1: usize,
        top5: Vec<usize>,
        reference_top5: Vec<usize>,
        top5_overlap: usize,
        /// The reference's 5th logit minus its 6th: how decisive the boundary is.
        fifth_gap: f64,
    }

    fn measure(actual: &[f32], reference: &[f32]) -> Metrics {
        if actual.len() != reference.len() {
            named_fail(
                "metric-length",
                format!(
                    "actual has {} logits, reference has {}",
                    actual.len(),
                    reference.len()
                ),
            );
        }
        let mut max_abs = 0.0f64;
        let mut total_signed = 0.0f64;
        for (&got, &want) in actual.iter().zip(reference) {
            if !got.is_finite() || !want.is_finite() {
                named_fail("non-finite-logit", "a compared logit is not finite");
            }
            let difference = f64::from(got) - f64::from(want);
            max_abs = max_abs.max(difference.abs());
            total_signed += difference;
        }
        let count = actual.len() as f64;
        let mean_signed = total_signed / count;
        let mut centered_squared = 0.0f64;
        let mut centered_max = 0.0f64;
        for (&got, &want) in actual.iter().zip(reference) {
            let centered = f64::from(got) - f64::from(want) - mean_signed;
            centered_max = centered_max.max(centered.abs());
            centered_squared += centered * centered;
        }

        let actual_top1 = argmax(actual);
        let reference_top1 = argmax(reference);
        let actual_top5 = top_k(actual, 5);
        let reference_top5 = top_k(reference, 5);
        let reference_top6 = top_k(reference, 6);
        let top5_overlap = actual_top5
            .iter()
            .filter(|token| reference_top5.contains(token))
            .count();
        let fifth_gap =
            f64::from(reference[reference_top6[4]]) - f64::from(reference[reference_top6[5]]);

        let max_reference = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_actual = actual.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut p = vec![0.0f64; reference.len()];
        let mut q = vec![0.0f64; actual.len()];
        let mut p_total = 0.0f64;
        let mut q_total = 0.0f64;
        for index in 0..reference.len() {
            p[index] = (f64::from(reference[index]) - f64::from(max_reference)).exp();
            q[index] = (f64::from(actual[index]) - f64::from(max_actual)).exp();
            p_total += p[index];
            q_total += q[index];
        }
        let kl = p
            .iter()
            .zip(&q)
            .map(|(p, q)| {
                let p = p / p_total;
                let q = q / q_total;
                p * (p / q).ln()
            })
            .sum();

        Metrics {
            max_abs,
            centered_max,
            centered_rmse: (centered_squared / count).sqrt(),
            kl,
            top1: actual_top1,
            reference_top1,
            top5: actual_top5,
            reference_top5,
            top5_overlap,
            fifth_gap,
        }
    }

    fn assert_phase(
        phase: &str,
        index: usize,
        int8: &[f32],
        fp32: &[f32],
        cpu_int8: &[f32],
        reference: &[f32],
        failures: &mut Vec<String>,
    ) {
        let int8_vs_ort = measure(int8, reference);
        let fp32_vs_int8 = measure(fp32, int8);
        let cpu_vs_vulkan = measure(cpu_int8, int8);
        let fp32_vs_ort = measure(fp32, reference);
        let kl_bound = if index == 0 {
            PREFILL_KL_BOUND
        } else {
            DECODE_KL_BOUND
        };

        println!("[fn3-trained-int8][summary] phase={phase}");
        println!(
            "[fn3-trained-int8][summary]   int8_vs_ort: top1={}/{} top5={}/5 kl={:.3e} (bound \
             {kl_bound:.0e}) centered_max={:.6} (bound {CENTERED_MAX_BOUND}) centered_rmse={:.6} \
             (bound {CENTERED_RMSE_BOUND}) max_abs={:.3e}",
            int8_vs_ort.top1,
            int8_vs_ort.reference_top1,
            int8_vs_ort.top5_overlap,
            int8_vs_ort.kl,
            int8_vs_ort.centered_max,
            int8_vs_ort.centered_rmse,
            int8_vs_ort.max_abs,
        );
        let fifth_gap_bound = 2.0 * int8_vs_ort.centered_max;
        let decisive_fifth = int8_vs_ort.fifth_gap > fifth_gap_bound;
        let required_top5 = if decisive_fifth { 5 } else { 4 };
        println!(
            "[fn3-trained-int8][summary]   int8_top5={:?} ort_top5={:?} fifth_gap={:.3e} \
             (2*centered_max={fifth_gap_bound:.3e}, required={required_top5}/5)",
            int8_vs_ort.top5, int8_vs_ort.reference_top5, int8_vs_ort.fifth_gap,
        );
        println!(
            "[fn3-trained-int8][summary]   fp32_vs_ort: top1={}/{} top5={}/5 max_abs={:.3e}; \
             fp32_vs_int8_max_abs={:.3e}; cpu_vs_vulkan_int8_max_abs={:.3e} (bound \
             {CPU_VS_VULKAN_INT8_MAX_ABS:.0e})",
            fp32_vs_ort.top1,
            fp32_vs_ort.reference_top1,
            fp32_vs_ort.top5_overlap,
            fp32_vs_ort.max_abs,
            fp32_vs_int8.max_abs,
            cpu_vs_vulkan.max_abs,
        );

        if int8_vs_ort.top1 != int8_vs_ort.reference_top1 {
            failures.push(format!(
                "{phase}: INT8 top-1 {} != ORT top-1 {}",
                int8_vs_ort.top1, int8_vs_ort.reference_top1
            ));
        }
        if fp32_vs_ort.top1 != fp32_vs_ort.reference_top1 {
            failures.push(format!(
                "{phase}: FP32 top-1 {} != ORT top-1 {}",
                fp32_vs_ort.top1, fp32_vs_ort.reference_top1
            ));
        }
        if fp32_vs_ort.top5_overlap != 5 {
            failures.push(format!(
                "{phase}: FP32 top-5 overlaps ORT {}/5 (FP32 {:?}, ORT {:?})",
                fp32_vs_ort.top5_overlap, fp32_vs_ort.top5, fp32_vs_ort.reference_top5
            ));
        }
        // #68 §6's top-5 rule: a 5/5 requirement is only truthful when the
        // reference's 5th-place gap is decisive. The trained decode boundary is
        // sub-millilogit, so the documented rule accepts 4/5 there; the raw
        // overlap is always printed above.
        if int8_vs_ort.top5_overlap < required_top5 {
            failures.push(format!(
                "{phase}: INT8 top-5 overlaps ORT {}/5, required {required_top5}/5 \
                 (INT8 {:?}, ORT {:?}, fifth_gap {:.3e}, 2*centered_max {:.3e})",
                int8_vs_ort.top5_overlap,
                int8_vs_ort.top5,
                int8_vs_ort.reference_top5,
                int8_vs_ort.fifth_gap,
                fifth_gap_bound,
            ));
        }
        if int8_vs_ort.kl > kl_bound {
            failures.push(format!(
                "{phase}: INT8-vs-ORT KL {:.3e} above bound {kl_bound:.3e}",
                int8_vs_ort.kl
            ));
        }
        if int8_vs_ort.centered_max > CENTERED_MAX_BOUND {
            failures.push(format!(
                "{phase}: centered max {:.6} above bound {CENTERED_MAX_BOUND}",
                int8_vs_ort.centered_max
            ));
        }
        if int8_vs_ort.centered_rmse > CENTERED_RMSE_BOUND {
            failures.push(format!(
                "{phase}: centered RMSE {:.6} above bound {CENTERED_RMSE_BOUND}",
                int8_vs_ort.centered_rmse
            ));
        }
        if cpu_vs_vulkan.max_abs > CPU_VS_VULKAN_INT8_MAX_ABS {
            failures.push(format!(
                "{phase}: CPU INT8 vs Vulkan INT8 max abs {:.3e} above {CPU_VS_VULKAN_INT8_MAX_ABS:.0e}",
                cpu_vs_vulkan.max_abs
            ));
        }
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (index, &value)| {
                if value > best.1 {
                    (index, value)
                } else {
                    best
                }
            })
            .0
    }

    fn top_k(values: &[f32], k: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
        order.truncate(k);
        order
    }
}
