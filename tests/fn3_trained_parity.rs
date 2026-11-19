//! FN3-30 (#76): trained GPT-2 FP32 parity, CPU and (optionally) Vulkan.
//!
//! This is the self-contained gate the stable review asked for. It runs the
//! real fn3 path — import, prepare, `Session::prefill`, two `Session::decode`
//! calls — against the committed ORT payloads and reports:
//!
//! * the elementwise bound (`atol = 5e-4`, `rtol = 1e-4`, zero violations is a
//!   pass);
//! * the **existing** suite-level `mean_abs` bound (`5e-5`), kept visible even
//!   when it fails;
//! * a scale-aware diagnostic (`5e-5 + 5e-6 * mean|ref|`) and the
//!   ranking-relevant centered RMS, so the stable-gate decision has the numbers
//!   in the test that will enforce it;
//! * top-1/top-5 agreement, top-1 margin and KL.
//!
//! It is `#[ignore]` because it needs the 498,126,358-byte trained graph. With
//! the model absent it prints the reason and passes; it never pretends a model
//! ran. The committed payloads are the authority for prefill and decode step 1;
//! decode step 2 has no committed payload, so its argmax is checked against the
//! manifest summary and the full logits are written for the Python comparison.
//!
//! ```bash
//! FN3_TRAINED_ONNX=/path/to/onnx/model.onnx \
//!   cargo test --release --no-default-features --test fn3_trained_parity \
//!   -- --ignored --nocapture
//!
//! FN3_TRAINED_ONNX=... FN3_TRAINED_VULKAN=1 FN3_VK_DEVICE_UUID=... \
//!   cargo test --release --features vulkan --test fn3_trained_parity \
//!   -- --ignored --nocapture
//! ```
//!
//! This test deliberately does not assert the proposed policy: the tolerance
//! decision belongs to the owner (#76). It asserts what is already policy (the
//! elementwise bound and the committed payload digest) and reports both the
//! current failure and the proposed bound.

#![allow(clippy::chunks_exact_to_as_chunks)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fastnn::fn3::import::{import_onnx_file, ModelImport};
use fastnn::fn3::model::{ModelWeights, Planner, PreparedModel, Session, SessionOptions};
use fastnn::fn3::shape::ShapeEnvelope;
use fastnn::fn3::{CpuBackend, DeviceId, LiveDims, Tensor, TensorMeta};

const PROMPT: [u32; 13] = [
    464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257,
];
const DECODE: [u32; 2] = [640, 11];
const TRAINED_SHA256: &str = "42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d";
const TRAINED_BYTES: usize = 498_126_358;
const VOCAB: usize = 50_257;
const ATOL: f32 = 5e-4;
const RTOL: f32 = 1e-4;
/// The existing suite-level bound. It is reported, not asserted: the observed
/// prefill and decode-step-2 means exceed it, and that is the finding.
const MEAN_ABS_BOUND: f64 = 5e-5;
/// The scale-aware diagnostic proposed in `docs/reports/fn3-trained-fp32.md`.
/// The absolute floor is the existing bound; the relative term is 3.4x the
/// worst observed CPU `mean_abs / mean|ref|` (1.47e-6) and 2.5x the staged
/// Vulkan worst (2.03e-6).
const SCALE_AWARE_FLOOR: f64 = 5e-5;
const SCALE_AWARE_RELATIVE: f64 = 5e-6;

fn sha256(bytes: &[u8]) -> String {
    fn compress(state: &mut [u32; 8], block: &[u8]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut w = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for block in message.chunks_exact(64) {
        compress(&mut state, block);
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

fn output_root() -> PathBuf {
    PathBuf::from(
        std::env::var("FN3_TRAINED_OUT")
            .unwrap_or_else(|_| "/tmp/opencode/fn3-trained-parity".into()),
    )
}

fn live_dims(seq: u64, past: u64) -> LiveDims {
    LiveDims::new(1, seq, past, 12, 12, 64).expect("trained GPT-2 total fits")
}

fn retag_weights(weights: &ModelWeights, device: DeviceId) -> ModelWeights {
    let mut named = Vec::with_capacity(weights.len());
    for index in 0..weights.len() {
        let tensor = weights.get(index).expect("weight index");
        let meta = TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), device)
            .expect("weight metadata");
        named.push((
            weights.name(index).expect("weight name").to_string(),
            Tensor::from_storage(tensor.storage().clone(), meta).expect("re-tagged weight"),
        ));
    }
    ModelWeights::with_names(named).expect("weights share one device")
}

fn prepared(
    import: &ModelImport,
    device: DeviceId,
    context: u64,
) -> (PreparedModel, ShapeEnvelope) {
    // The plan is specialised per session; this one is only the `PreparedModel`
    // placeholder, exactly as the e2e harness does it.
    let cpu = CpuBackend::new();
    let planner = Planner::new(import, &cpu);
    let envelope = planner
        .envelope(1, PROMPT.len() as u64, context, context + 1)
        .expect("trained window");
    let config = import.geometry().kv_cache_config(context, 1);
    let weights = retag_weights(&import.to_model_weights().expect("host weights"), device);
    let plan = planner
        .plan(&envelope, live_dims(PROMPT.len() as u64, 0))
        .expect("prefill plan")
        .execution()
        .clone();
    let model =
        PreparedModel::new(plan, weights, device, envelope, Some(config)).expect("prepared model");
    (model, envelope)
}

fn write_f32(path: &Path, values: &[f32]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("output directory");
    }
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path, bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
}

fn load_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
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

/// The metrics the stable-gate decision needs, all against one reference.
struct Metrics {
    max_abs: f64,
    mean_abs: f64,
    mean_signed: f64,
    centered_rms: f64,
    relative_rms: f64,
    rms_reference: f64,
    violations: usize,
    top1: usize,
    reference_top1: usize,
    top5_overlap: usize,
    reference_margin: f64,
    actual_margin: f64,
    kl: f64,
}

fn measure(actual: &[f32], reference: &[f32]) -> Metrics {
    assert_eq!(actual.len(), reference.len(), "logit vector length");
    let mut max_abs = 0.0f64;
    let mut total_abs = 0.0f64;
    let mut total_signed = 0.0f64;
    let mut squared = 0.0f64;
    let mut reference_squared = 0.0f64;
    let mut violations = 0usize;
    for (&got, &want) in actual.iter().zip(reference) {
        assert!(got.is_finite(), "non-finite fn3 logit");
        let difference = f64::from(got) - f64::from(want);
        let abs = difference.abs();
        max_abs = max_abs.max(abs);
        total_abs += abs;
        total_signed += difference;
        squared += difference * difference;
        reference_squared += f64::from(want) * f64::from(want);
        let bound = f64::from(ATOL) + f64::from(RTOL) * f64::from(want).abs();
        if abs > bound {
            violations += 1;
        }
    }
    let count = actual.len() as f64;
    let mean_signed = total_signed / count;
    let mut centered = 0.0f64;
    for (&got, &want) in actual.iter().zip(reference) {
        let difference = f64::from(got) - f64::from(want) - mean_signed;
        centered += difference * difference;
    }
    let actual_top1 = argmax(actual);
    let reference_top1 = argmax(reference);
    let actual_top5 = top_k(actual, 5);
    let reference_top5 = top_k(reference, 5);
    let overlap = actual_top5
        .iter()
        .filter(|token| reference_top5.contains(token))
        .count();
    let mut reference_order: Vec<usize> = (0..reference.len()).collect();
    reference_order.sort_by(|&a, &b| reference[b].total_cmp(&reference[a]));
    let mut actual_order: Vec<usize> = (0..actual.len()).collect();
    actual_order.sort_by(|&a, &b| actual[b].total_cmp(&actual[a]));

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
        mean_abs: total_abs / count,
        mean_signed,
        centered_rms: (centered / count).sqrt(),
        relative_rms: (squared / reference_squared).sqrt(),
        rms_reference: (reference_squared / count).sqrt(),
        violations,
        top1: actual_top1,
        reference_top1,
        top5_overlap: overlap,
        reference_margin: f64::from(reference[reference_order[0]])
            - f64::from(reference[reference_order[1]]),
        actual_margin: f64::from(actual[actual_order[0]]) - f64::from(actual[actual_order[1]]),
        kl,
    }
}

fn report(label: &str, metrics: &Metrics, mean_reference: f64) -> serde_json::Value {
    let scale_aware = SCALE_AWARE_FLOOR + SCALE_AWARE_RELATIVE * mean_reference;
    let existing_pass = metrics.mean_abs <= MEAN_ABS_BOUND;
    let proposed_pass = metrics.mean_abs <= scale_aware;
    println!(
        "{label}: max_abs {:.3e}, mean_abs {:.3e} (existing 5e-5: {}; \
         scale-aware {:.3e}: {}), signed {:.3e}, centered_rms {:.3e}, \
         relative_rms {:.3e}, violations {}, top1 {}/{} margin {:.4} (actual {:.4}), \
         top5 {}/5, KL {:.3e}",
        metrics.max_abs,
        metrics.mean_abs,
        if existing_pass { "PASS" } else { "FAIL" },
        scale_aware,
        if proposed_pass { "PASS" } else { "FAIL" },
        metrics.mean_signed,
        metrics.centered_rms,
        metrics.relative_rms,
        metrics.violations,
        metrics.top1,
        metrics.reference_top1,
        metrics.reference_margin,
        metrics.actual_margin,
        metrics.top5_overlap,
        metrics.kl
    );
    assert_eq!(
        metrics.violations, 0,
        "{label}: elementwise bound (atol {ATOL:e}, rtol {RTOL:e}) violated"
    );
    assert_eq!(
        metrics.top1, metrics.reference_top1,
        "{label}: top-1 disagrees"
    );
    assert_eq!(metrics.top5_overlap, 5, "{label}: top-5 disagrees");
    serde_json::json!({
        "max_abs": metrics.max_abs,
        "mean_abs": metrics.mean_abs,
        "mean_signed": metrics.mean_signed,
        "centered_rms": metrics.centered_rms,
        "relative_rms": metrics.relative_rms,
        "rms_reference": metrics.rms_reference,
        "mean_abs_bound_existing": MEAN_ABS_BOUND,
        "mean_abs_within_existing_bound": existing_pass,
        "scale_aware_bound": scale_aware,
        "mean_abs_within_scale_aware_bound": proposed_pass,
        "elementwise_violations": metrics.violations,
        "top1": metrics.top1,
        "reference_top1": metrics.reference_top1,
        "top1_agree": metrics.top1 == metrics.reference_top1,
        "reference_top1_margin": metrics.reference_margin,
        "actual_top1_margin": metrics.actual_margin,
        "top5_overlap": metrics.top5_overlap,
        "kl_reference_to_actual": metrics.kl,
    })
}

/// Fail with the known resident-Vulkan diagnosis instead of a bare error.
fn run_step(label: &str, what: &str, result: Result<Tensor, fastnn::fn3::Fn3Error>) -> Tensor {
    match result {
        Ok(tensor) => tensor,
        Err(error) => {
            let hint = if label == "vulkan" && std::env::var("FN3_VK_STAGED").as_deref() != Ok("1")
            {
                "\nThe resident Vulkan path cannot run this graph at this commit: the wte \
                 Gather consumes a resident Reshape(input_ids) index tensor and the gather \
                 kernel validates indices on the host. Set FN3_VK_STAGED=1 for the staged \
                 arithmetic arm; see docs/reports/fn3-trained-fp32.md §6.2."
            } else {
                ""
            };
            panic!("{label} {what} failed: {error}{hint}");
        }
    }
}

/// One executed arm: its JSON report and the logits of every phase. The
/// resident-vs-staged gate proves residency from the backend's execution
/// ledger, not from the returned tensor (a session's logits row is always an
/// explicit host materialisation).
struct Arm {
    report: serde_json::Value,
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    logits: BTreeMap<String, Vec<f32>>,
}

/// Run the pinned prompt and two decode steps on one backend with the default
/// (resident, when the backend supports it) session options.
fn run(
    import: &ModelImport,
    backend: &dyn fastnn::fn3::Backend,
    device: DeviceId,
    label: &str,
    out_dir: &Path,
    expected: &[(&str, &Path)],
) -> Arm {
    run_with_options(
        import,
        backend,
        device,
        label,
        out_dir,
        expected,
        SessionOptions::default(),
    )
}

/// Run the pinned prompt and two decode steps with explicit session options.
fn run_with_options(
    import: &ModelImport,
    backend: &dyn fastnn::fn3::Backend,
    device: DeviceId,
    label: &str,
    out_dir: &Path,
    expected: &[(&str, &Path)],
    options: SessionOptions,
) -> Arm {
    let (model, _envelope) = prepared(import, device, 32);
    run_session(&model, import, backend, label, out_dir, expected, options)
}

/// Run one arm over an already-prepared model.
///
/// Both comparison arms share one `PreparedModel` on purpose: the persistent
/// table is keyed by host allocation identity, and building a second model
/// after the first is dropped can reuse an address and produce a false
/// persistent hit. One immutable model is also what a real caller shares
/// between sessions.
fn run_session(
    model: &PreparedModel,
    import: &ModelImport,
    backend: &dyn fastnn::fn3::Backend,
    label: &str,
    out_dir: &Path,
    expected: &[(&str, &Path)],
    options: SessionOptions,
) -> Arm {
    let mut session =
        Session::with_runtime_options(model, import, backend, options).expect("session");
    let mut phases = serde_json::Map::new();
    let mut captured = BTreeMap::new();

    let logits = run_step(label, "prefill", session.prefill(&PROMPT));
    let host = logits.to_host().expect("logits to host");
    let values = host.as_f32_slice().expect("host logits").to_vec();
    assert_eq!(values.len(), VOCAB);
    captured.insert("prefill".to_string(), values.clone());
    write_f32(
        &out_dir.join(format!("{label}.prefill.p13.last_logits.f32")),
        &values,
    );
    if let Some((_, path)) = expected.iter().find(|(name, _)| *name == "prefill") {
        let reference = load_f32(path);
        let metrics = measure(&values, &reference);
        let mean_reference =
            reference.iter().map(|v| f64::from(*v).abs()).sum::<f64>() / reference.len() as f64;
        phases.insert(
            "prefill".to_string(),
            report(&format!("{label} prefill"), &metrics, mean_reference),
        );
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/trained-gpt2");
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("manifest.json")).expect("manifest"),
    )
    .expect("manifest json");
    for (step, token) in DECODE.iter().enumerate() {
        let step_number = step + 1;
        let logits = run_step(
            label,
            &format!("decode {step_number}"),
            session.decode(&[*token]),
        );
        let host = logits.to_host().expect("logits to host");
        let values = host.as_f32_slice().expect("host logits").to_vec();
        assert_eq!(values.len(), VOCAB);
        captured.insert(format!("decode.step{step_number}"), values.clone());
        write_f32(
            &out_dir.join(format!("{label}.decode.step{step_number}.logits.f32")),
            &values,
        );
        let name = format!("decode.step{step_number}");
        let reference_path = expected
            .iter()
            .find(|(expected, _)| *expected == name)
            .map(|(_, path)| *path);
        if let Some(path) = reference_path {
            let reference = load_f32(path);
            let metrics = measure(&values, &reference);
            let mean_reference =
                reference.iter().map(|v| f64::from(*v).abs()).sum::<f64>() / reference.len() as f64;
            phases.insert(
                name.clone(),
                report(&format!("{label} {name}"), &metrics, mean_reference),
            );
        } else {
            let recorded = manifest["tensors"]
                .as_array()
                .expect("tensors")
                .iter()
                .find(|tensor| tensor["name"] == format!("{name}.logits"))
                .expect("manifest entry")["summary"]["argmax_flat"]
                .as_i64()
                .expect("recorded argmax") as usize;
            let actual = argmax(&values);
            println!(
                "{label} {name}: summary-only reference; argmax {actual} \
                 (manifest {recorded}), top5 {:?}",
                top_k(&values, 5)
            );
            assert_eq!(actual, recorded, "{label} {name}: argmax vs manifest");
            phases.insert(
                name,
                serde_json::json!({
                    "argmax": actual,
                    "recorded_argmax": recorded,
                    "reference_kind": "summary-only",
                    "top5": top_k(&values, 5),
                }),
            );
        }
    }
    Arm {
        report: serde_json::Value::Object(phases),
        logits: captured,
    }
}

#[test]
#[ignore = "needs the 498,126,358-byte trained ONNX graph; set FN3_TRAINED_ONNX"]
fn trained_gpt2_fp32_parity_cpu_and_vulkan() {
    let Some(path) = std::env::var("FN3_TRAINED_ONNX").ok().map(PathBuf::from) else {
        println!("SKIP: FN3_TRAINED_ONNX is not set; no trained parity run was attempted");
        return;
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(bytes.len(), TRAINED_BYTES, "not the pinned trained graph");
    assert_eq!(sha256(&bytes), TRAINED_SHA256, "wrong trained graph bytes");
    let import = import_onnx_file(&path).expect("trained import");

    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/trained-gpt2/tensors");
    let prefill = root.join("prefill.p13.last_logits.f32");
    let step1 = root.join("decode.step1.logits.f32");
    let expected = [
        ("prefill", prefill.as_path()),
        ("decode.step1", step1.as_path()),
    ];
    let out = output_root();
    std::fs::create_dir_all(&out).expect("output root");
    let mut report_json = serde_json::Map::new();

    let backend = CpuBackend::new();
    let cpu = run(&import, &backend, DeviceId::Cpu, "cpu", &out, &expected).report;
    report_json.insert("cpu".to_string(), cpu);

    #[cfg(feature = "vulkan")]
    if std::env::var("FN3_TRAINED_VULKAN").as_deref() == Ok("1") {
        use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
        use fastnn::fn3::device::DeviceRequest;
        let require_physical = std::env::var("FN3_VK_REQUIRE_PHYSICAL").as_deref() == Ok("1");
        let pinned = std::env::var("FN3_VK_DEVICE_UUID")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty());
        let adapter = adapters()
            .iter()
            .find(|adapter| {
                adapter.has_compute_queue()
                    && !adapter.is_software()
                    && pinned
                        .as_deref()
                        .map(|uuid| adapter.uuid_string().eq_ignore_ascii_case(uuid))
                        .unwrap_or(true)
            })
            .map(|adapter| adapter.index);
        let Some(index) = adapter else {
            assert!(
                !require_physical,
                "FN3_VK_REQUIRE_PHYSICAL=1 but no physical compute device is present"
            );
            println!("SKIP Vulkan: no physical Vulkan compute device is present");
            return;
        };
        let backend = VulkanBackend::open_with(
            &DeviceRequest::Vulkan { index: Some(index) },
            VulkanConfig::validated(),
        )
        .expect("open physical Vulkan device");
        let device = backend.device_id();
        println!(
            "vulkan device: {} ({})",
            backend.adapter().name,
            backend.adapter().uuid_string()
        );
        let vulkan = run(&import, &*backend, device, "vulkan", &out, &expected).report;
        report_json.insert("vulkan".to_string(), vulkan);
    }

    let report_path = out.join("parity-report.json");
    std::fs::write(
        &report_path,
        serde_json::to_string_pretty(&serde_json::Value::Object(report_json)).expect("json"),
    )
    .expect("write report");
    println!("wrote {}", report_path.display());
}

/// The reference's 5th-place gap: how far the 5th logit is ahead of the 6th.
#[cfg(feature = "vulkan")]
fn fifth_gap(reference: &[f32]) -> f64 {
    let mut order: Vec<usize> = (0..reference.len()).collect();
    order.sort_by(|&a, &b| reference[b].total_cmp(&reference[a]));
    f64::from(reference[order[4]]) - f64::from(reference[order[5]])
}

/// The largest centred residual between two logit vectors.
#[cfg(feature = "vulkan")]
fn centered_max(actual: &[f32], reference: &[f32]) -> f64 {
    let mean: f64 = actual
        .iter()
        .zip(reference)
        .map(|(&got, &want)| f64::from(got) - f64::from(want))
        .sum::<f64>()
        / actual.len() as f64;
    actual
        .iter()
        .zip(reference)
        .map(|(&got, &want)| (f64::from(got) - f64::from(want) - mean).abs())
        .fold(0.0f64, f64::max)
}

/// #68's margin rule, applied to a cross-arm comparison: a top-5 difference is
/// only a failure when the reference's 5th-place gap is above twice the
/// centred error, and a top-1 difference is never accepted.
#[cfg(feature = "vulkan")]
fn assert_cross_arm_agreement(label: &str, resident: &[f32], staged: &[f32]) {
    let metrics = measure(resident, staged);
    assert_eq!(
        metrics.violations, 0,
        "{label}: resident vs staged exceeded atol {ATOL:e} + rtol {RTOL:e} * |staged|"
    );
    assert_eq!(
        metrics.top1, metrics.reference_top1,
        "{label}: resident and staged disagree on top-1"
    );
    let gap = fifth_gap(staged);
    let centered = centered_max(resident, staged);
    if gap > 2.0 * centered {
        assert!(
            metrics.top5_overlap >= 4,
            "{label}: top-5 overlap {} with a 5th-place gap of {gap:.3} and centred error \
             {centered:.3e}",
            metrics.top5_overlap
        );
    }
    println!(
        "{label}: resident vs staged max_abs {:.3e}, mean_abs {:.3e}, centered_rms {:.3e}, \
         violations {}, top1 {}/{}, top5 {}/5, 5th gap {:.4}, centred max {:.3e}",
        metrics.max_abs,
        metrics.mean_abs,
        metrics.centered_rms,
        metrics.violations,
        metrics.top1,
        metrics.reference_top1,
        metrics.top5_overlap,
        gap,
        centered
    );
}

/// FN3-26 (#72) gate: the trained graph must run on the resident default *and*
/// agree with the staged arm, on the named physical GPU.
///
/// This is the arm that failed before the identity-`Reshape` fold:
/// `Gather(wte, Reshape(input_ids))` validated a device-resident index tensor on
/// the host and returned `NotImplemented`. It is deliberately not a silent
/// skip: `FN3_TRAINED_VULKAN=1` means "run on the physical GPU", and a missing
/// device or a missing validation layer is a failure. The CPU arm is covered by
/// `trained_gpt2_fp32_parity_cpu_and_vulkan`.
#[test]
#[ignore = "needs the 498,126,358-byte trained ONNX graph and a physical Vulkan device"]
fn trained_gpt2_resident_matches_staged_on_physical_vulkan() {
    let Some(path) = std::env::var("FN3_TRAINED_ONNX").ok().map(PathBuf::from) else {
        println!(
            "SKIP: FN3_TRAINED_ONNX is not set; the resident-vs-staged arm needs the pinned \
             trained graph"
        );
        return;
    };
    if std::env::var("FN3_TRAINED_VULKAN").as_deref() != Ok("1") {
        println!("SKIP: FN3_TRAINED_VULKAN is not 1; set it to run the GPU arm");
        return;
    }
    assert!(
        !matches!(
            std::env::var("FN3_VK_STAGED").as_deref(),
            Ok("1") | Ok("true") | Ok("yes") | Ok("on")
        ),
        "FN3_VK_STAGED is set; the comparison uses explicit SessionOptions, and the resident arm \
         would be silently replaced by staging"
    );

    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(bytes.len(), TRAINED_BYTES, "not the pinned trained graph");
    assert_eq!(sha256(&bytes), TRAINED_SHA256, "wrong trained graph bytes");

    #[cfg(not(feature = "vulkan"))]
    {
        panic!(
            "FN3_TRAINED_VULKAN=1 but this test binary was built without the `vulkan` feature; \
             build with --features vulkan"
        );
    }

    #[cfg(feature = "vulkan")]
    {
        use fastnn::fn3::backend::vulkan::instance::validation_counts;
        use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
        use fastnn::fn3::device::DeviceRequest;
        use fastnn::fn3::OpId;

        let import = import_onnx_file(&path).expect("trained import");

        let pinned = std::env::var("FN3_VK_DEVICE_UUID")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty());
        let index = adapters()
            .iter()
            .find(|adapter| {
                adapter.has_compute_queue()
                    && !adapter.is_software()
                    && pinned
                        .as_deref()
                        .map(|uuid| adapter.uuid_string().eq_ignore_ascii_case(uuid))
                        .unwrap_or(true)
            })
            .map(|adapter| adapter.index)
            .unwrap_or_else(|| {
                panic!(
                    "FN3_TRAINED_VULKAN=1 but no physical Vulkan compute device is present \
                     (pinned uuid {pinned:?}); this gate does not skip"
                )
            });
        let backend = VulkanBackend::open_with(
            &DeviceRequest::Vulkan { index: Some(index) },
            VulkanConfig::validated(),
        )
        .expect("open the physical Vulkan device");
        assert!(!backend.is_software(), "the gate requires a real GPU");
        assert!(
            backend.validation_enabled(),
            "the gate requires the validation layer"
        );
        println!(
            "[fn3-72][trained] device=\"{}\" uuid={} validation={}",
            backend.adapter().name,
            backend.adapter().uuid_string(),
            backend.validation_enabled()
        );

        let root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/trained-gpt2/tensors");
        let prefill = root.join("prefill.p13.last_logits.f32");
        let step1 = root.join("decode.step1.logits.f32");
        let expected = [
            ("prefill", prefill.as_path()),
            ("decode.step1", step1.as_path()),
        ];
        let out = output_root().join("resident-vs-staged");
        std::fs::create_dir_all(&out).expect("output root");

        // One immutable model shared by both sessions: the persistent table is
        // keyed by host allocation identity, so a second model built after the
        // first is dropped could alias a freed address.
        let (model, _envelope) = prepared(&import, backend.device_id(), 32);

        let before = validation_counts();

        // ---- resident (the default) --------------------------------------
        backend.reset_residency_stats();
        backend.residency().clear();
        let resident = run_session(
            &model,
            &import,
            &*backend,
            "vulkan-resident",
            &out,
            &expected,
            SessionOptions::default(),
        );
        // The ledger is the proof of *how* the arm ran: every resident record
        // joined the step batch and proves device execution, no input was read
        // back, and only the tiny graph-input uploads crossed from the host.
        // (A `Gather` and the mask consumers legitimately stage their *graph
        // input*; weights and intermediates do not.)
        let resident_records = backend.residency().records();
        assert!(!resident_records.is_empty(), "the resident arm ran nothing");
        assert!(resident_records
            .iter()
            .any(|record| record.op == OpId::Gather));
        for record in &resident_records {
            assert!(
                record.batch.is_some(),
                "resident op {} did not join the step batch",
                record.op
            );
            assert!(
                !record.inputs_read_back,
                "resident op {} read an input back on the host",
                record.op
            );
            assert!(
                record.proves_device_execution(),
                "resident op {} has no device-execution proof",
                record.op
            );
        }
        let resident_host_staged = resident_records
            .iter()
            .filter(|record| record.host_staged)
            .count();
        let resident_summary = backend.residency().summary();
        assert_eq!(
            resident_summary.input_readbacks, 0,
            "the resident arm read a device input back on the host"
        );
        assert_eq!(
            resident_summary.readback_bytes, 0,
            "a resident op record read bytes back to the host; only the explicit logits row may"
        );
        assert!(
            resident_summary.upload_bytes < 64 * 1024,
            "the resident arm uploaded {} bytes; only graph inputs may cross the host bridge",
            resident_summary.upload_bytes
        );
        println!(
            "[fn3-72][trained][resident] steps=3 records={} host_staged_input_ops={} \
             executions={} proven={} input_readbacks={} upload_bytes={} readback_bytes={}",
            resident_records.len(),
            resident_host_staged,
            resident_summary.executions,
            resident_summary.proven_device_executions,
            resident_summary.input_readbacks,
            resident_summary.upload_bytes,
            resident_summary.readback_bytes
        );

        // ---- staged comparison arm ---------------------------------------
        backend.reset_residency_stats();
        backend.residency().clear();
        let staged = run_session(
            &model,
            &import,
            &*backend,
            "vulkan-staged",
            &out,
            &expected,
            SessionOptions::staged(),
        );
        let staged_records = backend.residency().records();
        assert!(!staged_records.is_empty(), "the staged arm ran nothing");
        for record in &staged_records {
            assert!(
                record.host_staged,
                "staged arm kept op {} on the device",
                record.op
            );
            assert!(
                record.batch.is_none(),
                "staged op {} unexpectedly joined a batch",
                record.op
            );
        }
        let staged_summary = backend.residency().summary();
        println!(
            "[fn3-72][trained][staged] steps=3 records={} executions={} proven={} \
             input_readbacks={} upload_bytes={} readback_bytes={}",
            staged_records.len(),
            staged_summary.executions,
            staged_summary.proven_device_executions,
            staged_summary.input_readbacks,
            staged_summary.upload_bytes,
            staged_summary.readback_bytes
        );

        // ---- resident vs staged, per phase -------------------------------
        let mut comparison = serde_json::Map::new();
        for phase in ["prefill", "decode.step1", "decode.step2"] {
            let resident_logits = resident
                .logits
                .get(phase)
                .unwrap_or_else(|| panic!("resident {phase} logits"));
            let staged_logits = staged
                .logits
                .get(phase)
                .unwrap_or_else(|| panic!("staged {phase} logits"));
            assert_cross_arm_agreement(
                &format!("[fn3-72][trained] {phase}"),
                resident_logits,
                staged_logits,
            );
            // Every decode step's top-1 is asserted against the reference
            // independently inside `report`; step 2 has no committed payload,
            // so its reference is the manifest argmax and the staged arm.
            let metrics = measure(resident_logits, staged_logits);
            comparison.insert(
                phase.to_string(),
                serde_json::json!({
                    "max_abs": metrics.max_abs,
                    "mean_abs": metrics.mean_abs,
                    "centered_rms": metrics.centered_rms,
                    "elementwise_violations": metrics.violations,
                    "top1_resident": metrics.top1,
                    "top1_staged": metrics.reference_top1,
                    "top5_overlap": metrics.top5_overlap,
                }),
            );
        }

        let after = validation_counts();
        assert_eq!(
            after, before,
            "validation messages changed during the trained resident-vs-staged run"
        );
        println!(
            "[fn3-72][trained] validation errors {} warnings {}",
            after.0, after.1
        );

        let report_path = out.join("resident-vs-staged.json");
        std::fs::write(
            &report_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "device": {
                    "name": backend.adapter().name,
                    "uuid": backend.adapter().uuid_string(),
                    "validation_errors": after.0,
                    "validation_warnings": after.1,
                },
                "resident": resident.report,
                "staged": staged.report,
                "resident_vs_staged": comparison,
            }))
            .expect("json"),
        )
        .expect("write report");
        println!("wrote {}", report_path.display());
    }
}
