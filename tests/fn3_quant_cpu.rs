//! FN3-23 (#69): production CPU inference on the locked W8A16 format.
//!
//! Three layers of evidence:
//!
//! 1. **Format parity.** The fused kernel consumes the committed
//!    `fn3-quant-format-v1` fixtures directly. Code bytes are compared with the
//!    manifest and the independent Python reference byte for byte; kernel
//!    results are compared with a `f64` re-derivation from the manifest's
//!    dequantized values and with the committed Rust decoder
//!    (`QuantTensor::dequantize_into`) on a tail group and a non-zero
//!    zero-point case.
//! 2. **Registration and fallback.** The §6 policy, the `Support` mapping, an
//!    explicit FP32 fallback for an unregistered/strided `MatMul`, and a
//!    malformed-metadata rejection.
//! 3. **The pinned models.** `#[ignore]` tests import the tiny and trained
//!    GPT-2 graphs, run the real `Session` through a `QuantCpuBackend`, and
//!    report coverage, storage, latency and accuracy against the FP32 run.
//!
//! ```bash
//! cargo test --test fn3_quant_cpu
//! cargo test --release --test fn3_quant_cpu -- --ignored --nocapture
//! FN3_QUANT_MODE=fp32 cargo test --release --test fn3_quant_cpu measure_trained -- --ignored --nocapture
//! FN3_QUANT_MODE=w8a16 cargo test --release --test fn3_quant_cpu measure_trained -- --ignored --nocapture
//! ```
//!
//! The final quantized accuracy sign-off is **not** taken here: it belongs to
//! the trained FP32 investigation. The numbers are measured and printed; only
//! the format-level bounds are asserted.

#![allow(clippy::chunks_exact_to_as_chunks, clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fastnn::inference::backend::cpu::quant::{
    matmul_quantized, registration, ActivationSpec, CoverageReport, PackedView, PackedWeight,
    ProjectionKind, ProjectionSpec, QuantCpuBackend, QuantDecision, QuantPolicy, QuantWeightSet,
    WorkHint,
};
use fastnn::inference::import::{import_onnx_file, ModelImport};
use fastnn::inference::model::{ModelWeights, Plan, Planner, PreparedModel, Session};
use fastnn::inference::quant::{crc32, QuantBlob, QuantError};
use fastnn::inference::shape::{LiveDims, Shape, Strides};
use fastnn::inference::{CpuBackend, DType, DeviceId, Fn3Error, OpId, Tensor, TensorMeta};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn quant_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/quant")
}

fn quant_manifest() -> serde_json::Value {
    let path = quant_root().join("manifest.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn f32_tensor(shape: [u64; 2], values: &[f32]) -> Tensor {
    Tensor::from_f32(Shape::from_dims(shape), values).expect("f32 tensor")
}

fn cpu_out(shape: [u64; 2]) -> Tensor {
    Tensor::zeros(Shape::from_dims(shape), DType::F32, DeviceId::Cpu).expect("output tensor")
}

fn as_f64(value: &serde_json::Value) -> f64 {
    value.as_f64().expect("number")
}

// ---------------------------------------------------------------------------
// 1. Format parity: the committed blobs, the Python manifest, the Rust decoder
// ---------------------------------------------------------------------------

#[test]
fn fixture_unpack_is_byte_exact_and_the_kernel_matches_the_reference() {
    let root = quant_root();
    let manifest = quant_manifest();
    let fixtures = manifest["fixtures"].as_array().expect("fixtures");
    assert!(!fixtures.is_empty());

    let mut saw_tail = false;
    let mut saw_zero_point = false;

    for entry in fixtures {
        let relative = entry["path"].as_str().expect("fixture path");
        let bytes = std::fs::read(root.join(relative))
            .unwrap_or_else(|error| panic!("{relative}: {error}"));
        let blob = QuantBlob::parse(&bytes)
            .unwrap_or_else(|error| panic!("{relative}: valid blob rejected: {error}"));
        for want in entry["tensors"].as_array().expect("tensors") {
            let name = want["name"].as_str().expect("name");
            let tensor = blob
                .tensor(name)
                .unwrap_or_else(|| panic!("{relative}: tensor {name} missing"));
            let view = PackedView::from_quant_tensor(tensor)
                .unwrap_or_else(|error| panic!("{relative}/{name}: packed view: {error}"));
            let n_rows = tensor.n_rows();
            let k = tensor.k();
            let groups = tensor.groups_per_row();

            // Byte-exact unpack: codes including the row padding, then every
            // scale and zero-point bit pattern the manifest recorded.
            let mut packed = Vec::with_capacity(view.code_bytes());
            for row in 0..n_rows {
                for column in 0..k {
                    packed.push(tensor.code(row, column) as u8);
                }
                packed.extend(std::iter::repeat_n(0u8, view.row_stride() - k));
            }
            assert_eq!(
                packed,
                view.codes(),
                "{relative}/{name}: code bytes are not the manifest bytes"
            );
            let expected_hex = want["packed_hex"].as_str().expect("packed_hex");
            assert_eq!(
                hex(view.codes()),
                expected_hex,
                "{relative}/{name}: packed hex"
            );
            let scales = want["scales"].as_array().expect("scales");
            let zeros = want["zero_points"].as_array().expect("zeros");
            for row in 0..n_rows {
                for group in 0..groups {
                    let at = row * groups + group;
                    assert_eq!(
                        u16::from_le_bytes([view.scales()[at * 2], view.scales()[at * 2 + 1]]),
                        half::f16::from_f32(scales[at].as_f64().unwrap() as f32).to_bits(),
                        "{relative}/{name}: scale [{row}, {group}]"
                    );
                    assert_eq!(
                        i16::from_le_bytes([
                            view.zero_points()[at * 2],
                            view.zero_points()[at * 2 + 1]
                        ]),
                        zeros[at].as_i64().unwrap() as i16,
                        "{relative}/{name}: zero-point [{row}, {group}]"
                    );
                    if zeros[at].as_i64().unwrap() != 0 {
                        saw_zero_point = true;
                    }
                }
                let (start, end) = tensor.group_range(groups - 1);
                if end - start < 64 {
                    saw_tail = true;
                }
            }

            // A deterministic activation row, in the format's f32 arithmetic.
            let x: Vec<f32> = (0..k)
                .map(|column| (column as f32 * 0.13).sin() + 0.7)
                .collect();
            let lhs = f32_tensor([1, k as u64], &x);
            let mut out = cpu_out([1, n_rows as u64]);
            {
                let lhs_view = lhs.as_view().expect("lhs view");
                let mut out_view = out.as_mut_view().expect("output view");
                matmul_quantized(
                    OpId::MatMul,
                    &lhs_view,
                    &view,
                    &mut out_view,
                    WorkHint::serial(),
                )
                .unwrap_or_else(|error| panic!("{relative}/{name}: kernel: {error}"));
            }

            // Independent reference: the manifest's dequantized values (produced
            // by the Python encoder) accumulated in f64.
            let expected = want["expected"].as_array().expect("expected");
            let mut decoded = vec![0.0f32; n_rows * k];
            tensor
                .dequantize_into(&mut decoded)
                .expect("decoder buffer size");
            let got = out.as_f32_slice().expect("host output");
            for row in 0..n_rows {
                let mut reference = 0.0f64;
                for column in 0..k {
                    reference += f64::from(x[column]) * as_f64(&expected[row * k + column]);
                }
                let tolerance = 2e-6 * (1.0 + reference.abs());
                assert!(
                    (f64::from(got[row]) - reference).abs() <= tolerance,
                    "{relative}/{name}: row {row}: kernel {} vs reference {reference} \
                     (tolerance {tolerance})",
                    got[row]
                );
                // The committed Rust decoder, dequantize + one f32 accumulator.
                let mut decoder_dot = 0.0f32;
                for column in 0..k {
                    decoder_dot += x[column] * decoded[row * k + column];
                }
                let tolerance = 1e-5 * (1.0 + decoder_dot.abs());
                assert!(
                    (got[row] - decoder_dot).abs() <= tolerance,
                    "{relative}/{name}: row {row}: fused {} vs decoder {decoder_dot} \
                     (tolerance {tolerance})",
                    got[row]
                );
            }
        }
    }

    assert!(saw_tail, "the fixture set must include a tail group");
    assert!(
        saw_zero_point,
        "the fixture set must include a non-zero zero-point"
    );
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

#[test]
fn the_packer_matches_the_independent_python_encoder() {
    // Reference produced by `tools/quant/fn3_quant.py::quantize_int8_grouped`
    // on the logical [N, K] matrix below; the Rust packer is fed its [K, N]
    // transpose, exactly the MatMul right-hand side the graph carries.
    const K: usize = 70;
    const N: usize = 4;
    const CODES_CRC32: u32 = 0x09f7_9257;
    const SCALES_CRC32: u32 = 0xabed_3e89;
    const ZEROS_CRC32: u32 = 0xf832_64c7;

    let mut rhs = vec![0.0f32; K * N];
    for column in 0..K {
        rhs[column * N] = (column as f32 - 20.0) * 0.0375 + 0.11;
        rhs[column * N + 1] = 0.25;
        rhs[column * N + 2] = 0.5 + (column as f32) * 0.01;
        rhs[column * N + 3] = 0.0;
    }
    let weight = PackedWeight::from_f32_rhs("pack-reference", K, N, &rhs).expect("pack");
    let view = weight.view();
    assert_eq!(view.n_rows(), N);
    assert_eq!(view.k(), K);
    assert_eq!(view.groups_per_row(), 2);
    assert_eq!(view.group_range(0), 0..64);
    assert_eq!(view.group_range(1), 64..70);
    assert_eq!(crc32(view.codes()), CODES_CRC32, "code bytes");
    assert_eq!(crc32(view.scales()), SCALES_CRC32, "scale bytes");
    assert_eq!(crc32(view.zero_points()), ZEROS_CRC32, "zero-point bytes");
    // The first group is asymmetric with a non-zero zero-point; the constant
    // groups are symmetric; the all-positive group falls back to symmetric.
    assert_eq!(
        i16::from_le_bytes([view.zero_points()[0], view.zero_points()[1]]),
        -59
    );
    assert_eq!(weight.report().asymmetric_groups, 1);
    assert_eq!(weight.report().constant_groups, 4);
    assert!(weight.report().relative_frobenius < 0.01);
    assert_eq!(weight.report().packed_bytes(), (N * K + N * 2 * 4) as u64);
}

// ---------------------------------------------------------------------------
// 2. Registration, policy, coverage and fallback
// ---------------------------------------------------------------------------

fn trained_projection_specs() -> Vec<ProjectionSpec> {
    let mut specs = Vec::new();
    for layer in 0..12 {
        specs.push(ProjectionSpec::static_weight(
            format!("transformer.h.{layer}.attn.c_attn.weight"),
            &Shape::from_dims([768, 2304]),
            DType::F32,
        ));
        specs.push(ProjectionSpec::static_weight(
            format!("transformer.h.{layer}.attn.c_proj.weight"),
            &Shape::from_dims([768, 768]),
            DType::F32,
        ));
        specs.push(ProjectionSpec::static_weight(
            format!("transformer.h.{layer}.mlp.c_fc.weight"),
            &Shape::from_dims([768, 3072]),
            DType::F32,
        ));
        specs.push(ProjectionSpec::static_weight(
            format!("transformer.h.{layer}.mlp.c_proj.weight"),
            &Shape::from_dims([3072, 768]),
            DType::F32,
        ));
        specs.push(ProjectionSpec::computed(
            format!("transformer.h.{layer}.attn.score.rhs"),
            "a value computed in the step",
            &Shape::from_dims([64, 13]),
        ));
        specs.push(ProjectionSpec::computed(
            format!("transformer.h.{layer}.attn.value.rhs"),
            "a value computed in the step",
            &Shape::from_dims([13, 64]),
        ));
    }
    specs.push(ProjectionSpec::computed(
        "transformer.wte.weight_transposed",
        "a value computed in the step (a transpose)",
        &Shape::from_dims([768, 50257]),
    ));
    specs
}

#[test]
fn coverage_policy_matches_the_format_and_names_every_fallback() {
    let policy = QuantPolicy::default();
    let specs = trained_projection_specs();
    let mut decisions = BTreeMap::new();
    let mut quantized = 0;
    let mut fallback = 0;
    for spec in &specs {
        match policy.classify(spec) {
            QuantDecision::Quantize => quantized += 1,
            QuantDecision::Fallback(reason) => {
                *decisions.entry(reason.code()).or_insert(0usize) += 1;
                fallback += 1;
            }
        }
    }
    assert_eq!((quantized, fallback), (48, 25));
    assert_eq!(decisions.get("dynamic-rhs"), Some(&25));
    assert_eq!(specs.len(), 73);

    // The tiny graph has the same 4 projections per layer over 2 layers.
    let mut tiny_quantized = 0;
    for layer in 0..2 {
        for shape in [[128, 384], [128, 128], [128, 512], [512, 128]] {
            let spec = ProjectionSpec::static_weight(
                format!("transformer.h.{layer}.projection.{shape:?}"),
                &Shape::from_dims(shape),
                DType::F32,
            );
            if policy.classify(&spec) == QuantDecision::Quantize {
                tiny_quantized += 1;
            }
        }
    }
    assert_eq!(tiny_quantized, 8);

    // Every documented fallback is reachable and named.
    let fallbacks = [
        (
            ProjectionSpec::static_weight("small-k", &Shape::from_dims([32, 4096]), DType::F32),
            "k-below-min",
        ),
        (
            ProjectionSpec::static_weight("small-numel", &Shape::from_dims([64, 32]), DType::F32),
            "numel-below-min",
        ),
        (
            ProjectionSpec::static_weight("rank-3", &Shape::from_dims([2, 64, 64]), DType::F32),
            "rhs-rank",
        ),
        (
            ProjectionSpec::static_weight("i32", &Shape::from_dims([768, 768]), DType::I32),
            "weight-dtype",
        ),
        (
            ProjectionSpec::static_weight("shared", &Shape::from_dims([768, 768]), DType::F32)
                .with_shared_weight(true)
                .with_packed_copy(false),
            "shared-weight",
        ),
    ];
    for (spec, code) in fallbacks {
        assert_eq!(policy.classify(&spec).fallback_code(), Some(code));
    }

    // Registration maps onto the backend contract without widening it.
    let qkv = ProjectionSpec::static_weight(
        "transformer.h.0.attn.c_attn.weight",
        &Shape::from_dims([768, 2304]),
        DType::F32,
    );
    let contiguous = ActivationSpec {
        dtype: DType::F32,
        rank: 2,
        k: 768,
        contiguous: true,
    };
    assert_eq!(
        registration(&qkv, contiguous, &policy).support(),
        fastnn::inference::Support::Full
    );
    let partial = registration(
        &qkv,
        ActivationSpec {
            contiguous: false,
            ..contiguous
        },
        &policy,
    );
    assert!(matches!(
        partial.support(),
        fastnn::inference::Support::Partial { .. }
    ));
    assert_eq!(partial.code(), "activation-layout");
    assert!(!partial.is_admitted());

    // `CpuBackend::admits` is untouched: its default body is `supports`.
    use fastnn::inference::backend::Backend;
    let backend = CpuBackend::new();
    let shapes = [
        Shape::from_dims([13, 768]),
        Shape::from_dims([768, 2304]),
        Shape::from_dims([13, 2304]),
    ];
    let dtypes = [DType::F32, DType::F32, DType::F32];
    assert_eq!(
        backend.admits(OpId::MatMul, &shapes, &dtypes, &Default::default()),
        backend.supports(OpId::MatMul)
    );
    assert!(backend.supports(OpId::MatMul).is_full());
}

trait FallbackCode {
    fn fallback_code(&self) -> Option<&'static str>;
}

impl FallbackCode for QuantDecision {
    fn fallback_code(&self) -> Option<&'static str> {
        match self {
            QuantDecision::Quantize => None,
            QuantDecision::Fallback(reason) => Some(reason.code()),
        }
    }
}

fn synthetic_projection() -> (Tensor, PackedWeight) {
    let k = 64usize;
    let n = 128usize;
    let values: Vec<f32> = (0..k * n)
        .map(|index| (((index * 37) % 101) as f32 - 50.0) / 50.0)
        .collect();
    let source = f32_tensor([k as u64, n as u64], &values);
    let weight = PackedWeight::from_tensor("synthetic.proj.weight", &source).expect("pack");
    (source, weight)
}

#[test]
fn an_unregistered_or_strided_matmuls_falls_back_explicitly() {
    use fastnn::inference::backend::Backend;
    let policy = QuantPolicy::default();
    let (source, weight) = synthetic_projection();
    let mut set = QuantWeightSet::default();
    let spec = ProjectionSpec::static_weight(
        "synthetic.proj.weight",
        source.shape(),
        source.meta().dtype(),
    );
    set.register(spec, weight.clone(), &source)
        .expect("register");
    let backend = QuantCpuBackend::new(set, policy, WorkHint::serial());

    // A contiguous invocation takes the fused path.
    let k = weight.k();
    let n = weight.n_rows();
    let values: Vec<f32> = (0..k)
        .map(|column| (column as f32 * 0.07).cos() + 0.2)
        .collect();
    let lhs = f32_tensor([1, k as u64], &values);
    let mut out = cpu_out([1, n as u64]);
    {
        let lhs_view = lhs.as_view().expect("lhs");
        let rhs_view = source.as_view().expect("rhs");
        let mut out_view = out.as_mut_view().expect("out");
        backend
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &Default::default(),
            )
            .expect("quantized matmul");
    }
    let stats = backend.stats();
    assert_eq!(stats.quantized_calls, 1);
    assert_eq!(stats.fallback_calls, 0);
    assert!(stats.all_registered_weights_used());
    let kernel_view = backend
        .packed_view("synthetic.proj.weight")
        .expect("packed view");
    let mut direct = cpu_out([1, n as u64]);
    {
        let lhs_view = lhs.as_view().expect("lhs");
        let mut out_view = direct.as_mut_view().expect("out");
        matmul_quantized(
            OpId::MatMul,
            &lhs_view,
            &kernel_view,
            &mut out_view,
            WorkHint::serial(),
        )
        .expect("direct kernel");
    }
    assert_eq!(
        out.as_f32_slice().unwrap(),
        direct.as_f32_slice().unwrap(),
        "the backend must call the fused kernel, not a different arithmetic"
    );

    // An unpacked right-hand side (different storage) is a counted fallback.
    let other = f32_tensor([k as u64, n as u64], source.as_f32_slice().unwrap());
    let mut out = cpu_out([1, n as u64]);
    {
        let lhs_view = lhs.as_view().expect("lhs");
        let rhs_view = other.as_view().expect("rhs");
        let mut out_view = out.as_mut_view().expect("out");
        backend
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &Default::default(),
            )
            .expect("fallback matmul");
    }
    let mut reference = cpu_out([1, n as u64]);
    {
        let lhs_view = lhs.as_view().expect("lhs");
        let rhs_view = other.as_view().expect("rhs");
        let mut out_view = reference.as_mut_view().expect("out");
        CpuBackend::new()
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &Default::default(),
            )
            .expect("reference matmul");
    }
    assert_eq!(
        out.as_f32_slice().unwrap(),
        reference.as_f32_slice().unwrap()
    );
    let stats = backend.stats();
    assert_eq!(stats.fallback_calls, 1);
    assert_eq!(stats.fallback_reasons.get("unpacked-rhs"), Some(&1));

    // A strided activation is a `Partial` registration: the kernel rejects it
    // with UnsupportedOp before writing, and the backend delegates.
    let storage = Tensor::from_f32(Shape::from_dims([128]), &vec![1.0f32; 128]).expect("storage");
    let meta = TensorMeta::strided(
        Shape::from_dims([1, k as u64]),
        Strides::from_elements(vec![2, 2]),
        0,
        DType::F32,
        DeviceId::Cpu,
    )
    .expect("strided metadata");
    let strided = Tensor::from_storage(storage.storage().clone(), meta).expect("strided tensor");
    assert!(!strided.meta().is_contiguous());
    let error = matmul_quantized(
        OpId::MatMul,
        &strided.as_view().unwrap(),
        &kernel_view,
        &mut cpu_out([1, n as u64]).as_mut_view().unwrap(),
        WorkHint::serial(),
    )
    .expect_err("the kernel must reject a strided activation");
    assert!(matches!(error, Fn3Error::UnsupportedOp { .. }));

    let mut out = cpu_out([1, n as u64]);
    {
        let lhs_view = strided.as_view().expect("lhs");
        let rhs_view = source.as_view().expect("rhs");
        let mut out_view = out.as_mut_view().expect("out");
        backend
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &Default::default(),
            )
            .expect("delegated matmul");
    }
    let mut reference = cpu_out([1, n as u64]);
    {
        let lhs_view = strided.as_view().expect("lhs");
        let rhs_view = source.as_view().expect("rhs");
        let mut out_view = reference.as_mut_view().expect("out");
        CpuBackend::new()
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &Default::default(),
            )
            .expect("reference matmul");
    }
    assert_eq!(
        out.as_f32_slice().unwrap(),
        reference.as_f32_slice().unwrap()
    );
    let stats = backend.stats();
    assert_eq!(stats.quantized_calls, 1);
    assert!(stats.fallback_calls >= 2);
    assert_eq!(stats.fallback_reasons.get("activation-layout"), Some(&1));

    // A view of the same storage with a different shape is not mistaken for
    // the registered weight.
    let transposed_meta = TensorMeta::contiguous(
        Shape::from_dims([n as u64, k as u64]),
        DType::F32,
        DeviceId::Cpu,
    )
    .expect("transposed metadata");
    let transposed =
        Tensor::from_storage(source.storage().clone(), transposed_meta).expect("transposed");
    assert!(backend
        .weights()
        .lookup(&transposed.as_view().unwrap())
        .is_none());
}

#[test]
fn work_hint_partitions_are_deterministic_and_bit_identical() {
    let k = 130usize; // two full groups and a two-element tail
    let n = 16usize;
    let values: Vec<f32> = (0..k * n)
        .map(|index| ((index * 17 % 211) as f32 - 105.0) / 37.0)
        .collect();
    let source = f32_tensor([k as u64, n as u64], &values);
    let weight = PackedWeight::from_tensor("determinism.proj.weight", &source).expect("pack");
    let x: Vec<f32> = (0..3 * k)
        .map(|index| ((index * 29 % 97) as f32 - 48.0) / 19.0)
        .collect();
    let lhs = f32_tensor([3, k as u64], &x);

    let run = |hint: WorkHint| -> Vec<f32> {
        let mut out = cpu_out([3, n as u64]);
        matmul_quantized(
            OpId::MatMul,
            &lhs.as_view().unwrap(),
            &weight.view(),
            &mut out.as_mut_view().unwrap(),
            hint,
        )
        .expect("kernel");
        out.as_f32_slice().unwrap().to_vec()
    };

    let serial = run(WorkHint::serial());
    assert_eq!(serial, run(WorkHint::serial()), "repeat run");
    for threads in [2, 3, 8] {
        for min_rows in [1, 3, 7] {
            let hinted = run(WorkHint::new(threads, min_rows).unwrap());
            for (index, (a, b)) in serial.iter().zip(&hinted).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "threads {threads}, min_rows {min_rows}, element {index}"
                );
            }
        }
    }
    // A one-row weight is a single chunk for every hint.
    for threads in [1, 4, 16] {
        let hint = WorkHint::new(threads, 4).unwrap();
        assert_eq!(hint.chunks(1).collect::<Vec<_>>(), vec![0..1]);
    }
}

#[test]
fn batched_activations_use_the_same_group_arithmetic() {
    use fastnn::inference::backend::Backend;
    let k = 130usize;
    let n = 16usize;
    let values: Vec<f32> = (0..k * n)
        .map(|index| ((index * 13 % 149) as f32 - 74.0) / 31.0)
        .collect();
    let source = f32_tensor([k as u64, n as u64], &values);
    let weight = PackedWeight::from_tensor("batch.proj.weight", &source).expect("pack");
    let x: Vec<f32> = (0..2 * 3 * k)
        .map(|index| ((index * 11 % 83) as f32 - 41.0) / 17.0)
        .collect();
    let lhs = Tensor::from_f32(Shape::from_dims([2, 3, k as u64]), &x).expect("batched lhs");
    let shape = Shape::from_dims([2, 3, n as u64]);

    let mut fused = Tensor::zeros(shape.clone(), DType::F32, DeviceId::Cpu).expect("output");
    matmul_quantized(
        OpId::MatMul,
        &lhs.as_view().unwrap(),
        &weight.view(),
        &mut fused.as_mut_view().unwrap(),
        WorkHint::serial(),
    )
    .expect("fused batch");

    // The reference matmul uses the *dequantized* packed weight, so the only
    // difference left is the per-group scale application.
    let view = weight.view();
    let mut dequantized = vec![0.0f32; k * n];
    for row in 0..n {
        for column in 0..k {
            let group = column / 64;
            let at = row * view.groups_per_row() + group;
            let scale = half::f16::from_bits(u16::from_le_bytes([
                view.scales()[at * 2],
                view.scales()[at * 2 + 1],
            ]))
            .to_f32();
            let zero =
                i16::from_le_bytes([view.zero_points()[at * 2], view.zero_points()[at * 2 + 1]]);
            dequantized[column * n + row] = ((view.codes()[row * view.row_stride() + column] as i8)
                as i32
                - i32::from(zero)) as f32
                * scale;
        }
    }
    let dequantized = f32_tensor([k as u64, n as u64], &dequantized);

    let mut reference = Tensor::zeros(shape, DType::F32, DeviceId::Cpu).expect("output");
    {
        let lhs_view = lhs.as_view().unwrap();
        let rhs_view = dequantized.as_view().unwrap();
        let mut out_view = reference.as_mut_view().unwrap();
        CpuBackend::new()
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &Default::default(),
            )
            .expect("reference batch");
    }
    for (index, (&got, &want)) in fused
        .as_f32_slice()
        .unwrap()
        .iter()
        .zip(reference.as_f32_slice().unwrap())
        .enumerate()
    {
        let tolerance = 1e-5 * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tolerance,
            "batched element {index}: fused {got} vs reference {want}"
        );
    }
}

#[test]
fn malformed_metadata_is_rejected_before_the_kernel_runs() {
    let root = quant_root();
    for (file, kind) in [
        ("malformed/unsupported_qbits.fn3q", "unsupported-qbits"),
        (
            "malformed/zero_point_out_of_range.fn3q",
            "zero-point-out-of-range",
        ),
        ("malformed/scale_nan.fn3q", "bad-scale"),
        (
            "malformed/trailing_or_truncated_bytes.fn3q",
            "trailing-or-truncated-bytes",
        ),
    ] {
        let bytes =
            std::fs::read(root.join(file)).unwrap_or_else(|error| panic!("{file}: {error}"));
        match QuantBlob::parse(&bytes) {
            Ok(_) => panic!("{file}: malformed blob was accepted"),
            Err(error) => {
                assert_eq!(error.kind(), kind, "{file}");
                assert!(!error.to_string().is_empty());
            }
        }
    }
    let bytes = std::fs::read(root.join("v1/tail_k70.fn3q")).expect("valid fixture");
    let blob = QuantBlob::parse(&bytes).expect("valid blob");
    let tensor = blob.tensor("tail_k70").expect("tensor");

    // A packed view whose code / scale / zero-point lengths disagree with its
    // geometry is rejected before any arithmetic.
    let view = PackedView::from_quant_tensor(tensor).expect("view");
    assert!(matches!(
        PackedView::new(
            view.n_rows(),
            view.k(),
            view.row_stride(),
            &view.codes()[..view.codes().len() - 1],
            view.scales(),
            view.zero_points(),
        ),
        Err(Fn3Error::InvalidShape { .. })
    ));
    assert!(matches!(
        PackedView::new(
            view.n_rows(),
            view.k(),
            view.k() - 1,
            view.codes(),
            view.scales(),
            view.zero_points(),
        ),
        Err(Fn3Error::InvalidShape { .. })
    ));
    assert!(matches!(
        PackedView::new(
            view.n_rows(),
            view.k(),
            view.row_stride(),
            view.codes(),
            &[],
            &[]
        ),
        Err(Fn3Error::InvalidShape { .. })
    ));

    // A kernel whose K disagrees with the packed weight reports the mismatch
    // and leaves the output untouched.
    let lhs = f32_tensor([1, 64], &[1.0f32; 64]);
    let mut out = cpu_out([1, 1]);
    let before = out.as_f32_slice().unwrap().to_vec();
    let error = matmul_quantized(
        OpId::MatMul,
        &lhs.as_view().unwrap(),
        &view,
        &mut out.as_mut_view().unwrap(),
        WorkHint::serial(),
    )
    .expect_err("K mismatch");
    assert!(matches!(error, Fn3Error::ShapeMismatch { .. }));
    assert_eq!(out.as_f32_slice().unwrap(), before);

    // The reader's typed error is the one the independent Python reference
    // shares; the kernel entry point never sees an unvalidated blob.
    let qbits = QuantBlob::parse(
        &std::fs::read(root.join("malformed/unsupported_qbits.fn3q")).expect("file"),
    )
    .expect_err("qbits");
    assert!(matches!(qbits, QuantError::UnsupportedQBits { .. }));
    assert_eq!(qbits.kind(), "unsupported-qbits");
}

// ---------------------------------------------------------------------------
// 3. The pinned models: coverage, accuracy and measurements
// ---------------------------------------------------------------------------

const TINY_ONNX_BYTES: u64 = 27_897_974;
const TINY_ONNX_SHA256: &str = "e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1";
const TRAINED_ONNX_BYTES: u64 = 498_126_358;
const TRAINED_ONNX_SHA256: &str =
    "42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d";

const PROMPT: [u32; 13] = [
    464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257,
];
const DECODE: [u32; 2] = [640, 11];
const CONTEXT: u64 = 32;

/// Streaming SHA-256, so verifying a 498 MB model does not become the peak
/// memory of the measurement run.
struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    length: u64,
}

impl Sha256 {
    fn new() -> Sha256 {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0u8; 64],
            buffered: 0,
            length: 0,
        }
    }

    fn absorb(&mut self, mut data: &[u8]) {
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                compress(&mut self.state, &self.buffer);
                self.buffered = 0;
            }
        }
        while data.len() >= 64 {
            compress(&mut self.state, &data[..64]);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        self.absorb(data);
    }

    fn finish(mut self) -> String {
        let bit_len = self.length.wrapping_mul(8);
        let mut padding = [0u8; 72];
        padding[0] = 0x80;
        let pad = if self.buffered < 56 {
            56 - self.buffered
        } else {
            120 - self.buffered
        };
        self.absorb(&padding[..pad]);
        self.absorb(&bit_len.to_be_bytes());
        debug_assert_eq!(self.buffered, 0);
        let mut text = String::with_capacity(64);
        for word in self.state {
            text.push_str(&format!("{word:08x}"));
        }
        text
    }
}

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
        let s0 =
            w[index - 15].rotate_right(7) ^ w[index - 15].rotate_right(18) ^ (w[index - 15] >> 3);
        let s1 =
            w[index - 2].rotate_right(17) ^ w[index - 2].rotate_right(19) ^ (w[index - 2] >> 10);
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

fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finish())
}

fn model_path(variable: &str, default: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn verified_model(path: &Path, bytes: u64, sha256: &str) -> bool {
    if !path.exists() {
        println!("SKIP: {} is not present", path.display());
        return false;
    }
    let metadata =
        std::fs::metadata(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(
        metadata.len(),
        bytes,
        "{} is not the pinned byte length",
        path.display()
    );
    let digest = sha256_file(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(digest, sha256, "{} is not the pinned model", path.display());
    true
}

struct Loaded {
    import: ModelImport,
    model: PreparedModel,
    plan: Plan,
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

fn load_model(path: &Path, heads: u64, kv_heads: u64, head_dim: u64) -> Loaded {
    let import = import_onnx_file(path).expect("import");
    let backend = CpuBackend::new();
    let planner = Planner::new(&import, &backend);
    let envelope = planner
        .envelope(1, PROMPT.len() as u64, CONTEXT, CONTEXT + 1)
        .expect("envelope");
    let dims =
        LiveDims::new(1, PROMPT.len() as u64, 0, heads, kv_heads, head_dim).expect("live dims");
    let plan = planner.plan(&envelope, dims).expect("prefill plan");
    let config = import.geometry().kv_cache_config(CONTEXT, 1);
    let weights = import.to_model_weights().expect("host weights");
    let weights = retag_weights(&weights, DeviceId::Cpu);
    let model = PreparedModel::new(
        plan.execution().clone(),
        weights,
        DeviceId::Cpu,
        envelope,
        Some(config),
    )
    .expect("prepared model");
    Loaded {
        import,
        model,
        plan,
    }
}

fn run_logits<'a>(
    loaded: &'a Loaded,
    backend: &'a dyn fastnn::inference::Backend,
    prompt: &[u32],
    decode: &[u32],
) -> Vec<Vec<f32>> {
    let mut session =
        Session::with_runtime(&loaded.model, &loaded.import, backend).expect("session");
    let mut logits = Vec::with_capacity(1 + decode.len());
    let prefill = session.prefill(prompt).expect("prefill");
    logits.push(prefill.as_f32_slice().expect("host logits").to_vec());
    for &token in decode {
        let step = session.decode(&[token]).expect("decode");
        logits.push(step.as_f32_slice().expect("host logits").to_vec());
    }
    logits
}

#[derive(Debug, Clone, Copy)]
struct Accuracy {
    max_abs: f32,
    mean_abs: f64,
    centered_max: f32,
    centered_rmse: f64,
    kl: f64,
    top1_reference: usize,
    top1_actual: usize,
    top1_margin: f32,
    top5_overlap: usize,
    fifth_gap: f32,
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&value| f64::from(value - maximum).exp())
        .collect();
    let total: f64 = exps.iter().sum();
    exps.into_iter().map(|value| value / total).collect()
}

fn top_indices(logits: &[f32], count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&left, &right| {
        logits[right]
            .partial_cmp(&logits[left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.cmp(&right))
    });
    order.truncate(count);
    order
}

fn accuracy(reference: &[f32], actual: &[f32]) -> Accuracy {
    assert_eq!(reference.len(), actual.len());
    let mut max_abs = 0.0f32;
    let mut total_abs = 0.0f64;
    let mut error = Vec::with_capacity(reference.len());
    for (&want, &got) in reference.iter().zip(actual) {
        assert!(got.is_finite(), "non-finite quantized logit");
        let difference = got - want;
        max_abs = max_abs.max(difference.abs());
        total_abs += f64::from(difference.abs());
        error.push(difference);
    }
    let mean_error = error.iter().map(|&value| f64::from(value)).sum::<f64>() / error.len() as f64;
    let mut centered_max = 0.0f32;
    let mut centered_squares = 0.0f64;
    for value in &error {
        let centered = f64::from(*value) - mean_error;
        centered_max = centered_max.max(centered.abs() as f32);
        centered_squares += centered * centered;
    }
    let centered_rmse = (centered_squares / error.len() as f64).sqrt();

    let reference_probs = softmax(reference);
    let actual_probs = softmax(actual);
    let kl = reference_probs
        .iter()
        .zip(&actual_probs)
        .map(|(&p, &q)| if p > 0.0 { p * (p.ln() - q.ln()) } else { 0.0 })
        .sum::<f64>()
        .max(0.0);

    let reference_top = top_indices(reference, 6);
    let actual_top = top_indices(actual, 5);
    let top5_overlap = reference_top
        .iter()
        .take(5)
        .filter(|index| actual_top.contains(index))
        .count();
    Accuracy {
        max_abs,
        mean_abs: total_abs / reference.len() as f64,
        centered_max,
        centered_rmse,
        kl,
        top1_reference: reference_top[0],
        top1_actual: actual_top[0],
        top1_margin: reference[reference_top[0]] - reference[reference_top[1]],
        top5_overlap,
        fifth_gap: reference[reference_top[4]] - reference[reference_top[5]],
    }
}

fn print_accuracy(label: &str, accuracy: &Accuracy) {
    println!(
        "  {label}: max_abs {:.3e}, mean_abs {:.3e}, centered_max {:.3e}, \
         centered_rmse {:.3e}, KL {:.3e}, top1 {}/{} (margin {:.3}), top5 {}/5 (gap {:.3})",
        accuracy.max_abs,
        accuracy.mean_abs,
        accuracy.centered_max,
        accuracy.centered_rmse,
        accuracy.kl,
        accuracy.top1_actual,
        accuracy.top1_reference,
        accuracy.top1_margin,
        accuracy.top5_overlap,
        accuracy.fifth_gap
    );
}

/// The #68 §6 bounds. A failure is reported as a failure; no bound is widened.
fn assert_accuracy_gates(label: &str, accuracy: &Accuracy, kl_bound: f64) {
    assert!(
        accuracy.max_abs <= 1.5,
        "{label}: elementwise logits max_abs {:.3e} exceeds 1.5",
        accuracy.max_abs
    );
    assert!(
        accuracy.centered_max <= 0.8,
        "{label}: centered max {:.3e} exceeds 0.8",
        accuracy.centered_max
    );
    assert!(
        accuracy.centered_rmse <= 0.25,
        "{label}: centered RMSE {:.3e} exceeds 0.25",
        accuracy.centered_rmse
    );
    assert!(
        accuracy.kl <= kl_bound,
        "{label}: KL {:.3e} exceeds {kl_bound:.0e}",
        accuracy.kl
    );
    if accuracy.top1_margin > 1.0 {
        assert_eq!(
            accuracy.top1_actual, accuracy.top1_reference,
            "{label}: top-1 flipped with a reference margin of {:.3}",
            accuracy.top1_margin
        );
    }
    if accuracy.fifth_gap > 2.0 * accuracy.centered_max {
        assert!(
            accuracy.top5_overlap >= 4,
            "{label}: top-5 overlap {} with a 5th-place gap of {:.3}",
            accuracy.top5_overlap,
            accuracy.fifth_gap
        );
    }
}

fn print_coverage(loaded: &Loaded, set: &QuantWeightSet, expected: (usize, usize)) {
    let coverage: &CoverageReport = set.coverage();
    assert_eq!(
        (coverage.quantized(), coverage.fallback()),
        expected,
        "coverage counts"
    );
    println!(
        "coverage: {} quantized, {} fallback",
        coverage.quantized(),
        coverage.fallback()
    );
    for kind in [
        ProjectionKind::Qkv,
        ProjectionKind::AttnOut,
        ProjectionKind::MlpIn,
        ProjectionKind::MlpOut,
        ProjectionKind::LmHead,
        ProjectionKind::Other,
    ] {
        let quantized = coverage.quantized_of(kind);
        let fallback = coverage
            .projections()
            .iter()
            .filter(|row| row.kind == kind && matches!(row.decision, QuantDecision::Fallback(_)))
            .count();
        if quantized + fallback > 0 {
            println!(
                "  {:<9} quantized {:3}  fallback {:3}",
                kind.label(),
                quantized,
                fallback
            );
        }
    }
    for (code, count) in coverage.fallback_reasons() {
        println!("  fallback {code}: {count}");
    }

    let mut max_relative = 0.0f64;
    let mut total_relative = 0.0f64;
    for row in coverage.projections() {
        if let Some(relative) = row.relative_frobenius {
            max_relative = max_relative.max(relative);
            total_relative += relative;
        }
    }
    let quantized = coverage.quantized();
    assert!(quantized > 0);
    assert!(
        max_relative <= 0.008,
        "per-projection relative Frobenius {max_relative:.3e} exceeds 0.008"
    );
    println!(
        "per-projection relative Frobenius: mean {:.3e}, max {:.3e} (bound 0.008)",
        total_relative / quantized as f64,
        max_relative
    );

    let fp32 = coverage.fp32_bytes();
    let packed = coverage.packed_bytes();
    let metadata: u64 = set
        .entries()
        .iter()
        .map(|entry| entry.weight().report().metadata_bytes)
        .sum();
    println!(
        "projection storage: fp32 {fp32} B, w8a16 {packed} B ({:.2}%), metadata {metadata} B \
         ({:.2}% of packed), saved {} B",
        packed as f64 / fp32 as f64 * 100.0,
        metadata as f64 / packed as f64 * 100.0,
        fp32 - packed
    );
    let whole_model: u64 = (0..loaded.model.weights().len())
        .map(|index| loaded.model.weights().get(index).unwrap().storage().len() as u64)
        .sum();
    println!(
        "model weights: {whole_model} B FP32 in memory; this backend keeps them and adds {packed} B \
         of packed copies (a prepared artifact that replaces the FP32 projections would use {} B)",
        whole_model - fp32 + packed
    );
    if let Some(row) = coverage
        .projections()
        .iter()
        .find(|row| row.kind == ProjectionKind::LmHead)
    {
        let fp32_lm_head = row.k.saturating_mul(row.n).saturating_mul(4);
        let packed_lm_head = row
            .k
            .saturating_mul(row.n)
            .saturating_add(row.k.div_ceil(64).saturating_mul(row.n).saturating_mul(4));
        println!(
            "  lm_head {} x {}: {fp32_lm_head} B stays FP32 (computed transpose); a packed copy \
             would be {packed_lm_head} B",
            row.n, row.k
        );
    }
}

fn print_stats(backend: &QuantCpuBackend, steps: usize) {
    let stats = backend.stats();
    println!(
        "dispatch: {} quantized ({} per step), {} fallback ({} per step)",
        stats.quantized_calls,
        stats.quantized_calls / steps as u64,
        stats.fallback_calls,
        stats.fallback_calls / steps as u64
    );
    for (code, count) in &stats.fallback_reasons {
        println!("  fallback {code}: {count}");
    }
    assert!(stats.all_registered_weights_used());
    let slowest = stats
        .weights
        .iter()
        .max_by(|left, right| left.mean_nanos().total_cmp(&right.mean_nanos()))
        .expect("at least one weight");
    let fastest = stats
        .weights
        .iter()
        .min_by(|left, right| left.mean_nanos().total_cmp(&right.mean_nanos()))
        .expect("at least one weight");
    println!(
        "fused kernel mean per projection: {:.3} ms .. {:.3} ms (slowest {}, fastest {})",
        slowest.mean_nanos() / 1e6,
        fastest.mean_nanos() / 1e6,
        slowest.name,
        fastest.name
    );
}

struct ModelSpec {
    label: &'static str,
    path_variable: &'static str,
    default_path: &'static str,
    bytes: u64,
    sha256: &'static str,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    expected_coverage: (usize, usize),
    prefill_runs: usize,
    decode_runs: usize,
}

impl ModelSpec {
    fn path(&self) -> PathBuf {
        model_path(self.path_variable, self.default_path)
    }

    fn verify(&self) -> Option<PathBuf> {
        let path = self.path();
        if verified_model(&path, self.bytes, self.sha256) {
            Some(path)
        } else {
            None
        }
    }
}

const TINY: ModelSpec = ModelSpec {
    label: "tiny-gpt2",
    path_variable: "FN3_TINY_ONNX",
    default_path: "/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx",
    bytes: TINY_ONNX_BYTES,
    sha256: TINY_ONNX_SHA256,
    heads: 2,
    kv_heads: 2,
    head_dim: 64,
    expected_coverage: (8, 5),
    prefill_runs: 5,
    decode_runs: 16,
};

const TRAINED: ModelSpec = ModelSpec {
    label: "trained-gpt2",
    path_variable: "FN3_TRAINED_ONNX",
    default_path: "/tmp/opencode/fn3-68/models/trained-gpt2/onnx/model.onnx",
    bytes: TRAINED_ONNX_BYTES,
    sha256: TRAINED_ONNX_SHA256,
    heads: 12,
    kv_heads: 12,
    head_dim: 64,
    expected_coverage: (48, 25),
    prefill_runs: 3,
    decode_runs: 16,
};

fn evaluate(spec: &ModelSpec, gate: bool) {
    let Some(path) = spec.verify() else {
        return;
    };
    let loaded = load_model(&path, spec.heads, spec.kv_heads, spec.head_dim);
    let reference_backend = CpuBackend::new();
    let reference = run_logits(&loaded, &reference_backend, &PROMPT, &DECODE);

    let policy = QuantPolicy::default();
    let started = Instant::now();
    let set = QuantWeightSet::from_plan(&loaded.plan, loaded.model.weights(), policy)
        .expect("quantized weight set");
    let pack_time = started.elapsed();
    println!(
        "{}: packed {} weights in {:.3} s",
        spec.label,
        set.len(),
        pack_time.as_secs_f64()
    );
    print_coverage(&loaded, &set, spec.expected_coverage);

    let backend = QuantCpuBackend::new(set, policy, WorkHint::serial());
    let actual = run_logits(&loaded, &backend, &PROMPT, &DECODE);
    assert_eq!(reference.len(), actual.len());
    println!("accuracy vs the FP32 run:");
    for (position, (want, got)) in reference.iter().zip(&actual).enumerate() {
        let metrics = accuracy(want, got);
        let name = match position {
            0 => "prefill".to_string(),
            1 => "decode step 1".to_string(),
            other => format!("decode step {other}"),
        };
        print_accuracy(&name, &metrics);
        if gate {
            let bound = if position == 0 { 1e-3 } else { 0.02 };
            assert_accuracy_gates(&name, &metrics, bound);
        }
    }
    print_stats(&backend, 1 + DECODE.len());
}

#[test]
#[ignore = "needs the 27,897,974-byte tiny GPT-2 graph"]
fn tiny_gpt2_w8a16_matches_fp32() {
    evaluate(&TINY, false);
}

#[test]
#[ignore = "needs the 498,126,358-byte trained GPT-2 graph"]
fn trained_gpt2_w8a16_matches_fp32() {
    evaluate(&TRAINED, true);
}

// ---------------------------------------------------------------------------
// 4. Measurements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Fp32,
    W8a16,
    Both,
}

fn selected_mode() -> Mode {
    match std::env::var("FN3_QUANT_MODE")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "fp32" => Mode::Fp32,
        "w8a16" | "quant" => Mode::W8a16,
        _ => Mode::Both,
    }
}

fn percentile(sorted: &[Duration], percent: usize) -> Duration {
    let rank = (percent * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

fn print_latency(label: &str, samples: &[Duration]) {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let p50 = percentile(&sorted, 50);
    let p95 = percentile(&sorted, 95);
    println!(
        "{label}: n={} p50 {:.3} ms, p95 {:.3} ms, min {:.3} ms, max {:.3} ms",
        sorted.len(),
        p50.as_secs_f64() * 1e3,
        p95.as_secs_f64() * 1e3,
        sorted[0].as_secs_f64() * 1e3,
        sorted[sorted.len() - 1].as_secs_f64() * 1e3
    );
}

fn latencies(
    loaded: &Loaded,
    backend: &dyn fastnn::inference::Backend,
    prefill_runs: usize,
    decode_runs: usize,
) -> (Vec<Duration>, Vec<Duration>) {
    let mut prefill = Vec::with_capacity(prefill_runs);
    for _ in 0..prefill_runs {
        let mut session =
            Session::with_runtime(&loaded.model, &loaded.import, backend).expect("session");
        let started = Instant::now();
        let _ = session.prefill(&PROMPT).expect("prefill");
        prefill.push(started.elapsed());
    }
    let mut session =
        Session::with_runtime(&loaded.model, &loaded.import, backend).expect("session");
    let _ = session.prefill(&PROMPT).expect("prefill");
    let mut decode = Vec::with_capacity(decode_runs);
    for _ in 0..decode_runs {
        let started = Instant::now();
        let _ = session.decode(&[DECODE[0]]).expect("decode");
        decode.push(started.elapsed());
    }
    (prefill, decode)
}

fn time_loop(iterations: usize, mut call: impl FnMut()) -> f64 {
    let started = Instant::now();
    for _ in 0..iterations {
        call();
    }
    started.elapsed().as_secs_f64() * 1e9 / iterations as f64
}

fn dispatch_overhead(loaded: &Loaded, backend: &QuantCpuBackend) {
    use fastnn::inference::backend::Backend;
    let inner = CpuBackend::new();

    // A 64x64 MatMul whose right-hand side is not registered.
    let k = 64usize;
    let n = 64usize;
    let rhs = f32_tensor([k as u64, n as u64], &vec![0.01f32; k * n]);
    let lhs = f32_tensor([1, k as u64], &vec![0.5f32; k]);
    let mut out = cpu_out([1, n as u64]);
    let inner_ns = time_loop(200, || {
        let lhs_view = lhs.as_view().expect("lhs");
        let rhs_view = rhs.as_view().expect("rhs");
        let mut out_view = out.as_mut_view().expect("out");
        inner
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &Default::default(),
            )
            .expect("inner matmul");
    });
    let delegated_ns = time_loop(200, || {
        let lhs_view = lhs.as_view().expect("lhs");
        let rhs_view = rhs.as_view().expect("rhs");
        let mut out_view = out.as_mut_view().expect("out");
        backend
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &Default::default(),
            )
            .expect("delegated matmul");
    });

    // A registered projection: backend dispatch against the kernel directly.
    let entry = backend
        .weights()
        .entries()
        .first()
        .expect("one registered weight");
    let name = entry.spec().name.clone();
    let source = loaded
        .model
        .weights()
        .named(&name)
        .expect("the packed weight's source tensor");
    let k = entry.weight().k();
    let n = entry.weight().n_rows();
    let lhs = f32_tensor([1, k as u64], &vec![0.01f32; k]);
    let mut out = cpu_out([1, n as u64]);
    let view = entry.weight().view();
    let kernel_ns = time_loop(50, || {
        let lhs_view = lhs.as_view().expect("lhs");
        let mut out_view = out.as_mut_view().expect("out");
        matmul_quantized(
            OpId::MatMul,
            &lhs_view,
            &view,
            &mut out_view,
            WorkHint::serial(),
        )
        .expect("kernel");
    });
    let dispatch_ns = time_loop(50, || {
        let lhs_view = lhs.as_view().expect("lhs");
        let rhs_view = source.as_view().expect("rhs");
        let mut out_view = out.as_mut_view().expect("out");
        backend
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &Default::default(),
            )
            .expect("quantized matmul");
    });
    println!(
        "dispatch overhead: unregistered MatMul {:.0} ns/call (inner {:.0}, +{:.0}); \
         fused projection {:.0} ns/call (kernel {:.0}, +{:.0}, {} x {})",
        delegated_ns,
        inner_ns,
        delegated_ns - inner_ns,
        dispatch_ns,
        kernel_ns,
        dispatch_ns - kernel_ns,
        n,
        k
    );
}

fn proc_status(field: &str) -> Option<String> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            return Some(rest.trim_start_matches(':').trim().to_string());
        }
    }
    None
}

fn print_peak(label: &str) {
    for field in ["VmHWM", "VmRSS"] {
        if let Some(value) = proc_status(field) {
            println!("{label} {field}: {value}");
        }
    }
}

fn measure(spec: &ModelSpec) {
    let Some(path) = spec.verify() else {
        return;
    };
    let loaded = load_model(&path, spec.heads, spec.kv_heads, spec.head_dim);
    let mode = selected_mode();
    println!("{}: mode {mode:?}", spec.label);

    if mode != Mode::W8a16 {
        let backend = CpuBackend::new();
        let (prefill, decode) = latencies(&loaded, &backend, spec.prefill_runs, spec.decode_runs);
        print_latency(&format!("{} fp32 prefill", spec.label), &prefill);
        print_latency(&format!("{} fp32 decode (batch 1)", spec.label), &decode);
    }

    if mode != Mode::Fp32 {
        let policy = QuantPolicy::default();
        let started = Instant::now();
        let set = QuantWeightSet::from_plan(&loaded.plan, loaded.model.weights(), policy)
            .expect("quantized weight set");
        let pack_time = started.elapsed();
        println!(
            "{}: packed {} weights in {:.3} s",
            spec.label,
            set.len(),
            pack_time.as_secs_f64()
        );
        print_coverage(&loaded, &set, spec.expected_coverage);
        let backend = QuantCpuBackend::new(set, policy, WorkHint::serial());
        let (prefill, decode) = latencies(&loaded, &backend, spec.prefill_runs, spec.decode_runs);
        print_latency(&format!("{} w8a16 prefill", spec.label), &prefill);
        print_latency(&format!("{} w8a16 decode (batch 1)", spec.label), &decode);
        backend.reset_stats();
        let _ = run_logits(&loaded, &backend, &PROMPT, &DECODE[..1]);
        print_stats(&backend, 2);
        dispatch_overhead(&loaded, &backend);
    }

    print_peak(spec.label);
}

#[test]
#[ignore = "needs the 27,897,974-byte tiny GPT-2 graph"]
fn measure_tiny() {
    measure(&TINY);
}

#[test]
#[ignore = "needs the 498,126,358-byte trained GPT-2 graph"]
fn measure_trained() {
    measure(&TRAINED);
}
