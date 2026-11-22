//! FN3-22 (#68): the locked quantization format and its trained-model evidence.
//!
//! Two independent layers of evidence live here:
//!
//! 1. **Format conformance.** `fixtures/quant/` stores hand-checkable
//!    `fn3-quant-format-v1` blobs. The Rust decoder in [`fastnn::inference::quant`]
//!    must agree with the Python reference in `tools/quant/` byte for byte, and
//!    malformed metadata must produce typed errors rather than panics.
//! 2. **Trained-model parity.** An opt-in, release-mode run of the pinned
//!    `onnx-community/gpt2-ONNX` graph through the real fn3 CPU path, writing
//!    full logits that `tools/quant/compare_trained_run.py` compares with an
//!    independent ONNX Runtime run.
//!
//! The trained-model test is `#[ignore]` and needs `FN3_TRAINED_ONNX`; without
//! it the suite is still a complete format gate and never pretends a model ran.
//!
//! ```bash
//! cargo test --release --no-default-features --test fn3_quant_format -- --ignored --nocapture
//! cargo test --release --no-default-features --test fn3_quant_format
//! ```

#![allow(clippy::chunks_exact_to_as_chunks)]

use std::path::{Path, PathBuf};

use fastnn::inference::import::{import_onnx_file, ModelImport};
use fastnn::inference::model::{ModelWeights, Planner, PreparedModel, Session};
use fastnn::inference::shape::ShapeEnvelope;
use fastnn::inference::{CpuBackend, DeviceId, LiveDims, Tensor, TensorMeta};

// ---------------------------------------------------------------------------
// Trained GPT-2 parity: the real model, the real release build, full logits
// ---------------------------------------------------------------------------

const PROMPT: [u32; 13] = [
    464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257,
];
const DECODE: [u32; 2] = [640, 11];
const TRAINED_SHA256: &str = "42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d";
const TRAINED_BYTES: usize = 498_126_358;
const LOGITS: usize = 50_257;
const ATOL: f32 = 5e-4;
const RTOL: f32 = 1e-4;

fn sha256(bytes: &[u8]) -> String {
    // The parity harness carries the same compact SHA-256; reusing it keeps the
    // test crate dependency-free.
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

fn trained_path() -> Option<PathBuf> {
    std::env::var("FN3_TRAINED_ONNX").ok().map(PathBuf::from)
}

fn output_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("FN3_TRAINED_OUT").unwrap_or_else(|_| "/tmp/opencode/fn3-68/fn3".into()),
    )
}

fn dims(seq: u64, past: u64) -> LiveDims {
    LiveDims::new(1, seq, past, 12, 12, 64).expect("trained GPT-2 total fits")
}

fn prepared(import: &ModelImport) -> (PreparedModel, ShapeEnvelope) {
    let backend = CpuBackend::new();
    let planner = Planner::new(import, &backend);
    let context = 32;
    let envelope = planner
        .envelope(1, PROMPT.len() as u64, context, context + 1)
        .expect("trained window");
    let config = import.geometry().kv_cache_config(context, 1);
    let weights = import.to_model_weights().expect("host weights");
    let weights = retag_weights(&weights, DeviceId::Cpu);
    let plan = planner
        .plan(&envelope, dims(PROMPT.len() as u64, 0))
        .expect("prefill plan")
        .execution()
        .clone();
    let model = PreparedModel::new(plan, weights, DeviceId::Cpu, envelope, Some(config))
        .expect("prepared model");
    (model, envelope)
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

/// Compare one logit vector with a stored fixture under the trained tolerance.
fn compare_fixture(label: &str, actual: &[f32], reference: &[f32]) -> (f32, f64) {
    assert_eq!(actual.len(), reference.len(), "{label}: length");
    let mut max_abs = 0.0f32;
    let mut total = 0.0f64;
    for (index, (&got, &want)) in actual.iter().zip(reference).enumerate() {
        assert!(got.is_finite(), "{label}: element {index} is not finite");
        let difference = (got - want).abs();
        max_abs = max_abs.max(difference);
        total += f64::from(difference);
        let bound = ATOL + RTOL * want.abs();
        assert!(
            difference <= bound,
            "{label}: element {index} is {got} vs {want} ({difference:e} > {bound:e})"
        );
    }
    let mean_abs = total / reference.len() as f64;
    (max_abs, mean_abs)
}

#[test]
#[ignore = "needs the 498,126,358-byte trained ONNX graph; set FN3_TRAINED_ONNX"]
fn trained_gpt2_fp32_prefill_and_decode_match_ort() {
    let Some(path) = trained_path() else {
        println!("SKIP: FN3_TRAINED_ONNX is not set; no trained-model run was attempted");
        return;
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(bytes.len(), TRAINED_BYTES, "not the pinned trained graph");
    assert_eq!(sha256(&bytes), TRAINED_SHA256, "wrong trained graph bytes");
    let import = import_onnx_file(&path).expect("trained import");
    let (model, _envelope) = prepared(&import);
    let backend = CpuBackend::new();
    let mut session = Session::with_runtime(&model, &import, &backend).expect("session");

    let out = output_dir();
    let prefill = session.prefill(&PROMPT).expect("trained prefill");
    let prefill = prefill.as_f32_slice().expect("host logits");
    assert_eq!(prefill.len(), LOGITS);
    write_f32(&out.join("fn3.prefill.p13.last_logits.f32"), prefill);
    println!(
        "fn3 prefill: 13 tokens, cache {}, argmax {}",
        session.cache_len(),
        argmax(prefill)
    );

    let mut step = 0;
    for token in DECODE {
        step += 1;
        let logits = session.decode(&[token]).expect("trained decode");
        let logits = logits.as_f32_slice().expect("host logits");
        assert_eq!(logits.len(), LOGITS);
        write_f32(
            &out.join(format!("fn3.decode.step{step}.logits.f32")),
            logits,
        );
        println!(
            "fn3 decode {step}: token {token}, extent {}, argmax {}",
            session.cache_len(),
            argmax(logits)
        );
    }

    // The committed trained fixtures are the same ORT run this test's Python
    // comparison reproduces byte for byte, so assert against them directly.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity/trained-gpt2");
    for (name, actual) in [
        ("prefill.p13.last_logits", prefill),
        (
            "decode.step1.logits",
            &std::fs::read(out.join("fn3.decode.step1.logits.f32"))
                .map(|bytes| {
                    bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect::<Vec<_>>()
                })
                .expect("step1 logits"),
        ),
    ] {
        let payload = root.join(format!("tensors/{name}.f32"));
        let Ok(bytes) = std::fs::read(&payload) else {
            println!("SKIP fixture {name}: {} is not present", payload.display());
            continue;
        };
        let reference: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let (max_abs, mean_abs) = compare_fixture(name, actual, &reference);
        println!(
            "{name}: {} values, max_abs {max_abs:e}, mean_abs {mean_abs:e}, \
             top1 {} vs {}",
            reference.len(),
            argmax(actual),
            argmax(&reference)
        );
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

// ---------------------------------------------------------------------------
// Format fixtures: the independent Python reference and the Rust decoder
// ---------------------------------------------------------------------------

use fastnn::inference::quant::{QuantBlob, QuantError};

fn quant_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/quant")
}

fn quant_manifest() -> serde_json::Value {
    let path = quant_root().join("manifest.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn as_f32(value: &serde_json::Value) -> f32 {
    value.as_f64().expect("number") as f32
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
fn format_fixtures_decode_like_the_python_reference() {
    let root = quant_root();
    let manifest = quant_manifest();
    assert_eq!(manifest["format"], "fn3-quant-format-v1");
    assert_eq!(manifest["format_version"], 1);
    let fixtures = manifest["fixtures"].as_array().expect("fixtures");
    assert!(!fixtures.is_empty(), "at least one fixture");
    for entry in fixtures {
        let relative = entry["path"].as_str().expect("fixture path");
        let bytes = std::fs::read(root.join(relative))
            .unwrap_or_else(|error| panic!("{relative}: {error}"));
        assert_eq!(
            bytes.len() as u64,
            entry["bytes"].as_u64().unwrap(),
            "{relative}: bytes"
        );
        let blob = QuantBlob::parse(&bytes)
            .unwrap_or_else(|error| panic!("{relative}: valid blob rejected: {error}"));
        let declared = entry["tensors"].as_array().expect("tensors");
        assert_eq!(
            blob.tensors().len(),
            declared.len(),
            "{relative}: tensor count"
        );
        for want in declared {
            let name = want["name"].as_str().expect("name");
            let tensor = blob
                .tensor(name)
                .unwrap_or_else(|| panic!("{relative}: tensor {name} missing"));
            assert_eq!(
                tensor.n_rows() as u64,
                want["n_rows"].as_u64().unwrap(),
                "{relative}/{name}: n_rows"
            );
            assert_eq!(
                tensor.k() as u64,
                want["k"].as_u64().unwrap(),
                "{relative}/{name}: k"
            );
            assert_eq!(
                tensor.qbits() as u64,
                want["qbits"].as_u64().unwrap(),
                "{relative}/{name}: qbits"
            );
            assert_eq!(
                tensor.group_size() as u64,
                want["group_size"].as_u64().unwrap(),
                "{relative}/{name}: group_size"
            );
            assert_eq!(
                tensor.groups_per_row() as u64,
                want["groups_per_row"].as_u64().unwrap(),
                "{relative}/{name}: groups"
            );
            assert_eq!(
                tensor.row_stride_bytes() as u64,
                want["row_stride_bytes"].as_u64().unwrap(),
                "{relative}/{name}: row stride"
            );
            assert_eq!(
                tensor.data_offset() as u64,
                want["data_offset"].as_u64().unwrap(),
                "{relative}/{name}: data offset"
            );
            let scales = want["scales"].as_array().expect("scales");
            let zeros = want["zero_points"].as_array().expect("zeros");
            for row in 0..tensor.n_rows() {
                for group in 0..tensor.groups_per_row() {
                    let index = row * tensor.groups_per_row() + group;
                    assert_eq!(
                        tensor.scale(row, group).to_bits(),
                        as_f32(&scales[index]).to_bits(),
                        "{relative}/{name}: scale[{row},{group}]"
                    );
                    assert_eq!(
                        i32::from(tensor.zero_point(row, group)),
                        zeros[index].as_i64().unwrap() as i32,
                        "{relative}/{name}: zero[{row},{group}]"
                    );
                }
            }

            let mut packed = Vec::new();
            for row in 0..tensor.n_rows() {
                for col in 0..tensor.k() {
                    packed.push(tensor.code(row, col) as u8);
                }
                packed.extend(std::iter::repeat_n(
                    0u8,
                    tensor.row_stride_bytes() - tensor.k(),
                ));
            }
            assert_eq!(
                hex(&packed),
                want["packed_hex"].as_str().expect("packed hex"),
                "{relative}/{name}: packed bytes"
            );

            let expected = want["expected"].as_array().expect("expected");
            let actual = tensor.dequantize();
            assert_eq!(
                actual.len(),
                expected.len(),
                "{relative}/{name}: expected length"
            );
            for (index, (&got, want)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    as_f32(want).to_bits(),
                    "{relative}/{name}: dequantized value {index}"
                );
            }
        }
    }
}

#[test]
fn format_fixtures_are_hand_checkable() {
    // The `basic` fixture's first row is codes -8..=7 at scale 0.5, so byte 0
    // is 0xF8 and value (0, 0) is exactly -4.0.  Row 1 is code -4 with
    // zero-point -4 at scale 1.0, so value (1, 63) is exactly 0.0.
    let bytes = std::fs::read(quant_root().join("v1/basic.fn3q")).expect("basic fixture");
    let blob = QuantBlob::parse(&bytes).expect("basic parses");
    let tensor = blob.tensor("basic").expect("basic tensor");
    assert_eq!(tensor.codes()[0], 0xF8);
    assert_eq!(tensor.codes()[64], 0xFC);
    assert_eq!(tensor.value(0, 0), Some(-4.0));
    assert_eq!(tensor.value(0, 63), Some(3.5));
    assert_eq!(tensor.value(1, 0), Some(0.0));
    assert_eq!(tensor.value(1, 63), Some(0.0));
    assert_eq!(tensor.value(0, 64), None);
    assert_eq!(tensor.value(2, 0), None);
    assert_eq!(tensor.scale(0, 0), 0.5);
    assert_eq!(tensor.zero_point(1, 0), -4);

    // The tail fixture's second group is six elements long and its last value
    // is (60 - 30) * 0.125 = 3.75.
    let bytes = std::fs::read(quant_root().join("v1/tail_k70.fn3q")).expect("tail fixture");
    let blob = QuantBlob::parse(&bytes).expect("tail parses");
    let tensor = blob.tensor("tail_k70").expect("tail tensor");
    assert_eq!(tensor.groups_per_row(), 2);
    assert_eq!(tensor.group_range(0), (0, 64));
    assert_eq!(tensor.group_range(1), (64, 70));
    assert_eq!(tensor.row_stride_bytes(), 80);
    assert_eq!(tensor.value(0, 69), Some(3.75));
}

#[test]
fn malformed_fixtures_return_the_documented_typed_error() {
    let root = quant_root();
    let manifest = quant_manifest();
    let malformed = manifest["malformed"].as_array().expect("malformed");
    assert!(malformed.len() >= 20, "the malformed matrix stays broad");
    for entry in malformed {
        let relative = entry["path"].as_str().expect("path");
        let expected = entry["expected_error"].as_str().expect("expected error");
        let bytes = std::fs::read(root.join(relative))
            .unwrap_or_else(|error| panic!("{relative}: {error}"));
        assert_eq!(
            bytes.len() as u64,
            entry["bytes"].as_u64().unwrap(),
            "{relative}: bytes"
        );
        match QuantBlob::parse(&bytes) {
            Ok(_) => panic!("{relative}: malformed blob was accepted"),
            Err(error) => assert_eq!(
                error.kind(),
                expected,
                "{relative}: got {error} ({})",
                error.kind()
            ),
        }
    }
}

#[test]
fn dequantize_into_requires_the_exact_buffer_length() {
    let bytes = std::fs::read(quant_root().join("v1/basic.fn3q")).expect("basic fixture");
    let blob = QuantBlob::parse(&bytes).expect("basic parses");
    let tensor = blob.tensor("basic").expect("basic tensor");
    let mut short = vec![0.0f32; tensor.n_rows() * tensor.k() - 1];
    match tensor.dequantize_into(&mut short) {
        Err(QuantError::OutputTooSmall { expected, found }) => {
            assert_eq!(expected, tensor.n_rows() * tensor.k());
            assert_eq!(found, expected - 1);
        }
        other => panic!("expected OutputTooSmall, got {other:?}"),
    }
}

