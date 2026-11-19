#![allow(clippy::chunks_exact_to_as_chunks)] // the dependency-free SHA-256 helper

//! FN3-30 (#76): per-instruction intermediate probe for the trained GPT-2.
//!
//! The whole-model parity test in `tests/fn3_quant_format.rs` compares only the
//! final logits. This test executes the same specialised release plan one
//! instruction at a time and writes the named value slots that a per-layer
//! comparison needs, so an error can be attributed to a stage instead of a
//! model.
//!
//! It is opt-in and self-contained:
//!
//! ```bash
//! FN3_TRAINED_ONNX=/path/to/onnx/model.onnx \
//! FN3_PROBE_OUT=/tmp/fn3-probe \
//! cargo test --release --no-default-features --test fn3_trained_layers \
//!   -- --ignored --nocapture
//! ```
//!
//! Without `FN3_TRAINED_ONNX` the test prints a skip reason and returns; it
//! never pretends a model ran. The output is consumed by
//! `tools/diagnostics/trained-fp32/compare_layers.py`.
//!
//! # Why the plan is executed here instead of through `Session`
//!
//! `Session` intentionally exposes only the decoder interface (logits and KV),
//! not internal slots. The plan format already names every slot after the ONNX
//! value it holds, and `Plan::materialise` plus `Instruction` are public, so
//! this probe can run exactly the same instructions the session would run
//! without adding a debug hook to the runtime. Nothing here changes the
//! numerical path: every instruction is dispatched to the same `CpuBackend`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use fastnn::inference::import::import_onnx_file;
use fastnn::inference::model::{ModelWeights, Planner};
use fastnn::inference::shape::ShapeEnvelope;
use fastnn::inference::{
    Backend, CpuBackend, DeviceId, LiveDims, Tensor, TensorMeta, TensorMut, TensorView,
};

const PROMPT: [u32; 13] = [
    464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257,
];
const DECODE: [u32; 2] = [640, 11];
const TRAINED_SHA256: &str = "42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d";
const TRAINED_BYTES: usize = 498_126_358;

/// Stages captured for every block. Suffix matching is used because the ONNX
/// node names carry the layer index in the middle.
fn wanted(name: &str) -> bool {
    if matches!(
        name,
        "/transformer/Add_output_0" | "/transformer/ln_f/Add_1_output_0" | "logits"
    ) {
        return true;
    }
    if name.starts_with("present.") {
        return true;
    }
    let Some(rest) = name.strip_prefix("/transformer/h.") else {
        return false;
    };
    let (layer, tail) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
    if layer.parse::<u32>().is_err() {
        return false;
    }
    const EVERY_LAYER: [&str; 11] = [
        "/ln_1/Add_1_output_0",
        "/attn/c_attn/Gemm_output_0",
        "/attn/Softmax_output_0",
        "/attn/MatMul_1_output_0",
        "/attn/c_proj/Gemm_output_0",
        "/Add_output_0",
        "/ln_2/Add_1_output_0",
        "/mlp/c_fc/Gemm_output_0",
        "/mlp/act/Mul_3_output_0",
        "/mlp/c_proj/Gemm_output_0",
        "/Add_1_output_0",
    ];
    if EVERY_LAYER.contains(&tail) {
        return true;
    }
    // Layer 0 and the final norm get the decomposition internals, which is
    // where an epsilon or mean/variance-order difference would show first.
    if layer == "0" {
        const LN_INTERNALS: [&str; 6] = [
            "/ln_1/ReduceMean_output_0",
            "/ln_1/ReduceMean_1_output_0",
            "/ln_1/Add_output_0",
            "/ln_1/Sqrt_output_0",
            "/ln_1/Div_output_0",
            "/ln_1/Mul_output_0",
        ];
        if LN_INTERNALS.contains(&tail) {
            return true;
        }
        const GELU_INTERNALS: [&str; 4] = [
            "/mlp/act/Mul_output_0",
            "/mlp/act/Add_output_0",
            "/mlp/act/Tanh_output_0",
            "/mlp/act/Add_1_output_0",
        ];
        if GELU_INTERNALS.contains(&tail) {
            return true;
        }
        // Attention scores before and after the additive mask, plus the
        // scaled Q/K that feed them.
        const ATTN_INTERNALS: [&str; 4] = [
            "/attn/Mul_output_0",
            "/attn/Mul_1_output_0",
            "/attn/MatMul_output_0",
            "/attn/Add_output_0",
        ];
        if ATTN_INTERNALS.contains(&tail) {
            return true;
        }
    }
    false
}

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

fn live_dims(seq: u64, past: u64) -> LiveDims {
    LiveDims::new(1, seq, past, 12, 12, 64).expect("trained GPT-2 total fits")
}

/// Execute one plan instruction at a time, exactly as `ExecutionPlan::execute`
/// does, snapshotting the requested output slots after each dispatch.
fn execute_and_capture(
    backend: &CpuBackend,
    plan: &fastnn::inference::model::Plan,
    slots: &mut [Option<Tensor>],
    out_dir: &Path,
    index: &mut Vec<serde_json::Value>,
) {
    for (position, instruction) in plan.instructions().iter().enumerate() {
        // Outputs are taken first: an input view borrows `slots`, and the plan
        // guarantees an instruction never reads the slot it writes.
        let mut outputs: Vec<Tensor> = instruction
            .outputs
            .iter()
            .map(|&slot| {
                slots[slot as usize].take().unwrap_or_else(|| {
                    panic!("instruction {position}: output slot {slot} is empty")
                })
            })
            .collect();
        let mut inputs: Vec<TensorView<'_>> = Vec::with_capacity(instruction.inputs.len());
        for &slot in &instruction.inputs {
            let tensor = slots[slot as usize]
                .as_ref()
                .unwrap_or_else(|| panic!("instruction {position}: input slot {slot} is empty"));
            inputs.push(tensor.as_view().expect("input view"));
        }
        {
            let mut views: Vec<TensorMut<'_>> = outputs
                .iter_mut()
                .map(|tensor| tensor.as_mut_view().expect("output view"))
                .collect();
            backend
                .execute_op(instruction.op, &inputs, &mut views, &instruction.params)
                .unwrap_or_else(|error| {
                    panic!(
                        "instruction {position} ({}) failed: {error}",
                        instruction.op
                    )
                });
        }
        for (&slot, tensor) in instruction.outputs.iter().zip(outputs.iter()) {
            let name = plan
                .slot(slot)
                .expect("slot description")
                .name()
                .to_string();
            if wanted(&name) {
                let values = tensor.as_f32_slice().expect("captured slot is f32");
                let file = format!("{:04}-{}.f32", index.len(), name.replace(['/', '.'], "_"));
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(out_dir.join(&file), &bytes)
                    .unwrap_or_else(|error| panic!("{}: {error}", out_dir.join(&file).display()));
                let shape: Vec<u64> = tensor.shape().dims().to_vec();
                index.push(serde_json::json!({
                    "name": name,
                    "shape": shape,
                    "file": file,
                    "elements": values.len(),
                    "instruction": position,
                    "op": instruction.op.to_string(),
                }));
            }
        }
        for (slot, tensor) in instruction.outputs.iter().zip(outputs) {
            slots[*slot as usize] = Some(tensor);
        }
    }
}

fn bind_i64(
    plan: &fastnn::inference::model::Plan,
    slots: &mut [Option<Tensor>],
    name: &str,
    values: &[i64],
) {
    let input = plan
        .input(name)
        .unwrap_or_else(|| panic!("plan input {name}"));
    let tensor = Tensor::from_i64(input.shape().clone(), values).expect("i64 input");
    slots[input.slot() as usize] = Some(tensor);
}

fn bind_tensor(
    plan: &fastnn::inference::model::Plan,
    slots: &mut [Option<Tensor>],
    name: &str,
    tensor: Tensor,
) {
    let input = plan
        .input(name)
        .unwrap_or_else(|| panic!("plan input {name}"));
    assert_eq!(input.shape(), tensor.shape(), "{name}: shape");
    slots[input.slot() as usize] = Some(tensor);
}

/// Collect `present.N.key/value` outputs so the next decode step can consume
/// them as `past_key_values.N.key/value`.
fn collect_present(
    plan: &fastnn::inference::model::Plan,
    slots: &[Option<Tensor>],
) -> BTreeMap<String, Tensor> {
    let mut present = BTreeMap::new();
    for layer in 0..12 {
        for kind in ["key", "value"] {
            let name = format!("present.{layer}.{kind}");
            let slot = plan.slot_of(&name).unwrap_or_else(|| panic!("slot {name}"));
            let tensor = slots[slot as usize]
                .as_ref()
                .unwrap_or_else(|| panic!("{name} was not written"))
                .clone();
            present.insert(name, tensor);
        }
    }
    present
}

/// Run one phase: plan, materialise, bind the decoder interface, execute with
/// capture, return the `present` tensors.
#[allow(clippy::too_many_arguments)]
fn run_phase(
    backend: &CpuBackend,
    planner: &Planner<'_>,
    envelope: &ShapeEnvelope,
    weights: &ModelWeights,
    out_root: &Path,
    label: &str,
    tokens: &[u32],
    past: u64,
    past_tensors: Option<&BTreeMap<String, Tensor>>,
) -> BTreeMap<String, Tensor> {
    let seq = tokens.len() as u64;
    let dims = live_dims(seq, past);
    let plan = planner.plan(envelope, dims).expect("phase plan");
    plan.check_supported(backend).expect("phase plan supported");
    let out_dir = out_root.join(label);
    std::fs::create_dir_all(&out_dir).expect("probe directory");
    let mut slots = plan
        .materialise(weights, DeviceId::Cpu)
        .expect("materialise");

    let token_values: Vec<i64> = tokens.iter().map(|&t| i64::from(t)).collect();
    bind_i64(&plan, &mut slots, "input_ids", &token_values);
    let positions: Vec<i64> = (past..past + seq).map(|v| v as i64).collect();
    bind_i64(&plan, &mut slots, "position_ids", &positions);
    let mask = vec![1i64; (past + seq) as usize];
    bind_i64(&plan, &mut slots, "attention_mask", &mask);
    if let Some(past_tensors) = past_tensors {
        for layer in 0..12 {
            for kind in ["key", "value"] {
                let name = format!("past_key_values.{layer}.{kind}");
                let tensor = past_tensors
                    .get(&name.replace("past_key_values", "present"))
                    .unwrap_or_else(|| panic!("{name} has no present source"))
                    .clone();
                bind_tensor(&plan, &mut slots, &name, tensor);
            }
        }
    } else {
        // Prefill: the graph still declares every past input; an empty prefix is
        // a zero-element tensor with the declared `[1, 12, 0, 64]` shape.
        for layer in 0..12 {
            for kind in ["key", "value"] {
                let name = format!("past_key_values.{layer}.{kind}");
                let input = plan
                    .input(&name)
                    .unwrap_or_else(|| panic!("plan input {name}"));
                let tensor = Tensor::zeros(input.shape().clone(), input.dtype(), DeviceId::Cpu)
                    .expect("empty past");
                slots[input.slot() as usize] = Some(tensor);
            }
        }
    }

    let mut index = Vec::new();
    let started = std::time::Instant::now();
    execute_and_capture(backend, &plan, &mut slots, &out_dir, &mut index);
    let elapsed = started.elapsed();

    let present = collect_present(&plan, &slots);
    let index_path = out_dir.join("index.json");
    std::fs::write(
        &index_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "phase": label,
            "tokens": tokens,
            "past": past,
            "seq": seq,
            "total": past + seq,
            "plan_id": plan.identity().to_hex(),
            "instructions": plan.instructions().len(),
            "slots": plan.slots().len(),
            "elapsed_ms": elapsed.as_secs_f64() * 1e3,
            "captured": index,
        }))
        .expect("index json"),
    )
    .expect("write index");

    for name in present.keys() {
        let slot = plan.slot_of(name).expect("present slot");
        let tensor = slots[slot as usize].as_ref().expect("present slot value");
        let values = tensor.as_f32_slice().expect("present f32");
        let file = format!("present-{}.f32", name.replace('.', "_"));
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out_dir.join(&file), bytes).expect("present bytes");
    }

    println!(
        "phase {label}: {seq} token(s), past {past}, plan {} ({} instructions, {} slots), \
         {} captured tensors, {:.1} ms",
        plan.identity().to_hex(),
        plan.instructions().len(),
        plan.slots().len(),
        index.len(),
        elapsed.as_secs_f64() * 1e3
    );
    present
}

#[test]
#[ignore = "needs the 498,126,358-byte trained ONNX graph; set FN3_TRAINED_ONNX"]
fn trained_gpt2_per_layer_intermediates() {
    let Some(path) = std::env::var("FN3_TRAINED_ONNX").ok().map(PathBuf::from) else {
        println!("SKIP: FN3_TRAINED_ONNX is not set; no per-layer probe was attempted");
        return;
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(bytes.len(), TRAINED_BYTES, "not the pinned trained graph");
    assert_eq!(sha256(&bytes), TRAINED_SHA256, "wrong trained graph bytes");
    let import = import_onnx_file(&path).expect("trained import");
    let backend = CpuBackend::new();
    let planner = Planner::new(&import, &backend);
    let context = 32;
    let envelope = planner
        .envelope(1, PROMPT.len() as u64, context, context + 1)
        .expect("trained window");
    let weights = retag_weights(
        &import.to_model_weights().expect("host weights"),
        DeviceId::Cpu,
    );
    let out_root = PathBuf::from(
        std::env::var("FN3_PROBE_OUT")
            .unwrap_or_else(|_| "/tmp/opencode/trained-fp32/fn3-layers".into()),
    );
    std::fs::create_dir_all(&out_root).expect("probe root");

    let present = run_phase(
        &backend, &planner, &envelope, &weights, &out_root, "prefill", &PROMPT, 0, None,
    );
    let present = run_phase(
        &backend,
        &planner,
        &envelope,
        &weights,
        &out_root,
        "decode1",
        &DECODE[..1],
        PROMPT.len() as u64,
        Some(&present),
    );
    let _ = run_phase(
        &backend,
        &planner,
        &envelope,
        &weights,
        &out_root,
        "decode2",
        &DECODE[1..],
        PROMPT.len() as u64 + 1,
        Some(&present),
    );
    println!("wrote per-layer probe to {}", out_root.display());
}
