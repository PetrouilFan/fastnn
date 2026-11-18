//! Deterministic synthetic GPT-2-shaped decoder stack for the 2.x baseline.
//!
//! The baseline revision has no weight artifact or whole-model loader, so the
//! harness builds a real AOT graph with real (deterministically synthesised)
//! f32 weights and runs it through the existing `GraphExecutor`:
//!
//! `hidden [seq, 768] -> 12 x (pre-LN attention + pre-LN MLP) -> LN -> lm_head`
//!
//! Multi-head attention is expressed with rank-2 matmuls (one per head) to stay
//! inside the operator set exercised by the existing CPU tests. There is no KV
//! cache in 2.x, so the decode measurement runs a one-token forward pass; this
//! limitation is recorded in the result metadata rather than hidden.

use crate::schema::{Checksum, Precision};
use fastnn::ir::builder::{GraphBuilder, GraphTensor};
use fastnn::ir::{ComputeGraph, DimExpr, IrDType, TensorType};

#[derive(Clone, Copy, Debug)]
pub struct Gpt2Spec {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub vocab: usize,
    /// Sequence length baked into the graph (prefill length or 1 for decode).
    pub seq: usize,
    pub seed: u64,
    pub include_lm_head: bool,
}

impl Gpt2Spec {
    pub fn gpt2(seq: usize, layers: usize, seed: u64) -> Self {
        Self {
            hidden: 768,
            heads: 12,
            head_dim: 64,
            layers,
            intermediate: 3072,
            vocab: 50_257,
            seq,
            seed,
            include_lm_head: true,
        }
    }

    pub fn tiny(seq: usize, seed: u64) -> Self {
        Self {
            hidden: 96,
            heads: 2,
            head_dim: 48,
            layers: 1,
            intermediate: 192,
            vocab: 1_024,
            seq,
            seed,
            include_lm_head: true,
        }
    }

    pub fn parameter_count(&self) -> u64 {
        let h = self.hidden as u64;
        let qkv = h * 3 * h;
        let attn_out = h * h;
        let mlp = h * self.intermediate as u64 + self.intermediate as u64 * h;
        // LayerNorm gamma/beta (4h), QKV bias (3h), attention-out bias (h),
        // MLP fc bias (intermediate), MLP proj bias (h).
        let vectors = 9 * h + self.intermediate as u64;
        let per_layer = qkv + attn_out + mlp + vectors;
        // Final LayerNorm gamma/beta plus the output projection.
        let head = 2 * h
            + if self.include_lm_head {
                h * self.vocab as u64
            } else {
                0
            };
        self.layers as u64 * per_layer + head
    }

    pub fn weights_bytes(&self) -> u64 {
        self.parameter_count() * 4
    }
}

pub struct BuiltModel {
    pub graph: ComputeGraph,
    pub spec: Gpt2Spec,
    pub output_shape: Vec<u64>,
    pub checksum: Checksum,
    pub parameter_count: u64,
    pub weights_bytes: u64,
}

/// splitmix64: tiny, fully specified, and stable across platforms.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[-scale, scale)`.
    fn next_symmetric(&mut self, scale: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (unit * 2.0 - 1.0) * scale
    }
}

#[derive(Default)]
struct WeightHasher {
    hash: u64,
    bytes: u64,
    parameters: u64,
}

impl WeightHasher {
    fn new() -> Self {
        Self {
            hash: 0xcbf2_9ce4_8422_2325,
            bytes: 0,
            parameters: 0,
        }
    }

    fn push_f32(&mut self, value: f32) {
        for byte in value.to_le_bytes() {
            self.hash ^= u64::from(byte);
            self.hash = self.hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.bytes += 4;
        self.parameters += 1;
    }

    fn finish(&self, subject: &str) -> Checksum {
        Checksum {
            subject: subject.to_string(),
            algorithm: "fnv1a64".to_string(),
            value: format!("{:016x}", self.hash),
        }
    }
}

struct Builder<'a> {
    gb: &'a GraphBuilder,
    rng: SplitMix64,
    hasher: WeightHasher,
}

impl<'a> Builder<'a> {
    fn matrix(&mut self, rows: usize, cols: usize) -> GraphTensor {
        // GPT-2 initialises with N(0, 0.02); uniform ±0.02 is used here so the
        // generator stays allocation-light and platform-independent.
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(self.rng.next_symmetric(0.02));
        }
        for value in &data {
            self.hasher.push_f32(*value);
        }
        let tensor_type = TensorType::new(
            vec![DimExpr::Known(rows as u64), DimExpr::Known(cols as u64)],
            IrDType::F32,
        );
        self.gb.constant(bytemuck::cast_slice(&data), tensor_type)
    }

    fn vector(&mut self, len: usize, scale: f32) -> GraphTensor {
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(self.rng.next_symmetric(scale));
        }
        for value in &data {
            self.hasher.push_f32(*value);
        }
        self.gb.constant(
            bytemuck::cast_slice(&data),
            TensorType::new(vec![DimExpr::Known(len as u64)], IrDType::F32),
        )
    }

    fn scalar(&mut self, value: f32) -> GraphTensor {
        // Scaling constants are graph attributes, not learned parameters.
        self.gb.constant(
            bytemuck::cast_slice(&[value]),
            TensorType::new(vec![], IrDType::F32),
        )
    }
}

fn layer(b: &mut Builder<'_>, x: &GraphTensor, spec: &Gpt2Spec) -> GraphTensor {
    let gb = b.gb;
    let h = spec.hidden;
    let hd = spec.head_dim;

    // Pre-LN self-attention.
    let gamma1 = b.vector(h, 0.02);
    let beta1 = b.vector(h, 0.002);
    let x_norm = gb.layer_norm(x, &gamma1, &beta1, 1e-5);
    let w_qkv = b.matrix(h, 3 * h);
    let b_qkv = b.vector(3 * h, 0.002);
    let qkv = gb.add(&gb.matmul(&x_norm, &w_qkv), &b_qkv);
    let q = gb.slice(&qkv, 1, 0, h);
    let k = gb.slice(&qkv, 1, h, 2 * h);
    let v = gb.slice(&qkv, 1, 2 * h, 3 * h);

    let scale = 1.0 / (hd as f32).sqrt();
    let scale_tensor = b.scalar(scale);
    let mut head_outputs = Vec::with_capacity(spec.heads);
    for head in 0..spec.heads {
        let start = head * hd;
        let end = start + hd;
        let q_head = gb.slice(&q, 1, start, end);
        let k_head = gb.slice(&k, 1, start, end);
        let v_head = gb.slice(&v, 1, start, end);
        let k_transposed = gb.transpose(&k_head);
        let scores = gb.matmul(&q_head, &k_transposed);
        let scaled = gb.mul_scalar(&scores, &scale_tensor);
        let probabilities = gb.softmax(&scaled, 1);
        head_outputs.push(gb.matmul(&probabilities, &v_head));
    }
    let head_refs: Vec<&GraphTensor> = head_outputs.iter().collect();
    let attention = gb.concat(&head_refs, 1);

    let w_out = b.matrix(h, h);
    let b_out = b.vector(h, 0.002);
    let projected = gb.add(&gb.matmul(&attention, &w_out), &b_out);
    let x1 = gb.add(x, &projected);

    // Pre-LN MLP.
    let gamma2 = b.vector(h, 0.02);
    let beta2 = b.vector(h, 0.002);
    let x_norm2 = gb.layer_norm(&x1, &gamma2, &beta2, 1e-5);
    let w_fc = b.matrix(h, spec.intermediate);
    let b_fc = b.vector(spec.intermediate, 0.002);
    let hidden = gb.gelu(&gb.add(&gb.matmul(&x_norm2, &w_fc), &b_fc));
    let w_proj = b.matrix(spec.intermediate, h);
    let b_proj = b.vector(h, 0.002);
    let mlp = gb.add(&gb.matmul(&hidden, &w_proj), &b_proj);
    gb.add(&x1, &mlp)
}

pub fn build(spec: &Gpt2Spec) -> Result<BuiltModel, String> {
    if spec.hidden != spec.heads * spec.head_dim {
        return Err(format!(
            "invalid head geometry: hidden {} != heads {} * head_dim {}",
            spec.hidden, spec.heads, spec.head_dim
        ));
    }
    if spec.seq == 0 {
        return Err("sequence length must be positive".to_string());
    }

    let gb = GraphBuilder::new();
    let input = gb.input_with_dims(
        &[
            DimExpr::Known(spec.seq as u64),
            DimExpr::Known(spec.hidden as u64),
        ],
        IrDType::F32,
    );

    let mut builder = Builder {
        gb: &gb,
        rng: SplitMix64::new(spec.seed),
        hasher: WeightHasher::new(),
    };

    let mut x = input;
    for _ in 0..spec.layers {
        x = layer(&mut builder, &x, spec);
    }

    let gamma_f = builder.vector(spec.hidden, 0.02);
    let beta_f = builder.vector(spec.hidden, 0.002);
    let final_norm = gb.layer_norm(&x, &gamma_f, &beta_f, 1e-5);

    let output = if spec.include_lm_head {
        let w_lm = builder.matrix(spec.hidden, spec.vocab);
        gb.matmul(&final_norm, &w_lm)
    } else {
        final_norm
    };

    let mut graph = gb.to_graph();
    graph.inputs = gb.recorded_input_ids();
    graph.outputs = vec![output.node_id()];

    let output_shape: Vec<u64> = output
        .shape()
        .iter()
        .map(|dim| dim.evaluate().unwrap_or(0))
        .collect();

    Ok(BuiltModel {
        checksum: builder.hasher.finish("weights"),
        parameter_count: builder.hasher.parameters,
        weights_bytes: builder.hasher.bytes,
        graph,
        spec: *spec,
        output_shape,
    })
}

/// Deterministic f32 fixture data for microbenchmarks (`[-scale, scale)`).
pub fn deterministic_f32(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..len).map(|_| rng.next_symmetric(scale)).collect()
}

pub fn precision() -> Precision {
    Precision {
        weights_dtype: "f32".to_string(),
        activations_dtype: "f32".to_string(),
        accumulation_dtype: "f32".to_string(),
        quantization: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_weights_and_checksum() {
        let spec = Gpt2Spec {
            hidden: 8,
            heads: 2,
            head_dim: 4,
            layers: 1,
            intermediate: 16,
            vocab: 32,
            seq: 2,
            seed: 99,
            include_lm_head: true,
        };
        let a = build(&spec).expect("build");
        let b = build(&spec).expect("build");
        assert_eq!(a.checksum.value, b.checksum.value);
        assert_eq!(a.parameter_count, b.parameter_count);
        assert_eq!(a.graph.outputs.len(), 1);
    }

    #[test]
    fn parameter_count_matches_spec() {
        let spec = Gpt2Spec::gpt2(4, 1, 1);
        let built = build(&spec).expect("build");
        assert_eq!(built.parameter_count, spec.parameter_count());
    }
}
