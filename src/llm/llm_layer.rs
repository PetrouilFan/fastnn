use crate::quants::{GgmlQuantizedTensor, QuantizedDType};

/// Configuration for a Gemma-4 transformer layer (pre-norm, GQA, RoPE).
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1536,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            intermediate_size: 4 * 1536,
            rms_norm_eps: 1e-6,
        }
    }
}

/// A single transformer layer for fast LLM inference.
///
/// All linear projections use `GgmlQuantizedTensor` GEMV (native, no dequant).
/// Attention supports **grouped query attention** (GQA) and RoPE.
/// FFN uses fused SiLU × gate + up + down.
pub struct LlmTransformerLayer {
    pub config: LlmConfig,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,

    pub q_proj: GgmlQuantizedTensor,
    pub k_proj: GgmlQuantizedTensor,
    pub v_proj: GgmlQuantizedTensor,
    pub o_proj: GgmlQuantizedTensor,

    pub ffn_gate: GgmlQuantizedTensor,
    pub ffn_up: GgmlQuantizedTensor,
    pub ffn_down: GgmlQuantizedTensor,

    pub q_norm_weight: Option<Vec<f32>>,
    pub k_norm_weight: Option<Vec<f32>>,
}

impl LlmTransformerLayer {
    pub fn new(config: LlmConfig) -> Self {
        use crate::quants::{Q4_K, Q4_0};
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;

        let q_proj = GgmlQuantizedTensor::from_gguf_bytes(
            QuantizedDType::Q4_K,
            [hidden * config.num_attention_heads, hidden],
            vec![0; (hidden * config.num_attention_heads / 256) * 144],
        );

        let k_proj = GgmlQuantizedTensor::from_gguf_bytes(
            QuantizedDType::Q4_K,
            [hidden * config.num_key_value_heads, hidden],
            vec![0; hidden * config.num_key_value_heads / 256 * 144],
        );

        let v_proj = GgmlQuantizedTensor::from_gguf_bytes(
            QuantizedDType::Q4_K,
            [hidden * config.num_key_value_heads, hidden],
            vec![0; hidden * config.num_key_value_heads / 256 * 144],
        );

        let o_proj = GgmlQuantizedTensor::from_gguf_bytes(
            QuantizedDType::Q4_K,
            [hidden, hidden],
            vec![0; hidden * hidden / 256 * 144],
        );

        let ffn_gate = GgmlQuantizedTensor::from_gguf_bytes(
            QuantizedDType::Q4_0,
            [inter, hidden],
            vec![0; inter * hidden / 32 * 18],
        );

        let ffn_up = GgmlQuantizedTensor::from_gguf_bytes(
            QuantizedDType::Q4_0,
            [inter, hidden],
            vec![0; inter * hidden / 32 * 18],
        );

        let ffn_down = GgmlQuantizedTensor::from_gguf_bytes(
            QuantizedDType::Q4_K,
            [hidden, inter],
            vec![0; hidden * inter / 256 * 144],
        );

        LlmTransformerLayer {
            config,
            attn_norm: vec![1.0; hidden],
            ffn_norm: vec![1.0; hidden],
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            ffn_gate,
            ffn_up,
            ffn_down,
            q_norm_weight: None,
            k_norm_weight: None,
        }
    }

    pub fn from_gguf_tensors(
        config: LlmConfig,
        q: GgmlQuantizedTensor,
        k: GgmlQuantizedTensor,
        v: GgmlQuantizedTensor,
        o: GgmlQuantizedTensor,
        gate: GgmlQuantizedTensor,
        up: GgmlQuantizedTensor,
        down: GgmlQuantizedTensor,
        attn_norm: &[f32],
        ffn_norm: &[f32],
    ) -> Self {
        LlmTransformerLayer {
            config,
            attn_norm: attn_norm.to_vec(),
            ffn_norm: ffn_norm.to_vec(),
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            ffn_gate: gate,
            ffn_up: up,
            ffn_down: down,
            q_norm_weight: None,
            k_norm_weight: None,
        }
    }
}
