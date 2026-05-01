use crate::llm::config::LayerConfig;
use crate::llm::kv_cache::KVCache;
use crate::llm::ops::Kernels;
use crate::quants::quantized_tensor::GgmlQuantizedTensor;
use crate::quants::QuantizedGemm;

#[cfg(feature = "debug")]
macro_rules! llm_debug {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}

#[cfg(not(feature = "debug"))]
macro_rules! llm_debug {
    ($($arg:tt)*) => {};
}

#[derive(Clone)]
pub struct AttentionLayer {
    pub q_weights: GgmlQuantizedTensor,
    pub k_weights: GgmlQuantizedTensor,
    pub v_weights: GgmlQuantizedTensor,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub o_weights: GgmlQuantizedTensor,
    pub layer_config: LayerConfig,
    pub kv_cache: KVCache,
}

fn checked_gemv(tensor: &GgmlQuantizedTensor, activation: &[f32], output: &mut [f32]) -> Result<(), Box<dyn std::error::Error>> {
    let shape = tensor.shape();
    if activation.len() != shape[1] {
        return Err(format!("GEMV: activation.len()={} != in_features={}", activation.len(), shape[1]).into());
    }
    if output.len() != shape[0] {
        return Err(format!("GEMV: output.len()={} != out_features={}", output.len(), shape[0]).into());
    }
    tensor.as_gemm().gemv(activation, output);
    Ok(())
}

impl AttentionLayer {
    pub fn new(
        q_weights: GgmlQuantizedTensor,
        k_weights: GgmlQuantizedTensor,
        v_weights: GgmlQuantizedTensor,
        o_weights: GgmlQuantizedTensor,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        layer_config: LayerConfig,
    ) -> Self {
        let kv_cache = KVCache::new(&layer_config, 4096);
        AttentionLayer {
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            q_norm,
            k_norm,
            layer_config,
            kv_cache,
        }
    }

    pub fn forward(
        &mut self,
        x: &mut [f32],
        pos: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let hidden_size = self.layer_config.hidden_size;
        let num_heads = self.layer_config.num_heads;
        let num_kv_heads = self.layer_config.num_kv_heads;
        let head_dim = self.layer_config.head_dim;
        let layer_idx = self.layer_config.layer_idx;

        let q_output_size = self.q_weights.shape()[0];
        let k_output_size = self.k_weights.shape()[0];

        llm_debug!("[Layer {}, pos {}] input hidden_state: len={}", layer_idx, pos, x.len());

        let mut q = vec![0.0; q_output_size];
        let mut k = vec![0.0; k_output_size];
        let mut v = vec![0.0; k_output_size];

        checked_gemv(&self.q_weights, x, &mut q)?;
        checked_gemv(&self.k_weights, x, &mut k)?;
        checked_gemv(&self.v_weights, x, &mut v)?;

        // Apply per-head RMSNorm to Q and K BEFORE RoPE (Gemma-style)
        if let Some(ref q_norm) = self.q_norm {
            Kernels::apply_q_norm(&mut q, q_norm);
        }
        if let Some(ref k_norm) = self.k_norm {
            Kernels::apply_q_norm(&mut k, k_norm);
        }

        // Apply RoPE to Q and K
        let rope_theta = self.layer_config.rope_theta;
        for head in 0..num_heads {
            let offset = head * head_dim;
            Kernels::rope(&mut q[offset..offset + head_dim], rope_theta, pos, head_dim);
        }
        for head in 0..num_kv_heads {
            let offset = head * head_dim;
            Kernels::rope(&mut k[offset..offset + head_dim], rope_theta, pos, head_dim);
        }

        self.kv_cache.append(&k, &v, pos)?;

        let kv_len = self.kv_cache.current_pos();
        let kv_dim = self.kv_cache.kv_dim();

        let gqa_ratio = num_heads / num_kv_heads;
        let mut attn_output = vec![0.0f32; q_output_size];

        // Sliding window attention masking
        let sliding_window = self.layer_config.sliding_window;
        let window_start = if pos >= sliding_window { pos - sliding_window + 1 } else { 0 };

        for qh in 0..num_heads {
            let kv_h = qh / gqa_ratio;
            let q_offset = qh * head_dim;
            let kv_offset = kv_h * head_dim;

            let mut max_score = f32::NEG_INFINITY;
            let mut scores = vec![0.0f32; kv_len];

            let scale = 1.0 / (head_dim as f32).sqrt();

            for t in 0..kv_len {
                if t < window_start {
                    scores[t] = f32::NEG_INFINITY;
                    continue;
                }

                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[q_offset + d] * self.kv_cache.get_k(t)[kv_offset + d];
                }
                let scaled = score * scale;
                scores[t] = scaled;
                if scaled > max_score {
                    max_score = scaled;
                }
            }

            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }

            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..kv_len {
                    val += scores[t] * self.kv_cache.get_v(t)[kv_offset + d];
                }
                attn_output[q_offset + d] += val;
            }
        }

        checked_gemv(&self.o_weights, &attn_output, x)?;

        llm_debug!("[Layer {}, pos {}] output hidden_state: len={}", layer_idx, pos, x.len());

        Ok(())
    }
}
