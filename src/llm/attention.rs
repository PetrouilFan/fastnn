use crate::llm::config::LayerConfig;
use crate::llm::kv_cache::KVCache;
use crate::llm::ops::Kernels;
use crate::quants::quantized_tensor::GgmlQuantizedTensor;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernels::cpu::from_slice_unaligned_f32x8;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use wide::f32x8;

#[inline]
fn dot_product(q: &[f32], q_offset: usize, k: &[f32], k_offset: usize, len: usize) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        let mut sum = 0.0f32;
        let mut i = 0;
        while i + 8 <= len {
            if q.len() < q_offset + i + 8 {
                eprintln!("ERROR in dot_product: q.len={}, q_offset={}, i={}, len={}", q.len(), q_offset, i, len);
                break;
            }
            if k.len() < k_offset + i + 8 {
                eprintln!("ERROR in dot_product: k.len={}, k_offset={}, i={}, len={}", k.len(), k_offset, i, len);
                break;
            }
            let q_vec = from_slice_unaligned_f32x8(&q[q_offset + i..]);
            let k_vec = from_slice_unaligned_f32x8(&k[k_offset + i..]);
            sum += (q_vec * k_vec).to_array().iter().sum::<f32>();
            i += 8;
        }
        while i < len {
            sum += q[q_offset + i] * k[k_offset + i];
            i += 1;
        }
        sum
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += q[q_offset + i] * k[k_offset + i];
        }
        sum
    }
}

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
    pub rope_freqs: Option<Vec<f32>>,
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
        rope_freqs: Option<Vec<f32>>,
    ) -> Self {
        let kv_cache = KVCache::new(&layer_config, 4096);
        AttentionLayer { q_weights, k_weights, v_weights, o_weights, q_norm, k_norm, layer_config, kv_cache, rope_freqs }
    }

    pub fn forward(
        &mut self,
        x: &mut [f32],
        pos: usize,
        shared_kv: Option<&KVCache>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let num_heads = self.layer_config.num_heads;
        let num_kv_heads = self.layer_config.num_kv_heads;
        let head_dim = self.layer_config.head_dim;
        let layer_idx = self.layer_config.layer_idx;

        let q_output_size = self.q_weights.shape()[0];

        llm_debug!("[Layer {}, pos {}] input hidden_state: len={}", layer_idx, pos, x.len());

        let mut q = vec![0.0; q_output_size];
        checked_gemv(&self.q_weights, x, &mut q)?;

        if let Some(ref q_norm) = self.q_norm {
            Kernels::apply_q_norm(&mut q, q_norm);
        }

        let rope_theta = self.layer_config.rope_theta;
        let freq_factors = if self.layer_config.head_dim > 256 {
            self.rope_freqs.as_deref()
        } else {
            None
        };
        for head in 0..num_heads {
            let offset = head * head_dim;
            Kernels::rope(&mut q[offset..offset + head_dim], rope_theta, pos, head_dim, freq_factors);
        }

        if self.layer_config.has_own_kv {
            let k_output_size = self.k_weights.shape()[0];
            let mut k = vec![0.0; k_output_size];
            let mut v = vec![0.0; k_output_size];
            checked_gemv(&self.k_weights, x, &mut k)?;
            checked_gemv(&self.v_weights, x, &mut v)?;

            if let Some(ref k_norm) = self.k_norm {
                Kernels::apply_q_norm(&mut k, k_norm);
            }
            Kernels::apply_v_norm(&mut v, head_dim);

            for head in 0..num_kv_heads {
                let offset = head * head_dim;
                Kernels::rope(&mut k[offset..offset + head_dim], rope_theta, pos, head_dim, freq_factors);
            }

            self.kv_cache.append(&k, &v, pos)?;
        }

        let kv_len;
        let kv_dim;

        let k_ref: &[f32];
        let v_ref: &[f32];

        if self.layer_config.has_own_kv {
            kv_len = self.kv_cache.current_pos();
            kv_dim = self.kv_cache.kv_dim();
            k_ref = &self.kv_cache.k_cache;
            v_ref = &self.kv_cache.v_cache;
        } else {
            let shared = shared_kv.ok_or_else(|| format!("Layer {} needs shared KV but none provided", layer_idx))?;
            kv_len = shared.current_pos();
            kv_dim = shared.kv_dim();
            k_ref = &shared.k_cache;
            v_ref = &shared.v_cache;
        }

        let gqa_ratio = num_heads / num_kv_heads;
        let mut attn_output = vec![0.0f32; q_output_size];

        let sliding_window = self.layer_config.sliding_window;
        let window_start = if pos >= sliding_window { pos - sliding_window + 1 } else { 0 };

        #[cfg(feature = "parallel")]
        {
            let q_slice: &[f32] = &q;
            let k_ref_slice: &[f32] = k_ref;
            let v_ref_slice: &[f32] = v_ref;
            let scale = 1.0f32;

            attn_output.par_chunks_mut(head_dim).enumerate().for_each(move |(qh, chunk)| {
                let kv_h = qh / gqa_ratio;
                let q_offset = qh * head_dim;
                let kv_offset = kv_h * head_dim;

                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; kv_len];

                for t in 0..kv_len {
                    if t < window_start {
                        scores[t] = f32::NEG_INFINITY;
                        continue;
                    }

                    let k_row_offset = t * kv_dim + kv_offset;
                    let score = dot_product(q_slice, q_offset, k_ref_slice, k_row_offset, head_dim);
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

                // V accumulation (SIMD on x86_64)
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let mut d = 0;
                    while d + 8 <= head_dim {
                        let mut vals = f32x8::splat(0.0f32);
                        for t in 0..kv_len {
                            let v_row_offset = t * kv_dim + kv_offset;
                            let slice = &v_ref_slice[v_row_offset + d..];
                            if slice.len() < 8 {
                                eprintln!("ERROR: v_row_offset={}, d={}, head_dim={}, kv_dim={}, kv_len={}, t={}, slice.len={}, v_ref_slice.len={}",
                                    v_row_offset, d, head_dim, kv_dim, kv_len, t, slice.len(), v_ref_slice.len());
                                // Fall back to scalar
                                let mut val = 0.0f32;
                                for t2 in 0..kv_len {
                                    let v_row_offset2 = t2 * kv_dim + kv_offset;
                                    val += scores[t2] * v_ref_slice[v_row_offset2 + d];
                                }
                                chunk[d] = val;
                                d += 1;
                                continue;
                            }
                            let v_vec = from_slice_unaligned_f32x8(slice);
                            vals = vals + v_vec * f32x8::splat(scores[t]);
                        }
                        chunk[d..d+8].copy_from_slice(&vals.to_array());
                        d += 8;
                    }
                    while d < head_dim {
                        let mut val = 0.0f32;
                        for t in 0..kv_len {
                            let v_row_offset = t * kv_dim + kv_offset;
                            val += scores[t] * v_ref_slice[v_row_offset + d];
                        }
                        chunk[d] = val;
                        d += 1;
                    }
                }
                #[cfg(any(not(feature = "simd"), not(target_arch = "x86_64")))]
                {
                    for d in 0..head_dim {
                        let mut val = 0.0f32;
                        for t in 0..kv_len {
                            let v_row_offset = t * kv_dim + kv_offset;
                            val += scores[t] * v_ref_slice[v_row_offset + d];
                        }
                        chunk[d] = val;
                    }
                }
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for qh in 0..num_heads {
                let kv_h = qh / gqa_ratio;
                let q_offset = qh * head_dim;
                let kv_offset = kv_h * head_dim;

                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; kv_len];

                let scale = 1.0f32;

                for t in 0..kv_len {
                    if t < window_start {
                        scores[t] = f32::NEG_INFINITY;
                        continue;
                    }

                    let k_row_offset = t * kv_dim + kv_offset;
                    let score = dot_product(&q, q_offset, k_ref, k_row_offset, head_dim);
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

                // V accumulation (SIMD on x86_64)
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    let mut d = 0;
                    while d + 8 <= head_dim {
                        let mut vals = f32x8::splat(0.0f32);
                        for t in 0..kv_len {
                            let v_row_offset = t * kv_dim + kv_offset;
                            let slice = &v_ref[v_row_offset + d..];
                            if slice.len() < 8 {
                                eprintln!("ERROR: v_row_offset={}, d={}, head_dim={}, kv_dim={}, kv_len={}, t={}, slice.len={}, v_ref.len={}",
                                    v_row_offset, d, head_dim, kv_dim, kv_len, t, slice.len(), v_ref.len());
                                // Fall back to scalar
                                let mut val = 0.0f32;
                                for t2 in 0..kv_len {
                                    let v_row_offset2 = t2 * kv_dim + kv_offset;
                                    val += scores[t2] * v_ref[v_row_offset2 + d];
                                }
                                attn_output[q_offset + d] = val;
                                d += 1;
                                continue;
                            }
                            let v_vec = from_slice_unaligned_f32x8(slice);
                            vals = vals + v_vec * f32x8::splat(scores[t]);
                        }
                        attn_output[q_offset + d..q_offset + d + 8].copy_from_slice(&vals.to_array());
                        d += 8;
                    }
                    while d < head_dim {
                        let mut val = 0.0f32;
                        for t in 0..kv_len {
                            let v_row_offset = t * kv_dim + kv_offset;
                            val += scores[t] * v_ref[v_row_offset + d];
                        }
                        attn_output[q_offset + d] = val;
                        d += 1;
                    }
                }
                #[cfg(any(not(feature = "simd"), not(target_arch = "x86_64")))]
                {
                    for d in 0..head_dim {
                        let mut val = 0.0f32;
                        for t in 0..kv_len {
                            let v_row_offset = t * kv_dim + kv_offset;
                            val += scores[t] * v_ref[v_row_offset + d];
                        }
                        attn_output[q_offset + d] += val;
                    }
                }
            }
        }

        checked_gemv(&self.o_weights, &attn_output, x)?;

        llm_debug!("[Layer {}, pos {}] output hidden_state: len={}", layer_idx, pos, x.len());

        Ok(())
    }
}
