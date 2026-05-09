use crate::dtypes::PackedWord;
use crate::nn::linear::Linear;
use crate::nn::Module;
use crate::packed_tensor::PackedTensor;
use crate::tensor::Tensor;
use crate::{impl_training_state, nn::TrainingState};

/// Quantized Multi-Head Attention with block-wise KV cache.
/// Uses packed precision for weights and activations to reduce memory bandwidth.
#[allow(dead_code)]
pub struct PackedMultiHeadAttention<T: PackedWord> {
    /// Packed QKV projection [d_model, d_model * 3]
    pub qkv_proj: PackedTensor<T>,
    /// Packed output projection [d_model, d_model]
    pub out_proj: PackedTensor<T>,
    pub num_heads: i64,
    pub head_dim: i64,
    pub d_model: i64,
    pub dropout_p: f32,
    pub causal: bool,
    training: TrainingState,
    /// Scale factor for attention scores
    scale: f32,
    // TODO: Implement KV cache for autoregressive decoding
    kv_cache: Option<(PackedTensor<T>, PackedTensor<T>)>,
}

impl<T: PackedWord> PackedMultiHeadAttention<T> {
    #[allow(dead_code)]
    /// Create a new quantized multi-head attention layer.
    pub fn new(d_model: i64, num_heads: i64, dropout_p: f32, causal: bool) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );

        let head_dim = d_model / num_heads;

        // Initialize with random weights (will be replaced with actual weights)
        let d_model_usize = d_model as usize;
        let qkv_data: Vec<f32> = (0..d_model_usize * d_model_usize * 3)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();
        let out_data: Vec<f32> = (0..d_model_usize * d_model_usize)
            .map(|i| (i as f32 * 0.01).cos() * 0.1)
            .collect();

        let qkv_proj =
            PackedTensor::<T>::from_f32_auto(&qkv_data, &[d_model_usize, d_model_usize * 3]);
        let out_proj = PackedTensor::<T>::from_f32_auto(&out_data, &[d_model_usize, d_model_usize]);

        PackedMultiHeadAttention {
            qkv_proj,
            out_proj,
            num_heads,
            head_dim,
            d_model,
            dropout_p,
            causal,
            training: TrainingState::new(),
            scale: 1.0 / (head_dim as f32).sqrt(),
            kv_cache: None,
        }
    }

    /// Forward pass with quantized inputs.
    /// Input should be a 3D tensor [batch, seq_len, d_model].
    pub fn forward_impl(&self, x: &Tensor) -> Tensor {
        let shape = x.shape_ref();
        assert_eq!(shape.len(), 3, "Input must be 3D [batch, seq_len, d_model]");
        let batch = shape[0] as usize;
        let seq_len = shape[1] as usize;
        let d_model = shape[2] as usize;

        // Convert to f32 for processing
        let x_data = x.to_numpy();

        // QKV projection: [batch, seq_len, d_model] @ [d_model*3, d_model]^T -> [batch, seq_len, d_model * 3]
        let qkv = self.quantized_matmul(&x_data, &self.qkv_proj, batch * seq_len, d_model, d_model * 3);

        // Split Q, K, V
        let q: Vec<f32> = qkv
            .iter()
            .take(batch * seq_len * d_model)
            .cloned()
            .collect();
        let k: Vec<f32> = qkv
            .iter()
            .skip(batch * seq_len * d_model)
            .take(batch * seq_len * d_model)
            .cloned()
            .collect();
        let v: Vec<f32> = qkv
            .iter()
            .skip(batch * seq_len * d_model * 2)
            .take(batch * seq_len * d_model)
            .cloned()
            .collect();

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = self.reshape_heads(&q, batch, seq_len);
        let k = self.reshape_heads(&k, batch, seq_len);
        let v = self.reshape_heads(&v, batch, seq_len);

        // Attention scores: Q @ K^T
        let attn_scores = self.compute_attention_scores(&q, &k, batch, seq_len);

        // Apply causal mask if enabled
        let attn_scores = if self.causal {
            self.apply_causal_mask(attn_scores, batch, seq_len)
        } else {
            attn_scores
        };

        // Softmax
        let attn_weights = self.softmax(&attn_scores, batch, seq_len);

        // Apply attention to values: attn_weights @ V
        let context = self.apply_attention(&attn_weights, &v, batch, seq_len);

        // Reshape back to [batch, seq_len, d_model]
        let context = self.reshape_from_heads(&context, batch, seq_len);

        // Output projection: [batch, seq_len, d_model] @ [d_model, d_model]^T -> [batch, seq_len, d_model]
        let output = self.quantized_matmul(&context, &self.out_proj, batch * seq_len, d_model, d_model);

        Tensor::from_vec(output, vec![batch as i64, seq_len as i64, d_model as i64])
    }

    /// Quantized matrix multiplication using packed weights.
    /// input: [m, k], weight: [k, n], output: [m, n]
    fn quantized_matmul(
        &self,
        input: &[f32],
        weight: &PackedTensor<T>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        debug_assert_eq!(weight.shape()[0], k);
        debug_assert_eq!(weight.shape()[1], n);
        let mut output = vec![0.0f32; m * n];
        for i in 0..m {
            let row_start = i * k;
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let w_idx = l * n + j;
                    let w = weight.get(w_idx);
                    sum += input[row_start + l] * w;
                }
                output[i * n + j] = sum;
            }
        }
        output
    }

    /// Reshape tensor to [batch, num_heads, seq_len, head_dim]
    fn reshape_heads(&self, x: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let d_model = self.d_model as usize;
        let num_heads = self.num_heads as usize;
        let head_dim = self.head_dim as usize;

        let mut reshaped = vec![0.0f32; batch * num_heads * seq_len * head_dim];

        for b in 0..batch {
            for s in 0..seq_len {
                for d in 0..d_model {
                    let src_idx = b * seq_len * d_model + s * d_model + d;
                    let head = d / head_dim;
                    let h_dim = d % head_dim;
                    let dst_idx = b * num_heads * seq_len * head_dim
                        + head * seq_len * head_dim
                        + s * head_dim
                        + h_dim;
                    reshaped[dst_idx] = x[src_idx];
                }
            }
        }

        reshaped
    }

    /// Compute attention scores: Q @ K^T
    fn compute_attention_scores(
        &self,
        q: &[f32],
        k: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let num_heads = self.num_heads as usize;
        let head_dim = self.head_dim as usize;

        let mut scores = vec![0.0f32; batch * num_heads * seq_len * seq_len];

        for b in 0..batch {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut sum = 0.0f32;
                        for d in 0..head_dim {
                            let q_idx = b * num_heads * seq_len * head_dim
                                + h * seq_len * head_dim
                                + i * head_dim
                                + d;
                            let k_idx = b * num_heads * seq_len * head_dim
                                + h * seq_len * head_dim
                                + j * head_dim
                                + d;
                            sum += q[q_idx] * k[k_idx];
                        }
                        scores[b * num_heads * seq_len * seq_len
                            + h * seq_len * seq_len
                            + i * seq_len
                            + j] = sum * self.scale;
                    }
                }
            }
        }

        scores
    }

    /// Apply causal mask (upper triangular with -inf)
    fn apply_causal_mask(&self, scores: Vec<f32>, batch: usize, seq_len: usize) -> Vec<f32> {
        let num_heads = self.num_heads as usize;
        let mut masked = scores;

        for b in 0..batch {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        let idx = b * num_heads * seq_len * seq_len
                            + h * seq_len * seq_len
                            + i * seq_len
                            + j;
                        masked[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        masked
    }

    /// Softmax along the last dimension
    fn softmax(&self, x: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let num_heads = self.num_heads as usize;
        let mut output = vec![0.0f32; x.len()];

        for b in 0..batch {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    let start =
                        b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len;
                    let end = start + seq_len;
                    let row = &x[start..end];

                    // Find max for numerical stability
                    let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                    // Compute exp and sum
                    let mut sum = 0.0f32;
                    for j in 0..seq_len {
                        let exp_val = (row[j] - max_val).exp();
                        output[start + j] = exp_val;
                        sum += exp_val;
                    }

                    // Normalize
                    for j in 0..seq_len {
                        output[start + j] /= sum;
                    }
                }
            }
        }

        output
    }

    /// Apply attention weights to values
    fn apply_attention(&self, attn: &[f32], v: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let num_heads = self.num_heads as usize;
        let head_dim = self.head_dim as usize;

        let mut context = vec![0.0f32; batch * num_heads * seq_len * head_dim];

        for b in 0..batch {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            let attn_idx = b * num_heads * seq_len * seq_len
                                + h * seq_len * seq_len
                                + i * seq_len
                                + j;
                            let v_idx = b * num_heads * seq_len * head_dim
                                + h * seq_len * head_dim
                                + j * head_dim
                                + d;
                            sum += attn[attn_idx] * v[v_idx];
                        }
                        let ctx_idx = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim
                            + i * head_dim
                            + d;
                        context[ctx_idx] = sum;
                    }
                }
            }
        }

        context
    }

    /// Reshape from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, d_model]
    fn reshape_from_heads(&self, x: &[f32], batch: usize, seq_len: usize) -> Vec<f32> {
        let num_heads = self.num_heads as usize;
        let head_dim = self.head_dim as usize;
        let d_model = self.d_model as usize;

        let mut reshaped = vec![0.0f32; batch * seq_len * d_model];

        for b in 0..batch {
            for s in 0..seq_len {
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        let src_idx = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim
                            + s * head_dim
                            + d;
                        let dst_idx = b * seq_len * d_model + s * d_model + h * head_dim + d;
                        reshaped[dst_idx] = x[src_idx];
                    }
                }
            }
        }

        reshaped
    }
}

impl<T: PackedWord> Module for PackedMultiHeadAttention<T> {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_impl(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {}

    impl_training_state!(self, self.training);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F16x2, F32x1, U4x8, U8x4};

    #[test]
    fn test_packed_attention_creation() {
        let attn_f32 = PackedMultiHeadAttention::<F32x1>::new(64, 8, 0.1, false);
        assert_eq!(attn_f32.d_model, 64);
        assert_eq!(attn_f32.num_heads, 8);

        let attn_u8 = PackedMultiHeadAttention::<U8x4>::new(64, 8, 0.1, false);
        assert_eq!(attn_u8.d_model, 64);
        assert_eq!(attn_u8.num_heads, 8);
    }

    #[test]
    fn test_packed_attention_forward() {
        let attn = PackedMultiHeadAttention::<F32x1>::new(64, 8, 0.1, false);
        let batch = 2;
        let seq_len = 16;
        let d_model = 64;

        let input_data: Vec<f32> = (0..batch * seq_len * d_model)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let input = Tensor::from_vec(
            input_data,
            vec![batch as i64, seq_len as i64, d_model as i64],
        );

        let output = attn.forward(&input);
        assert_eq!(
            output.shape_ref(),
            vec![batch as i64, seq_len as i64, d_model as i64]
        );

        // Check that output is finite
        let output_data = output.to_numpy();
        for &val in &output_data {
            assert!(val.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_reshape_heads() {
        let attn = PackedMultiHeadAttention::<F32x1>::new(64, 8, 0.1, false);
        let batch = 2;
        let seq_len = 16;
        let d_model = 64;

        let input: Vec<f32> = (0..batch * seq_len * d_model).map(|i| i as f32).collect();
        let reshaped = attn.reshape_heads(&input, batch, seq_len);

        assert_eq!(reshaped.len(), batch * 8 * seq_len * 8); // batch * num_heads * seq_len * head_dim

        // Check that reshaping is invertible
        let back = attn.reshape_from_heads(&reshaped, batch, seq_len);
        for i in 0..input.len() {
            assert!((input[i] - back[i]).abs() < 1e-4);
        }
    }
}

pub struct MultiHeadAttention {
    pub q_proj: Option<Linear>,
    pub k_proj: Option<Linear>,
    pub v_proj: Option<Linear>,
    pub out_proj: Linear,
    pub num_heads: i64,
    pub head_dim: i64,
    pub d_model: i64,
    #[allow(dead_code)]
    pub dropout_p: f32,
    pub causal: bool,
    training: TrainingState,
    // Fused projection for better memory locality
    pub qkv_proj: Option<Linear>,
    // Pre-allocated scale tensor to avoid per-forward allocation
    scale: f32,
}

impl MultiHeadAttention {
    #[allow(dead_code)]
    pub fn new(d_model: i64, num_heads: i64, dropout_p: f32) -> Self {
        Self::new_fused(d_model, num_heads, dropout_p, false)
    }

    #[allow(dead_code)]
    pub fn new_unfused(d_model: i64, num_heads: i64, dropout_p: f32, causal: bool) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );

        let head_dim = d_model / num_heads;

        let q_proj = Linear::new(d_model, d_model, false);
        let k_proj = Linear::new(d_model, d_model, false);
        let v_proj = Linear::new(d_model, d_model, false);
        let out_proj = Linear::new(d_model, d_model, true);

        MultiHeadAttention {
            q_proj: Some(q_proj),
            k_proj: Some(k_proj),
            v_proj: Some(v_proj),
            out_proj,
            num_heads,
            head_dim,
            d_model,
            dropout_p,
            causal,
            training: TrainingState::new(),
            qkv_proj: None,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn new_fused(d_model: i64, num_heads: i64, dropout_p: f32, causal: bool) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model must be divisible by num_heads"
        );

        let head_dim = d_model / num_heads;

        // Fused QKV projection for better memory locality
        let qkv_proj = Linear::new(d_model, d_model * 3, false);
        let out_proj = Linear::new(d_model, d_model, true);

        // No individual projections needed in fused mode
        // This saves 3x memory by not allocating unused weight matrices

        MultiHeadAttention {
            q_proj: None,
            k_proj: None,
            v_proj: None,
            out_proj,
            num_heads,
            head_dim,
            d_model,
            dropout_p,
            causal,
            training: TrainingState::new(),
            qkv_proj: Some(qkv_proj),
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn forward_impl(&self, x: &Tensor) -> Tensor {
        if x.ndim() < 3 {
            panic!(
                "MultiHeadAttention expected input with at least 3 dimensions (batch, seq_len, d_model), got shape {:?}",
                x.shape_ref()
            );
        }
        let batch = x.shape_ref()[0];
        let seq_len = x.shape_ref()[1];

        // Use fused QKV projection if available (better memory locality)
        let (q, k, v) = if let Some(ref qkv_proj) = self.qkv_proj {
            let qkv = qkv_proj.forward(x);
            // Assert proper QKV split dimensions
            let expected_qkv_size = self.d_model * 3;
            let actual_qkv_size = qkv.shape_ref()[2];
            assert_eq!(
                actual_qkv_size, expected_qkv_size,
                "Fused QKV projection output size mismatch: expected {} but got {}",
                expected_qkv_size, actual_qkv_size
            );
            // Reshape to [batch, seq_len, 3, num_heads, head_dim]
            let qkv = qkv.reshape(vec![batch, seq_len, 3, self.num_heads, self.head_dim]);

            // Split into Q, K, V without copying
            // Slice and permute directly to [batch, num_heads, seq_len, head_dim]
            let q = qkv
                .slice(2, 0, 1, 1)
                .squeeze(Some(2))
                .permute(vec![0, 2, 1, 3]); // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
            let k = qkv
                .slice(2, 1, 2, 1)
                .squeeze(Some(2))
                .permute(vec![0, 2, 1, 3]); // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
            let v = qkv
                .slice(2, 2, 3, 1)
                .squeeze(Some(2))
                .permute(vec![0, 2, 1, 3]); // [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]

            (q, k, v)
        } else {
            // Fallback to separate projections
            let q = self.q_proj.as_ref().unwrap().forward(x);
            let k = self.k_proj.as_ref().unwrap().forward(x);
            let v = self.v_proj.as_ref().unwrap().forward(x);

            // Reshape and transpose to [batch, num_heads, seq_len, head_dim]
            let q = q.reshape_permute(
                vec![batch, seq_len, self.num_heads, self.head_dim],
                vec![0, 2, 1, 3],
            );
            let k = k.reshape_permute(
                vec![batch, seq_len, self.num_heads, self.head_dim],
                vec![0, 2, 1, 3],
            );
            let v = v.reshape_permute(
                vec![batch, seq_len, self.num_heads, self.head_dim],
                vec![0, 2, 1, 3],
            );

            (q, k, v)
        };

        // 3. Compute attention scores with scale factor
        // Q @ K^T: [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        // Only make contiguous right before matmul
        let q = if q.is_contiguous() { q } else { q.contiguous() };
        let k_t = k.permute(vec![0, 1, 3, 2]);
        let k_t = if k_t.is_contiguous() {
            k_t
        } else {
            k_t.contiguous()
        };
        let mut attn_scores = q.matmul(&k_t);
        attn_scores = attn_scores.mul_scalar(self.scale);

        // 4. Apply causal mask if enabled
        if self.causal {
            // Create upper triangular mask with -inf above the diagonal
            let mut mask_data = vec![0.0f32; (seq_len * seq_len) as usize];
            for i in 0..seq_len as usize {
                for j in (i + 1)..seq_len as usize {
                    mask_data[i * seq_len as usize + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::from_vec(mask_data, vec![seq_len, seq_len]);
            attn_scores = attn_scores.add(&mask);
        }

        // 5. Apply softmax along the last dimension (seq_len)
        let attn_weights = attn_scores.softmax(3);

        // 5. Apply attention to values
        // attn_weights @ V: [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        let v = if v.is_contiguous() { v } else { v.contiguous() };
        let context = attn_weights.matmul(&v);

        // 6. Reshape back to [batch, seq_len, d_model]
        // Fused permute+reshape to avoid separate intermediate
        let context = context.reshape_permute(
            vec![batch, self.num_heads, seq_len, self.head_dim],
            vec![0, 2, 1, 3],
        );
        let context = context.reshape(vec![batch, seq_len, self.d_model]);

        // 7. Output projection
        self.out_proj.forward(&context)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_impl(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        if let Some(ref qkv_proj) = self.qkv_proj {
            params.extend(qkv_proj.parameters());
        } else {
            if let Some(ref q_proj) = self.q_proj {
                params.extend(q_proj.parameters());
            }
            if let Some(ref k_proj) = self.k_proj {
                params.extend(k_proj.parameters());
            }
            if let Some(ref v_proj) = self.v_proj {
                params.extend(v_proj.parameters());
            }
        }
        params.extend(self.out_proj.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = Vec::new();
        if let Some(ref qkv_proj) = self.qkv_proj {
            for (name, t) in qkv_proj.named_parameters() {
                params.push((format!("qkv_proj.{}", name), t));
            }
        } else {
            if let Some(ref q_proj) = self.q_proj {
                for (name, t) in q_proj.named_parameters() {
                    params.push((format!("q_proj.{}", name), t));
                }
            }
            if let Some(ref k_proj) = self.k_proj {
                for (name, t) in k_proj.named_parameters() {
                    params.push((format!("k_proj.{}", name), t));
                }
            }
            if let Some(ref v_proj) = self.v_proj {
                for (name, t) in v_proj.named_parameters() {
                    params.push((format!("v_proj.{}", name), t));
                }
            }
        }
        for (name, t) in self.out_proj.named_parameters() {
            params.push((format!("out_proj.{}", name), t));
        }
        params
    }

    fn zero_grad(&self) {
        if let Some(ref q_proj) = self.q_proj {
            q_proj.zero_grad();
        }
        if let Some(ref k_proj) = self.k_proj {
            k_proj.zero_grad();
        }
        if let Some(ref v_proj) = self.v_proj {
            v_proj.zero_grad();
        }
        if let Some(ref qkv_proj) = self.qkv_proj {
            qkv_proj.zero_grad();
        }
        self.out_proj.zero_grad();
    }

    impl_training_state!(self, self.training);
}
