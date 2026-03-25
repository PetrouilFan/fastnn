use crate::nn::linear::Linear;
use crate::nn::Module;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

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
    training: AtomicBool,
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
            training: AtomicBool::new(true),
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
            training: AtomicBool::new(true),
            qkv_proj: Some(qkv_proj),
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if x.ndim() < 3 {
            panic!(
                "MultiHeadAttention expected input with at least 3 dimensions (batch, seq_len, d_model), got shape {:?}",
                x.shape()
            );
        }
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        // Use fused QKV projection if available (better memory locality)
        let (q, k, v) = if let Some(ref qkv_proj) = self.qkv_proj {
            let qkv = qkv_proj.forward(x);
            // Assert proper QKV split dimensions
            let expected_qkv_size = self.d_model * 3;
            let actual_qkv_size = qkv.shape()[2];
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
        self.forward(x)
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
        self.out_proj.zero_grad();
    }

    fn train_mode(&self) {
        self.training.store(true, Ordering::Relaxed);
        if let Some(ref q_proj) = self.q_proj {
            q_proj.train_mode();
        }
        if let Some(ref k_proj) = self.k_proj {
            k_proj.train_mode();
        }
        if let Some(ref v_proj) = self.v_proj {
            v_proj.train_mode();
        }
        self.out_proj.train_mode();
    }

    fn eval_mode(&self) {
        self.training.store(false, Ordering::Relaxed);
        if let Some(ref q_proj) = self.q_proj {
            q_proj.eval_mode();
        }
        if let Some(ref k_proj) = self.k_proj {
            k_proj.eval_mode();
        }
        if let Some(ref v_proj) = self.v_proj {
            v_proj.eval_mode();
        }
        self.out_proj.eval_mode();
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }
}
