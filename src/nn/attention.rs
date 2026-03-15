use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::linear::Linear;
use crate::nn::Module;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: i64,
    pub head_dim: i64,
    pub d_model: i64,
    #[allow(dead_code)]
    pub dropout_p: f32,
    training: AtomicBool,
}

impl MultiHeadAttention {
    pub fn new(d_model: i64, num_heads: i64, dropout_p: f32) -> Self {
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
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            d_model,
            dropout_p,
            training: AtomicBool::new(true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        // 1. Project to Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // 2. Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape(vec![batch, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape(vec![batch, seq_len, self.num_heads, self.head_dim]);
        let v = v.reshape(vec![batch, seq_len, self.num_heads, self.head_dim]);

        // 3. Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.permute(vec![0, 2, 1, 3]);
        let k = k.permute(vec![0, 2, 1, 3]);
        let v = v.permute(vec![0, 2, 1, 3]);

        // Make contiguous after permute
        let q = q.contiguous();
        let k = k.contiguous();
        let v = v.contiguous();

        // 4. Reshape for batched matmul
        let batch_heads = batch * self.num_heads;
        let q = q.reshape(vec![batch_heads, seq_len, self.head_dim]);
        let k = k.reshape(vec![batch_heads, self.head_dim, seq_len]); // Transpose last two dims
        let v = v.reshape(vec![batch_heads, seq_len, self.head_dim]);

        // 5. Compute attention scores
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Q @ K^T
        let mut attn_scores = q.matmul(&k);
        attn_scores = attn_scores.mul(&Tensor::from_scalar(scale));

        // 6. Apply softmax
        let attn_weights = attn_scores.softmax(2); // axis=2

        // 7. Apply attention to values
        let context = attn_weights.matmul(&v);

        // 8. Reshape back to [batch, num_heads, seq_len, head_dim]
        let context = context.reshape(vec![batch, self.num_heads, seq_len, self.head_dim]);

        // 9. Transpose to [batch, seq_len, num_heads, head_dim]
        let context = context.permute(vec![0, 2, 1, 3]);
        let context = context.contiguous();

        // 10. Reshape to [batch, seq_len, d_model]
        let context = context.reshape(vec![batch, seq_len, self.d_model]);

        // 11. Output projection
        self.out_proj.forward(&context)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = Vec::new();
        for (name, t) in self.q_proj.named_parameters() {
            params.push((format!("q_proj.{}", name), t));
        }
        for (name, t) in self.k_proj.named_parameters() {
            params.push((format!("k_proj.{}", name), t));
        }
        for (name, t) in self.v_proj.named_parameters() {
            params.push((format!("v_proj.{}", name), t));
        }
        for (name, t) in self.out_proj.named_parameters() {
            params.push((format!("out_proj.{}", name), t));
        }
        params
    }

    fn zero_grad(&self) {
        self.q_proj.zero_grad();
        self.k_proj.zero_grad();
        self.v_proj.zero_grad();
        self.out_proj.zero_grad();
    }

    fn train_mode(&self) {
        self.training.store(true, Ordering::SeqCst);
        self.q_proj.train_mode();
        self.k_proj.train_mode();
        self.v_proj.train_mode();
        self.out_proj.train_mode();
    }

    fn eval_mode(&self) {
        self.training.store(false, Ordering::SeqCst);
        self.q_proj.eval_mode();
        self.k_proj.eval_mode();
        self.v_proj.eval_mode();
        self.out_proj.eval_mode();
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::SeqCst)
    }
}
