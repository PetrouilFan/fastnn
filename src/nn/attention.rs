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
        if x.ndim() < 3 {
            panic!(
                "MultiHeadAttention expected input with at least 3 dimensions (batch, seq_len, d_model), got shape {:?}",
                x.shape()
            );
        }
        let batch = x.shape()[0];
        let seq_len = x.shape()[1];

        // 1. Project to Q, K, V - single combined projection for better memory locality
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // 2. Reshape and transpose in one step to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape(vec![batch, seq_len, self.num_heads, self.head_dim])
            .permute(vec![0, 2, 1, 3])
            .contiguous()
            .reshape(vec![batch * self.num_heads, seq_len, self.head_dim]);

        let k = k
            .reshape(vec![batch, seq_len, self.num_heads, self.head_dim])
            .permute(vec![0, 2, 1, 3])
            .contiguous()
            .reshape(vec![batch * self.num_heads, self.head_dim, seq_len]); // Transpose for K^T

        let v = v
            .reshape(vec![batch, seq_len, self.num_heads, self.head_dim])
            .permute(vec![0, 2, 1, 3])
            .contiguous()
            .reshape(vec![batch * self.num_heads, seq_len, self.head_dim]);

        // 3. Compute attention scores with scale factor
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_scores = q.matmul(&k);
        attn_scores = attn_scores.mul(&Tensor::from_scalar(scale));

        // 4. Apply softmax
        let attn_weights = attn_scores.softmax(2);

        // 5. Apply attention to values
        let context = attn_weights.matmul(&v);

        // 6. Reshape back to [batch, seq_len, d_model]
        let context = context
            .reshape(vec![batch, self.num_heads, seq_len, self.head_dim])
            .permute(vec![0, 2, 1, 3])
            .contiguous()
            .reshape(vec![batch, seq_len, self.d_model]);

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
