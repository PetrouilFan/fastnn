use crate::nn::attention::MultiHeadAttention;
use crate::nn::dropout::Dropout;
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::norm::LayerNorm;
use crate::nn::Module;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

fn pos_encoding_cache() -> &'static Mutex<HashMap<(i64, i64, i64), Tensor>> {
    static CACHE: OnceLock<Mutex<HashMap<(i64, i64, i64), Tensor>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct TransformerBlock {
    pub self_attn: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    #[allow(dead_code)]
    pub dropout: Dropout,
    #[allow(dead_code)]
    pub d_model: i64,
    #[allow(dead_code)]
    pub ff_dim: i64,
    training: AtomicBool,
}

impl TransformerBlock {
    pub fn new(d_model: i64, num_heads: i64, ff_dim: i64, dropout_p: f32) -> Self {
        Self::new_with_config(d_model, num_heads, ff_dim, dropout_p, true)
    }

    pub fn new_with_config(
        d_model: i64,
        num_heads: i64,
        ff_dim: i64,
        dropout_p: f32,
        use_fused_attention: bool,
    ) -> Self {
        // Use standard attention for small models, fused for larger ones
        // Fused attention has overhead for small d_model but benefits larger models
        let use_fused = use_fused_attention && d_model >= 128;
        let self_attn = if use_fused {
            MultiHeadAttention::new_fused(d_model, num_heads, dropout_p, false)
        } else {
            MultiHeadAttention::new_unfused(d_model, num_heads, dropout_p, false)
        };

        let norm1 = LayerNorm::new(d_model, 1e-5);
        let norm2 = LayerNorm::new(d_model, 1e-5);
        let ff1 = Linear::new(d_model, ff_dim, true);
        let ff2 = Linear::new(ff_dim, d_model, true);
        let dropout = Dropout::new(dropout_p as f64);

        TransformerBlock {
            self_attn,
            norm1,
            norm2,
            ff1,
            ff2,
            dropout,
            d_model,
            ff_dim,
            training: AtomicBool::new(true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Layer 1: Self-attention with residual connection and dropout
        let x_norm1 = self.norm1.forward(x);
        let attn_output = self.self_attn.forward(&x_norm1);
        let attn_dropped = self.dropout.forward(&attn_output);
        let x = attn_dropped.add(x);

        // Layer 2: Feed-forward with residual connection and dropout
        let x_norm2 = self.norm2.forward(&x);

        // Fused feed-forward: linear -> gelu -> linear
        let ff_hidden = self.ff1.forward(&x_norm2);
        let ff_gelu = ff_hidden.gelu();
        let ff_out = self.ff2.forward(&ff_gelu);
        let ff_dropped = self.dropout.forward(&ff_out);

        ff_dropped.add(&x)
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        TransformerBlock::forward(self, x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = Vec::new();
        for (name, t) in self.self_attn.named_parameters() {
            params.push((format!("self_attn.{}", name), t));
        }
        for (name, t) in self.norm1.named_parameters() {
            params.push((format!("norm1.{}", name), t));
        }
        for (name, t) in self.norm2.named_parameters() {
            params.push((format!("norm2.{}", name), t));
        }
        for (name, t) in self.ff1.named_parameters() {
            params.push((format!("ff1.{}", name), t));
        }
        for (name, t) in self.ff2.named_parameters() {
            params.push((format!("ff2.{}", name), t));
        }
        params
    }

    fn zero_grad(&self) {
        self.self_attn.zero_grad();
        self.norm1.zero_grad();
        self.norm2.zero_grad();
        self.ff1.zero_grad();
        self.ff2.zero_grad();
    }

    fn train_mode(&self) {
        self.training.store(true, Ordering::Relaxed);
        self.self_attn.train_mode();
        self.norm1.train_mode();
        self.norm2.train_mode();
        self.ff1.train_mode();
        self.ff2.train_mode();
    }

    fn eval_mode(&self) {
        self.training.store(false, Ordering::Relaxed);
        self.self_attn.eval_mode();
        self.norm1.eval_mode();
        self.norm2.eval_mode();
        self.ff1.eval_mode();
        self.ff2.eval_mode();
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }
}

pub struct TransformerEncoder {
    pub embedding: Embedding,
    pub pos_embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub classifier: Linear,
    #[allow(dead_code)]
    pub d_model: i64,
    #[allow(dead_code)]
    pub max_seq_len: i64,
    #[allow(dead_code)]
    pub num_classes: i64,
    training: AtomicBool,
}

impl TransformerEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: i64,
        max_seq_len: i64,
        d_model: i64,
        num_heads: i64,
        num_layers: i64,
        ff_dim: i64,
        num_classes: i64,
        dropout_p: f32,
    ) -> Self {
        let embedding = Embedding::new(vocab_size, d_model);
        let pos_embedding = Embedding::new(max_seq_len, d_model);

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TransformerBlock::new(d_model, num_heads, ff_dim, dropout_p));
        }

        let norm = LayerNorm::new(d_model, 1e-5);
        let classifier = Linear::new(d_model, num_classes, true);

        TransformerEncoder {
            embedding,
            pos_embedding,
            layers,
            norm,
            classifier,
            d_model,
            max_seq_len,
            num_classes,
            training: AtomicBool::new(true),
        }
    }

    pub fn forward(&self, token_ids: &Tensor) -> Tensor {
        let shape = token_ids.shape();
        if shape.len() < 2 {
            panic!(
                "TransformerEncoder expected input with at least 2 dimensions (batch, seq_len), got shape {:?}",
                shape
            );
        }
        let batch = shape[0];
        let seq_len = shape[1];

        if seq_len > self.max_seq_len {
            panic!(
                "Sequence length {} exceeds maximum sequence length {}",
                seq_len, self.max_seq_len
            );
        }
        if seq_len == 0 {
            panic!("Sequence length cannot be 0");
        }

        let x = self.embedding.forward(token_ids);

        // Cached positional encoding - reuse for same (batch, seq_len, d_model)
        let cache_key = (batch, seq_len, self.pos_embedding.weight.shape()[1]);
        let positions_expanded = {
            let mut cache = pos_encoding_cache().lock().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                cached.clone()
            } else {
                let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
                let mut repeated_positions = Vec::with_capacity(batch as usize * seq_len as usize);
                for _ in 0..batch {
                    repeated_positions.extend_from_slice(&positions);
                }
                let t = Tensor::from_vec(repeated_positions, vec![batch, seq_len])
                    .requires_grad_(false);
                cache.insert(cache_key, t.clone());
                t
            }
        };

        let pos_emb = self.pos_embedding.forward(&positions_expanded);
        let x = x.add(&pos_emb);

        // Process layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        x = self.norm.forward(&x);

        // Extract CLS token (first token of sequence)
        let cls_token = x.slice(1, 0, 1, 1);
        let cls_token = cls_token.squeeze(Some(1));

        self.classifier.forward(&cls_token)
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, x: &Tensor) -> Tensor {
        TransformerEncoder::forward(self, x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters());
        params.extend(self.pos_embedding.parameters());
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.norm.parameters());
        params.extend(self.classifier.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = Vec::new();
        for (name, t) in self.embedding.named_parameters() {
            params.push((format!("embedding.{}", name), t));
        }
        for (name, t) in self.pos_embedding.named_parameters() {
            params.push((format!("pos_embedding.{}", name), t));
        }
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, t) in layer.named_parameters() {
                params.push((format!("layers.{}.{}", i, name), t));
            }
        }
        for (name, t) in self.norm.named_parameters() {
            params.push((format!("norm.{}", name), t));
        }
        for (name, t) in self.classifier.named_parameters() {
            params.push((format!("classifier.{}", name), t));
        }
        params
    }

    fn zero_grad(&self) {
        self.embedding.zero_grad();
        self.pos_embedding.zero_grad();
        for layer in &self.layers {
            layer.zero_grad();
        }
        self.norm.zero_grad();
        self.classifier.zero_grad();
    }

    fn train_mode(&self) {
        self.training.store(true, Ordering::Relaxed);
        self.embedding.train_mode();
        self.pos_embedding.train_mode();
        for layer in &self.layers {
            layer.train_mode();
        }
        self.norm.train_mode();
        self.classifier.train_mode();
    }

    fn eval_mode(&self) {
        self.training.store(false, Ordering::Relaxed);
        self.embedding.eval_mode();
        self.pos_embedding.eval_mode();
        for layer in &self.layers {
            layer.eval_mode();
        }
        self.norm.eval_mode();
        self.classifier.eval_mode();
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }
}
