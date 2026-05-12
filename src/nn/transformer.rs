use crate::dtypes::PackedWord;
use crate::nn::attention::{MultiHeadAttention, PackedMultiHeadAttention};
use crate::nn::dropout::Dropout;
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::norm::LayerNorm;
use crate::nn::Module;
use crate::packed_layer::PackedLinear;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

#[allow(dead_code)]
type PosCacheKey = (i64, i64, i64);

#[allow(dead_code)]
fn pos_encoding_cache() -> &'static Mutex<HashMap<PosCacheKey, Tensor>> {
    static CACHE: OnceLock<Mutex<HashMap<PosCacheKey, Tensor>>> = OnceLock::new();
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

        // Fused feed-forward: linear -> gelu
        let ff_gelu = x_norm2.fused_linear_gelu(&self.ff1.weight, self.ff1.bias.as_ref());
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
    // Precomputed position tensor for max_seq_len, sliced for actual seq_len
    pos_cache: OnceLock<Option<Tensor>>,
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

        // Precompute max-length position tensor for fast slice
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let pos_tensor = Tensor::from_vec(positions, vec![1, max_seq_len]).requires_grad_(false);

        let transformer = TransformerEncoder {
            embedding,
            pos_embedding,
            layers,
            norm,
            classifier,
            d_model,
            max_seq_len,
            num_classes,
            training: AtomicBool::new(true),
            pos_cache: OnceLock::new(),
        };
        let _ = transformer.pos_cache.set(Some(pos_tensor));
        transformer
    }

    pub fn forward(&self, token_ids: &Tensor) -> Tensor {
        let shape = token_ids.shape_ref();
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

        // Use precomputed position tensor, slice to actual seq_len
        let pos_tensor = self.pos_cache.get().unwrap().as_ref().unwrap();
        let pos_indices = pos_tensor.slice(0, 0, seq_len, 1);
        // Expand to [batch, seq_len] for batching, embedding handles the d_model conversion
        let pos_expanded = pos_indices.expand(vec![batch, seq_len]);
        let pos_emb = self.pos_embedding.forward(&pos_expanded);
        let x = x.add(&pos_emb);

        // Process layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        x = self.norm.forward(&x);

        // Extract CLS token (first token of sequence)
        let cls_token = x.slice(1, 0, 1, 1).squeeze(Some(1));
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

/// Quantized Transformer Block with packed precision weights.
/// Uses quantized linear layers and attention for reduced memory bandwidth.
#[allow(dead_code)]
pub struct PackedTransformerBlock<T: PackedWord> {
    /// Quantized self-attention
    pub self_attn: PackedMultiHeadAttention<T>,
    /// Layer normalization 1
    pub norm1: LayerNorm,
    /// Layer normalization 2
    pub norm2: LayerNorm,
    /// Quantized feed-forward layer 1
    pub ff1: PackedLinear<T>,
    /// Quantized feed-forward layer 2
    pub ff2: PackedLinear<T>,
    #[allow(dead_code)]
    pub dropout: Dropout,
    #[allow(dead_code)]
    pub d_model: i64,
    #[allow(dead_code)]
    pub ff_dim: i64,
    training: AtomicBool,
}

impl<T: PackedWord> PackedTransformerBlock<T> {
    #[allow(dead_code)]
    /// Create a new quantized transformer block.
    pub fn new(d_model: i64, num_heads: i64, ff_dim: i64, dropout_p: f32) -> Self {
        let self_attn =
            PackedMultiHeadAttention::<T>::new(d_model, num_heads, dropout_p, false, 2048);
        let norm1 = LayerNorm::new(d_model, 1e-5);
        let norm2 = LayerNorm::new(d_model, 1e-5);
        let ff1 = PackedLinear::<T>::new(d_model as usize, ff_dim as usize, true);
        let ff2 = PackedLinear::<T>::new(ff_dim as usize, d_model as usize, true);
        let dropout = Dropout::new(dropout_p as f64);

        PackedTransformerBlock {
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

    /// Forward pass with quantized operations.
    #[inline]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Layer 1: Self-attention with residual connection and dropout
        let x_norm1 = self.norm1.forward(x);
        let attn_output = self.self_attn.forward(&x_norm1);
        let attn_dropped = self.dropout.forward(&attn_output);
        let x = attn_dropped.add(x);

        // Layer 2: Feed-forward with residual connection and dropout
        let x_norm2 = self.norm2.forward(&x);

        // Convert to f32 for quantized linear layers
        let x_data = x_norm2.to_numpy();

        // Fused feed-forward: linear -> relu -> linear
        // ReLU is much cheaper than GELU (max vs tanh) and activations can stay packed
        let ff_hidden = self.ff1.forward(&x_data);
        let ff_relu: Vec<f32> = ff_hidden.iter().map(|&v| v.max(0.0)).collect();
        let ff_out = self.ff2.forward(&ff_relu);
        let ff_dropped = self.dropout.forward(&Tensor::from_vec(
            ff_out.clone(),
            vec![x.shape_ref()[0], x.shape_ref()[1], self.d_model],
        ));

        ff_dropped.add(&x)
    }
}

impl<T: PackedWord> Module for PackedTransformerBlock<T> {
    fn forward(&self, x: &Tensor) -> Tensor {
        PackedTransformerBlock::forward(self, x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Note: Packed layers don't have Tensor parameters in the traditional sense
        // They have packed weight representations
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {
        // Packed layers don't maintain gradients in the same way
    }

    fn train_mode(&self) {
        self.training.store(true, Ordering::Relaxed);
        self.self_attn.train_mode();
        self.norm1.train_mode();
        self.norm2.train_mode();
        self.dropout.train_mode();
    }

    fn eval_mode(&self) {
        self.training.store(false, Ordering::Relaxed);
        self.self_attn.eval_mode();
        self.norm1.eval_mode();
        self.norm2.eval_mode();
        self.dropout.eval_mode();
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }
}

/// Quantized Transformer Encoder with packed precision.
#[allow(dead_code)]
pub struct PackedTransformerEncoder<T: PackedWord> {
    /// Token embedding (kept in f32 for compatibility)
    pub embedding: Embedding,
    /// Positional embedding (kept in f32 for compatibility)
    pub pos_embedding: Embedding,
    /// Quantized transformer layers
    pub layers: Vec<PackedTransformerBlock<T>>,
    /// Final layer normalization
    pub norm: LayerNorm,
    /// Classifier head (quantized)
    pub classifier: PackedLinear<T>,
    #[allow(dead_code)]
    pub d_model: i64,
    #[allow(dead_code)]
    pub max_seq_len: i64,
    #[allow(dead_code)]
    pub num_classes: i64,
    training: AtomicBool,
}

impl<T: PackedWord> PackedTransformerEncoder<T> {
    #[allow(dead_code)]
    /// Create a new quantized transformer encoder.
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
            layers.push(PackedTransformerBlock::<T>::new(
                d_model, num_heads, ff_dim, dropout_p,
            ));
        }

        let norm = LayerNorm::new(d_model, 1e-5);
        let classifier = PackedLinear::<T>::new(d_model as usize, num_classes as usize, true);

        PackedTransformerEncoder {
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

    /// Forward pass through quantized transformer.
    pub fn forward(&self, token_ids: &Tensor) -> Tensor {
        let shape = token_ids.shape_ref();
        if shape.len() < 2 {
            panic!(
                "PackedTransformerEncoder expected input with at least 2 dimensions (batch, seq_len), got shape {:?}",
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

        // Use standard embeddings (kept in f32 for compatibility)
        let x = self.embedding.forward(token_ids);

        // Add positional embeddings
        let seq_len_usize = seq_len as usize;
        let batch_usize = batch as usize;
        let positions: Vec<f32> = (0..seq_len_usize).map(|i| i as f32).collect();
        let pos_tensor = Tensor::from_vec(positions, vec![1, seq_len]);
        let pos_expanded = pos_tensor.expand(vec![batch, seq_len]);
        let pos_emb = self.pos_embedding.forward(&pos_expanded);
        let x = x.add(&pos_emb);

        // Process through quantized layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        x = self.norm.forward(&x);

        // Extract CLS token (first token of sequence)
        let cls_token = x.slice(1, 0, 1, 1);
        let cls_token = cls_token.squeeze(Some(1));

        // Quantized classifier - handle batch > 1 properly
        let cls_numpy = cls_token.to_numpy();
        let d_model_usize = self.d_model as usize;
        let num_classes_usize = self.num_classes as usize;
        let mut output_data = Vec::with_capacity(batch_usize * num_classes_usize);
        for b in 0..batch_usize {
            let start = b * d_model_usize;
            let end = start + d_model_usize;
            let row_output = self.classifier.forward(&cls_numpy[start..end]);
            output_data.extend(row_output);
        }
        Tensor::from_vec(output_data, vec![batch, self.num_classes])
    }
}

impl<T: PackedWord> Module for PackedTransformerEncoder<T> {
    fn forward(&self, x: &Tensor) -> Tensor {
        PackedTransformerEncoder::forward(self, x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Note: Packed layers don't have Tensor parameters in the traditional sense
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {
        // Packed layers don't maintain gradients in the same way
    }

    fn train_mode(&self) {
        self.training.store(true, Ordering::Relaxed);
        for layer in &self.layers {
            layer.train_mode();
        }
    }

    fn eval_mode(&self) {
        self.training.store(false, Ordering::Relaxed);
        for layer in &self.layers {
            layer.eval_mode();
        }
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }
}
