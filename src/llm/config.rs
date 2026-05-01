use crate::io::gguf::{GgufFile, GgufTensorInfo};
use crate::quants::QuantizedDType;

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub layer_idx: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub has_q_norm: bool,
    pub has_k_norm: bool,
    pub qkv_shared: bool,
    pub sliding_window: usize,
    pub rope_theta: f32,
}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub shared_kv_layers: usize,
    pub layers: Vec<LayerConfig>,
}

impl LlmConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let vocab_size = Self::get_vocab_size(gguf);
        let hidden_size = Self::get_hidden_size(gguf);

        let num_layers = Self::get_metadata_u32(gguf, "gemma4.block_count").unwrap_or(35) as usize;
        let num_heads = Self::get_metadata_u32(gguf, "gemma4.attention.head_count").unwrap_or(8) as usize;
        let num_kv_heads = Self::get_metadata_u32(gguf, "gemma4.attention.head_count_kv").unwrap_or(1) as usize;
        let rms_norm_eps = Self::get_metadata_f32(gguf, "gemma4.attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let sliding_window = Self::get_metadata_u32(gguf, "gemma4.attention.sliding_window").unwrap_or(512) as usize;
        let shared_kv_layers = Self::get_metadata_u32(gguf, "gemma4.attention.shared_kv_layers").unwrap_or(20) as usize;

        let rope_dim = match Self::get_metadata_u32(gguf, "gemma4.rope.dimension_count") {
            Some(v) => v as usize,
            None => 256,  // Default if missing
        };
        let rope_dim_swa = match Self::get_metadata_u32(gguf, "gemma4.rope.dimension_count_swa") {
            Some(v) => v as usize,
            None => 512,  // Default if missing
        };
        let rope_theta = 10000.0;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let head_dim = if Self::is_full_attention_layer(gguf, i) {
                rope_dim  // Full attention: head_dim = 512
            } else {
                rope_dim_swa  // SWA: head_dim = 256
            };

            let q_shape = Self::get_tensor_shape(gguf, &format!("blk.{}.attn_q.weight", i));
            let k_shape = Self::get_tensor_shape(gguf, &format!("blk.{}.attn_k.weight", i));
            let ffn_shape = Self::get_tensor_shape(gguf, &format!("blk.{}.ffn_gate.weight", i));

            let intermediate_size = ffn_shape.map(|(out, _)| out).unwrap_or(6144);

            let has_q_norm = Self::tensor_exists(gguf, &format!("blk.{}.attn_q_norm.weight", i));
            let has_k_norm = Self::tensor_exists(gguf, &format!("blk.{}.attn_k_norm.weight", i));

            let qkv_shared = i < shared_kv_layers;

            layers.push(LayerConfig {
                layer_idx: i,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_size,
                has_q_norm,
                has_k_norm,
                qkv_shared,
                sliding_window,
                rope_theta,
            });
        }

        LlmConfig {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim: rope_dim,
            intermediate_size: layers.first().map(|l| l.intermediate_size).unwrap_or(6144),
            rms_norm_eps,
            rope_theta,
            sliding_window,
            shared_kv_layers,
            layers,
        }
    }

fn get_vocab_size(gguf: &GgufFile) -> usize {
        for tensor in &gguf.tensors {
            if tensor.name == "token_embd.weight" {
                return tensor.dims[1] as usize;
            }
        }
        262144
    }

    fn get_hidden_size(gguf: &GgufFile) -> usize {
        for tensor in &gguf.tensors {
            if tensor.name == "token_embd.weight" {
                return tensor.dims[0] as usize;
            }
        }
        1536
    }

    fn get_metadata_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
        for (k, v) in &gguf.metadata {
            if k == key {
                if v.len() >= 4 {
                    return Some(u32::from_le_bytes([v[0], v[1], v[2], v[3]]));
                }
            }
        }
        None
    }

    fn get_metadata_f32(gguf: &GgufFile, key: &str) -> Option<f32> {
        for (k, v) in &gguf.metadata {
            if k == key {
                if v.len() >= 4 {
                    return Some(f32::from_le_bytes([v[0], v[1], v[2], v[3]]));
                }
            }
        }
        None
    }

    fn tensor_exists(gguf: &GgufFile, name: &str) -> bool {
        gguf.tensors.iter().any(|t| &t.name == name)
    }

    fn get_tensor_shape(gguf: &GgufFile, name: &str) -> Option<(usize, usize)> {
        for tensor in &gguf.tensors {
            if tensor.name == name {
                // GGML convention: dims[0]=ne[0]=in_features (cols), dims[1]=ne[1]=out_features (rows)
                return Some((tensor.dims[1] as usize, tensor.dims[0] as usize));
            }
        }
        None
    }

    fn is_full_attention_layer(gguf: &GgufFile, layer_idx: usize) -> bool {
        let name = format!("blk.{}.attn_k.weight", layer_idx);
        if let Some((k_out, _k_in)) = Self::get_tensor_shape(gguf, &name) {
            return k_out > 256;
        }
        false
    }

    pub fn max_head_dim(&self) -> usize {
        self.layers.iter().map(|l| l.head_dim).max().unwrap_or(self.head_dim)
    }

    pub fn max_intermediate_size(&self) -> usize {
        self.layers.iter().map(|l| l.intermediate_size).max().unwrap_or(self.intermediate_size)
    }

    pub fn total_kv_cache_size(&self) -> usize {
        let mut total = 0usize;
        for layer in &self.layers {
            if layer.qkv_shared && layer.layer_idx > 0 {
                continue;
            }
            total += self.sliding_window * layer.num_kv_heads * layer.head_dim;
        }
        total * 2
    }
}
