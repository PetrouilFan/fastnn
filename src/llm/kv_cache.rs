use crate::llm::config::LayerConfig;

#[derive(Debug, Clone)]
pub struct KVCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub kv_dim: usize,
    pub max_seq_len: usize,
    pub current_pos: usize,
}

impl KVCache {
    pub fn new(layer_config: &LayerConfig, max_seq_len: usize) -> Self {
        let kv_dim = layer_config.num_kv_heads * layer_config.head_dim;
        let cache_size = max_seq_len * kv_dim;
        KVCache {
            k_cache: vec![0.0; cache_size],
            v_cache: vec![0.0; cache_size],
            kv_dim,
            max_seq_len,
            current_pos: 0,
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32], pos: usize) -> Result<(), String> {
        if k.len() != self.kv_dim || v.len() != self.kv_dim {
            return Err(format!(
                "KV dimensions don't match cache: k.len()={}, v.len()={}, kv_dim={}",
                k.len(), v.len(), self.kv_dim
            ));
        }

        if pos >= self.max_seq_len {
            return Err(format!("Position {} exceeds max_seq_len {}", pos, self.max_seq_len));
        }

        let offset = pos * self.kv_dim;
        self.k_cache[offset..offset + self.kv_dim].copy_from_slice(k);
        self.v_cache[offset..offset + self.kv_dim].copy_from_slice(v);
        self.current_pos = pos + 1;

        Ok(())
    }

    pub fn get_k(&self, pos: usize) -> &[f32] {
        let offset = pos * self.kv_dim;
        &self.k_cache[offset..offset + self.kv_dim]
    }

    pub fn get_v(&self, pos: usize) -> &[f32] {
        let offset = pos * self.kv_dim;
        &self.v_cache[offset..offset + self.kv_dim]
    }

    pub fn get_all_k(&self, len: usize) -> &[f32] {
        let total = len * self.kv_dim;
        &self.k_cache[..total.min(self.k_cache.len())]
    }

    pub fn get_all_v(&self, len: usize) -> &[f32] {
        let total = len * self.kv_dim;
        &self.v_cache[..total.min(self.v_cache.len())]
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    pub fn clear(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.current_pos = 0;
    }
}

pub type SharedKVCache = KVCache;