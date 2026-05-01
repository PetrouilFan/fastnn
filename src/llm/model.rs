use crate::io::gguf::GgufFile;
use crate::llm::config::LlmConfig;
use crate::llm::embedding::Embedding;
use crate::llm::ops::{Kernels, NormParams, FFNParams};
use crate::quants::quantized_tensor::GgmlQuantizedTensor;

#[cfg(feature = "debug")]
macro_rules! llm_debug {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}

#[cfg(not(feature = "debug"))]
macro_rules! llm_debug {
    ($($arg:tt)*) => {};
}

pub struct LlmModel {
    pub config: LlmConfig,
    pub embedding: Embedding,
    pub norm: NormParams,
    pub lm_head: GgmlQuantizedTensor,
    pub layers: Vec<TransformerLayer>,
    pub scratch: Vec<f32>,
}

#[derive(Clone)]
pub struct TransformerLayer {
    pub attention: crate::llm::attention::AttentionLayer,
    pub norm1: NormParams,
    pub norm2: NormParams,
    pub ffn: FFNParams,
    pub layer_idx: usize,
}

impl TransformerLayer {
    pub fn new(
        attention: crate::llm::attention::AttentionLayer,
        norm1: NormParams,
        norm2: NormParams,
        ffn: FFNParams,
        layer_idx: usize,
    ) -> Self {
        TransformerLayer {
            attention,
            norm1,
            norm2,
            ffn,
            layer_idx,
        }
    }

pub fn forward(
        &mut self,
        x: &mut [f32],
        pos: usize,
        _scratch: &mut [f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let hidden_size = self.ffn.hidden_size;
        let layer_idx = self.layer_idx;

        let mut residual = x.to_vec();

        Kernels::rmsnorm_inplace(x, &self.norm1.weight, self.norm1.eps);

        self.attention.forward(x, pos)?;

        for i in 0..hidden_size {
            x[i] = residual[i] + x[i];
        }

        residual = x.to_vec();

        Kernels::rmsnorm_inplace(x, &self.norm2.weight, self.norm2.eps);

        let ffn_inter = self.ffn.intermediate_size;
        let mut gate_out = vec![0.0; ffn_inter];
        let mut up_out = vec![0.0; ffn_inter];

        self.ffn.gate_weight.gemv(x, &mut gate_out);
        self.ffn.up_weight.gemv(x, &mut up_out);

        for i in 0..ffn_inter {
            let gate = gate_out[i];
            let silu = gate / (1.0 + (-gate).exp());
            gate_out[i] = silu * up_out[i];
        }

        let mut ffn_result = vec![0.0; hidden_size];
        self.ffn.down_weight.gemv(&gate_out, &mut ffn_result);

        for i in 0..hidden_size {
            x[i] = residual[i] + ffn_result[i];
        }

        Ok(())
    }
}

impl LlmModel {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, Box<dyn std::error::Error>> {
        let config = LlmConfig::from_gguf(gguf);
llm_debug!("LlmModel: config.vocab_size={}, config.hidden_size={}, config.num_layers={}, config.num_heads={}, config.num_kv_heads={}, config.head_dim={}", 
            config.vocab_size, config.hidden_size, config.num_layers, config.num_heads, config.num_kv_heads, config.head_dim);
        llm_debug!("LlmModel: config.layers[0].hidden_size={}, .num_heads={}, .num_kv_heads={}, .head_dim={}, .intermediate_size={}",
            config.layers[0].hidden_size, config.layers[0].num_heads, config.layers[0].num_kv_heads, config.layers[0].head_dim, config.layers[0].intermediate_size);

        let embedding = Embedding::from_gguf(gguf, &config)?;

        let output_norm_weight = Self::get_tensor(gguf, "output_norm.weight")?;
        let norm = NormParams {
            eps: config.rms_norm_eps,
            weight: Self::extract_norm_weights(&output_norm_weight, config.hidden_size)?,
        };

        let lm_head = gguf.get_tensor("output.weight")
            .or_else(|| gguf.get_tensor("token_embd.weight"))
            .ok_or("Missing output.weight (and no token_embd.weight fallback)")?;

        let mut layers = Vec::new();

        for layer in &config.layers {
            let q_weights = Self::get_tensor(gguf, &format!("blk.{}.attn_q.weight", layer.layer_idx))?;
            llm_debug!("Layer {}: attn_q weight shape=[{}, {}], layer.hidden_size={}", 
                layer.layer_idx, q_weights.shape()[0], q_weights.shape()[1], layer.hidden_size);
            let k_weights = Self::get_tensor(gguf, &format!("blk.{}.attn_k.weight", layer.layer_idx))?;
            let v_weights = Self::get_tensor(gguf, &format!("blk.{}.attn_v.weight", layer.layer_idx))?;
            let o_weights = Self::get_tensor(gguf, &format!("blk.{}.attn_output.weight", layer.layer_idx))?;

            let q_norm = Self::get_norm(gguf, &format!("blk.{}.attn_q_norm.weight", layer.layer_idx));
            let k_norm = Self::get_norm(gguf, &format!("blk.{}.attn_k_norm.weight", layer.layer_idx));

            let attention_layer = crate::llm::attention::AttentionLayer::new(
                q_weights,
                k_weights,
                v_weights,
                o_weights,
                q_norm,
                k_norm,
                layer.clone(),
            );

            let norm1_weight = Self::get_tensor(gguf, &format!("blk.{}.attn_norm.weight", layer.layer_idx))?;
            let norm1 = NormParams {
                eps: config.rms_norm_eps,
                weight: Self::extract_norm_weights(&norm1_weight, layer.hidden_size)?,
            };

            let norm2_weight = Self::get_tensor(gguf, &format!("blk.{}.ffn_norm.weight", layer.layer_idx))?;
            let norm2 = NormParams {
                eps: config.rms_norm_eps,
                weight: Self::extract_norm_weights(&norm2_weight, layer.hidden_size)?,
            };

            let gate_weight = Self::get_tensor(gguf, &format!("blk.{}.ffn_gate.weight", layer.layer_idx))?;
            let up_weight = Self::get_tensor(gguf, &format!("blk.{}.ffn_up.weight", layer.layer_idx))?;
            let down_weight = Self::get_tensor(gguf, &format!("blk.{}.ffn_down.weight", layer.layer_idx))?;

            let ffn = FFNParams {
                gate_weight,
                up_weight,
                down_weight,
                intermediate_size: layer.intermediate_size,
                hidden_size: layer.hidden_size,
            };

            layers.push(TransformerLayer::new(
                attention_layer,
                norm1,
                norm2,
                ffn,
                layer.layer_idx,
            ));
        }

        let scratch_size = config.max_intermediate_size() * 8;
        let scratch = vec![0.0; scratch_size];

        Ok(LlmModel {
            config,
            embedding,
            norm,
            lm_head,
            layers,
            scratch,
        })
    }

    pub fn forward_token(&mut self, token_idx: usize, pos: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
llm_debug!("forward_token: token_idx={}, pos={}, hidden_size={}", token_idx, pos, self.config.hidden_size);

        let mut hidden_state = vec![0.0; self.config.hidden_size];
        llm_debug!("forward_token: hidden_state created, len={}", hidden_state.len());

        self.embedding.lookup(token_idx, &mut hidden_state)?;
        llm_debug!("forward_token: after embedding lookup");

        for layer in &mut self.layers {
            llm_debug!("forward_token: calling layer {} forward, hidden_state len={}", layer.layer_idx, hidden_state.len());
            layer.forward(
                &mut hidden_state,
                pos,
                &mut self.scratch,
            )?;
        }

        Kernels::rmsnorm_inplace(&mut hidden_state, &self.norm.weight, self.norm.eps);

        llm_debug!("[forward_token] after final norm: hidden_state len={}", hidden_state.len());

        let mut logits = vec![0.0; self.config.vocab_size];
        llm_debug!("[forward_token] lm_head: weight shape=[{}, {}], activation len={}, output len={}",
            self.lm_head.shape()[1], self.lm_head.shape()[0], hidden_state.len(), logits.len());
        self.lm_head.gemv(
            &hidden_state,
            &mut logits,
        );

        Ok(logits)
    }

    pub fn generate(&mut self, prompt_tokens: &[usize], max_tokens: usize) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        llm_debug!("[generate] prompt_tokens={:?}, max_tokens={}", prompt_tokens, max_tokens);

        let mut output = Vec::new();
        let mut current_pos = 0;

        for &token in prompt_tokens {
            let logits = self.forward_token(token, current_pos)?;
            let next_token = self.sample(&logits);
            output.push(next_token);
            current_pos += 1;
        }

        for _ in 0..max_tokens {
            let logits = self.forward_token(*output.last().unwrap(), current_pos)?;
            let next_token = self.sample(&logits);
            output.push(next_token);
            current_pos += 1;
        }

        Ok(output)
    }

    fn sample(&self, logits: &[f32]) -> usize {
        let mut max_logit = logits[0];
        let mut max_idx = 0;

        for (i, &logit) in logits.iter().enumerate() {
            if logit > max_logit {
                max_logit = logit;
                max_idx = i;
            }
        }

        max_idx
    }

    fn get_tensor(gguf: &GgufFile, name: &str) -> Result<GgmlQuantizedTensor, Box<dyn std::error::Error>> {
        gguf.get_tensor(name)
            .ok_or_else(|| format!("Missing tensor: {}", name).into())
    }

    fn get_norm(gguf: &GgufFile, name: &str) -> Option<Vec<f32>> {
        let tensor = gguf.get_tensor(name)?;
        // For 1D norm tensors, dequantize by extracting as a single row
        // GGML: dims[0] = number of elements, dims[1] = 1 (or absent for 1D)
        let size = tensor.shape()[0].max(tensor.shape()[1]);
        let data = tensor.row(0);
        Some(data[..size].to_vec())
    }

    fn extract_norm_weights(tensor: &GgmlQuantizedTensor, hidden_size: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Dequantize the norm weights by extracting the first (and only) row
        let data = tensor.row(0);
        let size = tensor.shape()[0].max(tensor.shape()[1]);
        Ok(data[..size.min(hidden_size)].to_vec())
    }

    }