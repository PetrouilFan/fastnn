use crate::io::gguf::GgufFile;
use crate::llm::config::LlmConfig;
use crate::llm::embedding::Embedding;
use crate::llm::kv_cache::KVCache;
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
    pub has_ple: bool,
    pub per_layer_tok_embd: Option<GgmlQuantizedTensor>,
    pub per_layer_model_proj: Option<GgmlQuantizedTensor>,
    pub per_layer_proj_norm: Option<Vec<f32>>,
    pub ple_dim: usize,
    pub rope_freqs: Option<Vec<f32>>,
    pub shared_kv_swa: KVCache,
    pub shared_kv_full: KVCache,
}

pub struct PerLayerEmbedding {
    pub per_layer_tok_embd: GgmlQuantizedTensor,
    pub per_layer_model_proj: GgmlQuantizedTensor,
    pub per_layer_proj_norm: Vec<f32>,
    pub n_layers: usize,
    pub ple_dim: usize,
    pub hidden_size: usize,
    pub eps: f32,
}

pub struct PerLayerInput {
    pub data: Vec<f32>,
}

pub struct PleLayerParams {
    pub inp_gate: GgmlQuantizedTensor,
    pub proj: GgmlQuantizedTensor,
    pub post_norm: NormParams,
}

#[derive(Clone)]
pub struct TransformerLayer {
    pub attention: crate::llm::attention::AttentionLayer,
    pub norm1: NormParams,
    pub norm2: NormParams,
    pub post_attn_norm: NormParams,
    pub post_ffn_norm: NormParams,
    pub ffn: FFNParams,
    pub layer_output_scale: f32,
    pub layer_idx: usize,
    pub has_ple: bool,
    pub inp_gate: Option<GgmlQuantizedTensor>,
    pub proj: Option<GgmlQuantizedTensor>,
    pub ple_post_norm: Option<NormParams>,
}

impl TransformerLayer {
    pub fn new(
        attention: crate::llm::attention::AttentionLayer,
        norm1: NormParams,
        norm2: NormParams,
        post_attn_norm: NormParams,
        post_ffn_norm: NormParams,
        ffn: FFNParams,
        layer_output_scale: f32,
        layer_idx: usize,
        has_ple: bool,
        inp_gate: Option<GgmlQuantizedTensor>,
        proj: Option<GgmlQuantizedTensor>,
        ple_post_norm: Option<NormParams>,
    ) -> Self {
        TransformerLayer {
            attention,
            norm1,
            norm2,
            post_attn_norm,
            post_ffn_norm,
            ffn,
            layer_output_scale,
            layer_idx,
            has_ple,
            inp_gate,
            proj,
            ple_post_norm,
        }
    }

    pub fn forward(
        &mut self,
        x: &mut [f32],
        pos: usize,
        _scratch: &mut [f32],
        ple_input: Option<&[f32]>,
        shared_kv: Option<&KVCache>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let hidden_size = self.ffn.hidden_size;
        let layer_idx = self.layer_idx;

        fn rms(v: &[f32]) -> f32 {
            v.iter().map(|x| x * x).sum::<f32>().sqrt() / (v.len() as f32).sqrt()
        }

        let mut residual = x.to_vec();

        Kernels::rmsnorm_inplace(x, &self.norm1.weight, self.norm1.eps);
        llm_debug!("L{}: after norm1 rms={:.4}", layer_idx, rms(x));

        self.attention.forward(x, pos, shared_kv)?;
        Kernels::rmsnorm_inplace(x, &self.post_attn_norm.weight, self.post_attn_norm.eps);
        llm_debug!("L{}: after attn+post_norm rms={:.4}", layer_idx, rms(x));

        for i in 0..hidden_size {
            x[i] = residual[i] + x[i];
        }
        llm_debug!("L{}: after attn+residual rms={:.4}, mean={:.6}", layer_idx, rms(x), x.iter().sum::<f32>() / x.len() as f32);

        residual = x.to_vec();

        Kernels::rmsnorm_inplace(x, &self.norm2.weight, self.norm2.eps);
        llm_debug!("L{}: after norm2 rms={:.4}, mean={:.6}, first5={:.4},{:.4},{:.4},{:.4},{:.4}", 
            layer_idx, rms(x), 
            x.iter().sum::<f32>() / x.len() as f32,
            x[0], x[1], x[2], x[3], x[4]);

        let ffn_inter = self.ffn.intermediate_size;
        let mut gate_out = vec![0.0; ffn_inter];
        let mut up_out = vec![0.0; ffn_inter];

        self.ffn.gate_weight.gemv(x, &mut gate_out);
        self.ffn.up_weight.gemv(x, &mut up_out);

        if layer_idx <= 4 {
            llm_debug!("L{}: gate_out rms={:.4}, max={:.4}, min={:.4}", layer_idx, 
                gate_out.iter().map(|x| x * x).sum::<f32>().sqrt() / (gate_out.len() as f32).sqrt(),
                gate_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                gate_out.iter().cloned().fold(f32::INFINITY, f32::min));
            llm_debug!("L{}: up_out rms={:.4}, max={:.4}, min={:.4}", layer_idx, 
                up_out.iter().map(|x| x * x).sum::<f32>().sqrt() / (up_out.len() as f32).sqrt(),
                up_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                up_out.iter().cloned().fold(f32::INFINITY, f32::min));
        }

        for i in 0..ffn_inter {
            gate_out[i] = Kernels::gelu(gate_out[i]) * up_out[i];
        }

        if layer_idx <= 4 {
            llm_debug!("L{}: after silu*up rms={:.4}", layer_idx, 
                gate_out.iter().map(|x| x * x).sum::<f32>().sqrt() / (gate_out.len() as f32).sqrt());
        }

        let mut ffn_result = vec![0.0; hidden_size];
        self.ffn.down_weight.gemv(&gate_out, &mut ffn_result);

        if layer_idx <= 4 {
            let fr_mean = ffn_result.iter().sum::<f32>() / ffn_result.len() as f32;
            llm_debug!("L{}: ffn_result (before post_ffn_norm) rms={:.4}, mean={:.6}", layer_idx, rms(&ffn_result), fr_mean);
        }

        Kernels::rmsnorm_inplace(&mut ffn_result, &self.post_ffn_norm.weight, self.post_ffn_norm.eps);
        llm_debug!("L{}: ffn_result (after post_ffn_norm) rms={:.4}, mean={:.6}", layer_idx, rms(&ffn_result), ffn_result.iter().sum::<f32>() / ffn_result.len() as f32);

        for i in 0..hidden_size {
            x[i] = residual[i] + ffn_result[i];
        }
        llm_debug!("L{}: after ffn+residual rms={:.4}, mean={:.6}", layer_idx, rms(x), x.iter().sum::<f32>() / x.len() as f32);

        if self.has_ple {
            if let (Some(ple_data), Some(inp_gate), Some(proj), Some(ple_norm)) =
                (ple_input, &self.inp_gate, &self.proj, &self.ple_post_norm)
            {
                let ple_dim = inp_gate.shape()[0];
                let mut gate_result = vec![0.0; ple_dim];
                inp_gate.gemv(x, &mut gate_result);

                if layer_idx <= 4 {
                    let gr_rms = gate_result.iter().map(|x| x * x).sum::<f32>().sqrt() / (gate_result.len() as f32).sqrt();
                    llm_debug!("L{} PLE: inp_gate output rms={:.6}", layer_idx, gr_rms);
                }

        for v in gate_result.iter_mut() {
            *v = Kernels::gelu(*v);
        }
                for i in 0..ple_dim {
                    gate_result[i] *= ple_data[i];
                }
                let mut proj_result = vec![0.0; hidden_size];
                proj.gemv(&gate_result, &mut proj_result);
                Kernels::rmsnorm_inplace(&mut proj_result, &ple_norm.weight, ple_norm.eps);

                if layer_idx <= 4 {
                    let pr_rms = proj_result.iter().map(|x| x * x).sum::<f32>().sqrt() / (proj_result.len() as f32).sqrt();
                    let pr_mean = proj_result.iter().sum::<f32>() / proj_result.len() as f32;
                    llm_debug!("L{} PLE: proj_result rms={:.6}, mean={:.6}", layer_idx, pr_rms, pr_mean);
                }

                for i in 0..hidden_size {
                    x[i] += proj_result[i];
                }
            }
        }

        if self.layer_output_scale != 0.0 && self.layer_output_scale != 1.0 {
            for val in x.iter_mut() {
                *val *= self.layer_output_scale;
            }
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

    let rope_freqs: Option<Vec<f32>> = gguf.get_tensor("rope_freqs.weight").map(|t| {
        let data = t.row(0);
        let n = t.shape()[0].max(t.shape()[1]);
        data[..n].to_vec()
    });

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

        let layer_rope_freqs = if layer.head_dim > 256 { rope_freqs.clone() } else { None };

        let attention_layer = crate::llm::attention::AttentionLayer::new(
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            q_norm,
            k_norm,
            layer.clone(),
            layer_rope_freqs,
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

            let post_attn_norm = Self::get_norm(gguf, &format!("blk.{}.post_attention_norm.weight", layer.layer_idx))
                .unwrap_or_else(|| vec![0.0; layer.hidden_size]);

            let post_ffn_norm = Self::get_norm(gguf, &format!("blk.{}.post_ffw_norm.weight", layer.layer_idx))
                .unwrap_or_else(|| vec![0.0; layer.hidden_size]);

            let ffn = FFNParams {
                gate_weight,
                up_weight,
                down_weight,
                intermediate_size: layer.intermediate_size,
                hidden_size: layer.hidden_size,
            };

            let layer_output_scale = Self::get_layer_output_scale(gguf, layer.layer_idx);

            let ple_inp_gate = gguf.get_tensor(&format!("blk.{}.inp_gate.weight", layer.layer_idx));
            let ple_proj = gguf.get_tensor(&format!("blk.{}.proj.weight", layer.layer_idx));
            let ple_post_norm = Self::get_norm(gguf, &format!("blk.{}.post_norm.weight", layer.layer_idx));

            let has_ple = ple_inp_gate.is_some() && ple_proj.is_some() && ple_post_norm.is_some();
            llm_debug!("Layer {}: has_ple={}, inp_gate={}, proj={}, post_norm={}", 
                layer.layer_idx, has_ple,
                ple_inp_gate.as_ref().map(|t| format!("{:?}", t.shape())).unwrap_or_else(|| "None".to_string()),
                ple_proj.as_ref().map(|t| format!("{:?}", t.shape())).unwrap_or_else(|| "None".to_string()),
                ple_post_norm.as_ref().map(|t| format!("Some(len={})", t.len())).unwrap_or_else(|| "None".to_string()));

            layers.push(TransformerLayer::new(
                attention_layer,
                norm1,
                norm2,
                NormParams { eps: config.rms_norm_eps, weight: post_attn_norm },
                NormParams { eps: config.rms_norm_eps, weight: post_ffn_norm },
                ffn,
                layer_output_scale,
                layer.layer_idx,
                has_ple,
                ple_inp_gate,
                ple_proj,
                ple_post_norm.map(|w| NormParams { eps: config.rms_norm_eps, weight: w }),
            ));
        }

    let has_ple = layers.iter().any(|l| l.has_ple);
    let ple_dim = if has_ple {
        layers.iter().find_map(|l| l.inp_gate.as_ref()).map(|g| g.shape()[0]).unwrap_or(0)
    } else {
        0
    };

        let per_layer_tok_embd = if has_ple {
            gguf.get_tensor("per_layer_token_embd.weight")
        } else {
            None
        };
        let per_layer_model_proj = if has_ple {
            gguf.get_tensor("per_layer_model_proj.weight")
        } else {
            None
        };
        let per_layer_proj_norm = if has_ple {
            Self::get_norm(gguf, "per_layer_proj_norm.weight")
        } else {
            None
        };

    let scratch_size = config.max_intermediate_size() * 8;
    let scratch = vec![0.0; scratch_size];

    let n_layer_kv_from_start = config.n_layer_kv_from_start;
    let swa_kv_layer_idx = n_layer_kv_from_start - 2;
    let full_kv_layer_idx = n_layer_kv_from_start - 1;
    let shared_kv_swa = KVCache::new(&config.layers[swa_kv_layer_idx], 4096);
    let shared_kv_full = KVCache::new(&config.layers[full_kv_layer_idx], 4096);

    Ok(LlmModel {
        config,
        embedding,
        norm,
        lm_head,
        layers,
        scratch,
        has_ple,
        per_layer_tok_embd,
        per_layer_model_proj,
        per_layer_proj_norm,
        ple_dim,
        rope_freqs,
        shared_kv_swa,
        shared_kv_full,
    })
    }

    pub fn forward_token(&mut self, token_idx: usize, pos: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
llm_debug!("forward_token: token_idx={}, pos={}, hidden_size={}", token_idx, pos, self.config.hidden_size);

        let mut hidden_state = vec![0.0; self.config.hidden_size];
        llm_debug!("forward_token: hidden_state created, len={}", hidden_state.len());

        self.embedding.lookup(token_idx, &mut hidden_state)?;

        {
            let rms = hidden_state.iter().map(|x| x * x).sum::<f32>().sqrt() / (hidden_state.len() as f32).sqrt();
            let mean = hidden_state.iter().sum::<f32>() / hidden_state.len() as f32;
            llm_debug!("[forward_token] after embedding: token={}, RMS={:.4}, mean={:.4}", token_idx, rms, mean);
        }

        let n_layers = self.config.layers.len();
        let hidden_size = self.config.hidden_size;
        let ple_dim = self.ple_dim;

        let ple_inputs: Vec<Vec<f32>> = if self.has_ple {
            let ple_scale = (ple_dim as f32).sqrt();

            let tok_embd_row = self.per_layer_tok_embd.as_ref()
                .map(|embd| embd.row(token_idx))
                .unwrap_or_else(|| vec![0.0; n_layers * ple_dim]);

            let mut tok_embd_scaled = tok_embd_row;
            for v in tok_embd_scaled.iter_mut() {
                *v *= ple_scale;
            }

            let mut model_proj_out = vec![0.0; n_layers * ple_dim];
            if let Some(ref model_proj) = self.per_layer_model_proj {
                model_proj.gemv(&hidden_state, &mut model_proj_out);
            }
            let inv_sqrt_hidden = 1.0 / (hidden_size as f32).sqrt();
            for v in model_proj_out.iter_mut() {
                *v *= inv_sqrt_hidden;
            }

            let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();

            let mut ple_inputs = Vec::with_capacity(n_layers);
            for layer_idx in 0..n_layers {
                let offset = layer_idx * ple_dim;
                let mut combined = vec![0.0; ple_dim];

                let mut model_proj_slice = model_proj_out[offset..offset + ple_dim].to_vec();
                if let Some(ref proj_norm) = self.per_layer_proj_norm {
                    Kernels::rmsnorm_inplace(&mut model_proj_slice, proj_norm, self.config.rms_norm_eps);
                }

                for i in 0..ple_dim {
                    combined[i] = (model_proj_slice[i] + tok_embd_scaled[offset + i]) * inv_sqrt2;
                }

                ple_inputs.push(combined);
            }
            ple_inputs
        } else {
            (0..n_layers).map(|_| Vec::new()).collect()
        };

    for (i, layer) in self.layers.iter_mut().enumerate() {
        let rms_before = hidden_state.iter().map(|x| x * x).sum::<f32>().sqrt() / (hidden_state.len() as f32).sqrt();
        let ple_data = if self.has_ple && i < ple_inputs.len() {
            Some(ple_inputs[i].as_slice())
        } else {
            None
        };

        let shared_kv = if !self.config.layers[i].has_own_kv {
            if self.config.layers[i].head_dim <= 256 {
                Some(&self.shared_kv_swa as &KVCache)
            } else {
                Some(&self.shared_kv_full as &KVCache)
            }
        } else {
            None
        };

        layer.forward(
            &mut hidden_state,
            pos,
            &mut self.scratch,
            ple_data,
            shared_kv,
        )?;

        // After layers 13 and 14, snapshot their KV caches to shared storage
        if i == self.config.n_layer_kv_from_start - 2 {
            self.shared_kv_swa.k_cache.copy_from_slice(&layer.attention.kv_cache.k_cache);
            self.shared_kv_swa.v_cache.copy_from_slice(&layer.attention.kv_cache.v_cache);
            self.shared_kv_swa.current_pos = layer.attention.kv_cache.current_pos;
        }
        if i == self.config.n_layer_kv_from_start - 1 {
            self.shared_kv_full.k_cache.copy_from_slice(&layer.attention.kv_cache.k_cache);
            self.shared_kv_full.v_cache.copy_from_slice(&layer.attention.kv_cache.v_cache);
            self.shared_kv_full.current_pos = layer.attention.kv_cache.current_pos;
        }

        let rms_after = hidden_state.iter().map(|x| x * x).sum::<f32>().sqrt() / (hidden_state.len() as f32).sqrt();
        llm_debug!("Layer {}: RMS before={}, after={}", layer.layer_idx, rms_before, rms_after);
    }

        Kernels::rmsnorm_inplace(&mut hidden_state, &self.norm.weight, self.norm.eps);

        {
            let rms = hidden_state.iter().map(|x| x * x).sum::<f32>().sqrt() / (hidden_state.len() as f32).sqrt();
            let mean = hidden_state.iter().sum::<f32>() / hidden_state.len() as f32;
            llm_debug!("[forward_token] after final norm: RMS={:.4}, mean={:.4}", rms, mean);
        }

        let mut logits = vec![0.0; self.config.vocab_size];
        llm_debug!("[forward_token] lm_head: weight shape=[{}, {}], activation len={}, output len={}",
            self.lm_head.shape()[1], self.lm_head.shape()[0], hidden_state.len(), logits.len());

        {
            let lm_row0 = self.lm_head.row(0);
            let lm_rms = (lm_row0.iter().map(|x| x*x).sum::<f32>() / lm_row0.len() as f32).sqrt();
            let lm_mean = lm_row0.iter().sum::<f32>() / lm_row0.len() as f32;
            llm_debug!("[forward_token] lm_head row 0: rms={:.6}, mean={:.6}, first5={:.4},{:.4},{:.4},{:.4},{:.4}",
                lm_rms, lm_mean, lm_row0[0], lm_row0[1], lm_row0[2], lm_row0[3], lm_row0[4]);
        }

        self.lm_head.gemv(
            &hidden_state,
            &mut logits,
        );

        let cap = self.config.logit_softcapping;
        if cap > 0.0 {
            for logit in logits.iter_mut() {
                *logit = (*logit / cap).tanh() * cap;
            }
        }

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
        let data = tensor.row(0);
        let size = tensor.shape()[0].max(tensor.shape()[1]);
        Ok(data[..size.min(hidden_size)].to_vec())
    }

    fn get_layer_output_scale(gguf: &GgufFile, layer_idx: usize) -> f32 {
        let name = format!("blk.{}.layer_output_scale.weight", layer_idx);
        if let Some(tensor) = gguf.get_tensor(&name) {
            let data = tensor.row(0);
            if !data.is_empty() {
                return data[0];
            }
        }
        1.0
    }

}