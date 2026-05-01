use crate::io::gguf::GgufFile;
use crate::llm::config::LlmConfig;
use crate::quants::quantized_tensor::GgmlQuantizedTensor;

#[cfg(feature = "debug")]
macro_rules! llm_debug {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}

#[cfg(not(feature = "debug"))]
macro_rules! llm_debug {
    ($($arg:tt)*) => {};
}

pub struct Embedding {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub weights: GgmlQuantizedTensor,
}

impl Embedding {
    pub fn from_gguf(gguf: &GgufFile, config: &LlmConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let weights = gguf.get_tensor("token_embd.weight")
            .ok_or("Missing token_embd.weight tensor")?;

        Ok(Embedding {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            weights,
        })
    }

    pub fn lookup(&mut self, token_idx: usize, output: &mut [f32]) -> Result<(), Box<dyn std::error::Error>> {
        if token_idx >= self.vocab_size {
            return Err(format!("Token index {} out of range [0, {})", token_idx, self.vocab_size).into());
        }

        if output.len() < self.hidden_size {
            return Err("Output buffer too small".into());
        }

        let row = self.weights.row(token_idx);
        let scale = (self.hidden_size as f32).sqrt();
        for i in 0..self.hidden_size {
            output[i] = row[i] * scale;
        }

        llm_debug!("[Embedding] token_idx={}, hidden_size={}", token_idx, self.hidden_size);

        Ok(())
    }

    pub fn batch_lookup(&mut self, token_indices: &[usize], output: &mut Vec<Vec<f32>>) -> Result<(), Box<dyn std::error::Error>> {
        if token_indices.is_empty() {
            return Ok(());
        }

        if output.len() < token_indices.len() || output[0].len() < self.hidden_size {
            return Err("Output buffer too small".into());
        }

        for (i, &token_idx) in token_indices.iter().enumerate() {
            if token_idx >= self.vocab_size {
                return Err(format!("Token index {} out of range [0, {})", token_idx, self.vocab_size).into());
            }

            self.lookup(token_idx, &mut output[i])?;
        }

        Ok(())
    }
}