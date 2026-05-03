use crate::io::gguf::GgufFile;
use crate::llm::model::LlmModel;
use pyo3::prelude::*;
use std::path::Path;

#[pyclass]
pub struct PyLlmModel {
    model: LlmModel,
}

#[pymethods]
impl PyLlmModel {
    #[new]
    fn new(gguf_path: &str) -> PyResult<Self> {
        let path = Path::new(gguf_path);
        let gguf = GgufFile::from_path(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let model = LlmModel::from_gguf(&gguf).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyLlmModel { model })
    }

    fn generate(&mut self, prompt_tokens: Vec<usize>, max_tokens: usize) -> PyResult<Vec<usize>> {
        let result = self.model.generate(&prompt_tokens, max_tokens)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result)
    }

    fn forward_token(&mut self, token_idx: usize, pos: usize) -> PyResult<Vec<f32>> {
        let result = self.model.forward_token(token_idx, pos)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result)
    }

    fn get_vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }

    fn get_hidden_size(&self) -> usize {
        self.model.config.hidden_size
    }

    fn get_num_layers(&self) -> usize {
        self.model.config.num_layers
    }

    fn get_num_heads(&self) -> usize {
        self.model.config.num_heads
    }

    fn get_num_kv_heads(&self) -> usize {
        self.model.config.num_kv_heads
    }

    fn get_head_dim(&self) -> usize {
        self.model.config.head_dim
    }

    fn get_intermediate_size(&self) -> usize {
        self.model.config.intermediate_size
    }

    fn get_sliding_window(&self) -> usize {
        self.model.config.sliding_window
    }

    fn get_shared_kv_layers(&self) -> usize {
        self.model.config.shared_kv_layers
    }

    fn list_tensor_names(&self, filter: Option<&str>) -> Vec<String> {
        let gguf = GgufFile::from_path("/home/petrouil/Projects/fastnn_projects/llm/loader/gemma-4-E2B-it-Q4_K_M.gguf").unwrap();
        let names: Vec<String> = gguf.tensor_names().iter()
            .map(|s| s.to_string())
            .filter(|n| filter.as_ref().map_or(true, |f| n.contains(f)))
            .collect();
        names
    }

    fn get_tensor_info(&self, name: &str) -> PyResult<Vec<f32>> {
        let gguf = GgufFile::from_path("/home/petrouil/Projects/fastnn_projects/llm/loader/gemma-4-E2B-it-Q4_K_M.gguf")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let tensor = gguf.get_tensor(name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tensor {} not found", name)))?;
        let shape = tensor.shape();
        let total: usize = shape.iter().product();
        let mut result = Vec::with_capacity(total);
        for row in 0..shape[0] {
            result.extend_from_slice(&tensor.row(row));
        }
        Ok(result)
    }

    fn clear_cache(&mut self) {
        for layer in &mut self.model.layers {
            layer.attention.kv_cache.clear();
        }
        self.model.shared_kv_swa.clear();
        self.model.shared_kv_full.clear();
    }

    fn debug_gemv(&mut self, weight_name: &str, input: Vec<f32>) -> PyResult<Vec<f32>> {
        use crate::io::gguf::GgufFile;
        use crate::quants::quantized_tensor::GgmlQuantizedTensor;
        use std::path::Path;

        let path = Path::new("/home/petrouil/Projects/fastnn_projects/llm/loader/gemma-4-E2B-it-Q4_K_M.gguf");
        let gguf = GgufFile::from_path(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let tensor = gguf.get_tensor(weight_name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tensor {} not found", weight_name)))?;

        let out_size = tensor.shape()[0];
        let mut output = vec![0.0f32; out_size];
        tensor.gemv(&input, &mut output);
        Ok(output)
    }

    fn get_weight_row(&self, weight_name: &str, row_idx: usize) -> PyResult<Vec<f32>> {
        let tensor = match weight_name {
            name if name.starts_with("blk.") => {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() < 3 {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Invalid tensor name: {}", weight_name)));
                }
                let layer_idx: usize = parts[1].parse().unwrap_or(0);
                if layer_idx >= self.model.layers.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Layer index {} out of range", layer_idx)));
                }
                let layer = &self.model.layers[layer_idx];
                let suffix = &name[4 + parts[1].len() + 1..]; // after "blk.N."
                match suffix {
                    "ffn_gate.weight" => Some(layer.ffn.gate_weight.row(row_idx)),
                    "ffn_up.weight" => Some(layer.ffn.up_weight.row(row_idx)),
                    "ffn_down.weight" => Some(layer.ffn.down_weight.row(row_idx)),
                    "attn_q.weight" => Some(layer.attention.q_weights.row(row_idx)),
                    "attn_k.weight" => Some(layer.attention.k_weights.row(row_idx)),
                    "attn_v.weight" => Some(layer.attention.v_weights.row(row_idx)),
                    "attn_output.weight" => Some(layer.attention.o_weights.row(row_idx)),
                    _ => None,
                }
            }
            "output.weight" | "lm_head.weight" => Some(self.model.lm_head.row(row_idx)),
            _ => None,
        };

        tensor.ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Tensor {} not found or unsupported", weight_name)))
    }
}

#[pyfunction]
pub fn load_gguf_model(gguf_path: &str) -> PyResult<PyLlmModel> {
    PyLlmModel::new(gguf_path)
}

#[pyfunction]
pub fn generate_with_model(
    model: &mut PyLlmModel,
    prompt_tokens: Vec<usize>,
    max_tokens: usize,
) -> PyResult<Vec<usize>> {
    model.generate(prompt_tokens, max_tokens)
}

pub fn register_llm_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLlmModel>()?;
    m.add_function(wrap_pyfunction!(load_gguf_model, m.py())?)?;
    m.add_function(wrap_pyfunction!(generate_with_model, m.py())?)?;
    Ok(())
}