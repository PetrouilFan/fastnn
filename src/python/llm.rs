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

    fn clear_cache(&mut self) {
        for layer in &mut self.model.layers {
            layer.attention.kv_cache.clear();
        }
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