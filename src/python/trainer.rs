use crate::backend::cpu::CpuBackend;
use crate::backend::executor::{CompiledTrainingModel, GraphExecutor};
use crate::compiler::passes::training::{OptimizerConfig, TrainConfig};
use crate::ir::node::ComputeGraph;

#[pyclass]
pub struct PyCompiledTrainingModel {
    pub inner: CompiledTrainingModel<CpuBackend>,
}

#[pymethods]
impl PyCompiledTrainingModel {
    fn train_step(&mut self, inputs: Vec<Vec<u8>>) -> PyResult<f32> {
        let refs: Vec<&[u8]> = inputs.iter().map(|v| &v[..]).collect();
        self.inner
            .train_step(&refs)
            .map_err(|e| PyRuntimeError::new_err(format!("train_step failed: {e}")))
    }
}

impl From<CompiledTrainingModel<CpuBackend>> for PyCompiledTrainingModel {
    fn from(model: CompiledTrainingModel<CpuBackend>) -> Self {
        PyCompiledTrainingModel { inner: model }
    }
}

#[pyfunction]
pub fn compile_train_model(
    graph_bytes: &[u8],
    loss_node_id: usize,
    param_ids: Vec<usize>,
    param_data: Vec<Vec<u8>>,
    batch_input_ids: Vec<usize>,
    optimizer: &str,
    lr: f64,
    weight_decay: f64,
    beta1: Option<f64>,
    beta2: Option<f64>,
    eps: Option<f64>,
    quantize: Option<u8>,
) -> PyResult<PyCompiledTrainingModel> {
    let graph: ComputeGraph = bincode::deserialize(graph_bytes)
        .map_err(|e| PyRuntimeError::new_err(format!(
            "compile_train_model: failed to deserialize graph: {e}"
        )))?;

    let optimizer_config = match optimizer {
        "sgd" => OptimizerConfig::SGD {
            lr: lr as f32,
            weight_decay: weight_decay as f32,
        },
        "adamw" => OptimizerConfig::AdamW {
            lr: lr as f32,
            beta1: beta1.unwrap_or(0.9) as f32,
            beta2: beta2.unwrap_or(0.999) as f32,
            eps: eps.unwrap_or(1e-8) as f32,
            weight_decay: weight_decay as f32,
        },
        _ => return Err(PyRuntimeError::new_err(format!(
            "compile_train_model: unknown optimizer '{optimizer}', expected 'sgd' or 'adamw'"
        ))),
    };

    let config = TrainConfig {
        optimizer: optimizer_config,
        quantize,
    };

    let param_refs: Vec<&[u8]> = param_data.iter().map(|v| &v[..]).collect();
    let executor = GraphExecutor::new(CpuBackend);
    let model = executor
        .compile_train(
            &graph,
            loss_node_id,
            &param_ids,
            &param_refs,
            &batch_input_ids,
            None,
            &config,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("compile_train_model: {e}")))?;

    Ok(PyCompiledTrainingModel { inner: model })
}
