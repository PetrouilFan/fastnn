use crate::autograd::{no_grad_enter, no_grad_exit};
use crate::dispatcher::list_registered_ops as dispatcher_list_ops;
use crate::nn::{self as core_nn, Module};
use crate::optim::{self as core_optim, Optimizer};
use crate::storage::allocator_stats as storage_allocator_stats;
use crate::storage::{DType, Device};
use crate::tensor::{self as core_tensor, Tensor};
use crate::{autograd, dispatcher, io as core_io, residual, set_seeded_rng};
use parking_lot::RwLock;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::wrap_pyfunction;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

// Custom exception hierarchy for fastnn
pyo3::create_exception!(
    fastnn,
    FastnnError,
    PyRuntimeError,
    "Base exception for fastnn operations."
);
pyo3::create_exception!(
    fastnn,
    ShapeError,
    FastnnError,
    "Shape mismatch or invalid shape for operation."
);
pyo3::create_exception!(
    fastnn,
    DtypeError,
    FastnnError,
    "Data type mismatch or unsupported dtype."
);
pyo3::create_exception!(
    fastnn,
    DeviceError,
    FastnnError,
    "Device-related error (e.g., GPU unavailable)."
);
pyo3::create_exception!(
    fastnn,
    AutogradError,
    FastnnError,
    "Error during autograd/backward pass."
);
pyo3::create_exception!(
    fastnn,
    OptimizerError,
    FastnnError,
    "Error in optimizer step."
);
pyo3::create_exception!(
    fastnn,
    IoError,
    FastnnError,
    "Error during serialization/deserialization."
);
pyo3::create_exception!(
    fastnn,
    CudaError,
    FastnnError,
    "CUDA/GPU computation error."
);

// Thread-local default device storage
static DEFAULT_DEVICE: OnceLock<RwLock<Device>> = OnceLock::new();

fn get_default_device() -> Device {
    let guard = DEFAULT_DEVICE
        .get_or_init(|| RwLock::new(Device::Cpu))
        .read();
    *guard
}

fn set_default_device_internal(device: Device) {
    let mut guard = DEFAULT_DEVICE
        .get_or_init(|| RwLock::new(Device::Cpu))
        .write();
    *guard = device;
}

include!("tensor.rs");
include!("factories.rs");
include!("ops.rs");
include!("nn.rs");
include!("optim.rs");
include!("io.rs");
include!("llm.rs");

#[pymodule]
pub fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Set OpenBLAS and OMP to single-threaded mode by default for optimal
    // single-threaded performance. Users can override via environment variables.
    if std::env::var("OPENBLAS_NUM_THREADS").is_err() {
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    }
    if std::env::var("OMP_NUM_THREADS").is_err() {
        std::env::set_var("OMP_NUM_THREADS", "1");
    }

    let py = m.py();

    // Register exception hierarchy
    m.add("FastnnError", py.get_type::<FastnnError>())?;
    m.add("ShapeError", py.get_type::<ShapeError>())?;
    m.add("DtypeError", py.get_type::<DtypeError>())?;
    m.add("DeviceError", py.get_type::<DeviceError>())?;
    m.add("AutogradError", py.get_type::<AutogradError>())?;
    m.add("OptimizerError", py.get_type::<OptimizerError>())?;
    m.add("IoError", py.get_type::<IoError>())?;
    m.add("CudaError", py.get_type::<CudaError>())?;

    m.add_function(wrap_pyfunction!(tensor_from_data, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_factory, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_from_list, py)?)?;
    m.add_function(wrap_pyfunction!(zeros, py)?)?;
    m.add_function(wrap_pyfunction!(ones, py)?)?;
    m.add_function(wrap_pyfunction!(full, py)?)?;
    m.add_function(wrap_pyfunction!(arange, py)?)?;
    m.add_function(wrap_pyfunction!(linspace, py)?)?;
    m.add_function(wrap_pyfunction!(eye, py)?)?;
    m.add_function(wrap_pyfunction!(randn, py)?)?;
    m.add_function(wrap_pyfunction!(rand_uniform, py)?)?;
    m.add_function(wrap_pyfunction!(randint, py)?)?;
    m.add_function(wrap_pyfunction!(zeros_like, py)?)?;
    m.add_function(wrap_pyfunction!(ones_like, py)?)?;
    m.add_function(wrap_pyfunction!(full_like, py)?)?;
    m.add_function(wrap_pyfunction!(add, py)?)?;
    m.add_function(wrap_pyfunction!(sub, py)?)?;
    m.add_function(wrap_pyfunction!(mul, py)?)?;
    m.add_function(wrap_pyfunction!(div, py)?)?;
    m.add_function(wrap_pyfunction!(matmul, py)?)?;
    m.add_function(wrap_pyfunction!(batched_mlp_forward, py)?)?;
    m.add_function(wrap_pyfunction!(neg, py)?)?;
    m.add_function(wrap_pyfunction!(abs, py)?)?;
    m.add_function(wrap_pyfunction!(exp, py)?)?;
    m.add_function(wrap_pyfunction!(log, py)?)?;
    m.add_function(wrap_pyfunction!(sqrt, py)?)?;
    m.add_function(wrap_pyfunction!(clamp, py)?)?;
    m.add_function(wrap_pyfunction!(pow, py)?)?;
    m.add_function(wrap_pyfunction!(relu, py)?)?;
    m.add_function(wrap_pyfunction!(fused_add_relu, py)?)?;
    m.add_function(wrap_pyfunction!(fused_linear_relu, py)?)?;
    m.add_function(wrap_pyfunction!(fused_linear_gelu, py)?)?;
    m.add_function(wrap_pyfunction!(fused_conv_bn_silu, py)?)?;
    m.add_function(wrap_pyfunction!(gelu, py)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, py)?)?;
    m.add_function(wrap_pyfunction!(tanh, py)?)?;
    m.add_function(wrap_pyfunction!(silu, py)?)?;
    m.add_function(wrap_pyfunction!(softmax, py)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, py)?)?;
    m.add_function(wrap_pyfunction!(sum, py)?)?;
    m.add_function(wrap_pyfunction!(mean, py)?)?;
    m.add_function(wrap_pyfunction!(max, py)?)?;
    m.add_function(wrap_pyfunction!(min, py)?)?;
    m.add_function(wrap_pyfunction!(maximum, py)?)?;
    m.add_function(wrap_pyfunction!(minimum, py)?)?;
    m.add_function(wrap_pyfunction!(argmax, py)?)?;
    m.add_function(wrap_pyfunction!(argmin, py)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, py)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_loss, py)?)?;
    m.add_function(wrap_pyfunction!(_no_grad_enter, py)?)?;
    m.add_function(wrap_pyfunction!(_no_grad_exit, py)?)?;
    m.add_function(wrap_pyfunction!(checkpoint, py)?)?;
    m.add_function(wrap_pyfunction!(_set_seed, py)?)?;
    m.add_function(wrap_pyfunction!(_set_num_threads, py)?)?;
    m.add_function(wrap_pyfunction!(_get_num_threads, py)?)?;
    m.add_function(wrap_pyfunction!(_set_default_device, py)?)?;
    m.add_function(wrap_pyfunction!(allocator_stats, py)?)?;
    m.add_function(wrap_pyfunction!(clear_storage_pool, py)?)?;
    m.add_function(wrap_pyfunction!(list_registered_ops, py)?)?;
    m.add_function(wrap_pyfunction!(save_model, py)?)?;
    m.add_function(wrap_pyfunction!(load_model, py)?)?;

    m.add_class::<Linear>()?;
    m.add_class::<Conv2d>()?;
    m.add_class::<MaxPool2d>()?;
    m.add_class::<ConvTranspose2d>()?;
    m.add_class::<Conv1d>()?;
    m.add_class::<Conv3d>()?;
    m.add_class::<ResidualBlock>()?;
    m.add_class::<LayerNorm>()?;
    m.add_class::<BatchNorm1d>()?;
    m.add_class::<RMSNorm>()?;
    m.add_class::<GroupNorm>()?;
    m.add_class::<BatchNorm2d>()?;
    m.add_class::<Dropout>()?;
    m.add_class::<Dropout2d>()?;
    m.add_class::<Upsample>()?;
    m.add_class::<Embedding>()?;
    m.add_class::<ReLU>()?;
    m.add_class::<Gelu>()?;
    m.add_class::<Sigmoid>()?;
    m.add_class::<Tanh>()?;
    m.add_class::<SiLU>()?;
    m.add_class::<LeakyReLU>()?;
    m.add_class::<Softplus>()?;
    m.add_class::<Hardswish>()?;
    m.add_class::<Elu>()?;
    m.add_class::<Mish>()?;
    m.add_class::<AdaptiveAvgPool2d>()?;
    m.add_class::<Sequential>()?;
    m.add_class::<ModuleList>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PyMuon>()?;
    m.add_class::<PyLion>()?;
    m.add_class::<PyRMSprop>()?;
    m.add_class::<PyTransformerEncoder>()?;

    m.add_class::<PyTensor>()?;

    m.add_function(wrap_pyfunction!(bucket_allreduce, py)?)?;
    m.add_function(wrap_pyfunction!(cat, py)?)?;
    m.add_function(wrap_pyfunction!(stack, py)?)?;
    m.add_function(wrap_pyfunction!(einsum, py)?)?;
    m.add_function(wrap_pyfunction!(im2col, py)?)?;

    m.add_function(wrap_pyfunction!(bce_with_logits, py)?)?;
    m.add_function(wrap_pyfunction!(huber_loss, py)?)?;
    m.add_function(wrap_pyfunction!(flash_attention, py)?)?;
    m.add_function(wrap_pyfunction!(clip_grad_norm_, py)?)?;
    m.add_function(wrap_pyfunction!(clip_grad_value_, py)?)?;

    // Register LLM module
    register_llm_module(&m)?;

    Ok(())
}
