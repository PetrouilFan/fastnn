#![allow(clippy::too_many_arguments)]
use crate::nn::{self as core_nn, Module};
use crate::storage::allocator_stats as storage_allocator_stats;
use crate::storage::{DType, Device};
use crate::tensor::{self as core_tensor, Tensor};
use crate::{residual, set_seeded_rng};
use parking_lot::RwLock;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::wrap_pyfunction;
use pyo3::PyAny;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

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
    IoError,
    FastnnError,
    "Error during serialization/deserialization."
);

// Lock-free default device storage (AtomicU8 for variant, AtomicUsize for device_id)
static DEFAULT_DEVICE_VARIANT: AtomicU8 = AtomicU8::new(0);

pub(crate) fn get_default_device() -> Device {
    match DEFAULT_DEVICE_VARIANT.load(Ordering::Relaxed) {
        0 => Device::Cpu,
        _ => Device::Cpu,
    }
}

pub(crate) fn set_default_device_internal(device: Device) {
    match device {
        Device::Cpu => DEFAULT_DEVICE_VARIANT.store(0, Ordering::Relaxed),
    }
}

include!("tensor.rs");
include!("factories.rs");
include!("ops.rs");
include!("nn.rs");
include!("io.rs");

// FN3-18 (#64): the 3.0 inference surface. Declared by path so the binding
// source lives under `src/fn3/python/` (the ADR's reserved path for #64) while
// still being a module of `crate::python`, which is the only tree compiled with
// the `python` feature. `src/fn3/` itself stays free of PyO3.
#[path = "../fn3/python/mod.rs"]
pub mod fn3;

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

    // Initialize the Rayon global thread pool with pinned physical cores.
    #[cfg(feature = "parallel")]
    crate::backend::cpu::affinity::ensure_global_pool_initialized();

    let py = m.py();

    // Register exception hierarchy
    m.add("FastnnError", py.get_type::<FastnnError>())?;
    m.add("ShapeError", py.get_type::<ShapeError>())?;
    m.add("DtypeError", py.get_type::<DtypeError>())?;
    m.add("DeviceError", py.get_type::<DeviceError>())?;
    m.add("IoError", py.get_type::<IoError>())?;

    m.add_function(wrap_pyfunction!(tensor_from_data, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_from_bytes, py)?)?;
    m.add_function(wrap_pyfunction!(tensor_factory, py)?)?;
    m.add_function(wrap_pyfunction!(zeros, py)?)?;
    m.add_function(wrap_pyfunction!(empty, py)?)?;
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
    m.add_function(wrap_pyfunction!(leaky_relu, py)?)?;
    m.add_function(wrap_pyfunction!(elu, py)?)?;
    m.add_function(wrap_pyfunction!(softplus, py)?)?;
    m.add_function(wrap_pyfunction!(hardswish, py)?)?;
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
    m.add_function(wrap_pyfunction!(_set_seed, py)?)?;
    m.add_function(wrap_pyfunction!(_set_num_threads, py)?)?;
    m.add_function(wrap_pyfunction!(_get_num_threads, py)?)?;
    m.add_function(wrap_pyfunction!(_set_default_device, py)?)?;
    m.add_function(wrap_pyfunction!(allocator_stats, py)?)?;
    m.add_function(wrap_pyfunction!(clear_storage_pool, py)?)?;
    m.add_function(wrap_pyfunction!(save_model, py)?)?;
    m.add_function(wrap_pyfunction!(load_model, py)?)?;

    m.add_class::<Linear>()?;
    m.add_class::<Conv2d>()?;
    m.add_class::<MaxPool2d>()?;
    m.add_class::<AvgPool2d>()?;
    m.add_class::<AvgPool1d>()?;
    m.add_class::<MaxPool1d>()?;
    m.add_class::<ConvTranspose2d>()?;
    m.add_class::<Conv1d>()?;
    m.add_class::<Conv3d>()?;
    m.add_class::<ResidualBlock>()?;
    m.add_class::<FusedConvBn>()?;
    m.add_class::<FusedConvBnRelu>()?;
    m.add_class::<FusedConvBnGelu>()?;
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
    m.add_class::<Softmax>()?;
    m.add_class::<PReLU>()?;
    m.add_class::<Mish>()?;
    m.add_class::<AdaptiveAvgPool2d>()?;
    m.add_class::<Sequential>()?;
    m.add_class::<ModuleList>()?;
    m.add_class::<PyTransformerEncoder>()?;
    m.add_class::<AotExecutor>()?;

    m.add_class::<PyTensor>()?;

    m.add_function(wrap_pyfunction!(cat, py)?)?;
    m.add_function(wrap_pyfunction!(stack, py)?)?;
    m.add_function(wrap_pyfunction!(where_, py)?)?;
    m.add_function(wrap_pyfunction!(repeat, py)?)?;
    m.add_function(wrap_pyfunction!(expand, py)?)?;
    m.add_function(wrap_pyfunction!(slice, py)?)?;
    m.add_function(wrap_pyfunction!(topk, py)?)?;
    m.add_function(wrap_pyfunction!(gather, py)?)?;
    m.add_function(wrap_pyfunction!(einsum, py)?)?;
    m.add_function(wrap_pyfunction!(im2col, py)?)?;

    m.add_function(wrap_pyfunction!(flash_attention, py)?)?;
    m.add_function(wrap_pyfunction!(cumsum, py)?)?;
    m.add_function(wrap_pyfunction!(erf, py)?)?;

    // `_core.fn3` is the 3.0 inference surface. The 2.x entry points above are
    // the secondary eager/AOT API and carry no training names.
    fn3::register(m)?;
    Ok(())
}
