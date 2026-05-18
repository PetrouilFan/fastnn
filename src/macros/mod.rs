//! Macro definitions for reducing boilerplate across the codebase.
//! These macros are `#[macro_export]`'d and available crate-wide.

// Keep this file minimal — each category lives in its own submodule.
// Macros defined here with #[macro_export] are available without
// any path qualification.

#[macro_use]
pub mod tensor_ops;   // impl_scalar_op!, impl_cpu_fast_path!
#[macro_use]
pub mod optim;        // impl_optim!, base_optimizer macros
#[macro_use]
pub mod python_bindings; // impl_py_optim!
#[macro_use]
pub mod nn_common;    // impl_norm!, impl_pooling!
#[macro_use]
pub mod wgpu;         // dispatch_gpu_compute!, build_pipeline!
