//! The FN3-11 smoke kernel's declared spec, retained as a reference artifact.
//!
//! The kernel is deliberately **not** registered: FN3-13's pointwise family
//! registers the real `Add`, and [`super::KernelSet::register`] keeps the later
//! registration for an op, so a registered smoke kernel could never run. What
//! remains is the tiny known-good shader and its `KernelSpec`, which the
//! reflection, pipeline and dependency-map tests use as a stable input:
//!
//! - `SMOKE_ADD`: `out[i] = a[i] + b[i]`, contiguous same-shape `f32`, with a
//!   bounds-checked tail for element counts that are not a multiple of the
//!   workgroup size;
//! - `SMOKE_WORKGROUP`: the workgroup width the committed SPIR-V was built with;
//! - `smoke_add.comp.spv`: the shader those tests compile and reflect.
//!
//! The launch path that used to live here (allocate three device-local buffers,
//! lease host-visible staging, record `upload → barrier → dispatch → barrier →
//! readback` as one submission) is no longer needed: every shipped family has
//! its own kernel, and `pointwise::launch_add` is the implementation `Add`
//! dispatches.

use crate::inference::backend::vulkan::kernels::{grid, BindingSpec, KernelSpec};
use crate::inference::dtype::DType;

/// Declared kernel: `out[i] = a[i] + b[i]`.
pub const SMOKE_ADD: KernelSpec = KernelSpec {
    name: "smoke.add.f32",
    spirv: include_bytes!("../shaders/smoke_add.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "lhs"),
        BindingSpec::storage(1, "rhs"),
        BindingSpec::storage(2, "out"),
    ],
    push_constant_bytes: 4,
    workgroup_size: [64, 1, 1],
    dtypes: &[DType::F32],
    shape: "two contiguous same-shape f32 inputs, one contiguous f32 output of that shape",
    grid: grid::elementwise,
};

/// The workgroup size the shader was compiled with.
pub const SMOKE_WORKGROUP: u32 = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_matches_the_committed_shader_metadata() {
        assert_eq!(SMOKE_ADD.workgroup_size, [SMOKE_WORKGROUP, 1, 1]);
        assert_eq!(SMOKE_ADD.push_constant_bytes, 4);
        assert_eq!(SMOKE_ADD.bindings.len(), 3);
        assert!(SMOKE_ADD.validate().is_ok());
        assert_eq!(
            SMOKE_ADD.workgroups(65, &[65]).expect("grid computes"),
            [2, 1, 1]
        );
    }
}
