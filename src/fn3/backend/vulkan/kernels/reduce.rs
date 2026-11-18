//! GPU kernels owned by FN3-15 (#61).
//!
//! # Status
//!
//! The platform layer (FN3-11 / #57) is merged; this module is a scaffold so
//! the registration site in [`super::KernelSet::alpha`] exists before the
//! implementation lands. It registers nothing yet, which is honest: an op this
//! module does not register keeps its current support verdict and the CPU
//! reference answers instead.
//!
//! # Ops this module must cover
//!
//! - `reduce_mean`
//! - `reduce_sum`
//! - `arg_max`
//!
//! # Acceptance gate
//!
//! The `tests/fn3_cpu_reference.rs` fixtures are the numerical contract: same
//! pinned-graph inputs, tolerances from `docs/reports/fn3-03-parity-fixtures.md`.
//! A Vulkan kernel must reproduce those bounds, not merely "look right". Two
//! properties are contract rather than implementation detail: `matmul`
//! accumulates ascending-k in one `f32` accumulator, and softmax subtracts the
//! row max before `exp`, which is what makes `-FLT_MAX` all-masked rows exactly
//! uniform instead of NaN.

use super::KernelSet;

/// Register this module's kernels.
///
/// Called from [`super::KernelSet::alpha`]. Replacement is by op id, so a
/// registration here supersedes the FN3-11 smoke kernel for the same op.
pub fn register(_set: &mut KernelSet) {
    // TODO(FN3-15 (#61)): `_set.register(OpKernel { .. })` per op above.
}
