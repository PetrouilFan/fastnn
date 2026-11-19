//! The alpha CPU reference backend (FN3-07 / #53).
//!
//! # What is implemented
//!
//! Every [`OpId`] except [`OpId::MaskedAttention`], which is reserved for the
//! Vulkan attention kernels (FN3-15 / #61) and is reported as
//! `Support::Unsupported`. The authoritative list and the measurement of the
//! pinned graph's ONNX op inventory live in [`support`]; the kernels live in
//! [`kernels`].
//!
//! ```text
//! cpu/
//!   mod.rs        CpuBackend: capabilities, support, dispatch
//!   support.rs    honest support table + pinned-graph op inventory
//!   kernels/
//!     mod.rs      shared indexing/attribute helpers, optimization hand-off
//!     elementwise.rs  add sub mul div neg pow sqrt erf tanh gelu relu sin cos cast
//!     matmul.rs       matmul (rank >= 2, batched, broadcast batch dims)
//!     norm.rs         layernorm (trailing axis)
//!     softmax.rs      softmax (one axis, all-masked rows uniform)
//!     reduce.rs       reduce_mean reduce_sum argmax
//!     transform.rs    gather transpose reshape flatten squeeze unsqueeze
//!                     concat slice split expand shape constant_of_shape range
//!     logic.rs        equal less greater where trilu scatter_nd
//! ```
//!
//! # Numerical contract
//!
//! See [`kernels`] for the module-wide rules and
//! [`kernels::optimization_handoff`] for the symbols later performance work
//! (FN3-42 / #71) is expected to replace, together with the reference each one
//! must keep matching. In short: `f32` compute only, `MatMul` accumulates in
//! ascending `k`, reductions accumulate in ascending row-major order, and
//! `Erf`/`Gelu` are evaluated in `f64` and rounded once.
//!
//! # Error policy
//!
//! A request the backend does not implement is reported, never approximated:
//!
//! - an unimplemented op id is `UnsupportedOp` from [`support`];
//! - a shape or rank outside a kernel's documented contract is `UnsupportedOp`
//!   naming FN3-07 (#53) and the offending value;
//! - a wrong dtype, device, arity or attribute is the matching
//!   [`crate::fn3::error::Fn3Error`] variant with the op named;
//! - every failure happens before the first output write, so a failed
//!   `execute_op` leaves the caller's outputs untouched.

use std::fmt;

use super::{check_arity, Backend, Capabilities, CapabilityFeatures, OpParams, Support};
use crate::fn3::device::DeviceId;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::tensor::{TensorMut, TensorView};

pub(crate) mod kernels;
pub(crate) mod support;

/// Backend name used in every CPU diagnostic.
pub const CPU_BACKEND_NAME: &str = "cpu";

/// Capabilities of the alpha CPU reference.
pub fn cpu_capabilities() -> Capabilities {
    Capabilities {
        dtypes: &[DType::F32, DType::I64, DType::I32, DType::Bool],
        // Alpha policy limit for one buffer. Vulkan reports its own
        // `maxStorageBufferRange`; the CPU limit keeps a bad plan from
        // requesting an unbounded host allocation.
        max_buffer_bytes: 16 * 1024 * 1024 * 1024,
        max_shape_rank: 8,
        max_workgroup_size: [1, 1, 1],
        max_threads: std::thread::available_parallelism()
            .map(|parallelism| parallelism.get().min(u32::MAX as usize) as u32)
            .unwrap_or(1),
        supports_strided_views: true,
        features: CapabilityFeatures {
            broadcasting: true,
            // The fused id belongs to #61; the CPU proves the same result by
            // decomposition (see `support`).
            masked_attention: false,
            quantized: false,
            persistent_weights: true,
        },
    }
}

/// The alpha CPU reference backend.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    capabilities: Capabilities,
}

impl CpuBackend {
    /// Construct with the capabilities of this host.
    pub fn new() -> CpuBackend {
        CpuBackend {
            capabilities: cpu_capabilities(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        CpuBackend::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        CPU_BACKEND_NAME
    }

    fn device(&self) -> DeviceId {
        DeviceId::Cpu
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn supports(&self, op: OpId) -> Support {
        support::support(op)
    }

    fn execute_op(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        self.preflight(op)?;
        for view in inputs {
            view.meta().require_device(Some(op), self.device())?;
            self.capabilities.check_rank(view.shape().rank())?;
        }
        for output in outputs.iter() {
            output.meta().require_device(Some(op), self.device())?;
            self.capabilities.check_rank(output.shape().rank())?;
        }
        // A kernel below writes only after every check has passed; the
        // arguments are validated by arity/dtype/shape helpers before the first
        // output element is touched.
        self.dispatch(op, inputs, outputs, params)
    }
}

impl CpuBackend {
    fn dispatch(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        use kernels::{elementwise, logic, matmul, norm, reduce, softmax, transform};

        match op {
            // -- linear algebra ------------------------------------------
            OpId::MatMul => {
                check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
                kernels::require_no_attributes(op, params)?;
                matmul::matmul(op, &inputs[0], &inputs[1], &mut outputs[0])
            }

            // -- elementwise arithmetic ----------------------------------
            OpId::Add | OpId::Sub | OpId::Mul | OpId::Div | OpId::Pow => {
                check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
                kernels::require_no_attributes(op, params)?;
                match op {
                    OpId::Add => elementwise::add(op, &inputs[0], &inputs[1], &mut outputs[0]),
                    OpId::Sub => elementwise::sub(op, &inputs[0], &inputs[1], &mut outputs[0]),
                    OpId::Mul => elementwise::mul(op, &inputs[0], &inputs[1], &mut outputs[0]),
                    OpId::Div => elementwise::div(op, &inputs[0], &inputs[1], &mut outputs[0]),
                    OpId::Pow => elementwise::pow(op, &inputs[0], &inputs[1], &mut outputs[0]),
                    other => unreachable!("matched a binary op: {other}"),
                }
            }
            OpId::Neg
            | OpId::Sqrt
            | OpId::Erf
            | OpId::Tanh
            | OpId::Gelu
            | OpId::Relu
            | OpId::Sin
            | OpId::Cos => {
                check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
                kernels::require_no_attributes(op, params)?;
                match op {
                    OpId::Neg => elementwise::neg(op, &inputs[0], &mut outputs[0]),
                    OpId::Sqrt => elementwise::sqrt(op, &inputs[0], &mut outputs[0]),
                    OpId::Erf => elementwise::erf(op, &inputs[0], &mut outputs[0]),
                    OpId::Tanh => elementwise::tanh(op, &inputs[0], &mut outputs[0]),
                    OpId::Gelu => elementwise::gelu(op, &inputs[0], &mut outputs[0]),
                    OpId::Relu => elementwise::relu(op, &inputs[0], &mut outputs[0]),
                    OpId::Sin => elementwise::sin(op, &inputs[0], &mut outputs[0]),
                    OpId::Cos => elementwise::cos(op, &inputs[0], &mut outputs[0]),
                    other => unreachable!("matched a unary op: {other}"),
                }
            }

            // -- normalization and reduction ------------------------------
            OpId::Softmax => {
                check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
                kernels::require_only_axes(op, params)?;
                softmax::softmax(op, &inputs[0], &mut outputs[0], params)
            }
            OpId::LayerNorm => {
                check_arity(op, inputs.len(), outputs.len(), 3, 1)?;
                kernels::require_axes_and_epsilon(op, params)?;
                norm::layer_norm(
                    op,
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &mut outputs[0],
                    params,
                )
            }
            OpId::ReduceMean | OpId::ReduceSum => {
                check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
                kernels::require_axes_and_keep_dims(op, params)?;
                reduce::reduce(
                    op,
                    &inputs[0],
                    &mut outputs[0],
                    params,
                    op == OpId::ReduceMean,
                )
            }
            OpId::ArgMax => {
                check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
                kernels::require_axes_and_keep_dims(op, params)?;
                reduce::argmax(op, &inputs[0], &mut outputs[0], params)
            }

            // -- data movement --------------------------------------------
            OpId::Gather => {
                check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
                kernels::require_only_axes(op, params)?;
                transform::gather(op, &inputs[0], &inputs[1], &mut outputs[0], params)
            }
            OpId::Transpose => {
                check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
                kernels::require_only_axes(op, params)?;
                transform::transpose(op, &inputs[0], &mut outputs[0], params)
            }
            OpId::Reshape => {
                check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
                kernels::require_no_attributes(op, params)?;
                transform::reshape(op, &inputs[0], &inputs[1], &mut outputs[0])
            }
            OpId::Flatten => {
                check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
                kernels::require_only_axes(op, params)?;
                transform::flatten(op, &inputs[0], &mut outputs[0], params)
            }
            OpId::Squeeze | OpId::Unsqueeze => {
                check_flexible_arity(op, inputs.len(), outputs.len(), &[1, 2], 1)?;
                kernels::require_only_axes(op, params)?;
                let axes_input = inputs.get(1);
                if op == OpId::Squeeze {
                    transform::squeeze(op, &inputs[0], axes_input, &mut outputs[0], params)
                } else {
                    transform::unsqueeze(op, &inputs[0], axes_input, &mut outputs[0], params)
                }
            }
            OpId::Concat => {
                if inputs.is_empty() || outputs.len() != 1 {
                    return Err(Fn3Error::ArityMismatch {
                        op,
                        expected_inputs: 1,
                        actual_inputs: inputs.len(),
                        expected_outputs: 1,
                        actual_outputs: outputs.len(),
                    });
                }
                kernels::require_only_axes(op, params)?;
                transform::concat(op, inputs, &mut outputs[0], params)
            }
            OpId::Slice => {
                check_flexible_arity(op, inputs.len(), outputs.len(), &[2, 3, 4, 5], 1)?;
                kernels::require_no_attributes(op, params)?;
                transform::slice(
                    op,
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    inputs.get(3),
                    inputs.get(4),
                    &mut outputs[0],
                )
            }
            OpId::Split => {
                check_flexible_arity(op, inputs.len(), outputs.len(), &[1, 2], 1)?;
                kernels::require_only_axes(op, params)?;
                transform::split(op, &inputs[0], inputs.get(1), outputs, params)
            }
            OpId::Expand => {
                check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
                kernels::require_no_attributes(op, params)?;
                transform::expand(op, &inputs[0], &inputs[1], &mut outputs[0])
            }

            // -- shape ----------------------------------------------------
            OpId::Shape => {
                check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
                kernels::require_no_attributes(op, params)?;
                transform::shape_of(op, &inputs[0], &mut outputs[0])
            }
            OpId::ConstantOfShape => {
                check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
                kernels::require_no_attributes(op, params)?;
                transform::constant_of_shape(op, &inputs[0], &inputs[1], &mut outputs[0])
            }
            OpId::Range => {
                check_arity(op, inputs.len(), outputs.len(), 3, 1)?;
                kernels::require_no_attributes(op, params)?;
                transform::range(op, &inputs[0], &inputs[1], &inputs[2], &mut outputs[0])
            }

            // -- conversion and comparison --------------------------------
            OpId::Cast => {
                check_arity(op, inputs.len(), outputs.len(), 1, 1)?;
                kernels::require_no_attributes(op, params)?;
                elementwise::cast(op, &inputs[0], &mut outputs[0])
            }
            OpId::Equal | OpId::Less | OpId::Greater => {
                check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
                kernels::require_no_attributes(op, params)?;
                let comparison = match op {
                    OpId::Equal => logic::Comparison::Equal,
                    OpId::Less => logic::Comparison::Less,
                    _ => logic::Comparison::Greater,
                };
                logic::compare(op, comparison, &inputs[0], &inputs[1], &mut outputs[0])
            }
            OpId::Where => {
                check_arity(op, inputs.len(), outputs.len(), 3, 1)?;
                kernels::require_no_attributes(op, params)?;
                logic::where_select(op, &inputs[0], &inputs[1], &inputs[2], &mut outputs[0])
            }
            OpId::Trilu => {
                check_flexible_arity(op, inputs.len(), outputs.len(), &[1, 2], 1)?;
                kernels::require_only_axes(op, params)?;
                logic::trilu(op, &inputs[0], inputs.get(1), &mut outputs[0], params)
            }
            OpId::ScatterND => {
                check_arity(op, inputs.len(), outputs.len(), 3, 1)?;
                kernels::require_no_attributes(op, params)?;
                logic::scatter_nd(op, &inputs[0], &inputs[1], &inputs[2], &mut outputs[0])
            }

            // -- reserved fused id ---------------------------------------
            OpId::MaskedAttention => {
                // `preflight` already rejected this id; it is matched so that a
                // future capability change cannot silently fall through.
                Err(Fn3Error::UnsupportedOp {
                    op,
                    backend: CPU_BACKEND_NAME,
                    reason: "the fused attention id belongs to FN3-15 (#61); the CPU backend \
                             computes attention by decomposition and does not approximate the \
                             fused op"
                        .to_string(),
                })
            }
        }
    }
}

impl fmt::Display for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{CPU_BACKEND_NAME}")
    }
}

/// Arity check for an op whose input count is one of `allowed` and that needs
/// at least `min_outputs` outputs (`Split` writes one output per piece).
fn check_flexible_arity(
    op: OpId,
    inputs: usize,
    outputs: usize,
    allowed: &[usize],
    min_outputs: usize,
) -> Result<(), Fn3Error> {
    if allowed.contains(&inputs) && outputs >= min_outputs {
        Ok(())
    } else {
        Err(Fn3Error::ArityMismatch {
            op,
            expected_inputs: allowed.last().copied().unwrap_or(0),
            actual_inputs: inputs,
            expected_outputs: min_outputs,
            actual_outputs: outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fn3::shape::Shape;
    use crate::fn3::tensor::Tensor;

    fn run(
        op: OpId,
        inputs: &[&Tensor],
        out: &mut Tensor,
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        let views: Vec<TensorView<'_>> = inputs.iter().map(|v| v.as_view().unwrap()).collect();
        let mut out_view = out.as_mut_view().unwrap();
        CpuBackend::new().execute_op(op, &views, std::slice::from_mut(&mut out_view), params)
    }

    #[test]
    fn add_broadcasts_a_trailing_vector() {
        let lhs =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Tensor::from_f32(Shape::from_dims([3]), &[10.0, 20.0, 30.0]).unwrap();
        let mut out = Tensor::zeros(Shape::from_dims([2, 3]), DType::F32, DeviceId::Cpu).unwrap();
        run(OpId::Add, &[&lhs, &rhs], &mut out, &OpParams::default()).unwrap();
        assert_eq!(
            out.as_f32_slice().unwrap(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn unsupported_ops_are_reported_not_approximated() {
        let input = Tensor::from_f32(Shape::from_dims([2, 3]), &[0.0; 6]).unwrap();
        let mut out = Tensor::zeros(Shape::from_dims([2, 3]), DType::F32, DeviceId::Cpu).unwrap();
        // `MaskedAttention` is reserved for #61 and must stay unsupported.
        let error = run(
            OpId::MaskedAttention,
            &[&input],
            &mut out,
            &OpParams::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::UnsupportedOp {
                op: OpId::MaskedAttention,
                backend: "cpu",
                ..
            }
        ));
        assert!(error.to_string().contains("masked_attention"), "{error}");
        assert_eq!(out.as_f32_slice().unwrap(), &[0.0; 6]);
    }

    #[test]
    fn device_mismatch_is_rejected() {
        let lhs = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
        let storage = crate::fn3::storage::Storage::from_bytes(
            &[1.0f32.to_ne_bytes(), 2.0f32.to_ne_bytes()].concat(),
        )
        .unwrap();
        let meta = crate::fn3::tensor::TensorMeta::contiguous(
            Shape::from_dims([2]),
            DType::F32,
            DeviceId::Vulkan(crate::fn3::device::PhysicalDeviceId::new(0)),
        )
        .unwrap();
        let rhs = Tensor::from_storage(storage, meta).unwrap();
        let mut out = Tensor::zeros(Shape::from_dims([2]), DType::F32, DeviceId::Cpu).unwrap();
        let lhs_view = lhs.as_view().unwrap();
        let rhs_view = rhs.as_view().unwrap();
        let mut out_view = out.as_mut_view().unwrap();
        let error = CpuBackend::new()
            .execute_op(
                OpId::Add,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &OpParams::default(),
            )
            .unwrap_err();
        assert!(matches!(error, Fn3Error::DeviceMismatch { .. }));
        assert_eq!(error.op(), Some(OpId::Add));
    }

    #[test]
    fn matmul_is_rank_two_and_strided() {
        let lhs =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs =
            Tensor::from_f32(Shape::from_dims([3, 2]), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let mut out = Tensor::zeros(Shape::from_dims([2, 2]), DType::F32, DeviceId::Cpu).unwrap();
        run(OpId::MatMul, &[&lhs, &rhs], &mut out, &OpParams::default()).unwrap();
        assert_eq!(out.as_f32_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn softmax_is_supported_and_masked_attention_is_not() {
        let backend = CpuBackend::new();
        assert!(backend.supports(OpId::Softmax).is_full());
        assert!(backend.supports(OpId::LayerNorm).is_full());
        assert!(!backend.supports(OpId::MaskedAttention).is_supported());
        assert!(backend
            .supports(OpId::MaskedAttention)
            .reason()
            .unwrap()
            .contains("#61"));
    }

    #[test]
    fn capabilities_report_the_alpha_limits() {
        let capabilities = cpu_capabilities();
        assert!(capabilities.supports_dtype(DType::F32));
        assert!(capabilities.features.broadcasting);
        // The fused attention flag stays false until #61 lands the id.
        assert!(!capabilities.features.masked_attention);
        assert_eq!(capabilities.max_workgroup_size, [1, 1, 1]);
        assert_eq!(capabilities.max_shape_rank, 8);
    }

    #[test]
    fn view_is_not_needed_for_arity_errors() {
        let backend = CpuBackend::new();
        assert!(matches!(
            check_arity(OpId::Add, 1, 1, 2, 1),
            Err(Fn3Error::ArityMismatch { .. })
        ));
        assert!(backend.supports(OpId::Add).is_full());
    }

    #[test]
    fn flexible_arity_reports_the_actual_counts() {
        assert!(check_flexible_arity(OpId::Split, 2, 2, &[1, 2], 1).is_ok());
        assert!(check_flexible_arity(OpId::Split, 3, 2, &[1, 2], 1).is_err());
        assert!(check_flexible_arity(OpId::Split, 1, 0, &[1, 2], 1).is_err());
    }
}
