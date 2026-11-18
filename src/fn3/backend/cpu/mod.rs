//! The alpha CPU reference backend.
//!
//! # What is implemented
//!
//! Two ops, end to end, through the [`Backend`] contract:
//!
//! | Op | Contract |
//! |---|---|
//! | [`OpId::Add`] | `F32`, right-aligned NumPy broadcasting, strided views |
//! | [`OpId::MatMul`] | `F32`, rank-2 `[M, K] x [K, N] -> [M, N]`, strided views |
//!
//! Everything else reports [`Support::Unsupported`]. This is intentional: the
//! backend is the working pattern downstream workers copy (FN3-07 / #53), not a
//! partial library of approximations.
//!
//! # Numerical contract
//!
//! Both kernels are scalar reference implementations with a documented order:
//!
//! - `Add` evaluates element `(i, j, ...)` as `lhs[...] + rhs[...]` with
//!   right-aligned broadcasting.
//! - `MatMul` accumulates in `f32` in ascending `k`. This is the reference
//!   order every faster kernel must be tested against; changing it changes
//!   results in the last bits.
//!
//! Kernels index storage through metadata (`offset + sum(coord * stride)`), so
//! a transposed or sliced view needs no materialisation. Performance work
//! (blocking, SIMD, threading) belongs to FN3-07 and must keep these results
//! bit-comparable or publish a documented tolerance.

use std::fmt;

use super::{check_arity, Backend, Capabilities, CapabilityFeatures, OpParams, Support};
use crate::fn3::device::DeviceId;
use crate::fn3::dtype::DType;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMut, TensorView};

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

    fn add(
        &self,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
    ) -> Result<(), Fn3Error> {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        lhs.meta().require_dtype(Some(OpId::Add), DType::F32)?;
        rhs.meta().require_dtype(Some(OpId::Add), DType::F32)?;
        outputs[0]
            .meta()
            .require_dtype(Some(OpId::Add), DType::F32)?;

        let shape = lhs.shape().broadcast_with(rhs.shape(), OpId::Add)?;
        if *outputs[0].shape() != shape {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(OpId::Add),
                expected: format!("broadcast result {shape}"),
                actual: outputs[0].shape().to_string(),
            });
        }
        let numel = shape.numel()?;
        if numel == 0 {
            return Ok(());
        }

        let rank = shape.rank();
        let lhs_layout = BroadcastLayout::new(&shape, lhs);
        let rhs_layout = BroadcastLayout::new(&shape, rhs);
        let out_offset = outputs[0].meta().offset();
        let out_strides = outputs[0].strides().as_slice().to_vec();

        let lhs_data = lhs.storage_slice::<f32>()?;
        let rhs_data = rhs.storage_slice::<f32>()?;
        let out_data = outputs[0].storage_slice_mut::<f32>()?;

        let mut coordinates = vec![0u64; rank];
        for linear in 0..numel {
            decompose(linear, shape.dims(), &mut coordinates);
            let left =
                lhs_data[element_index(lhs_layout.offset, &lhs_layout.strides, &coordinates)];
            let right =
                rhs_data[element_index(rhs_layout.offset, &rhs_layout.strides, &coordinates)];
            out_data[element_index(out_offset, &out_strides, &coordinates)] = left + right;
        }
        Ok(())
    }

    fn matmul(
        &self,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
    ) -> Result<(), Fn3Error> {
        let lhs = &inputs[0];
        let rhs = &inputs[1];
        for view in [lhs, rhs] {
            if view.shape().rank() != 2 {
                return Err(Fn3Error::UnsupportedOp {
                    op: OpId::MatMul,
                    backend: CPU_BACKEND_NAME,
                    reason: format!(
                        "the alpha CPU reference implements rank-2 matmul only; {} has rank {} \
                         (batched matmul is owned by FN3-07 / #53)",
                        view.meta(),
                        view.shape().rank()
                    ),
                });
            }
        }
        lhs.meta().require_dtype(Some(OpId::MatMul), DType::F32)?;
        rhs.meta().require_dtype(Some(OpId::MatMul), DType::F32)?;
        outputs[0]
            .meta()
            .require_dtype(Some(OpId::MatMul), DType::F32)?;
        if outputs[0].shape().rank() != 2 {
            return Err(Fn3Error::RankMismatch {
                op: Some(OpId::MatMul),
                expected: 2,
                actual: outputs[0].shape().rank(),
            });
        }

        let rows = lhs.shape().dims()[0];
        let inner = lhs.shape().dims()[1];
        let rhs_inner = rhs.shape().dims()[0];
        let columns = rhs.shape().dims()[1];
        if inner != rhs_inner {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(OpId::MatMul),
                expected: format!("rhs inner dimension {inner} (lhs is {})", lhs.shape()),
                actual: format!("rhs {} has inner dimension {rhs_inner}", rhs.shape()),
            });
        }
        let expected_output = Shape::from_dims([rows, columns]);
        if *outputs[0].shape() != expected_output {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(OpId::MatMul),
                expected: expected_output.to_string(),
                actual: outputs[0].shape().to_string(),
            });
        }

        let out_offset = outputs[0].meta().offset();
        let out_strides = outputs[0].strides().as_slice().to_vec();
        let lhs_offset = lhs.meta().offset();
        let lhs_strides = lhs.strides().as_slice();
        let rhs_offset = rhs.meta().offset();
        let rhs_strides = rhs.strides().as_slice();

        let lhs_data = lhs.storage_slice::<f32>()?;
        let rhs_data = rhs.storage_slice::<f32>()?;
        let out_data = outputs[0].storage_slice_mut::<f32>()?;

        for row in 0..rows {
            for column in 0..columns {
                let mut accumulator = 0.0f32;
                for step in 0..inner {
                    let left = lhs_data[lhs_offset
                        + row as usize * lhs_strides[0] as usize
                        + step as usize * lhs_strides[1] as usize];
                    let right = rhs_data[rhs_offset
                        + step as usize * rhs_strides[0] as usize
                        + column as usize * rhs_strides[1] as usize];
                    accumulator += left * right;
                }
                out_data[out_offset
                    + row as usize * out_strides[0] as usize
                    + column as usize * out_strides[1] as usize] = accumulator;
            }
        }
        Ok(())
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
        match op {
            OpId::Add | OpId::MatMul => Support::Full,
            other => Support::Unsupported {
                reason: format!(
                    "the alpha CPU reference implements add (broadcast f32) and rank-2 matmul only; \
                     {} is owned by FN3-07 (#53)",
                    other.name()
                ),
            },
        }
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
        }
        for output in outputs.iter() {
            output.meta().require_device(Some(op), self.device())?;
        }
        match op {
            OpId::Add => {
                check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
                require_no_attributes(op, params)?;
                self.add(inputs, outputs)
            }
            OpId::MatMul => {
                check_arity(op, inputs.len(), outputs.len(), 2, 1)?;
                require_no_attributes(op, params)?;
                self.matmul(inputs, outputs)
            }
            other => Err(Fn3Error::UnsupportedOp {
                op: other,
                backend: CPU_BACKEND_NAME,
                reason: "not implemented in the alpha CPU reference".to_string(),
            }),
        }
    }
}

impl fmt::Display for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{CPU_BACKEND_NAME}")
    }
}

/// Reject attributes an op does not define, instead of silently ignoring them.
fn require_no_attributes(op: OpId, params: &OpParams) -> Result<(), Fn3Error> {
    if params.axes.is_empty() && !params.keep_dims && params.epsilon == 0.0 {
        Ok(())
    } else {
        Err(Fn3Error::InvalidShape {
            context: "cpu backend attributes",
            detail: format!(
                "op {op} accepts no attributes, but the plan supplies axes={:?}, keep_dims={}, epsilon={}",
                params.axes, params.keep_dims, params.epsilon
            ),
        })
    }
}

/// Effective strides of one operand promoted to the broadcast result shape.
struct BroadcastLayout {
    offset: usize,
    strides: Vec<i64>,
}

impl BroadcastLayout {
    fn new(result: &Shape, operand: &TensorView<'_>) -> BroadcastLayout {
        debug_assert!(
            result.rank() >= operand.shape().rank(),
            "broadcast result rank must be at least the operand rank"
        );
        let result_rank = result.rank();
        let operand_rank = operand.shape().rank();
        let mut strides = vec![0i64; result_rank];
        for axis in (result_rank - operand_rank)..result_rank {
            let operand_axis = axis - (result_rank - operand_rank);
            let operand_extent = operand.shape().dims()[operand_axis];
            strides[axis] = if operand_extent == 1 && result.dims()[axis] != 1 {
                0
            } else {
                operand.strides().as_slice()[operand_axis]
            };
        }
        BroadcastLayout {
            offset: operand.meta().offset(),
            strides,
        }
    }
}

/// Row-major coordinates of a linear index.
fn decompose(mut linear: u64, dims: &[u64], coordinates: &mut [u64]) {
    for axis in (0..dims.len()).rev() {
        let extent = dims[axis];
        coordinates[axis] = linear % extent;
        linear /= extent;
    }
}

/// Element index for a validated (non-negative) layout.
fn element_index(offset: usize, strides: &[i64], coordinates: &[u64]) -> usize {
    let mut index = offset;
    for (axis, &coordinate) in coordinates.iter().enumerate() {
        index += coordinate as usize * strides[axis] as usize;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fn3::shape::Shape;
    use crate::fn3::tensor::Tensor;

    fn add_via_backend(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        let shape = lhs.shape().broadcast_with(rhs.shape(), OpId::Add).unwrap();
        let mut out = Tensor::zeros(shape, DType::F32, DeviceId::Cpu).unwrap();
        let lhs_view = lhs.as_view().unwrap();
        let rhs_view = rhs.as_view().unwrap();
        let mut out_view = out.as_mut_view().unwrap();
        CpuBackend::new()
            .execute_op(
                OpId::Add,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &OpParams::default(),
            )
            .unwrap();
        out
    }

    #[test]
    fn add_broadcasts_a_trailing_vector() {
        let lhs =
            Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Tensor::from_f32(Shape::from_dims([3]), &[10.0, 20.0, 30.0]).unwrap();
        let out = add_via_backend(&lhs, &rhs);
        assert_eq!(
            out.as_f32_slice().unwrap(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn unsupported_ops_are_reported_not_approximated() {
        let input = Tensor::from_f32(Shape::from_dims([2, 3]), &[0.0; 6]).unwrap();
        let mut out = Tensor::zeros(Shape::from_dims([2, 3]), DType::F32, DeviceId::Cpu).unwrap();
        let view = input.as_view().unwrap();
        let mut out_view = out.as_mut_view().unwrap();
        let error = CpuBackend::new()
            .execute_op(
                OpId::Softmax,
                &[view],
                std::slice::from_mut(&mut out_view),
                &OpParams::default().with_axes([-1]),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::UnsupportedOp {
                op: OpId::Softmax,
                ..
            }
        ));
        assert!(error.to_string().contains("softmax"), "{error}");
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
        let lhs_view = lhs.as_view().unwrap();
        let rhs_view = rhs.as_view().unwrap();
        let mut out_view = out.as_mut_view().unwrap();
        CpuBackend::new()
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &OpParams::default(),
            )
            .unwrap();
        assert_eq!(out.as_f32_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn capabilities_report_the_alpha_limits() {
        let capabilities = cpu_capabilities();
        assert!(capabilities.supports_dtype(DType::F32));
        assert!(capabilities.features.broadcasting);
        assert!(!capabilities.features.masked_attention);
        assert_eq!(capabilities.max_workgroup_size, [1, 1, 1]);
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
}
