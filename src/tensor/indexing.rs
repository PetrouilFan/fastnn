use crate::autograd::{self, Edge};
use crate::error::{FastnnError, FastnnResult};
use crate::impl_cpu_fast_path;
use crate::impl_scalar_op;
use crate::ir::{DimExpr, IrDType, TensorType};
use crate::storage::{DType, Device, Storage};
use std::sync::Arc;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

use super::{Tensor, TensorImpl};

fn cpu_f32_fast_path_setup(t: &Tensor) -> Option<(usize, *const f32, *mut f32, Tensor)> {
    if t.device() == Device::Cpu && t.inner.dtype == DType::F32 && t.is_contiguous() {
        let numel = t.inner.numel() as usize;
        let mut output = Tensor::empty(t.shape(), DType::F32, Device::Cpu);
        let inner = Arc::make_mut(&mut output.inner);
        let storage = Arc::make_mut(&mut inner.storage);
        #[cfg_attr(not(feature = "gpu"), allow(irrefutable_let_patterns))]
        let Storage::Cpu(cpu_storage) = storage
        else {
            unreachable!()
        };
        let out_data = Arc::make_mut(&mut cpu_storage.data);
        let a_ptr = t.data_ptr_f32();
        let out_ptr = out_data.as_mut_ptr() as *mut f32;
        Some((numel, a_ptr, out_ptr, output))
    } else {
        None
    }
}

fn validate_selection_values(left: &Tensor, right: &Tensor, operation: &str) -> FastnnResult<()> {
    if left.device() != right.device() {
        return Err(FastnnError::device(format!(
            "{operation} operands must be on the same device"
        )));
    }
    if left.dtype() != right.dtype() {
        return Err(FastnnError::dtype(format!(
            "{operation} operand dtypes must match"
        )));
    }
    Tensor::broadcast_shapes(&left.shape(), &right.shape())?;
    Ok(())
}

impl TensorImpl {
    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        self.try_slice(dim, start, end, step)
            .expect("Tensor::slice failed")
    }

    pub fn try_slice(&self, dim: usize, start: i64, end: i64, step: i64) -> FastnnResult<Tensor> {
        if dim >= self.ndim() {
            return Err(FastnnError::shape(format!(
                "slice dimension {dim} is out of range for {} dimensions",
                self.ndim()
            )));
        }
        if step <= 0 {
            return Err(FastnnError::InvalidArgument(format!(
                "slice step must be positive, got {step}"
            )));
        }

        let size = self.sizes[dim];
        let start = if start < 0 {
            size.saturating_add(start)
        } else {
            start
        }
        .clamp(0, size);
        let end = if end < 0 {
            size.saturating_add(end)
        } else {
            end
        }
        .clamp(0, size);

        if start > end {
            return Err(FastnnError::OutOfBounds(format!(
                "slice range [{start}, {end}) is reversed for dimension {dim} of size {size}"
            )));
        }

        let mut sizes = self.sizes.clone();
        let span = end - start;
        sizes[dim] = span
            .checked_add(step - 1)
            .ok_or_else(|| FastnnError::Overflow("slice length overflow".into()))?
            / step;

        let storage_offset = start
            .checked_mul(self.strides[dim])
            .and_then(|offset| self.storage_offset.checked_add(offset))
            .ok_or_else(|| FastnnError::Overflow("slice storage offset overflow".into()))?;

        let mut strides = self.strides.clone();
        strides[dim] = strides[dim]
            .checked_mul(step)
            .ok_or_else(|| FastnnError::Overflow("slice stride overflow".into()))?;

        let mut tensor = self.new_view_from(sizes, strides, storage_offset);
        tensor.autograd_meta = None;
        tensor.requires_grad = false;
        Ok(tensor.into())
    }
}

impl Tensor {
    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        self.try_slice(dim, start, end, step)
            .expect("Tensor::slice failed")
    }

    pub fn try_slice(&self, dim: usize, start: i64, end: i64, step: i64) -> FastnnResult<Tensor> {
        let output = self.inner.try_slice(dim, start, end, step)?;

        Ok(if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("SliceBackward", inputs))
        } else {
            output
        })
    }

    pub fn maximum(&self, other: &Tensor) -> Tensor {
        self.try_maximum(other)
            .expect("Tensor::maximum execution failed")
    }

    pub fn try_maximum(&self, other: &Tensor) -> FastnnResult<Tensor> {
        validate_selection_values(self, other, "maximum")?;
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.maximum(&ins[0], &ins[1])])
            .map_err(|error| FastnnError::Internal(error.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| FastnnError::Internal("maximum execution returned no output".into()))?;
        Ok(
            if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
                let inputs = vec![self.clone(), other.clone()];
                Self::attach_grad_fn(output, autograd::make_node_info("MaximumBackward", inputs))
            } else {
                output
            },
        )
    }

    impl_scalar_op!(gt_scalar, _CMP_GT_OQ, >);

    pub fn sign(&self) -> Tensor {
        self.try_sign().expect("Tensor::sign failed")
    }

    pub fn try_sign(&self) -> FastnnResult<Tensor> {
        Tensor::exec_aot(&[self], |g, ins| vec![g.sign(&ins[0])])
            .map_err(|error| FastnnError::Computation(error.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| FastnnError::Internal("sign execution returned no output".into()))
    }

    pub fn minimum(&self, other: &Tensor) -> Tensor {
        self.try_minimum(other)
            .expect("Tensor::minimum execution failed")
    }

    pub fn try_minimum(&self, other: &Tensor) -> FastnnResult<Tensor> {
        validate_selection_values(self, other, "minimum")?;
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.minimum(&ins[0], &ins[1])])
            .map_err(|error| FastnnError::Internal(error.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| FastnnError::Internal("minimum execution returned no output".into()))?;
        Ok(
            if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
                let inputs = vec![self.clone(), other.clone()];
                Self::attach_grad_fn(output, autograd::make_node_info("MinimumBackward", inputs))
            } else {
                output
            },
        )
    }

    pub fn ge_tensor(&self, other: &Tensor) -> Tensor {
        self.try_ge_tensor(other).expect("Tensor::ge_tensor failed")
    }

    pub fn try_ge_tensor(&self, other: &Tensor) -> FastnnResult<Tensor> {
        validate_selection_values(self, other, "ge_tensor")?;
        // ge(a,b) = not(b > a) = not(gt_scalar(b - a, 0))
        Tensor::exec_aot(&[self, other], |g, ins| {
            let diff = g.sub(&ins[1], &ins[0]); // b - a
            let zero = g.constant(&0.0f32.to_le_bytes(), TensorType::new(vec![], IrDType::F32));
            let gt = g.gt_scalar(&diff, &zero); // b - a > 0  (i.e. b > a)
            vec![g.logical_not(&gt)] // not(b > a) = a >= b
        })
        .map_err(|error| FastnnError::Computation(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| FastnnError::Internal("ge_tensor execution returned no output".into()))
    }

    pub fn le_tensor(&self, other: &Tensor) -> Tensor {
        self.try_le_tensor(other).expect("Tensor::le_tensor failed")
    }

    pub fn try_le_tensor(&self, other: &Tensor) -> FastnnResult<Tensor> {
        validate_selection_values(self, other, "le_tensor")?;
        // le(a,b) = not(a > b) = not(gt_scalar(a - b, 0))
        Tensor::exec_aot(&[self, other], |g, ins| {
            let diff = g.sub(&ins[0], &ins[1]); // a - b
            let zero = g.constant(&0.0f32.to_le_bytes(), TensorType::new(vec![], IrDType::F32));
            let gt = g.gt_scalar(&diff, &zero); // a - b > 0 (i.e. a > b)
            vec![g.logical_not(&gt)] // not(a > b) = a <= b
        })
        .map_err(|error| FastnnError::Computation(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| FastnnError::Internal("le_tensor execution returned no output".into()))
    }

    impl_scalar_op!(lt_scalar, _CMP_LT_OQ, <);

    impl_scalar_op!(eq_scalar, _CMP_EQ_OQ, ==);

    impl_cpu_fast_path!(add_scalar, _mm256_add_ps, +, AddScalarBackward);

    impl_cpu_fast_path!(div_scalar, _mm256_div_ps, /, DivScalarBackward);

    pub fn logical_not(&self) -> Tensor {
        self.try_logical_not().expect("Tensor::logical_not failed")
    }

    pub fn try_logical_not(&self) -> FastnnResult<Tensor> {
        Tensor::exec_aot(&[self], |g, ins| vec![g.logical_not(&ins[0])])
            .map_err(|error| FastnnError::Computation(error.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| FastnnError::Internal("logical_not execution returned no output".into()))
    }

    pub fn argmax(&self, dim: Option<usize>) -> Tensor {
        self.try_argmax(dim).expect("Tensor::argmax failed")
    }

    pub fn try_argmax(&self, dim: Option<usize>) -> FastnnResult<Tensor> {
        if let Some(axis) = dim {
            if axis >= self.ndim() {
                return Err(FastnnError::shape(format!(
                    "argmax dimension {axis} is out of range for {} dimensions",
                    self.ndim()
                )));
            }
        }
        let result = Tensor::exec_aot(&[self], |g, ins| {
            let axis = dim.map(|d| DimExpr::Known(d as u64));
            let out = g.argmax(&ins[0], axis);
            vec![out]
        })
        .map_err(|error| FastnnError::Computation(error.to_string()))?;
        result
            .into_iter()
            .next()
            .ok_or_else(|| FastnnError::Internal("argmax execution returned no output".into()))
    }

    pub fn cat(tensors: &[Tensor], dim: i32) -> Tensor {
        Self::try_cat(tensors, dim).expect("Tensor::cat failed")
    }

    pub fn try_cat(tensors: &[Tensor], dim: i32) -> FastnnResult<Tensor> {
        if tensors.is_empty() {
            return Err(FastnnError::shape("cat requires at least one tensor"));
        }
        let ndim = tensors[0].ndim();
        let normalized_dim = if dim < 0 {
            ndim as i64 + i64::from(dim)
        } else {
            i64::from(dim)
        };
        if normalized_dim < 0 || normalized_dim as usize >= ndim {
            return Err(FastnnError::shape(format!(
                "cat dimension {dim} is out of range for {ndim} dimensions"
            )));
        }
        let dim = normalized_dim as usize;
        for t in tensors {
            if t.ndim() != ndim {
                return Err(FastnnError::shape("cat tensors must have the same rank"));
            }
            if t.dtype() != tensors[0].dtype() {
                return Err(FastnnError::dtype("cat tensors must have the same dtype"));
            }
            if t.device() != tensors[0].device() {
                return Err(FastnnError::device(
                    "cat tensors must be on the same device",
                ));
            }
            for d in 0..ndim {
                if d != dim && tensors[0].inner.sizes[d] != t.inner.sizes[d] {
                    return Err(FastnnError::shape(format!(
                        "cat tensor sizes differ at dimension {d}"
                    )));
                }
            }
        }

        let inputs: Vec<&Tensor> = tensors.iter().collect();
        let output = Tensor::exec_aot(&inputs, |g, ins| {
            let refs: Vec<_> = ins.iter().collect();
            vec![g.concat(&refs, dim)]
        })
        .map_err(|error| FastnnError::Computation(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| FastnnError::Internal("cat execution returned no output".into()))?;

        if autograd::is_grad_enabled() && tensors.iter().any(|t| t.requires_grad()) {
            let _split_sizes: Vec<usize> = tensors
                .iter()
                .map(|t| t.inner.sizes[dim] as usize)
                .collect();
            let mut edges = Vec::new();
            for (i, t) in tensors.iter().enumerate() {
                if let Some(node) = t.grad_fn() {
                    edges.push(Edge(node, i));
                }
            }
            let inputs = tensors.to_vec();
            Ok(Self::attach_grad_fn(
                output,
                autograd::make_node_info("CatBackward", inputs),
            ))
        } else {
            Ok(output)
        }
    }

    pub fn stack(tensors: &[Tensor], dim: i32) -> Tensor {
        Self::try_stack(tensors, dim).expect("Tensor::stack failed")
    }

    pub fn try_stack(tensors: &[Tensor], dim: i32) -> FastnnResult<Tensor> {
        if tensors.is_empty() {
            return Err(FastnnError::shape("stack requires at least one tensor"));
        }

        let first_shape = tensors[0].shape();
        let ndim = first_shape.len();

        let normalized_dim = if dim < 0 {
            ndim as i64 + i64::from(dim) + 1
        } else {
            i64::from(dim)
        };
        if normalized_dim < 0 || normalized_dim as usize > ndim {
            return Err(FastnnError::shape(format!(
                "stack dimension {dim} is out of range for output rank {}",
                ndim + 1
            )));
        }
        let dim = normalized_dim as usize;

        for t in tensors {
            if t.shape() != first_shape {
                return Err(FastnnError::shape("stack tensors must have the same shape"));
            }
        }

        let unsqueezed: Vec<Tensor> = tensors
            .iter()
            .map(|tensor| tensor.try_unsqueeze(dim))
            .collect::<FastnnResult<_>>()?;

        Tensor::try_cat(&unsqueezed, dim as i32)
    }

    pub fn repeat(&self, repeats: &[i64]) -> Tensor {
        self.try_repeat(repeats).expect("Tensor::repeat failed")
    }

    pub fn try_repeat(&self, repeats: &[i64]) -> FastnnResult<Tensor> {
        if repeats.len() < self.ndim() {
            return Err(FastnnError::shape(format!(
                "repeat requires at least {} repeat values, got {}",
                self.ndim(),
                repeats.len()
            )));
        }
        if repeats.iter().any(|&repeat| repeat < 0) {
            return Err(FastnnError::shape(format!(
                "repeat values must be non-negative, got {repeats:?}"
            )));
        }
        let reps: Vec<usize> = repeats.iter().map(|&r| r as usize).collect();
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.repeat(&ins[0], &reps)])
            .map_err(|error| FastnnError::Computation(error.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| FastnnError::Internal("repeat execution returned no output".into()))?;

        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            Ok(Self::attach_grad_fn(
                output,
                autograd::make_node_info("RepeatBackward", inputs),
            ))
        } else {
            Ok(output)
        }
    }

    pub fn where_tensor(&self, condition: &Tensor, other: &Tensor) -> Tensor {
        self.try_where_tensor(condition, other)
            .expect("Tensor::where_tensor execution failed")
    }

    pub fn try_where_tensor(&self, condition: &Tensor, other: &Tensor) -> FastnnResult<Tensor> {
        validate_selection_values(self, other, "where")?;
        if condition.device() != self.device() {
            return Err(FastnnError::device(
                "where condition and values must be on the same device",
            ));
        }
        Tensor::broadcast_shapes(&condition.shape(), &self.shape())?;
        Tensor::broadcast_shapes(&condition.shape(), &other.shape())?;
        let output = Tensor::exec_aot(&[condition, self, other], |g, ins| {
            vec![g.where_tensor(&ins[0], &ins[1], &ins[2])]
        })
        .map_err(|error| FastnnError::Internal(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| FastnnError::Internal("where execution returned no output".into()))?;

        Ok(
            if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
                let inputs = vec![self.clone(), other.clone()];
                Self::attach_grad_fn(output, autograd::make_node_info("WhereBackward", inputs))
            } else {
                output
            },
        )
    }

    pub fn gather(&self, axis: i64, indices: &Tensor) -> Tensor {
        self.try_gather(axis, indices)
            .expect("Tensor::gather failed")
    }

    pub fn try_gather(&self, axis: i64, indices: &Tensor) -> FastnnResult<Tensor> {
        let rank = self.ndim() as i64;
        let normalized_axis = if axis < 0 { rank + axis } else { axis };
        if normalized_axis < 0 || normalized_axis >= rank {
            return Err(FastnnError::shape(format!(
                "gather axis {axis} is out of range for {rank} dimensions"
            )));
        }
        if self.device() != indices.device() {
            return Err(FastnnError::device(
                "gather data and indices must be on the same device",
            ));
        }
        if indices.dtype() != DType::F32 {
            return Err(FastnnError::dtype(
                "gather currently requires F32 index tensors",
            ));
        }
        let normalized_axis = normalized_axis as usize;
        let output = Tensor::exec_aot(&[self, indices], |g, ins| {
            vec![g.gather(&ins[0], &ins[1], normalized_axis)]
        })
        .map_err(|error| FastnnError::Computation(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| FastnnError::Internal("gather execution returned no output".into()))?;
        Ok(
            if autograd::is_grad_enabled() && (self.requires_grad() || indices.requires_grad()) {
                let inputs = vec![self.clone(), indices.clone()];
                Self::attach_grad_fn(output, autograd::make_node_info("GatherBackward", inputs))
            } else {
                output
            },
        )
    }

    pub fn nonzero(&self) -> Vec<i64> {
        self.try_nonzero().expect("Tensor::nonzero failed")
    }

    pub fn try_nonzero(&self) -> FastnnResult<Vec<i64>> {
        let cpu_t = if !self.is_contiguous() {
            self.try_contiguous()?
        } else {
            self.clone()
        };
        let data = cpu_t.try_as_f32_slice()?;
        let shape = cpu_t.shape_ref();
        let ndim = shape.len();

        if ndim == 0 {
            return Ok(Vec::new());
        }

        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            let next = usize::try_from(shape[i + 1])
                .map_err(|_| FastnnError::shape("nonzero shape contains a negative dimension"))?;
            strides[i] = strides[i + 1]
                .checked_mul(next)
                .ok_or_else(|| FastnnError::Overflow("nonzero stride overflow".into()))?;
        }

        let capacity = data
            .len()
            .checked_mul(ndim)
            .ok_or_else(|| FastnnError::Overflow("nonzero output capacity overflow".into()))?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(capacity)
            .map_err(|error| FastnnError::Allocation(error.to_string()))?;
        for (i, &value) in data.iter().enumerate() {
            if value != 0.0 {
                let mut remaining = i;
                for &stride in &strides {
                    result.push(
                        i64::try_from(remaining / stride).map_err(|_| {
                            FastnnError::Overflow("nonzero index exceeds i64".into())
                        })?,
                    );
                    remaining %= stride;
                }
            }
        }
        Ok(result)
    }
}
