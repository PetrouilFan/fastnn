use crate::autograd::{self, Edge};
use crate::impl_cpu_fast_path;
use crate::impl_scalar_op;
use crate::ir::node::{DimExpr, IrDType, TensorType};
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
        let Storage::Cpu(cpu_storage) = storage else { unreachable!() };
        let out_data = Arc::make_mut(&mut cpu_storage.data);
        let a_ptr = t.data_ptr_f32();
        let out_ptr = out_data.as_mut_ptr() as *mut f32;
        Some((numel, a_ptr, out_ptr, output))
    } else {
        None
    }
}

impl TensorImpl {
    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        if dim >= self.ndim() {
            panic!(
                "slice: dimension {} is out of range for {}-dimensional tensor",
                dim,
                self.ndim()
            );
        }

        let size = self.sizes[dim];
        let start = if start < 0 { size + start } else { start };
        let end = if end < 0 { size + end } else { end };
        let start = start.max(0) as usize;
        let end = (end.min(size)) as usize;

        if start > end {
            panic!(
                "slice: invalid range [{}, {}) for dimension {} of size {}",
                start, end, dim, size
            );
        }

        let mut sizes = self.sizes.clone();
        let numel = ((end as i64 - start as i64 + step - 1) / step) as usize;
        sizes[dim] = numel as i64;

        let storage_offset = self.storage_offset + (start as i64) * self.strides[dim];

        let mut strides = self.strides.clone();
        strides[dim] *= step;

        let mut t = self.new_view_from(sizes, strides, storage_offset);
        t.autograd_meta = None;
        t.requires_grad = false;
        t.into()
    }
}

impl Tensor {
    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        let output = self.inner.slice(dim, start, end, step);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("SliceBackward", inputs))
        } else {
            output
        }
    }

    pub fn maximum(&self, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.maximum(&ins[0], &ins[1])])
            .expect("Tensor::maximum: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = autograd::make_edges(self, other);
            let inputs = vec![self.clone(), other.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("MaximumBackward", inputs))
        } else {
            output
        }
    }

    impl_scalar_op!(gt_scalar, _CMP_GT_OQ, >);

    pub fn sign(&self) -> Tensor {
        Tensor::exec_aot(&[self], |g, ins| vec![g.sign(&ins[0])])
            .expect("Tensor::sign: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn minimum(&self, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.minimum(&ins[0], &ins[1])])
            .expect("Tensor::minimum: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = autograd::make_edges(self, other);
            let inputs = vec![self.clone(), other.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("MinimumBackward", inputs))
        } else {
            output
        }
    }

    pub fn ge_tensor(&self, other: &Tensor) -> Tensor {
        // ge(a,b) = not(b > a) = not(gt_scalar(b - a, 0))
        Tensor::exec_aot(&[self, other], |g, ins| {
            let diff = g.sub(&ins[1], &ins[0]); // b - a
            let zero = g.constant(
                &0.0f32.to_le_bytes(),
                TensorType::new(vec![], IrDType::F32),
            );
            let gt = g.gt_scalar(&diff, &zero); // b - a > 0  (i.e. b > a)
            vec![g.logical_not(&gt)] // not(b > a) = a >= b
        })
        .expect("Tensor::ge_tensor: AOT execution failed")
        .into_iter()
        .next()
        .unwrap()
    }

    pub fn le_tensor(&self, other: &Tensor) -> Tensor {
        // le(a,b) = not(a > b) = not(gt_scalar(a - b, 0))
        Tensor::exec_aot(&[self, other], |g, ins| {
            let diff = g.sub(&ins[0], &ins[1]); // a - b
            let zero = g.constant(
                &0.0f32.to_le_bytes(),
                TensorType::new(vec![], IrDType::F32),
            );
            let gt = g.gt_scalar(&diff, &zero); // a - b > 0 (i.e. a > b)
            vec![g.logical_not(&gt)] // not(a > b) = a <= b
        })
        .expect("Tensor::le_tensor: AOT execution failed")
        .into_iter()
        .next()
        .unwrap()
    }

    impl_scalar_op!(lt_scalar, _CMP_LT_OQ, <);

    impl_scalar_op!(eq_scalar, _CMP_EQ_OQ, ==);

    impl_cpu_fast_path!(add_scalar, _mm256_add_ps, +, AddScalarBackward);

    impl_cpu_fast_path!(div_scalar, _mm256_div_ps, /, DivScalarBackward);

    pub fn logical_not(&self) -> Tensor {
        Tensor::exec_aot(&[self], |g, ins| vec![g.logical_not(&ins[0])])
            .expect("Tensor::logical_not: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn argmax(&self, dim: Option<usize>) -> Tensor {
        let result = Tensor::exec_aot(&[self], |g, ins| {
            let axis = dim.map(|d| DimExpr::Known(d as u64));
            let out = g.argmax(&ins[0], axis);
            vec![out]
        })
        .expect("Tensor::argmax: AOT execution failed");
        result.into_iter().next().unwrap()
    }

    pub fn cat(tensors: &[Tensor], dim: i32) -> Tensor {
        if tensors.is_empty() {
            panic!("Tensor::cat: empty input");
        }
        let ndim = tensors[0].ndim();
        let dim = if dim < 0 { ndim as i32 + dim } else { dim } as usize;
        if dim >= ndim {
            panic!("Tensor::cat: dimension out of range");
        }
        for t in tensors {
            if t.ndim() != ndim {
                panic!("Tensor::cat: tensors must have same number of dimensions");
            }
            for d in 0..ndim {
                if d != dim && tensors[0].inner.sizes[d] != t.inner.sizes[d] {
                    panic!("Tensor::cat: tensor sizes mismatch at dimension {}", d);
                }
            }
        }

        let inputs: Vec<&Tensor> = tensors.iter().collect();
        let output = Tensor::exec_aot(&inputs, |g, ins| {
            let refs: Vec<_> = ins.iter().collect();
            vec![g.concat(&refs, dim)]
        })
        .expect("Tensor::cat: AOT execution failed")
        .into_iter()
        .next()
        .unwrap();

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
            Self::attach_grad_fn(output, autograd::make_node_info("CatBackward", inputs))
        } else {
            output
        }
    }

    pub fn stack(tensors: &[Tensor], dim: i32) -> Tensor {
        if tensors.is_empty() {
            panic!("stack: need at least one tensor");
        }

        let first_shape = tensors[0].shape();
        let ndim = first_shape.len();

        let dim = if dim < 0 {
            (ndim as i32 + dim + 1) as usize
        } else {
            dim as usize
        };
        if dim > ndim {
            panic!("stack: dimension out of range");
        }

        for t in tensors {
            if t.shape() != first_shape {
                panic!("stack: tensors must have same shape");
            }
        }

        let unsqueezed: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(dim)).collect();

        Tensor::cat(&unsqueezed, dim as i32)
    }

    pub fn repeat(&self, repeats: &[i64]) -> Tensor {
        if repeats.len() < self.ndim() {
            panic!("repeat: number of repeats must be >= number of dimensions");
        }
        let reps: Vec<usize> = repeats.iter().map(|&r| r as usize).collect();
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.repeat(&ins[0], &reps)])
            .expect("Tensor::repeat: AOT failed")
            .into_iter()
            .next()
            .unwrap();

        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("RepeatBackward", inputs))
        } else {
            output
        }
    }

    pub fn where_tensor(&self, condition: &Tensor, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[condition, self, other], |g, ins| {
            vec![g.where_tensor(&ins[0], &ins[1], &ins[2])]
        })
        .expect("Tensor::where_tensor: AOT failed")
        .into_iter()
        .next()
        .unwrap();

        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = autograd::make_edges(self, other);
            let inputs = vec![self.clone(), other.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("WhereBackward", inputs))
        } else {
            output
        }
    }

    pub fn gather(&self, axis: i64, indices: &Tensor) -> Tensor {
        let norm_axis = if axis < 0 {
            (self.ndim() as i64 + axis) as usize
        } else {
            axis as usize
        };
        let output = Tensor::exec_aot(&[self, indices], |g, ins| {
            vec![g.gather(&ins[0], &ins[1], norm_axis)]
        })
        .expect("Tensor::gather: AOT failed")
        .into_iter()
        .next()
        .unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || indices.requires_grad()) {
            let inputs = vec![self.clone(), indices.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("GatherBackward", inputs))
        } else {
            output
        }
    }

    pub fn nonzero(&self) -> Vec<i64> {
        let data = self.as_f32_slice();
        let shape = self.shape_ref();
        let ndim = shape.len();

        if ndim == 0 {
            return Vec::new();
        }

        // Precompute strides for fast flat-index decomposition
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as usize;
        }

        let total = data.len();
        let mut result = Vec::with_capacity(total.saturating_mul(ndim));
        for i in 0..total {
            if data[i] != 0.0 {
                let mut remaining = i;
                for d in 0..ndim {
                    result.push((remaining / strides[d]) as i64);
                    remaining %= strides[d];
                }
            }
        }
        result
    }
}
