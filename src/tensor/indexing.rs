use crate::autograd;
use crate::dispatcher::{device_to_dispatch_key, dispatch};
use crate::storage::Device;
use smallvec::SmallVec;
use std::sync::Arc;

use super::{Tensor, TensorImpl};

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
        t.into()
    }
}

impl Tensor {
    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        let output = self.inner.slice(dim, start, end, step);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::SliceBackward::new(
                self.clone(),
                dim,
                start,
                end,
                step,
                edges,
            ));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn maximum(&self, other: &Tensor) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("maximum", dispatch_key, &[self, other]);
        result[0].clone()
    }

    pub fn gt_scalar(&self, threshold: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "gt_scalar",
            dispatch_key,
            &[self, &Tensor::from_scalar(threshold)],
        );
        result[0].clone()
    }

    pub fn sign(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("sign", dispatch_key, &[self]);
        result[0].clone()
    }

    pub fn minimum(&self, other: &Tensor) -> Tensor {
        let dispatch_key = match (self.device(), other.device()) {
            (Device::Wgpu(id), _) => device_to_dispatch_key(Device::Wgpu(id)),
            (_, Device::Wgpu(id)) => device_to_dispatch_key(Device::Wgpu(id)),
            _ => device_to_dispatch_key(Device::Cpu),
        };
        let result = dispatch("minimum", dispatch_key, &[self, other]);
        result[0].clone()
    }

    pub fn ge_tensor(&self, other: &Tensor) -> Tensor {
        let numel = self.numel() as usize;
        let self_data = self.as_f32_slice();
        let other_data = other.as_f32_slice();
        let mut output_data = vec![0.0f32; numel];
        for i in 0..numel {
            output_data[i] = if self_data[i] >= other_data[i] {
                1.0
            } else {
                0.0
            };
        }
        Tensor::from_vec(output_data, self.shape())
    }

    pub fn le_tensor(&self, other: &Tensor) -> Tensor {
        let numel = self.numel() as usize;
        let self_data = self.as_f32_slice();
        let other_data = other.as_f32_slice();
        let mut output_data = vec![0.0f32; numel];
        for i in 0..numel {
            output_data[i] = if self_data[i] <= other_data[i] {
                1.0
            } else {
                0.0
            };
        }
        Tensor::from_vec(output_data, self.shape())
    }

    pub fn lt_scalar(&self, threshold: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "lt_scalar",
            dispatch_key,
            &[self, &Tensor::from_scalar(threshold)],
        );
        result[0].clone()
    }

    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "add_scalar",
            dispatch_key,
            &[self, &Tensor::from_scalar(scalar)],
        );
        result[0].clone()
    }

    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "div_scalar",
            dispatch_key,
            &[self, &Tensor::from_scalar(scalar)],
        );
        result[0].clone()
    }

    pub fn logical_not(&self) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch("logical_not", dispatch_key, &[self]);
        result[0].clone()
    }

    pub fn cat(tensors: &[Tensor], dim: i32) -> Tensor {
        if tensors.is_empty() {
            panic!("cat: need at least one tensor");
        }
        let ndim = tensors[0].ndim();
        let dim = if dim < 0 { ndim as i32 + dim } else { dim } as usize;
        if dim >= ndim {
            panic!("cat: dimension out of range");
        }
        let mut output_shape: SmallVec<[i64; 8]> = tensors[0].inner.sizes.clone();
        let mut total_dim_size: i64 = 0;
        for t in tensors {
            if t.ndim() != ndim {
                panic!("cat: tensors must have same number of dimensions");
            }
            total_dim_size += t.inner.sizes[dim];
            for d in 0..ndim {
                if d != dim && output_shape[d] != t.inner.sizes[d] {
                    panic!("cat: tensor sizes mismatch at dimension {}", d);
                }
            }
        }
        output_shape[dim] = total_dim_size;
        let numel: i64 = output_shape.iter().product();
        let mut output_data = vec![0.0f32; numel as usize];
        let mut out_strides: SmallVec<[i64; 8]> = SmallVec::new();
        let mut s = 1i64;
        for d in (0..ndim).rev() {
            out_strides.push(s);
            s *= output_shape[d];
        }
        out_strides.reverse();
        let mut out_offset = 0i64;
        for t in tensors {
            let t_data = t.as_f32_slice();
            let t_strides = &t.inner.strides;
            let t_sizes = &t.inner.sizes;
            let mut indices = vec![0i64; ndim];
            let dim_size = t.inner.sizes[dim];
            for _ in 0..t.numel() as usize {
                let mut t_lin = t.inner.storage_offset;
                let mut o_lin = out_offset;
                for d in 0..ndim {
                    t_lin += indices[d] * t_strides[d];
                    o_lin += indices[d] * out_strides[d];
                }
                output_data[o_lin as usize] = t_data[t_lin as usize];
                for d in (0..ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < t_sizes[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            out_offset += dim_size * out_strides[dim];
        }
        Tensor::from_vec(output_data, output_shape.into_vec())
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
        let mut new_shape: SmallVec<[i64; 8]> = SmallVec::new();
        let mut total: i64 = 1;
        for (i, &r) in repeats.iter().enumerate() {
            let orig = if i < repeats.len() - self.ndim() {
                1
            } else {
                self.inner.sizes[i - (repeats.len() - self.ndim())]
            };
            new_shape.push(orig * r);
            total *= orig * r;
        }
        let mut output_data = vec![0.0f32; total as usize];
        let src = self.to_cpu();
        let src_data = src.as_f32_slice();
        let src_strides = &self.inner.strides;
        let src_sizes = &self.inner.sizes;
        let ndim = self.ndim();
        let offset = repeats.len() - ndim;
        let mut src_indices = vec![0i64; ndim];
        for out_idx in 0..total as usize {
            let mut rep_indices: Vec<i64> = Vec::with_capacity(repeats.len());
            let mut remaining = out_idx as i64;
            for d in (0..repeats.len()).rev() {
                let dim_size = new_shape[d];
                rep_indices.push(remaining % dim_size);
                remaining /= dim_size;
            }
            rep_indices.reverse();
            for d in 0..ndim {
                src_indices[d] = rep_indices[d + offset] % src_sizes[d];
            }
            let mut src_lin = self.inner.storage_offset;
            for d in 0..ndim {
                src_lin += src_indices[d] * src_strides[d];
            }
            output_data[out_idx] = src_data[src_lin as usize];
        }
        Tensor::from_vec(output_data, new_shape.into_vec())
    }

    pub fn where_tensor(&self, condition: &Tensor, other: &Tensor) -> Tensor {
        let numel = self.numel() as usize;
        let self_data = self.as_f32_slice();
        let cond_data = condition.as_f32_slice();
        let other_data = other.as_f32_slice();
        let mut output_data = vec![0.0f32; numel];
        for i in 0..numel {
            output_data[i] = if cond_data[i] != 0.0 {
                self_data[i]
            } else {
                other_data[i]
            };
        }
        Tensor::from_vec(output_data, self.shape())
    }
}
