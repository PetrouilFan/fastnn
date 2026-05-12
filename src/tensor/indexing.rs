use crate::autograd::{self, Edge};
use crate::dispatcher::{device_to_dispatch_key, dispatch};
use crate::ir::node::DimExpr;
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
            let backward = Arc::new(autograd::SliceBackward::new());
            Self::attach_grad_fn(output, backward)
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
            let edges = autograd::make_edges(self, other);
            let backward = Arc::new(autograd::MaximumBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn gt_scalar(&self, threshold: f32) -> Tensor {
        let scalar = Tensor::from_scalar(threshold);
        Tensor::exec_aot(&[self, &scalar], |g, ins| vec![g.gt_scalar(&ins[0], &ins[1])])
            .expect("Tensor::gt_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

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
            let edges = autograd::make_edges(self, other);
            let backward = Arc::new(autograd::MinimumBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn ge_tensor(&self, other: &Tensor) -> Tensor {
        let numel = self.numel() as usize;
        let self_contig = if !self.is_contiguous() {
            self.contiguous()
        } else {
            self.clone()
        };
        let other_contig = if !other.is_contiguous() {
            other.contiguous()
        } else {
            other.clone()
        };
        let self_data = self_contig.as_f32_slice();
        let other_data = other_contig.as_f32_slice();
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
        let self_contig = if !self.is_contiguous() {
            self.contiguous()
        } else {
            self.clone()
        };
        let other_contig = if !other.is_contiguous() {
            other.contiguous()
        } else {
            other.clone()
        };
        let self_data = self_contig.as_f32_slice();
        let other_data = other_contig.as_f32_slice();
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
        let scalar = Tensor::from_scalar(threshold);
        Tensor::exec_aot(&[self, &scalar], |g, ins| vec![g.lt_scalar(&ins[0], &ins[1])])
            .expect("Tensor::lt_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn eq_scalar(&self, threshold: f32) -> Tensor {
        let scalar = Tensor::from_scalar(threshold);
        Tensor::exec_aot(&[self, &scalar], |g, ins| vec![g.eq_scalar(&ins[0], &ins[1])])
            .expect("Tensor::eq_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let s = Tensor::from_scalar(scalar);
        let output = Tensor::exec_aot(&[self, &s], |g, ins| vec![g.add_scalar(&ins[0], &ins[1])])
            .expect("Tensor::add_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::AddScalarBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let s = Tensor::from_scalar(scalar);
        let output = Tensor::exec_aot(&[self, &s], |g, ins| vec![g.div_scalar(&ins[0], &ins[1])])
            .expect("Tensor::div_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::DivScalarBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn logical_not(&self) -> Tensor {
        Tensor::exec_aot(&[self], |g, ins| vec![g.logical_not(&ins[0])])
            .expect("Tensor::logical_not: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn argmax(&self, dim: Option<usize>) -> Tensor {
        let result = if self.device() == Device::Cpu {
            Tensor::exec_aot(&[self], |g, ins| {
                let axis = dim.map(|d| DimExpr::Known(d as u64));
                let out = g.argmax(&ins[0], axis);
                vec![out]
            })
            .expect("Tensor::argmax: AOT execution failed")
        } else {
            let dispatch_key = device_to_dispatch_key(self.device());
            dispatch("argmax", dispatch_key, &[self])
                .expect("Tensor::argmax: dispatch failed")
        };
        result.into_iter().next().unwrap()
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
                let mut t_lin = 0i64;
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
        let output = Tensor::from_vec(output_data, output_shape.into_vec());

        if autograd::is_grad_enabled() && tensors.iter().any(|t| t.requires_grad()) {
            let split_sizes: Vec<usize> = tensors
                .iter()
                .map(|t| t.inner.sizes[dim] as usize)
                .collect();
            let mut edges = Vec::new();
            for (i, t) in tensors.iter().enumerate() {
                if let Some(node) = t.grad_fn() {
                    edges.push(Edge(node, i));
                }
            }
            let backward = Arc::new(autograd::CatBackward::new());
            Self::attach_grad_fn(output, backward)
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
        let mut src = self.to_cpu();
        if !src.is_contiguous() {
            src = src.contiguous();
        }
        let src_data = src.as_f32_slice();
        let src_strides = &src.inner.strides;
        let src_sizes = &src.inner.sizes;
        let ndim = src.ndim();
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
            let mut src_lin = 0i64;
            for d in 0..ndim {
                src_lin += src_indices[d] * src_strides[d];
            }
            output_data[out_idx] = src_data[src_lin as usize];
        }
        let output = Tensor::from_vec(output_data, new_shape.into_vec());

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::RepeatBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn where_tensor(&self, condition: &Tensor, other: &Tensor) -> Tensor {
        let numel = self.numel() as usize;
        let self_contig = if !self.is_contiguous() {
            self.contiguous()
        } else {
            self.clone()
        };
        let cond_contig = if !condition.is_contiguous() {
            condition.contiguous()
        } else {
            condition.clone()
        };
        let other_contig = if !other.is_contiguous() {
            other.contiguous()
        } else {
            other.clone()
        };
        let self_data = self_contig.as_f32_slice();
        let cond_data = cond_contig.as_f32_slice();
        let other_data = other_contig.as_f32_slice();
        let mut output_data = vec![0.0f32; numel];
        for i in 0..numel {
            output_data[i] = if cond_data[i] != 0.0 {
                self_data[i]
            } else {
                other_data[i]
            };
        }
        let output = Tensor::from_vec(output_data, self.shape());

        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let edges = autograd::make_edges(self, other);
            let backward = Arc::new(autograd::WhereBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn gather(&self, axis: i64, indices: &Tensor) -> Tensor {
        let indices_data = indices.as_i64_slice();
        let self_contig = if !self.is_contiguous() {
            self.contiguous()
        } else {
            self.clone()
        };
        let x_data = self_contig.as_f32_slice();
        let shape = self.shape_ref();
        let axis = if axis < 0 {
            shape.len() as i64 + axis
        } else {
            axis
        } as usize;

        // ONNX Gather semantics:
        // - If indices is 0-D (scalar), the gathered dim is *removed*
        //   (output rank = data.rank - 1).
        // - If indices is 1+D, the gathered dim is *replaced* by the indices shape
        //   (output rank = data.rank - 1 + indices.rank).
        let indices_ndim = indices.ndim();
        let scalar_idx = indices_ndim == 0;

        let out_shape: Vec<i64> = if scalar_idx {
            let mut s = shape.to_vec();
            s.remove(axis);
            s
        } else {
            let mut s = shape.to_vec();
            if indices_ndim == 1 {
                s[axis] = indices_data.len() as i64;
            } else {
                // N-D indices: replace gathered dim with all indices dims
                s.remove(axis);
                for d in (0..indices_ndim).rev() {
                    s.insert(axis, indices.inner.sizes[d]);
                }
            }
            s
        };
        let mut out_data = vec![0.0f32; out_shape.iter().product::<i64>() as usize];

        let inner = if axis + 1 < shape.len() {
            shape[axis + 1..].iter().product::<i64>() as usize
        } else {
            1
        };
        let outer = if axis > 0 {
            shape[..axis].iter().product::<i64>() as usize
        } else {
            1
        };

        if scalar_idx {
            let idx = indices_data[0] as usize;
            if idx >= shape[axis] as usize {
                panic!(
                    "gather: index {} is out of bounds for dimension {} (size {})",
                    idx, axis, shape[axis]
                );
            }
            for o in 0..outer {
                let src_off = (o * shape[axis] as usize + idx) * inner;
                let dst_off = o * inner;
                out_data[dst_off..dst_off + inner]
                    .copy_from_slice(&x_data[src_off..src_off + inner]);
            }
        } else {
            for o in 0..outer {
                for (i, &idx) in indices_data.iter().enumerate() {
                    if idx as usize >= shape[axis] as usize {
                        panic!(
                            "gather: index {} is out of bounds for dimension {} (size {})",
                            idx, axis, shape[axis]
                        );
                    }
                    let src_off = (o * shape[axis] as usize + idx as usize) * inner;
                    let dst_off = (o * out_shape[axis] as usize + i) * inner;
                    out_data[dst_off..dst_off + inner]
                        .copy_from_slice(&x_data[src_off..src_off + inner]);
                }
            }
        }

        Tensor::from_vec(out_data, out_shape)
    }

    pub fn nonzero(&self) -> Vec<Vec<i64>> {
        let data = self.as_f32_slice();
        let shape = self.shape_ref();
        let mut result = Vec::new();

        if shape.is_empty() {
            return result;
        }

        let total = data.len();
        for i in 0..total {
            if data[i] != 0.0 {
                // Convert flat index to multi-dimensional index
                let mut idx = i;
                let mut coords = Vec::with_capacity(shape.len());
                for &dim in shape.iter().rev() {
                    coords.push((idx % dim as usize) as i64);
                    idx /= dim as usize;
                }
                coords.reverse();
                result.push(coords);
            }
        }
        result
    }
}
