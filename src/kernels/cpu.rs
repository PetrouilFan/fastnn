use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
#[cfg(feature = "blas")]
use crate::kernels::blas::{self, MIN_BLAS_SIZE};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use smallvec::smallvec;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

fn create_output(tensor: &Tensor, shape: Vec<i64>) -> Tensor {
    let sizes: smallvec::SmallVec<[i64; 8]> = shape.into();
    let numel: i64 = sizes.iter().product();
    let nbytes = (numel * tensor.dtype().size() as i64) as usize;
    let storage = Arc::new(Storage::new(tensor.dtype(), tensor.device(), nbytes));
    Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))
}

fn add_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    // Get mutable access to output storage
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;
    let b_storage_offset = b.inner.storage_offset as usize;

    for idx in 0..numel {
        let mut remaining = idx;
        let mut a_idx = 0;
        let mut b_idx = 0;

        for i in (0..out_shape.len()).rev() {
            let dim_idx = remaining % out_shape[i] as usize;
            remaining /= out_shape[i] as usize;

            if i >= a_shape.len() || a_shape[i] == 1 {
            } else {
                a_idx += dim_idx * a_strides[i] as usize;
            }

            if i >= b_shape.len() || b_shape[i] == 1 {
            } else {
                b_idx += dim_idx * b_strides[i] as usize;
            }
        }

        a_idx += a_storage_offset;
        b_idx += b_storage_offset;

        unsafe {
            let a_val = *a_ptr.add(a_idx);
            let b_val = *b_ptr.add(b_idx);
            *out_ptr.add(idx) = a_val + b_val;
        }
    }

    vec![output]
}

fn sub_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let b_data = b.to_numpy();
    let mut output_data = output.to_numpy();

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    for idx in 0..output_data.len() {
        let mut remaining = idx;
        let mut a_idx = 0;
        let mut b_idx = 0;

        for i in (0..out_shape.len()).rev() {
            let dim_idx = remaining % out_shape[i] as usize;
            remaining /= out_shape[i] as usize;

            if i >= a_shape.len() || a_shape[i] == 1 {
            } else {
                a_idx += dim_idx as usize * a.inner.strides[i] as usize;
            }

            if i >= b_shape.len() || b_shape[i] == 1 {
            } else {
                b_idx += dim_idx as usize * b.inner.strides[i] as usize;
            }
        }

        a_idx += a.inner.storage_offset as usize;
        b_idx += b.inner.storage_offset as usize;

        let a_val = a_data.get(a_idx).copied().unwrap_or(0.0);
        let b_val = b_data.get(b_idx).copied().unwrap_or(0.0);
        output_data[idx] = a_val - b_val;
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn mul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let b_data = b.to_numpy();
    let mut output_data = output.to_numpy();

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    for idx in 0..output_data.len() {
        let mut remaining = idx;
        let mut a_idx = 0;
        let mut b_idx = 0;

        for i in (0..out_shape.len()).rev() {
            let dim_idx = remaining % out_shape[i] as usize;
            remaining /= out_shape[i] as usize;

            if i >= a_shape.len() || a_shape[i] == 1 {
            } else {
                a_idx += dim_idx as usize * a.inner.strides[i] as usize;
            }

            if i >= b_shape.len() || b_shape[i] == 1 {
            } else {
                b_idx += dim_idx as usize * b.inner.strides[i] as usize;
            }
        }

        a_idx += a.inner.storage_offset as usize;
        b_idx += b.inner.storage_offset as usize;

        let a_val = a_data.get(a_idx).copied().unwrap_or(0.0);
        let b_val = b_data.get(b_idx).copied().unwrap_or(0.0);
        output_data[idx] = a_val * b_val;
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn div_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let b_data = b.to_numpy();
    let mut output_data = output.to_numpy();

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    for idx in 0..output_data.len() {
        let mut remaining = idx;
        let mut a_idx = 0;
        let mut b_idx = 0;

        for i in (0..out_shape.len()).rev() {
            let dim_idx = remaining % out_shape[i] as usize;
            remaining /= out_shape[i] as usize;

            if i >= a_shape.len() || a_shape[i] == 1 {
            } else {
                a_idx += dim_idx as usize * a.inner.strides[i] as usize;
            }

            if i >= b_shape.len() || b_shape[i] == 1 {
            } else {
                b_idx += dim_idx as usize * b.inner.strides[i] as usize;
            }
        }

        a_idx += a.inner.storage_offset as usize;
        b_idx += b.inner.storage_offset as usize;

        let a_val = a_data.get(a_idx).copied().unwrap_or(0.0);
        let b_val = b_data.get(b_idx).copied().unwrap_or(0.0);
        output_data[idx] = a_val / b_val;
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn neg_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    for idx in 0..output_data.len() {
        let val = a_data.get(idx).copied().unwrap_or(0.0);
        output_data[idx] = -val;
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn abs_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    for idx in 0..output_data.len() {
        let val = a_data.get(idx).copied().unwrap_or(0.0);
        output_data[idx] = val.abs();
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn exp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    for idx in 0..output_data.len() {
        let val = a_data.get(idx).copied().unwrap_or(0.0);
        output_data[idx] = val.exp();
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn log_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    for idx in 0..output_data.len() {
        let val = a_data.get(idx).copied().unwrap_or(0.0);
        output_data[idx] = val.ln();
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn sqrt_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    for idx in 0..output_data.len() {
        let val = a_data.get(idx).copied().unwrap_or(0.0);
        output_data[idx] = val.sqrt();
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    // Get mutable access to output storage
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let val = *a_ptr.add(idx);
            *out_ptr.add(idx) = val.max(0.0);
        }
    }

    vec![output]
}

fn gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    for idx in 0..output_data.len() {
        let x = a_data.get(idx).copied().unwrap_or(0.0);
        output_data[idx] = 0.5 * x * (1.0 + (x * x * 0.7978845608028654).tanh());
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn sigmoid_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    for idx in 0..output_data.len() {
        let x = a_data.get(idx).copied().unwrap_or(0.0);
        output_data[idx] = 1.0 / (1.0 + (-x).exp());
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn tanh_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    for idx in 0..output_data.len() {
        let x = a_data.get(idx).copied().unwrap_or(0.0);
        output_data[idx] = x.tanh();
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    for idx in 0..output_data.len() {
        let x = a_data.get(idx).copied().unwrap_or(0.0);
        output_data[idx] = x / (1.0 + (-x).exp());
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn matmul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() < 2 || b_shape.len() < 2 {
        panic!("matmul: both tensors must have at least 2 dimensions");
    }

    let m = a_shape[a_shape.len() - 2] as i32;
    let k = a_shape[a_shape.len() - 1] as i32;
    let n = b_shape[b_shape.len() - 1] as i32;

    if b_shape[b_shape.len() - 2] as i32 != k {
        panic!("matmul: {} != {}", b_shape[b_shape.len() - 2], k);
    }

    #[cfg(feature = "blas")]
    {
        if a.dtype() == DType::F32
            && b.dtype() == DType::F32
            && a.is_contiguous()
            && b.is_contiguous()
        {
            let total_elements = (m * k + k * n + m * n) as usize;
            if total_elements >= MIN_BLAS_SIZE {
                let a_data = a.to_numpy();
                let b_data = b.to_numpy();
                let c_data = blas::matmul_blas(&a_data, &b_data, m, k, n);

                let mut output_shape: Vec<i64> = vec![];
                if a_shape.len() > 2 {
                    for i in 0..a_shape.len() - 2 {
                        output_shape.push(a_shape[i].max(b_shape[i]));
                    }
                }
                output_shape.push(m as i64);
                output_shape.push(n as i64);

                let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
                let storage = Arc::new(Storage::from_vec(c_data, a.dtype(), a.device()));
                return vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))];
            }
        }
    }

    let batch_a = if a_shape.len() > 2 {
        a_shape[..a_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch_b = if b_shape.len() > 2 {
        b_shape[..b_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch = batch_a.max(batch_b);

    let mut output_shape: Vec<i64> = vec![];
    if a_shape.len() > 2 {
        for i in 0..a_shape.len() - 2 {
            output_shape.push(a_shape[i].max(b_shape[i]));
        }
    }
    output_shape.push(m as i64);
    output_shape.push(n as i64);

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let a_data = a.to_numpy();
    let b_data = b.to_numpy();
    let mut output_data = output.to_numpy();

    let a_rows = a_shape[a_shape.len() - 2] as usize;
    let a_cols = a_shape[a_shape.len() - 1] as usize;
    let b_cols = b_shape[b_shape.len() - 1] as usize;

    const TILE_SIZE: usize = 32;

    for bat in 0..batch {
        for i_tile in (0..m as usize).step_by(TILE_SIZE) {
            for j_tile in (0..n as usize).step_by(TILE_SIZE) {
                for k_tile in (0..k as usize).step_by(TILE_SIZE) {
                    let i_max = (i_tile + TILE_SIZE).min(m as usize);
                    let j_max = (j_tile + TILE_SIZE).min(n as usize);
                    let k_max = (k_tile + TILE_SIZE).min(k as usize);

                    for i in i_tile..i_max {
                        for j in j_tile..j_max {
                            let mut sum = 0.0f32;

                            let mut kk = k_tile;
                            while kk + 4 <= k_max {
                                let a0 = unsafe {
                                    *a_data.get_unchecked(bat * a_rows * a_cols + i * a_cols + kk)
                                };
                                let a1 = unsafe {
                                    *a_data
                                        .get_unchecked(bat * a_rows * a_cols + i * a_cols + kk + 1)
                                };
                                let a2 = unsafe {
                                    *a_data
                                        .get_unchecked(bat * a_rows * a_cols + i * a_cols + kk + 2)
                                };
                                let a3 = unsafe {
                                    *a_data
                                        .get_unchecked(bat * a_rows * a_cols + i * a_cols + kk + 3)
                                };

                                let b0 = unsafe {
                                    *b_data
                                        .get_unchecked(bat * k as usize * b_cols + kk * b_cols + j)
                                };
                                let b1 = unsafe {
                                    *b_data.get_unchecked(
                                        bat * k as usize * b_cols + (kk + 1) * b_cols + j,
                                    )
                                };
                                let b2 = unsafe {
                                    *b_data.get_unchecked(
                                        bat * k as usize * b_cols + (kk + 2) * b_cols + j,
                                    )
                                };
                                let b3 = unsafe {
                                    *b_data.get_unchecked(
                                        bat * k as usize * b_cols + (kk + 3) * b_cols + j,
                                    )
                                };

                                sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
                                kk += 4;
                            }

                            while kk < k_max {
                                let a_val = unsafe {
                                    *a_data.get_unchecked(bat * a_rows * a_cols + i * a_cols + kk)
                                };
                                let b_val = unsafe {
                                    *b_data
                                        .get_unchecked(bat * k as usize * b_cols + kk * b_cols + j)
                                };
                                sum += a_val * b_val;
                                kk += 1;
                            }

                            let out_idx = bat * m as usize * n as usize + i * n as usize + j;
                            unsafe { *output_data.get_unchecked_mut(out_idx) = sum };
                        }
                    }
                }
            }
        }
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn linear_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[0];

    let x_flat = x.reshape(vec![batch_size, in_features]);
    let w_t = w.transpose(0, 1);

    let mut result = (x_flat.matmul(&w_t)).reshape(
        x_shape[..x_shape.len() - 1]
            .iter()
            .copied()
            .chain(std::iter::once(out_features))
            .collect(),
    );

    if let Some(b) = bias {
        result = result.add(b);
    }

    vec![result]
}

fn sum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();

    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    let total_blocks = (strides_before * strides_after) as usize;

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut sum = 0.0f32;
        for d in 0..dim_size {
            let idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            if idx < a_data.len() {
                sum += a_data[idx];
            }
        }

        output_data[block] = sum;
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn mean_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let dim_size = a_shape[dim] as f32;

    let sum_result = sum_kernel(args);
    let mut sum_tensor = sum_result[0].clone();

    if keepdim {
        sum_tensor = sum_tensor.unsqueeze(dim);
    }

    let scale = Tensor::full(vec![], 1.0 / dim_size, DType::F32, Device::Cpu);
    vec![sum_tensor * scale]
}

fn max_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    let total_blocks = (strides_before * strides_after) as usize;

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut max_val = f32::MIN;
        for d in 0..dim_size {
            let idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            if idx < a_data.len() {
                max_val = max_val.max(a_data[idx]);
            }
        }

        output_data[block] = max_val;
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn min_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };
    let keepdim = if args.len() > 2 {
        args[2].item() != 0.0
    } else {
        false
    };

    let a_shape = a.shape();
    let ndim = a_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let mut output_shape: Vec<i64> = a_shape.clone();
    if keepdim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    if output_shape.is_empty() {
        output_shape = vec![1];
    }

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut output_data = output.to_numpy();

    let dim_size = a_shape[dim] as usize;
    let mut strides_before = 1i64;
    let mut strides_after = 1i64;

    for i in 0..dim {
        strides_before *= a_shape[i];
    }
    for i in (dim + 1)..ndim {
        strides_after *= a_shape[i];
    }

    let total_blocks = (strides_before * strides_after) as usize;

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut min_val = f32::MAX;
        for d in 0..dim_size {
            let idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            if idx < a_data.len() {
                min_val = min_val.min(a_data[idx]);
            }
        }

        output_data[block] = min_val;
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn softmax_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let dim_size = x_shape[dim] as usize;

    let max_vals = max_kernel(&[
        x,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(1.0),
    ])[0]
        .clone();
    let max_exp = x.sub(&max_vals.clone().unsqueeze(dim)).exp();
    let sum_exp = max_exp.sum(dim as i32, true);

    vec![max_exp.div(&sum_exp)]
}

fn log_softmax_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let dim = if args.len() > 1 {
        args[1].item() as usize
    } else {
        0
    };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let dim = if dim >= ndim { ndim - 1 } else { dim };

    let max_vals = max_kernel(&[
        x,
        &Tensor::from_scalar(dim as f32),
        &Tensor::from_scalar(1.0),
    ])[0]
        .clone();
    let x_shifted = x.sub(&max_vals.clone().unsqueeze(dim));
    let log_sum_exp = x_shifted.exp().sum(dim as i32, true).ln();

    vec![x_shifted.sub(&log_sum_exp)]
}

fn mse_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let pred = args[0];
    let target = args[1];
    let reduction = if args.len() > 2 {
        match args[2].item() as i32 {
            0 => "none",
            1 => "mean",
            _ => "sum",
        }
    } else {
        "mean"
    };

    let diff = pred.sub(target);
    let loss = diff.mul(&diff);

    match reduction {
        "none" => vec![loss],
        "mean" => vec![loss
            .sum(0, false)
            .div(&Tensor::from_scalar(loss.numel() as f32))],
        "sum" => vec![loss.sum(0, false)],
        _ => vec![loss.sum(0, false)],
    }
}

fn cross_entropy_loss_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let logits = args[0];
    let targets = args[1];
    let reduction = if args.len() > 2 {
        match args[2].item() as i32 {
            0 => "none",
            1 => "mean",
            _ => "sum",
        }
    } else {
        "mean"
    };

    let logits_data = logits.to_numpy();
    let targets_data = targets.to_numpy();

    let batch_size = logits.shape()[0] as usize;
    let num_classes = logits.shape()[1] as usize;

    let mut total_loss = 0.0f32;
    let mut losses = vec![0.0f32; batch_size];

    for b in 0..batch_size {
        let max_logit = (0..num_classes)
            .map(|c| logits_data[b * num_classes + c])
            .fold(f32::MIN, f32::max);

        let mut sum_exp = 0.0f32;
        for c in 0..num_classes {
            sum_exp += (logits_data[b * num_classes + c] - max_logit).exp();
        }
        let log_sum_exp = sum_exp.ln();

        let target_class = targets_data[b] as usize;
        let class_logit = logits_data[b * num_classes + target_class];

        losses[b] = log_sum_exp - class_logit;
        total_loss += losses[b];
    }

    match reduction {
        "none" => {
            let output = Tensor::zeros(logits.shape(), DType::F32, Device::Cpu);
            vec![output]
        }
        "mean" => vec![Tensor::from_scalar(total_loss / batch_size as f32)],
        "sum" | _ => vec![Tensor::from_scalar(total_loss)],
    }
}

fn conv2d_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };
    let stride = if args.len() > 3 {
        args[3].item() as i64
    } else {
        1
    };
    let padding = if args.len() > 4 {
        args[4].item() as i64
    } else {
        0
    };
    let dilation = if args.len() > 5 {
        args[5].item() as i64
    } else {
        1
    };
    let _groups = if args.len() > 6 {
        args[6].item() as i64
    } else {
        1
    };

    let x_shape = x.shape();
    let w_shape = w.shape();

    let batch_size = x_shape[0];
    let in_channels = x_shape[1];
    let in_height = x_shape[2];
    let in_width = x_shape[3];

    let out_channels = w_shape[0];
    let kernel_height = w_shape[2];
    let kernel_width = w_shape[3];

    let out_height = (in_height + 2 * padding - dilation * (kernel_height - 1) - 1) / stride + 1;
    let out_width = (in_width + 2 * padding - dilation * (kernel_width - 1) - 1) / stride + 1;

    let output_shape = vec![batch_size, out_channels, out_height, out_width];
    let mut output = Tensor::zeros(output_shape.clone(), x.dtype(), x.device());

    let x_data = x.to_numpy();
    let w_data = w.to_numpy();
    let mut output_data = output.to_numpy();

    for n in 0..batch_size as usize {
        for oc in 0..out_channels as usize {
            for oh in 0..out_height as usize {
                for ow in 0..out_width as usize {
                    let mut sum = 0.0f32;

                    for ic in 0..in_channels as usize {
                        for kh in 0..kernel_height as usize {
                            for kw in 0..kernel_width as usize {
                                let ih = oh as i64 * stride - padding + kh as i64 * dilation;
                                let iw = ow as i64 * stride - padding + kw as i64 * dilation;

                                if ih >= 0 && ih < in_height && iw >= 0 && iw < in_width {
                                    let x_idx = ((n * in_channels as usize + ic)
                                        * in_height as usize
                                        + ih as usize)
                                        * in_width as usize
                                        + iw as usize;
                                    let w_idx = ((oc * in_channels as usize + ic)
                                        * kernel_height as usize
                                        + kh as usize)
                                        * kernel_width as usize
                                        + kw as usize;

                                    sum += x_data.get(x_idx).copied().unwrap_or(0.0)
                                        * w_data.get(w_idx).copied().unwrap_or(0.0);
                                }
                            }
                        }
                    }

                    if let Some(b) = bias {
                        sum += b.to_numpy().get(oc).copied().unwrap_or(0.0);
                    }

                    let out_idx = ((n * out_channels as usize + oc) * out_height as usize
                        + oh as usize)
                        * out_width as usize
                        + ow as usize;
                    if out_idx < output_data.len() {
                        output_data[out_idx] = sum;
                    }
                }
            }
        }
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(output_data, x.dtype(), x.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn layer_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let _normalized_shape = args[1].shape();
    let weight = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };
    let bias = if args.len() > 3 && args[3].numel() > 0 {
        Some(args[3])
    } else {
        None
    };
    let eps = if args.len() > 4 {
        args[4].item() as f64
    } else {
        1e-5
    };

    let x_shape = x.shape();
    let ndim = x_shape.len();
    let normalized_shape: i64 = x_shape.iter().skip(ndim - 1).product();

    let mean = x.mean((ndim - 1) as i32, true);
    let var = x
        .sub(&mean.clone())
        .mul(&x.sub(&mean.clone()))
        .mean((ndim - 1) as i32, true);
    let std = var.add(&Tensor::from_scalar(eps as f32)).sqrt();

    let mut normalized = x.sub(&mean).div(&std);

    if let Some(w) = weight {
        normalized = normalized.mul(w);
    }
    if let Some(b) = bias {
        normalized = normalized.add(b);
    }

    vec![normalized]
}

fn batch_norm_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let weight = if args.len() > 1 && args[1].numel() > 0 {
        Some(args[1])
    } else {
        None
    };
    let bias = if args.len() > 2 && args[2].numel() > 0 {
        Some(args[2])
    } else {
        None
    };
    let training = if args.len() > 5 {
        args[5].item() != 0.0
    } else {
        false
    };
    let eps = if args.len() > 6 {
        args[6].item() as f64
    } else {
        1e-5
    };

    let x_shape = x.shape();
    let num_features = x_shape[1];

    let mean = x.mean(0, false).unsqueeze(0);
    let var = x
        .sub(&mean.clone())
        .mul(&x.sub(&mean.clone()))
        .mean(0, false)
        .unsqueeze(0);
    let std = var.add(&Tensor::from_scalar(eps as f32)).sqrt();

    let mut normalized = x.sub(&mean).div(&std);

    if let Some(w) = weight {
        normalized = normalized.mul(&w.unsqueeze(0));
    }
    if let Some(b) = bias {
        normalized = normalized.add(&b.unsqueeze(0));
    }

    vec![normalized]
}

fn embedding_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let weight = args[0];
    let indices = args[1];

    let weight_shape = weight.shape();
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];

    let indices_shape = indices.shape();
    let batch_size: i64 = indices_shape.iter().product();

    let indices_data = indices.to_numpy();
    let weight_data = weight.to_numpy();

    let mut output_data = vec![0.0f32; (batch_size * embedding_dim) as usize];

    for i in 0..batch_size as usize {
        let idx = indices_data.get(i).copied().unwrap_or(0.0) as usize;
        if idx < num_embeddings as usize {
            for j in 0..embedding_dim as usize {
                let w_idx = idx * embedding_dim as usize + j;
                let o_idx = i * embedding_dim as usize + j;
                output_data[o_idx] = weight_data.get(w_idx).copied().unwrap_or(0.0);
            }
        }
    }

    let output_shape: Vec<i64> = indices_shape
        .iter()
        .chain(std::iter::once(&embedding_dim))
        .copied()
        .collect();
    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(
        output_data,
        weight.dtype(),
        weight.device(),
    ));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn zeros_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0]
        .to_numpy()
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<_>>();
    let dtype = DType::from_str(
        &args[1]
            .to_numpy()
            .iter()
            .map(|&x| x as u8 as char)
            .collect::<String>(),
    )
    .unwrap_or(DType::F32);
    let device = Device::Cpu;

    vec![Tensor::zeros(shape, dtype, device)]
}

fn ones_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0]
        .to_numpy()
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<_>>();
    let dtype = DType::F32;
    let device = Device::Cpu;

    vec![Tensor::ones(shape, dtype, device)]
}

fn full_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0]
        .to_numpy()
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<_>>();
    let value = args[1].item();
    let dtype = DType::F32;
    let device = Device::Cpu;

    vec![Tensor::full(shape, value, dtype, device)]
}

fn arange_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let start = args[0].item();
    let end = args[1].item();
    let step = if args.len() > 2 { args[2].item() } else { 1.0 };

    let numel = ((end - start) / step).ceil() as usize;
    let values: Vec<f32> = (0..numel).map(|i| start + i as f32 * step).collect();

    vec![Tensor::from_vec(values, vec![numel as i64])]
}

fn linspace_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let start = args[0].item();
    let end = args[1].item();
    let steps = args[2].item() as usize;

    let values: Vec<f32> = (0..steps)
        .map(|i| {
            let t = i as f32 / (steps - 1) as f32;
            start * (1.0 - t) + end * t
        })
        .collect();

    vec![Tensor::from_vec(values, vec![steps as i64])]
}

fn eye_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let n = args[0].item() as usize;
    let m = if args.len() > 1 {
        args[1].item() as usize
    } else {
        n
    };

    let mut values = vec![0.0f32; n * m];
    for i in 0..n.min(m) {
        values[i * m + i] = 1.0;
    }

    vec![Tensor::from_vec(values, vec![n as i64, m as i64])]
}

fn randn_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0]
        .to_numpy()
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<_>>();
    let numel: i64 = shape.iter().product();

    let mut values = vec![0.0f32; numel as usize];
    for v in &mut values {
        let u1: f32 = rand::random();
        let u2: f32 = rand::random();
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    vec![Tensor::from_vec(values, shape)]
}

fn rand_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0]
        .to_numpy()
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<_>>();
    let numel: i64 = shape.iter().product();

    let values: Vec<f32> = (0..numel as usize).map(|_| rand::random::<f32>()).collect();

    vec![Tensor::from_vec(values, shape)]
}

fn randint_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape = args[0]
        .to_numpy()
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<_>>();
    let low = args[1].item() as i32;
    let high = args[2].item() as i32;

    let numel: i64 = shape.iter().product();
    let values: Vec<f32> = (0..numel as usize)
        .map(|_| (rand::random::<i32>() % (high - low) + low) as f32)
        .collect();

    vec![Tensor::from_vec(values, shape)]
}

fn read_f32(slice: &[u8], dtype: DType) -> f32 {
    match dtype {
        DType::F32 => {
            let ptr = slice.as_ptr() as *const f32;
            unsafe { *ptr }
        }
        DType::F64 => {
            let ptr = slice.as_ptr() as *const f64;
            unsafe { *ptr as f32 }
        }
        DType::I32 => {
            let ptr = slice.as_ptr() as *const i32;
            unsafe { *ptr as f32 }
        }
        DType::I64 => {
            let ptr = slice.as_ptr() as *const i64;
            unsafe { *ptr as f32 }
        }
        _ => 0.0,
    }
}

fn write_f32(slice: &[u8], val: f32) {
    let ptr = slice.as_ptr() as *mut u8;
    unsafe {
        *(ptr as *mut f32) = val;
    }
}

#[ctor::ctor]
fn register_kernels() {
    register("add", DispatchKey::Cpu, add_kernel as KernelFn);
    register("sub", DispatchKey::Cpu, sub_kernel as KernelFn);
    register("mul", DispatchKey::Cpu, mul_kernel as KernelFn);
    register("div", DispatchKey::Cpu, div_kernel as KernelFn);
    register("neg", DispatchKey::Cpu, neg_kernel as KernelFn);
    register("abs", DispatchKey::Cpu, abs_kernel as KernelFn);
    register("exp", DispatchKey::Cpu, exp_kernel as KernelFn);
    register("log", DispatchKey::Cpu, log_kernel as KernelFn);
    register("sqrt", DispatchKey::Cpu, sqrt_kernel as KernelFn);
    register("relu", DispatchKey::Cpu, relu_kernel as KernelFn);
    register("gelu", DispatchKey::Cpu, gelu_kernel as KernelFn);
    register("sigmoid", DispatchKey::Cpu, sigmoid_kernel as KernelFn);
    register("tanh", DispatchKey::Cpu, tanh_kernel as KernelFn);
    register("silu", DispatchKey::Cpu, silu_kernel as KernelFn);
    register("matmul", DispatchKey::Cpu, matmul_kernel as KernelFn);
    register("linear", DispatchKey::Cpu, linear_kernel as KernelFn);
    register("sum", DispatchKey::Cpu, sum_kernel as KernelFn);
    register("mean", DispatchKey::Cpu, mean_kernel as KernelFn);
    register("max", DispatchKey::Cpu, max_kernel as KernelFn);
    register("min", DispatchKey::Cpu, min_kernel as KernelFn);
    register("softmax", DispatchKey::Cpu, softmax_kernel as KernelFn);
    register(
        "log_softmax",
        DispatchKey::Cpu,
        log_softmax_kernel as KernelFn,
    );
    register("mse_loss", DispatchKey::Cpu, mse_loss_kernel as KernelFn);
    register(
        "cross_entropy_loss",
        DispatchKey::Cpu,
        cross_entropy_loss_kernel as KernelFn,
    );
    register("conv2d", DispatchKey::Cpu, conv2d_kernel as KernelFn);
    register(
        "layer_norm",
        DispatchKey::Cpu,
        layer_norm_kernel as KernelFn,
    );
    register(
        "batch_norm",
        DispatchKey::Cpu,
        batch_norm_kernel as KernelFn,
    );
    register("embedding", DispatchKey::Cpu, embedding_kernel as KernelFn);
    register("zeros", DispatchKey::Cpu, zeros_kernel as KernelFn);
    register("ones", DispatchKey::Cpu, ones_kernel as KernelFn);
    register("full", DispatchKey::Cpu, full_kernel as KernelFn);
    register("arange", DispatchKey::Cpu, arange_kernel as KernelFn);
    register("linspace", DispatchKey::Cpu, linspace_kernel as KernelFn);
    register("eye", DispatchKey::Cpu, eye_kernel as KernelFn);
    register("randn", DispatchKey::Cpu, randn_kernel as KernelFn);
    register("rand", DispatchKey::Cpu, rand_kernel as KernelFn);
}
