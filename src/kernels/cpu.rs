use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
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

    if a.is_contiguous() && b.is_contiguous() && numel > 10000 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let a_data: Vec<f32> = (0..numel).map(|i| unsafe { *a_ptr.add(i) }).collect();
            let b_data: Vec<f32> = (0..numel).map(|i| unsafe { *b_ptr.add(i) }).collect();
            let result: Vec<f32> = a_data
                .par_iter()
                .zip(b_data.par_iter())
                .map(|(x, y)| x + y)
                .collect();
            unsafe {
                std::ptr::copy_nonoverlapping(result.as_ptr(), out_ptr, numel);
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let a_val = *a_ptr.add(idx);
                    let b_val = *b_ptr.add(idx);
                    *out_ptr.add(idx) = a_val + b_val;
                }
            }
        }
    } else {
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
    }

    vec![output]
}

fn sub_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

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
            *out_ptr.add(idx) = a_val - b_val;
        }
    }

    vec![output]
}

fn mul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

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
            *out_ptr.add(idx) = a_val * b_val;
        }
    }

    vec![output]
}

fn div_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

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
            *out_ptr.add(idx) = a_val / b_val;
        }
    }

    vec![output]
}

fn neg_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let val = *a_ptr.add(idx);
            *out_ptr.add(idx) = -val;
        }
    }

    vec![output]
}

fn abs_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());
    let a_data = a.to_numpy();
    let mut out_ptr = output.to_numpy();

    for idx in 0..out_ptr.len() {
        let val = a_data.get(idx).copied().unwrap_or(0.0);
        out_ptr[idx] = val.abs();
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(out_ptr, a.dtype(), a.device()));
    vec![Tensor::new(crate::tensor::TensorImpl::new(storage, sizes))]
}

fn exp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let val = *a_ptr.add(idx);
            *out_ptr.add(idx) = val.exp();
        }
    }

    vec![output]
}

fn log_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let val = *a_ptr.add(idx);
            *out_ptr.add(idx) = val.ln();
        }
    }

    vec![output]
}

fn sqrt_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let val = *a_ptr.add(idx);
            *out_ptr.add(idx) = val.sqrt();
        }
    }

    vec![output]
}

fn relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

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

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 10000 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let a_data: Vec<f32> = (0..numel).map(|i| unsafe { *a_ptr.add(i) }).collect();
            let result: Vec<f32> = a_data
                .par_iter()
                .map(|&x| 0.5 * x * (1.0 + (x * x * 0.7978845608028654).tanh()))
                .collect();
            unsafe {
                std::ptr::copy_nonoverlapping(result.as_ptr(), out_ptr, numel);
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let x = *a_ptr.add(idx);
                    *out_ptr.add(idx) = 0.5 * x * (1.0 + (x * x * 0.7978845608028654).tanh());
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = 0.5 * x * (1.0 + (x * x * 0.7978845608028654).tanh());
            }
        }
    }

    vec![output]
}

fn sigmoid_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let x = *a_ptr.add(idx);
            *out_ptr.add(idx) = 1.0 / (1.0 + (-x).exp());
        }
    }

    vec![output]
}

fn tanh_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let x = *a_ptr.add(idx);
            *out_ptr.add(idx) = x.tanh();
        }
    }

    vec![output]
}

fn silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::zeros(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let x = *a_ptr.add(idx);
            *out_ptr.add(idx) = x / (1.0 + (-x).exp());
        }
    }

    vec![output]
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

    // Use custom tiled matmul
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

    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    let a_rows = a_shape[a_shape.len() - 2] as usize;
    let a_cols = a_shape[a_shape.len() - 1] as usize;
    let b_cols = b_shape[b_shape.len() - 1] as usize;

    single_threaded_matmul(
        a_ptr, b_ptr, out_ptr, batch, m, n, k, a_rows, a_cols, b_cols,
    );

    vec![output]
}

#[inline]
fn single_threaded_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_rows: usize,
    a_cols: usize,
    b_cols: usize,
) {
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
                                let a0 =
                                    unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                                let a1 = unsafe {
                                    *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 1)
                                };
                                let a2 = unsafe {
                                    *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 2)
                                };
                                let a3 = unsafe {
                                    *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk + 3)
                                };

                                let b0 = unsafe {
                                    *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j)
                                };
                                let b1 = unsafe {
                                    *b_ptr.add(bat * k as usize * b_cols + (kk + 1) * b_cols + j)
                                };
                                let b2 = unsafe {
                                    *b_ptr.add(bat * k as usize * b_cols + (kk + 2) * b_cols + j)
                                };
                                let b3 = unsafe {
                                    *b_ptr.add(bat * k as usize * b_cols + (kk + 3) * b_cols + j)
                                };

                                sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
                                kk += 4;
                            }

                            while kk < k_max {
                                let a_val =
                                    unsafe { *a_ptr.add(bat * a_rows * a_cols + i * a_cols + kk) };
                                let b_val = unsafe {
                                    *b_ptr.add(bat * k as usize * b_cols + kk * b_cols + j)
                                };
                                sum += a_val * b_val;
                                kk += 1;
                            }

                            let out_idx = bat * m as usize * n as usize + i * n as usize + j;
                            unsafe { *out_ptr.add(out_idx) = sum };
                        }
                    }
                }
            }
        }
    }
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

    let a_ptr = a.data_ptr() as *const f32;
    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let a_storage_offset = a.inner.storage_offset as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

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
    let a_numel = a.numel() as usize;

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut sum = 0.0f32;
        for d in 0..dim_size {
            let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            let idx = a_storage_offset + linear_idx;
            if idx < a_numel {
                unsafe {
                    sum += *a_ptr.add(idx);
                }
            }
        }

        unsafe {
            *out_ptr.add(block) = sum;
        }
    }

    vec![output]
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

    let a_ptr = a.data_ptr() as *const f32;
    let a_storage_offset = a.inner.storage_offset as usize;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

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
    let a_numel = a.numel() as usize;

    for block in 0..total_blocks {
        let block_before = block / (strides_after as usize);
        let block_after = block % (strides_after as usize);

        let mut max_val = f32::MIN;
        for d in 0..dim_size {
            let linear_idx = (block_before * dim_size + d) * strides_after as usize + block_after;
            let idx = a_storage_offset + linear_idx;
            if idx < a_numel {
                unsafe {
                    max_val = max_val.max(*a_ptr.add(idx));
                }
            }
        }

        unsafe {
            *out_ptr.add(block) = max_val;
        }
    }

    vec![output]
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
    let mut out_ptr = output.to_numpy();

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

        out_ptr[block] = min_val;
    }

    let sizes: smallvec::SmallVec<[i64; 8]> = output_shape.into();
    let storage = Arc::new(Storage::from_vec(out_ptr, a.dtype(), a.device()));
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

    let x_ptr = x.data_ptr() as *const f32;
    let w_ptr = w.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);

    let in_channels = in_channels as usize;
    let out_channels = out_channels as usize;
    let out_height = out_height as usize;
    let out_width = out_width as usize;
    let in_height = in_height as usize;
    let in_width = in_width as usize;
    let kernel_height = kernel_height as usize;
    let kernel_width = kernel_width as usize;
    let stride = stride as usize;
    let padding = padding as usize;
    let dilation = dilation as usize;
    let batch_size = batch_size as usize;

    // Convert pointers to slices for parallel execution
    let x_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            x_ptr,
            (batch_size * in_channels * in_height * in_width) as usize,
        )
        .to_vec()
    };
    let w_data: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            w_ptr,
            (out_channels * in_channels * kernel_height * kernel_width) as usize,
        )
        .to_vec()
    };
    let bias_data: Option<Vec<f32>> = bias.map(|b| {
        let ptr = b.data_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, out_channels as usize).to_vec() }
    });

    // Pre-allocate output buffer
    let out_numel = (batch_size * out_channels * out_height * out_width) as usize;
    let mut out_data = vec![0.0f32; out_numel];

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let total_outputs = batch_size * out_channels * out_height * out_width;

        let results: Vec<f32> = (0..total_outputs)
            .into_par_iter()
            .map(|idx| {
                let n = idx / (out_channels * out_height * out_width);
                let rem = idx % (out_channels * out_height * out_width);
                let oc = rem / (out_height * out_width);
                let rem2 = rem % (out_height * out_width);
                let oh = rem2 / out_width;
                let ow = rem2 % out_width;

                let mut sum = 0.0f32;

                for ic in 0..in_channels {
                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            let ih = oh * stride + kh * dilation;
                            let iw = ow * stride + kw * dilation;

                            if ih < in_height && iw < in_width {
                                let x_idx =
                                    ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                                let w_idx = ((oc * in_channels + ic) * kernel_height + kh)
                                    * kernel_width
                                    + kw;

                                sum += x_data[x_idx] * w_data[w_idx];
                            }
                        }
                    }
                }

                if let Some(ref bias_vals) = bias_data {
                    sum += bias_vals[oc];
                }

                sum
            })
            .collect();

        // Copy result back to output storage
        let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(results.as_ptr(), out_ptr, out_numel);
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for n in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0f32;

                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * stride + kh * dilation;
                                    let iw = ow * stride + kw * dilation;

                                    if ih < in_height && iw < in_width {
                                        let x_idx = ((n * in_channels + ic) * in_height + ih)
                                            * in_width
                                            + iw;
                                        let w_idx = ((oc * in_channels + ic) * kernel_height + kh)
                                            * kernel_width
                                            + kw;

                                        sum += unsafe { *x_ptr.add(x_idx) }
                                            * unsafe { *w_ptr.add(w_idx) };
                                    }
                                }
                            }
                        }

                        if let Some(bias_tensor) = bias {
                            let bias_ptr = bias_tensor.data_ptr() as *const f32;
                            sum += unsafe { *bias_ptr.add(oc) };
                        }

                        let out_idx = ((n * out_channels + oc) * out_height + oh) * out_width + ow;
                        unsafe { *out_ptr.add(out_idx) = sum };
                    }
                }
            }
        }
    }

    vec![output]
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

    let output_shape: Vec<i64> = indices_shape
        .iter()
        .chain(std::iter::once(&embedding_dim))
        .copied()
        .collect();
    let mut output = Tensor::zeros(output_shape.clone(), weight.dtype(), weight.device());

    let indices_ptr = indices.data_ptr() as *const f32;
    let weight_ptr = weight.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let out_ptr = output_storage.data.as_mut_ptr() as *mut f32;

    for i in 0..batch_size as usize {
        let idx = unsafe { *indices_ptr.add(i) } as usize;
        if idx < num_embeddings as usize {
            for j in 0..embedding_dim as usize {
                let w_idx = idx * embedding_dim as usize + j;
                let o_idx = i * embedding_dim as usize + j;
                unsafe {
                    *out_ptr.add(o_idx) = *weight_ptr.add(w_idx);
                }
            }
        }
    }

    vec![output]
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
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let numel: i64 = shape.iter().product();

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut values = vec![0.0f32; numel as usize];

    for v in &mut values {
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    vec![Tensor::from_vec(values, shape)]
}

fn rand_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let numel: i64 = shape.iter().product();

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let values: Vec<f32> = (0..numel as usize).map(|_| rng.gen()).collect();

    vec![Tensor::from_vec(values, shape)]
}

fn randint_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let shape_tensor = args[0];
    let shape: Vec<i64> = shape_tensor.inner.sizes.iter().copied().collect();
    let low = args[1].item() as i32;
    let high = args[2].item() as i32;

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let numel: i64 = shape.iter().product();
    let values: Vec<f32> = (0..numel as usize)
        .map(|_| (rng.gen::<i32>() % (high - low) + low) as f32)
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
