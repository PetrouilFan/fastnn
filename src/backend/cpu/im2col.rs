//! Im2col kernel for CPU convolution.
use crate::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
pub unsafe fn im2col_kernel(
    x: &Tensor,
    kernel_height: usize,
    kernel_width: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_height: usize,
    out_width: usize,
) -> Tensor {
    let x_shape = x.shape_ref();
    let batch_size = x_shape[0] as usize;
    let in_channels = x_shape[1] as usize;
    let in_height = x_shape[2] as usize;
    let in_width = x_shape[3] as usize;

    let col_rows = batch_size * out_height * out_width;
    let col_cols = in_channels * kernel_height * kernel_width;

    let mut col_data = vec![0.0f32; col_rows * col_cols];

    let x_ptr = x.data_ptr() as *const f32;

    #[cfg(feature = "parallel")]
    {
        if batch_size > 1 {
            use rayon::prelude::*;

            let col_rows_per_batch = out_height * out_width;

            let x_usize = x_ptr as usize;
            let col_usize = col_data.as_mut_ptr() as usize;

            (0..batch_size).into_par_iter().for_each(|n| {
                unsafe {
                    let x_p = x_usize as *const f32;
                    let col_p = (col_usize as *mut f32).add(n * col_rows_per_batch * col_cols);

                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let col_row = oh * out_width + ow;

                            for ic in 0..in_channels {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let ih = (oh * stride + kh * dilation).wrapping_sub(padding);
                                        let iw = (ow * stride + kw * dilation).wrapping_sub(padding);

                                        let col_col = ((ic * kernel_height) + kh) * kernel_width + kw;

                                        if ih < in_height && iw < in_width {
                                            let x_idx = ((n * in_channels + ic) * in_height + ih)
                                                * in_width
                                                + iw;

                                            *col_p.add(col_row * col_cols + col_col) =
                                                *x_p.add(x_idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        } else {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let col_row = oh * out_width + ow;

                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = (oh * stride + kh * dilation).wrapping_sub(padding);
                                let iw = (ow * stride + kw * dilation).wrapping_sub(padding);

                                let col_col = ((ic * kernel_height) + kh) * kernel_width + kw;

                                if ih < in_height && iw < in_width {
                                    let x_idx = (ic * in_height + ih) * in_width + iw;

                                    col_data[col_row * col_cols + col_col] =
                                        unsafe { *x_ptr.add(x_idx) };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for n in 0..batch_size {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let col_row = (n * out_height + oh) * out_width + ow;

                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = (oh * stride + kh * dilation).wrapping_sub(padding);
                                let iw = (ow * stride + kw * dilation).wrapping_sub(padding);

                                let col_col = ((ic * kernel_height) + kh) * kernel_width + kw;

                                if ih < in_height && iw < in_width {
                                    let x_idx =
                                        ((n * in_channels + ic) * in_height + ih) * in_width + iw;

                                    col_data[col_row * col_cols + col_col] =
                                        unsafe { *x_ptr.add(x_idx) };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(col_data, vec![col_rows as i64, col_cols as i64])
}
