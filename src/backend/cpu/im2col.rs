//! Im2col kernel for CPU convolution — operates on a single image (no batch).

#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn im2col_kernel(
    data: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    col: &mut [f32],
) {
    let dk = (kernel_size - 1) * dilation + 1;
    let h_out = if h + 2 * padding >= dk {
        (h + 2 * padding - dk) / stride + 1
    } else {
        0
    };
    let w_out = if w + 2 * padding >= dk {
        (w + 2 * padding - dk) / stride + 1
    } else {
        0
    };
    let col_w = c * kernel_size * kernel_size;

    for oh in 0..h_out {
        for ow in 0..w_out {
            let row = oh * w_out + ow;
            for ic in 0..c {
                for kh in 0..kernel_size {
                    for kw in 0..kernel_size {
                        let ih = (oh * stride + kh * dilation).wrapping_sub(padding);
                        let iw = (ow * stride + kw * dilation).wrapping_sub(padding);
                        let dst =
                            row * col_w + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                        if ih < h && iw < w {
                            let src = (ic * h + ih) * w + iw;
                            col[dst] = data[src];
                        } else {
                            col[dst] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

/// Im2col kernel supporting non-square kernels (kh != kw).
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn im2col_kernel_rect(
    data: &[f32],
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    col: &mut [f32],
) {
    let dh = (kh - 1) * dilation + 1;
    let dw = (kw - 1) * dilation + 1;
    let h_out = if h + 2 * padding >= dh {
        (h + 2 * padding - dh) / stride + 1
    } else {
        0
    };
    let w_out = if w + 2 * padding >= dw {
        (w + 2 * padding - dw) / stride + 1
    } else {
        0
    };
    let col_w = c * kh * kw;

    for oh in 0..h_out {
        for ow in 0..w_out {
            let row = oh * w_out + ow;
            for ic in 0..c {
                for kkh in 0..kh {
                    for kkw in 0..kw {
                        let ih = oh * stride + kkh * dilation;
                        let iw = ow * stride + kkw * dilation;
                        let dst = row * col_w + ic * kh * kw + kkh * kw + kkw;
                        if ih < h + padding && iw < w + padding && ih >= padding && iw >= padding {
                            let src = (ic * h + (ih - padding)) * w + (iw - padding);
                            col[dst] = data[src];
                        } else {
                            col[dst] = 0.0;
                        }
                    }
                }
            }
        }
    }
}
