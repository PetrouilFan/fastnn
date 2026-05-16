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
                        if ih < h && iw < w {
                            let src = (ic * h + ih) * w + iw;
                            let dst =
                                row * col_w + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                            col[dst] = data[src];
                        }
                    }
                }
            }
        }
    }
}
