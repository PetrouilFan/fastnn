#[cfg(test)]
mod tests {
    use crate::backend::cpu::im2col;

    #[test]
    fn test_im2col_simd_vs_scalar() {
        let c = 3;
        let h = 5;
        let w = 5;
        let kh = 3;
        let kw = 3;
        let stride = 1;
        let padding = 1;
        let dilation = 1;
        
        let col_w = c * kh * kw;
        let h_out = h;
        let w_out = w;
        
        let mut input = vec![0.0f32; c * h * w];
        for i in 0..c * h * w {
            input[i] = i as f32 * 0.1;
        }
        
        // Scalar
        let mut col_scalar = vec![f32::NAN; h_out * w_out * col_w];
        unsafe {
            im2col::im2col_kernel_rect(
                &input, c, h, w, kh, kw, stride, padding, dilation,
                &mut col_scalar,
            );
        }
        
        // SIMD
        let mut col_simd = vec![f32::NAN; h_out * w_out * col_w];
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if crate::backend::cpu::microkernels::has_avx2() {
            unsafe {
                im2col::im2col_kernel_rect_avx2(
                    &input, c, h, w, &mut col_simd,
                );
            }
        } else {
            // Fallback to scalar if no AVX2
            unsafe {
                im2col::im2col_kernel_rect(
                    &input, c, h, w, kh, kw, stride, padding, dilation,
                    &mut col_simd,
                );
            }
        }
        
        // Compare
        let mut max_diff = 0.0f32;
        let mut first_err_idx = 0;
        for i in 0..col_scalar.len() {
            let diff = (col_scalar[i] - col_simd[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                first_err_idx = i;
            }
        }
        
        println!("Max diff: {} at index {}", max_diff, first_err_idx);
        println!("Scalar[first_err]: {}", col_scalar[first_err_idx]);
        println!("SIMD[first_err]: {}", col_simd[first_err_idx]);
        
        if max_diff > 1e-6 {
            // Show region around error
            let start = first_err_idx.saturating_sub(5);
            let end = (first_err_idx + 5).min(col_scalar.len());
            for i in start..end {
                println!("  [{}] scalar={} simd={}", i, col_scalar[i], col_simd[i]);
            }
        }
        
        assert!(max_diff < 1e-6, "SIMD im2col diverges from scalar (max_diff={})", max_diff);
    }
}
