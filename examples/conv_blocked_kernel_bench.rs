use std::time::Instant;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

use fastnn::backend::cpu::microkernels::{conv2d_f32_im2col_gemm, ConvActivation};

#[cfg(feature = "openblas")]
#[link(name = "openblas")]
extern "C" {
    fn cblas_sgemm(
        layout: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl Shape {
    fn h_out(self) -> usize {
        (self.h + 2 * self.padding).saturating_sub(self.dilation * (self.kh.saturating_sub(1)) + 1)
            / self.stride
            + 1
    }

    fn w_out(self) -> usize {
        (self.w + 2 * self.padding).saturating_sub(self.dilation * (self.kw.saturating_sub(1)) + 1)
            / self.stride
            + 1
    }

    fn spatial(self) -> usize {
        self.h_out() * self.w_out()
    }

    fn k(self) -> usize {
        (self.c / self.groups.max(1)) * self.kh * self.kw
    }
}

fn fill(len: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((x >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn materialize_im2col_nk(input: &[f32], shape: Shape, out: &mut [f32]) {
    let h_out = shape.h_out();
    let w_out = shape.w_out();
    let spatial = shape.spatial();
    let k = shape.k();
    debug_assert_eq!(out.len(), spatial * k);
    out.fill(0.0);

    for cc in 0..shape.c {
        for r in 0..shape.kh {
            for s in 0..shape.kw {
                let k_idx = (cc * shape.kh + r) * shape.kw + s;
                for oh in 0..h_out {
                    let ih_unpadded = oh * shape.stride + r * shape.dilation;
                    if ih_unpadded < shape.padding {
                        continue;
                    }
                    let ih = ih_unpadded - shape.padding;
                    if ih >= shape.h {
                        continue;
                    }
                    for ow in 0..w_out {
                        let iw_unpadded = ow * shape.stride + s * shape.dilation;
                        if iw_unpadded < shape.padding {
                            continue;
                        }
                        let iw = iw_unpadded - shape.padding;
                        if iw >= shape.w {
                            continue;
                        }
                        let n_idx = oh * w_out + ow;
                        let src = cc * shape.h * shape.w + ih * shape.w + iw;
                        out[n_idx * k + k_idx] = input[src];
                    }
                }
            }
        }
    }
}

fn materialize_im2col_kn(input: &[f32], shape: Shape, out: &mut [f32]) {
    let h_out = shape.h_out();
    let w_out = shape.w_out();
    let spatial = shape.spatial();
    let k = shape.k();
    debug_assert_eq!(out.len(), spatial * k);
    out.fill(0.0);

    for cc in 0..shape.c {
        for r in 0..shape.kh {
            for s in 0..shape.kw {
                let k_idx = (cc * shape.kh + r) * shape.kw + s;
                let col_row = k_idx * spatial;
                for oh in 0..h_out {
                    let ih_unpadded = oh * shape.stride + r * shape.dilation;
                    if ih_unpadded < shape.padding {
                        continue;
                    }
                    let ih = ih_unpadded - shape.padding;
                    if ih >= shape.h {
                        continue;
                    }
                    for ow in 0..w_out {
                        let iw_unpadded = ow * shape.stride + s * shape.dilation;
                        if iw_unpadded < shape.padding {
                            continue;
                        }
                        let iw = iw_unpadded - shape.padding;
                        if iw >= shape.w {
                            continue;
                        }
                        let n_idx = oh * w_out + ow;
                        let src = cc * shape.h * shape.w + ih * shape.w + iw;
                        out[col_row + n_idx] = input[src];
                    }
                }
            }
        }
    }
}

fn transpose_weight_km(weight: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k * m];
    for mm in 0..m {
        for kk in 0..k {
            out[kk * m + mm] = weight[mm * k + kk];
        }
    }
    out
}

#[cfg(feature = "openblas")]
fn openblas_im2col_transb(shape: Shape, col_nk: &[f32], weight: &[f32], out: &mut [f32]) {
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            shape.f as i32,
            shape.spatial() as i32,
            shape.k() as i32,
            1.0,
            weight.as_ptr(),
            shape.k() as i32,
            col_nk.as_ptr(),
            shape.k() as i32,
            0.0,
            out.as_mut_ptr(),
            shape.spatial() as i32,
        );
    }
}

fn apply_bias_silu(shape: Shape, bias: &[f32], out: &mut [f32]) {
    let spatial = shape.spatial();
    for (oc, b) in bias.iter().enumerate().take(shape.f) {
        let row = oc * spatial;
        for x in &mut out[row..row + spatial] {
            *x = silu(*x + b);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn blocked_weight_t_gemm_avx2(
    shape: Shape,
    col_kn: &[f32],
    weight_km: &[f32],
    out: &mut [f32],
) {
    let m = shape.f;
    let k = shape.k();
    let n = shape.spatial();
    out.fill(0.0);

    for m0 in (0..m).step_by(4) {
        for n0 in (0..n).step_by(8) {
            let nb = (n - n0).min(8);
            if m0 + 4 <= m && nb == 8 {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                for kk in 0..k {
                    let x = _mm256_loadu_ps(col_kn.as_ptr().add(kk * n + n0));
                    let w0 = _mm256_set1_ps(*weight_km.get_unchecked(kk * m + m0));
                    let w1 = _mm256_set1_ps(*weight_km.get_unchecked(kk * m + m0 + 1));
                    let w2 = _mm256_set1_ps(*weight_km.get_unchecked(kk * m + m0 + 2));
                    let w3 = _mm256_set1_ps(*weight_km.get_unchecked(kk * m + m0 + 3));
                    acc0 = _mm256_fmadd_ps(w0, x, acc0);
                    acc1 = _mm256_fmadd_ps(w1, x, acc1);
                    acc2 = _mm256_fmadd_ps(w2, x, acc2);
                    acc3 = _mm256_fmadd_ps(w3, x, acc3);
                }
                _mm256_storeu_ps(out.as_mut_ptr().add(m0 * n + n0), acc0);
                _mm256_storeu_ps(out.as_mut_ptr().add((m0 + 1) * n + n0), acc1);
                _mm256_storeu_ps(out.as_mut_ptr().add((m0 + 2) * n + n0), acc2);
                _mm256_storeu_ps(out.as_mut_ptr().add((m0 + 3) * n + n0), acc3);
            } else {
                for mm in m0..(m0 + 4).min(m) {
                    for nn in n0..(n0 + nb) {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            acc += weight_km[kk * m + mm] * col_kn[kk * n + nn];
                        }
                        out[mm * n + nn] = acc;
                    }
                }
            }
        }
    }
}

fn blocked_weight_t_gemm_scalar(shape: Shape, col_nk: &[f32], weight_km: &[f32], out: &mut [f32]) {
    let m = shape.f;
    let k = shape.k();
    let n = shape.spatial();
    out.fill(0.0);

    const NB: usize = 16;
    const MB: usize = 4;
    for n0 in (0..n).step_by(NB) {
        let nb = (n - n0).min(NB);
        for m0 in (0..m).step_by(MB) {
            let mb = (m - m0).min(MB);
            for nn in 0..nb {
                for mm in 0..mb {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc += weight_km[kk * m + m0 + mm] * col_nk[(n0 + nn) * k + kk];
                    }
                    out[(m0 + mm) * n + n0 + nn] = acc;
                }
            }
        }
    }
}

fn blocked_weight_t_gemm(shape: Shape, col_nk: &[f32], weight_km: &[f32], out: &mut [f32]) {
    blocked_weight_t_gemm_scalar(shape, col_nk, weight_km, out);
}

fn blocked_weight_t_gemm_kn(shape: Shape, col_kn: &[f32], weight_km: &[f32], out: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { blocked_weight_t_gemm_avx2(shape, col_kn, weight_km, out) };
            return;
        }
    }

    let m = shape.f;
    let k = shape.k();
    let n = shape.spatial();
    out.fill(0.0);
    for mm in 0..m {
        for nn in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += weight_km[kk * m + mm] * col_kn[kk * n + nn];
            }
            out[mm * n + nn] = acc;
        }
    }
}

fn bench<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    mean(&times)
}

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(80);
    let warmup = 5;
    let shape = Shape {
        name: "yolo_m64_k576_n400",
        n: 1,
        c: 64,
        h: 20,
        w: 20,
        f: 64,
        kh: 3,
        kw: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
    };
    let input = fill(shape.n * shape.c * shape.h * shape.w, 1);
    let weight = fill(shape.f * shape.k(), 2);
    let bias = fill(shape.f, 3);
    let weight_km = transpose_weight_km(&weight, shape.f, shape.k());
    let mut col_nk = vec![0.0f32; shape.spatial() * shape.k()];
    let mut col_kn = vec![0.0f32; shape.spatial() * shape.k()];
    materialize_im2col_nk(&input, shape, &mut col_nk);
    materialize_im2col_kn(&input, shape, &mut col_kn);

    let mut fastnn_out = vec![0.0f32; shape.f * shape.spatial()];
    let mut blocked_out = fastnn_out.clone();
    let mut avx2_out = fastnn_out.clone();
    let mut blocked_no_epilogue = fastnn_out.clone();
    let mut avx2_no_epilogue = fastnn_out.clone();

    conv2d_f32_im2col_gemm(
        &input,
        &weight,
        &bias,
        &mut fastnn_out,
        shape.n,
        shape.c,
        shape.h,
        shape.w,
        shape.f,
        shape.kh,
        shape.kw,
        shape.stride,
        shape.padding,
        shape.dilation,
        shape.groups,
        Some(ConvActivation::Silu),
    );
    blocked_weight_t_gemm(shape, &col_nk, &weight_km, &mut blocked_no_epilogue);
    blocked_out.copy_from_slice(&blocked_no_epilogue);
    apply_bias_silu(shape, &bias, &mut blocked_out);
    blocked_weight_t_gemm_kn(shape, &col_kn, &weight_km, &mut avx2_no_epilogue);
    avx2_out.copy_from_slice(&avx2_no_epilogue);
    apply_bias_silu(shape, &bias, &mut avx2_out);

    let fastnn_ms = bench(warmup, iters, || {
        conv2d_f32_im2col_gemm(
            &input,
            &weight,
            &bias,
            &mut fastnn_out,
            shape.n,
            shape.c,
            shape.h,
            shape.w,
            shape.f,
            shape.kh,
            shape.kw,
            shape.stride,
            shape.padding,
            shape.dilation,
            shape.groups,
            Some(ConvActivation::Silu),
        );
    });

    let materialize_ms = bench(warmup, iters, || {
        materialize_im2col_nk(&input, shape, &mut col_nk)
    });
    let materialize_kn_ms = bench(warmup, iters, || {
        materialize_im2col_kn(&input, shape, &mut col_kn)
    });
    let blocked_gemm_ms = bench(warmup, iters, || {
        blocked_weight_t_gemm(shape, &col_nk, &weight_km, &mut blocked_no_epilogue)
    });
    let avx2_gemm_ms = bench(warmup, iters, || {
        blocked_weight_t_gemm_kn(shape, &col_kn, &weight_km, &mut avx2_no_epilogue)
    });
    let epilogue_ms = bench(warmup, iters, || {
        blocked_out.copy_from_slice(&blocked_no_epilogue);
        apply_bias_silu(shape, &bias, &mut blocked_out);
    });
    let avx2_epilogue_ms = bench(warmup, iters, || {
        avx2_out.copy_from_slice(&avx2_no_epilogue);
        apply_bias_silu(shape, &bias, &mut avx2_out);
    });

    #[cfg(feature = "openblas")]
    let openblas_gemm_ms = {
        let mut openblas_out = vec![0.0f32; shape.f * shape.spatial()];
        bench(warmup, iters, || {
            openblas_im2col_transb(shape, &col_nk, &weight, &mut openblas_out)
        })
    };

    let blocked_total_ms = materialize_ms + blocked_gemm_ms + epilogue_ms;
    let avx2_total_ms = materialize_kn_ms + avx2_gemm_ms + avx2_epilogue_ms;
    let flops = 2.0 * shape.f as f64 * shape.k() as f64 * shape.spatial() as f64;

    println!(
        "shape={} gemm=({}, {}, {}) iters={} warmup={} openblas_feature={}",
        shape.name,
        shape.f,
        shape.k(),
        shape.spatial(),
        iters,
        warmup,
        cfg!(feature = "openblas")
    );
    println!("fastnn_current_total_ms={fastnn_ms:.4}");
    println!("blocked_total_ms={blocked_total_ms:.4}");
    println!("blocked_im2col_ms={materialize_ms:.4}");
    println!("blocked_gemm_ms={blocked_gemm_ms:.4}");
    println!("blocked_epilogue_ms={epilogue_ms:.4}");
    println!("avx2_total_ms={avx2_total_ms:.4}");
    println!("avx2_im2col_kn_ms={materialize_kn_ms:.4}");
    println!("avx2_gemm_ms={avx2_gemm_ms:.4}");
    println!("avx2_epilogue_ms={avx2_epilogue_ms:.4}");
    #[cfg(feature = "openblas")]
    println!("openblas_gemm_only_ms={openblas_gemm_ms:.4}");
    println!(
        "blocked_vs_fastnn_ratio={:.3}",
        blocked_total_ms / fastnn_ms
    );
    println!("avx2_vs_fastnn_ratio={:.3}", avx2_total_ms / fastnn_ms);
    println!(
        "max_abs_vs_fastnn={:.3e}",
        max_abs(&fastnn_out, &blocked_out)
    );
    println!(
        "avx2_max_abs_vs_fastnn={:.3e}",
        max_abs(&fastnn_out, &avx2_out)
    );
    println!(
        "blocked_gemm_gflops={:.1}",
        flops / (blocked_gemm_ms * 1.0e6)
    );
    println!("avx2_gemm_gflops={:.1}", flops / (avx2_gemm_ms * 1.0e6));
}
