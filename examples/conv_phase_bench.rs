use std::time::Instant;

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

#[cfg(feature = "openblas")]
#[inline]
fn openblas_conv_gemm_enabled() -> bool {
    std::env::var("FASTNN_DISABLE_OPENBLAS_CONV_GEMM")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(true)
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
    fn flops(self) -> f64 {
        2.0 * self.n as f64 * self.f as f64 * self.k() as f64 * self.spatial() as f64
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

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[allow(clippy::too_many_arguments)]
fn build_im2col(
    input: &[f32],
    col_matrix: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    spatial_size: usize,
    col_w: usize,
) {
    let c_per_group = c / groups.max(1);
    for g in 0..groups {
        let input_group_off = g * c_per_group * h * w;
        let group_col_off = g * n * spatial_size * col_w;
        for nn in 0..n {
            let col_start = group_col_off + nn * spatial_size * col_w;
            unsafe {
                fastnn::backend::cpu::im2col::im2col_dispatch(
                    &input[nn * (c * h * w) + input_group_off..],
                    c_per_group,
                    h,
                    w,
                    kh,
                    kw,
                    stride,
                    padding,
                    dilation,
                    &mut col_matrix[col_start..col_start + spatial_size * col_w],
                    false, // benchmark does not track min/max
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gemm_only(
    weight: &[f32],
    col_matrix: &[f32],
    output: &mut [f32],
    n: usize,
    f: usize,
    groups: usize,
    spatial_size: usize,
    col_w: usize,
) {
    let f_per_group = f / groups.max(1);
    let batch_stride = f * spatial_size;
    for g in 0..groups {
        let f_start = g * f_per_group;
        let weight_off = f_start * col_w;
        let group_out_off = g * f_per_group * spatial_size;
        let group_col_off = g * n * spatial_size * col_w;
        for nn in 0..n {
            let col_start = group_col_off + nn * spatial_size * col_w;
            unsafe {
                conv_phase_sgemm(
                    f_per_group,
                    col_w,
                    spatial_size,
                    weight.as_ptr().add(weight_off),
                    col_matrix.as_ptr().add(col_start),
                    output.as_mut_ptr().add(group_out_off + nn * batch_stride),
                );
            }
        }
    }
}

#[cfg(feature = "openblas")]
unsafe fn conv_phase_sgemm(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b_im2col_nk: *const f32,
    c: *mut f32,
) {
    if openblas_conv_gemm_enabled()
        && m <= i32::MAX as usize
        && n <= i32::MAX as usize
        && k <= i32::MAX as usize
    {
        const CBLAS_ROW_MAJOR: i32 = 101;
        const CBLAS_NO_TRANS: i32 = 111;
        const CBLAS_TRANS: i32 = 112;
        // General Conv im2col stores B physically as [N,K] row-major; consume it
        // as logical [K,N] with TransB, matching conv2d_f32_im2col_gemm.
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            b_im2col_nk,
            k as i32,
            0.0,
            c,
            n as i32,
        );
    } else {
        fastnn::backend::cpu::sgemm::sgemm(
            m,
            k,
            n,
            1.0,
            a,
            k as isize,
            1isize,
            b_im2col_nk,
            1isize,
            k as isize,
            0.0,
            c,
            n as isize,
            1isize,
        );
    }
}

#[cfg(not(feature = "openblas"))]
unsafe fn conv_phase_sgemm(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b_im2col_nk: *const f32,
    c: *mut f32,
) {
    fastnn::backend::cpu::sgemm::sgemm(
        m,
        k,
        n,
        1.0,
        a,
        k as isize,
        1isize,
        b_im2col_nk,
        1isize,
        k as isize,
        0.0,
        c,
        n as isize,
        1isize,
    );
}

fn apply_bias_silu(output: &mut [f32], bias: &[f32], n: usize, f: usize, spatial_size: usize) {
    let batch_stride = f * spatial_size;
    for nn in 0..n {
        let out_base = nn * batch_stride;
        for (oc, bias_val) in bias.iter().copied().enumerate().take(f) {
            let row_start = out_base + oc * spatial_size;
            for s in 0..spatial_size {
                output[row_start + s] = silu(output[row_start + s] + bias_val);
            }
        }
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn bench_one(shape: Shape, warmup: usize, iters: usize) {
    let spatial = shape.spatial();
    let col_w = shape.k();
    let input = fill(shape.n * shape.c * shape.h * shape.w, 1);
    let weight = fill(shape.f * col_w, 2);
    let bias = fill(shape.f, 3);
    let mut output_total = vec![0.0f32; shape.n * shape.f * spatial];
    let mut output_phased = output_total.clone();
    let mut col_matrix = vec![0.0f32; shape.groups * shape.n * spatial * col_w];

    for _ in 0..warmup {
        conv2d_f32_im2col_gemm(
            &input,
            &weight,
            &bias,
            &mut output_total,
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
        build_im2col(
            &input,
            &mut col_matrix,
            shape.n,
            shape.c,
            shape.h,
            shape.w,
            shape.kh,
            shape.kw,
            shape.stride,
            shape.padding,
            shape.dilation,
            shape.groups,
            spatial,
            col_w,
        );
        run_gemm_only(
            &weight,
            &col_matrix,
            &mut output_phased,
            shape.n,
            shape.f,
            shape.groups,
            spatial,
            col_w,
        );
        apply_bias_silu(&mut output_phased, &bias, shape.n, shape.f, spatial);
    }

    let mut total_ms = Vec::with_capacity(iters);
    let mut im2col_ms = Vec::with_capacity(iters);
    let mut gemm_ms = Vec::with_capacity(iters);
    let mut epilogue_ms = Vec::with_capacity(iters);

    for _ in 0..iters {
        let t0 = Instant::now();
        conv2d_f32_im2col_gemm(
            &input,
            &weight,
            &bias,
            &mut output_total,
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
        total_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let t0 = Instant::now();
        build_im2col(
            &input,
            &mut col_matrix,
            shape.n,
            shape.c,
            shape.h,
            shape.w,
            shape.kh,
            shape.kw,
            shape.stride,
            shape.padding,
            shape.dilation,
            shape.groups,
            spatial,
            col_w,
        );
        im2col_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let t0 = Instant::now();
        run_gemm_only(
            &weight,
            &col_matrix,
            &mut output_phased,
            shape.n,
            shape.f,
            shape.groups,
            spatial,
            col_w,
        );
        gemm_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let t0 = Instant::now();
        apply_bias_silu(&mut output_phased, &bias, shape.n, shape.f, spatial);
        epilogue_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let max_abs = output_total
        .iter()
        .zip(output_phased.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let total = mean(&total_ms);
    let im2col = mean(&im2col_ms);
    let gemm = mean(&gemm_ms);
    let epilogue = mean(&epilogue_ms);
    let phased = im2col + gemm + epilogue;
    println!(
        "{:<30} gemm=({},{},{}) total={:.4}ms phased={:.4}ms im2col={:.4}ms({:>5.1}%) gemm={:.4}ms({:>5.1}%) silu={:.4}ms({:>5.1}%) gf/s={:.1} max_abs={:.3e}",
        shape.name,
        shape.f,
        shape.k(),
        shape.spatial(),
        total,
        phased,
        im2col,
        100.0 * im2col / phased,
        gemm,
        100.0 * gemm / phased,
        epilogue,
        100.0 * epilogue / phased,
        shape.flops() / (total * 1.0e6),
        max_abs,
    );
}

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(60);
    let warmup = 5;
    let shapes = [
        Shape {
            name: "stem_f16_c3_sp25600",
            n: 1,
            c: 3,
            h: 320,
            w: 320,
            f: 16,
            kh: 3,
            kw: 3,
            stride: 2,
            padding: 1,
            dilation: 1,
            groups: 1,
        },
        Shape {
            name: "yolo_f32_c16_sp6400",
            n: 1,
            c: 16,
            h: 80,
            w: 80,
            f: 32,
            kh: 3,
            kw: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        },
        Shape {
            name: "yolo_f16_c16_sp6400",
            n: 1,
            c: 16,
            h: 80,
            w: 80,
            f: 16,
            kh: 3,
            kw: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        },
        Shape {
            name: "yolo_f32_c32_sp1600",
            n: 1,
            c: 32,
            h: 40,
            w: 40,
            f: 32,
            kh: 3,
            kw: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        },
        Shape {
            name: "yolo_f64_c64_sp400",
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
        },
        Shape {
            name: "yolo_f64_c64_sp1600",
            n: 1,
            c: 64,
            h: 40,
            w: 40,
            f: 64,
            kh: 3,
            kw: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        },
        Shape {
            name: "yolo_f80_c80_sp1600",
            n: 1,
            c: 80,
            h: 40,
            w: 40,
            f: 80,
            kh: 3,
            kw: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        },
        Shape {
            name: "yolo_f128_c128_sp100",
            n: 1,
            c: 128,
            h: 10,
            w: 10,
            f: 128,
            kh: 3,
            kw: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
        },
    ];
    println!(
        "iters={} warmup={} gemm_backend={} openblas_disabled={}",
        iters,
        warmup,
        gemm_backend_name(),
        std::env::var("FASTNN_DISABLE_OPENBLAS_CONV_GEMM").unwrap_or_default()
    );
    for shape in shapes {
        bench_one(shape, warmup, iters);
    }
}

fn gemm_backend_name() -> &'static str {
    #[cfg(feature = "openblas")]
    {
        if openblas_conv_gemm_enabled() {
            "openblas"
        } else {
            "matrixmultiply"
        }
    }
    #[cfg(not(feature = "openblas"))]
    {
        "matrixmultiply"
    }
}
