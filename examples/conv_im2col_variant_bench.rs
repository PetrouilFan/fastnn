use std::time::Instant;

use fastnn::backend::cpu::microkernels::{conv2d_f32_im2col_gemm, ConvActivation};

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

#[allow(clippy::too_many_arguments)]
fn conv2d_kmajor_im2col_gemm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
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
) {
    let c_per_group = c / groups.max(1);
    let f_per_group = f / groups.max(1);
    let h_out =
        (h + 2 * padding).saturating_sub(dilation * (kh.saturating_sub(1)) + 1) / stride + 1;
    let w_out =
        (w + 2 * padding).saturating_sub(dilation * (kw.saturating_sub(1)) + 1) / stride + 1;
    let spatial = h_out * w_out;
    let col_w = c_per_group * kh * kw;
    let batch_stride = f * spatial;

    for g in 0..groups {
        let input_group_off = g * c_per_group * h * w;
        let f_start = g * f_per_group;
        let weight_off = f_start * col_w;
        let group_out_off = g * f_per_group * spatial;
        let mut col = vec![0.0f32; col_w * spatial]; // [K,N], row-major

        for nn in 0..n {
            col.fill(0.0);
            for cc in 0..c_per_group {
                for r in 0..kh {
                    for s in 0..kw {
                        let k_idx = (cc * kh + r) * kw + s;
                        for oh in 0..h_out {
                            let ih_unpadded = oh * stride + r * dilation;
                            if ih_unpadded < padding {
                                continue;
                            }
                            let ih = ih_unpadded - padding;
                            if ih >= h {
                                continue;
                            }
                            for ow in 0..w_out {
                                let iw_unpadded = ow * stride + s * dilation;
                                if iw_unpadded < padding {
                                    continue;
                                }
                                let iw = iw_unpadded - padding;
                                if iw >= w {
                                    continue;
                                }
                                let spatial_idx = oh * w_out + ow;
                                let src =
                                    nn * c * h * w + input_group_off + cc * h * w + ih * w + iw;
                                col[k_idx * spatial + spatial_idx] = input[src];
                            }
                        }
                    }
                }
            }
            unsafe {
                matrixmultiply::sgemm(
                    f_per_group,
                    col_w,
                    spatial,
                    1.0,
                    weight.as_ptr().add(weight_off),
                    col_w as isize,
                    1,
                    col.as_ptr(),
                    spatial as isize,
                    1,
                    0.0,
                    output.as_mut_ptr().add(group_out_off + nn * batch_stride),
                    spatial as isize,
                    1,
                );
            }
        }

        if !bias.is_empty() {
            for nn in 0..n {
                let out_base = group_out_off + nn * batch_stride;
                for oc in 0..f_per_group {
                    let b = bias[f_start + oc];
                    let row = out_base + oc * spatial;
                    for x in &mut output[row..row + spatial] {
                        *x = silu(*x + b);
                    }
                }
            }
        } else {
            for nn in 0..n {
                let out_base = group_out_off + nn * batch_stride;
                for oc in 0..f_per_group {
                    let row = out_base + oc * spatial;
                    for x in &mut output[row..row + spatial] {
                        *x = silu(*x);
                    }
                }
            }
        }
    }
}

fn bench_one(shape: Shape, warmup: usize, iters: usize) {
    let input = fill(shape.n * shape.c * shape.h * shape.w, 1);
    let weight = fill(shape.f * shape.k(), 2);
    let bias = fill(shape.f, 3);
    let mut out_ref = vec![0.0f32; shape.n * shape.f * shape.spatial()];
    let mut out_kmajor = out_ref.clone();

    for _ in 0..warmup {
        conv2d_f32_im2col_gemm(
            &input,
            &weight,
            &bias,
            &mut out_ref,
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
        conv2d_kmajor_im2col_gemm(
            &input,
            &weight,
            &bias,
            &mut out_kmajor,
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
        );
    }

    let mut current_times = Vec::with_capacity(iters);
    let mut kmajor_times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        conv2d_f32_im2col_gemm(
            &input,
            &weight,
            &bias,
            &mut out_ref,
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
        current_times.push(t0.elapsed().as_secs_f64() * 1000.0);

        let t0 = Instant::now();
        conv2d_kmajor_im2col_gemm(
            &input,
            &weight,
            &bias,
            &mut out_kmajor,
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
        );
        kmajor_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let max_abs = out_ref
        .iter()
        .zip(out_kmajor.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_current = current_times.iter().sum::<f64>() / iters as f64;
    let mean_kmajor = kmajor_times.iter().sum::<f64>() / iters as f64;
    println!(
        "{:<32} gemm=({},{},{}) current={:.4}ms kmajor={:.4}ms ratio={:.3} max_abs={:.3e}",
        shape.name,
        shape.f,
        shape.k(),
        shape.spatial(),
        mean_current,
        mean_kmajor,
        mean_kmajor / mean_current,
        max_abs
    );
}

fn main() {
    let iters = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(40);
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
    println!("iters={} warmup={}", iters, warmup);
    for shape in shapes {
        bench_one(shape, warmup, iters);
    }
}
