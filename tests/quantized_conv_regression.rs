//! Regression test: heap corruption in quantized conv path with YOLO-like shapes.
//!
//! Exercises dispatch_quantized_conv2d with U4x8, U8x4, and F16x2 packed types
//! through both row-major and block-major paths, stressing the TlsVecPool
//! recycling path with varied spatial dimensions and many iterations.

use fastnn::backends::TlsVecPool;
use fastnn::dtypes::{F16x2, PackedWord, U4x8, U8x4};
use fastnn::kernels::cpu::dispatch_quantized_conv2d;
use fastnn::packed_tensor::PackedTensor;
use fastnn::tensor::Tensor;
use rand::Rng;

/// Create a random f32 tensor with the given shape.
fn random_tensor(shape: &[i64]) -> Tensor {
    let mut rng = rand::thread_rng();
    let numel: usize = shape.iter().product::<i64>() as usize;
    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-1.0..1.0)).collect();
    Tensor::from_vec(data, shape.to_vec())
}

/// Create a packed weight matrix [oc, ic * kh * kw] from random f32 data.
fn random_packed_weight<T: PackedWord>(oc: usize, ic: usize, kh: usize, kw: usize) -> PackedTensor<T> {
    let mut rng = rand::thread_rng();
    let numel = oc * ic * kh * kw;
    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-1.0..1.0)).collect();
    PackedTensor::<T>::from_f32_auto(&data, &[oc, ic * kh * kw])
}

/// Run a quantized conv forward pass and verify all outputs are finite.
unsafe fn run_quantized_conv<T: PackedWord>(
    weight: &PackedTensor<T>,
    input: &Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let result = dispatch_quantized_conv2d::<T>(
        input,
        weight,
        None, // no bias
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    );
    // Verify all elements are finite (catches NaN/inf from memory corruption)
    let data = result.as_f32_slice();
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "Non-finite output at index {}: {}", i, v);
    }
    result
}

// =========================================================================
// 3×3 CONV tests — YOLO11n-style
// =========================================================================

/// YOLO11n first conv: 3×3 stride=2, in=3, out=16, 640×640 → 320×320
/// Edge case: in_ch=3 < U4x8::ITEMS (8) — tests small-K packing
fn test_yolo_first_conv_3x3<T: PackedWord>() {
    let oc = 16;
    let ic = 3;
    let kh = 3;
    let kw = 3;
    let h = 640;
    let w = 640;
    let stride = 2;
    let pad = 1;

    let weight = random_packed_weight::<T>(oc, ic, kh, kw);
    let input = random_tensor(&[1, ic as i64, h as i64, w as i64]);

    unsafe {
        // Row-major path (block_size=1)
        let _out = run_quantized_conv::<T>(&weight, &input, kh, stride, pad, 1, 1);

        // Block-major path
        let weight_block = weight.to_block_major(4);
        let _out_block = run_quantized_conv::<T>(&weight_block, &input, kh, stride, pad, 1, 1);
    }
}

/// YOLO11n mid conv: 3×3 stride=1, in=32, out=64, 160×160
fn test_yolo_mid_conv_3x3<T: PackedWord>() {
    let oc = 64;
    let ic = 32;
    let kh = 3;
    let kw = 3;
    let h = 160;
    let w = 160;
    let stride = 1;
    let pad = 1;

    let weight = random_packed_weight::<T>(oc, ic, kh, kw);
    let input = random_tensor(&[1, ic as i64, h as i64, w as i64]);

    unsafe {
        let _out = run_quantized_conv::<T>(&weight, &input, kh, stride, pad, 1, 1);
        let weight_block = weight.to_block_major(4);
        let _out_block = run_quantized_conv::<T>(&weight_block, &input, kh, stride, pad, 1, 1);
    }
}

/// YOLO11n deep conv: 3×3 stride=1, in=128, out=128, 80×80
fn test_yolo_deep_conv_3x3<T: PackedWord>() {
    let oc = 128;
    let ic = 128;
    let kh = 3;
    let kw = 3;
    let h = 80;
    let w = 80;
    let stride = 1;
    let pad = 1;

    let weight = random_packed_weight::<T>(oc, ic, kh, kw);
    let input = random_tensor(&[1, ic as i64, h as i64, w as i64]);

    unsafe {
        let _out = run_quantized_conv::<T>(&weight, &input, kh, stride, pad, 1, 1);
        let weight_block = weight.to_block_major(4);
        let _out_block = run_quantized_conv::<T>(&weight_block, &input, kh, stride, pad, 1, 1);
    }
}

// =========================================================================
// 1×1 CONV tests — YOLO11n bottleneck/C2f style
// =========================================================================

/// YOLO11n 1×1 conv: in=64, out=64, 160×160
fn test_yolo_1x1_conv<T: PackedWord>() {
    let oc = 64;
    let ic = 64;
    let h = 160;
    let w = 160;
    let input = random_tensor(&[1, ic as i64, h as i64, w as i64]);
    let weight = random_packed_weight::<T>(oc, ic, 1, 1);

    unsafe {
        let _out = run_quantized_conv::<T>(&weight, &input, 1, 1, 0, 1, 1);
        let weight_block = weight.to_block_major(4);
        let _out_block = run_quantized_conv::<T>(&weight_block, &input, 1, 1, 0, 1, 1);
    }
}

/// YOLO11n 1×1 projection: in=256, out=128, 40×40 (non-power-of-2 spatial)
fn test_yolo_proj_1x1_odd_spatial<T: PackedWord>() {
    let oc = 128;
    let ic = 256;
    let h = 40;
    let w = 40;
    let input = random_tensor(&[1, ic as i64, h as i64, w as i64]);
    let weight = random_packed_weight::<T>(oc, ic, 1, 1);

    unsafe {
        let _out = run_quantized_conv::<T>(&weight, &input, 1, 1, 0, 1, 1);
        let weight_block = weight.to_block_major(4);
        let _out_block = run_quantized_conv::<T>(&weight_block, &input, 1, 1, 0, 1, 1);
    }
}

// =========================================================================
// Edge cases with small K (in_ch < ITEMS_PER_WORD)
// =========================================================================

/// U4x8 has ITEMS=8; in_ch=2 means k_packed=1 per row — extreme edge case.
fn test_small_k_conv<T: PackedWord>() {
    let oc = 8;
    let ic = 2;
    let h = 32;
    let w = 32;
    let input = random_tensor(&[2, ic as i64, h as i64, w as i64]);

    // 3×3 conv with tiny in_ch
    let weight_3x3 = random_packed_weight::<T>(oc, ic, 3, 3);
    unsafe {
        let _out = run_quantized_conv::<T>(&weight_3x3, &input, 3, 1, 1, 1, 1);
        let wb = weight_3x3.to_block_major(4);
        let _outb = run_quantized_conv::<T>(&wb, &input, 3, 1, 1, 1, 1);
    }

    // 1×1 conv with tiny in_ch
    let weight_1x1 = random_packed_weight::<T>(oc, ic, 1, 1);
    unsafe {
        let _out = run_quantized_conv::<T>(&weight_1x1, &input, 1, 1, 0, 1, 1);
        let wb = weight_1x1.to_block_major(4);
        let _outb = run_quantized_conv::<T>(&wb, &input, 1, 1, 0, 1, 1);
    }
}

// =========================================================================
// Non-power-of-2 spatial sizes to trigger alignment edge cases
// =========================================================================

fn test_odd_spatial_conv<T: PackedWord>() {
    let oc = 32;
    let ic = 16;
    let h = 65;
    let w = 63;
    let input = random_tensor(&[1, ic as i64, h as i64, w as i64]);

    // 3×3 stride=2 with odd input
    let weight_3x3 = random_packed_weight::<T>(oc, ic, 3, 3);
    unsafe {
        let _out = run_quantized_conv::<T>(&weight_3x3, &input, 3, 2, 1, 1, 1);
        let wb = weight_3x3.to_block_major(4);
        let _outb = run_quantized_conv::<T>(&wb, &input, 3, 2, 1, 1, 1);
    }
}

// =========================================================================
// Multi-iteration stress test — exercises TlsVecPool recycling
// =========================================================================

fn stress_test_quantized_conv<T: PackedWord>() {
    // Vary batch sizes and spatial dims across iterations
    let configs: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)> = vec![
        // (n, ic, oc, h, w, kh, stride, pad)
        (1, 3, 16, 320, 320, 3, 2, 1),   // first conv after stride
        (2, 16, 32, 160, 160, 3, 1, 1),  // batch=2, mid conv
        (1, 32, 64, 80, 80, 3, 1, 1),    // deep conv
        (1, 64, 64, 80, 80, 1, 1, 0),    // 1×1 bottleneck
        (1, 128, 256, 40, 40, 1, 1, 0),  // 1×1 expansion
        (3, 3, 16, 640, 640, 3, 2, 1),   // batch=3, first conv
        (1, 64, 128, 40, 40, 3, 2, 1),   // stride=2 3×3
        (1, 5, 32, 33, 37, 3, 1, 1),     // odd spatial, odd channels
    ];

    for iteration in 0..50 {
        for (n, ic, oc, h, w, kh, stride, pad) in &configs {
            let input = random_tensor(&[*n as i64, *ic as i64, *h as i64, *w as i64]);

            // Row-major weight
            let weight = random_packed_weight::<T>(*oc, *ic, *kh, *kh);

            unsafe {
                let _out = run_quantized_conv::<T>(
                    &weight,
                    &input,
                    *kh,
                    *stride,
                    *pad,
                    1,
                    1,
                );
            }

            // Block-major weight (same as DAG with_packed)
            let weight_block = weight.to_block_major(4);

            unsafe {
                let _out_block = run_quantized_conv::<T>(
                    &weight_block,
                    &input,
                    *kh,
                    *stride,
                    *pad,
                    1,
                    1,
                );
            }

            // Also test with varying block sizes
            if *oc >= 8 {
                let weight_block2 = weight.to_block_major(8);
                unsafe {
                    let _outb2 = run_quantized_conv::<T>(
                        &weight_block2,
                        &input,
                        *kh,
                        *stride,
                        *pad,
                        1,
                        1,
                    );
                }
            }

            // TlsVecPool-only stress: allocate and drop many ScopedVecs
            for _ in 0..10 {
                let pool_alloc = TlsVecPool::alloc(1000);
                drop(pool_alloc);
                let pool_alloc_z = TlsVecPool::alloc_zeroed(1000);
                drop(pool_alloc_z);
            }
        }

        if iteration % 10 == 0 {
            // Occasionally use different alloc sizes to fragment the pool
            for sz in [128, 256, 512, 1024, 2048, 4096] {
                let v = TlsVecPool::alloc(sz);
                drop(v);
                let vz = TlsVecPool::alloc_zeroed(sz);
                drop(vz);
            }
        }
    }
}

// =========================================================================
// Test entry points
// =========================================================================

#[test]
fn test_quantized_conv_3x3_first_layer_u4x8() {
    test_yolo_first_conv_3x3::<U4x8>();
}

#[test]
fn test_quantized_conv_3x3_first_layer_u8x4() {
    test_yolo_first_conv_3x3::<U8x4>();
}

#[test]
fn test_quantized_conv_3x3_first_layer_f16x2() {
    test_yolo_first_conv_3x3::<F16x2>();
}

#[test]
fn test_quantized_conv_3x3_mid_layer_u4x8() {
    test_yolo_mid_conv_3x3::<U4x8>();
}

#[test]
fn test_quantized_conv_3x3_mid_layer_u8x4() {
    test_yolo_mid_conv_3x3::<U8x4>();
}

#[test]
fn test_quantized_conv_3x3_deep_layer_u4x8() {
    test_yolo_deep_conv_3x3::<U4x8>();
}

#[test]
fn test_quantized_conv_3x3_deep_layer_u8x4() {
    test_yolo_deep_conv_3x3::<U8x4>();
}

#[test]
fn test_quantized_conv_1x1_bottleneck_u4x8() {
    test_yolo_1x1_conv::<U4x8>();
}

#[test]
fn test_quantized_conv_1x1_bottleneck_u8x4() {
    test_yolo_1x1_conv::<U8x4>();
}

#[test]
fn test_quantized_conv_1x1_bottleneck_f16x2() {
    test_yolo_1x1_conv::<F16x2>();
}

#[test]
fn test_quantized_conv_1x1_proj_odd_spatial_u4x8() {
    test_yolo_proj_1x1_odd_spatial::<U4x8>();
}

#[test]
fn test_quantized_conv_small_k_u4x8() {
    test_small_k_conv::<U4x8>();
}

#[test]
fn test_quantized_conv_small_k_u8x4() {
    test_small_k_conv::<U8x4>();
}

#[test]
fn test_quantized_conv_odd_spatial_u4x8() {
    test_odd_spatial_conv::<U4x8>();
}

#[test]
fn test_quantized_conv_odd_spatial_u8x4() {
    test_odd_spatial_conv::<U8x4>();
}

#[test]
fn test_quantized_conv_odd_spatial_f16x2() {
    test_odd_spatial_conv::<F16x2>();
}

/// Stress test: 50 iterations × 8 configs × 3 paths × 2 types = 2400 conv calls
#[test]
fn stress_test_quantized_conv_u4x8() {
    stress_test_quantized_conv::<U4x8>();
}

#[test]
fn stress_test_quantized_conv_u8x4() {
    stress_test_quantized_conv::<U8x4>();
}

#[test]
fn stress_test_quantized_conv_f16x2() {
    stress_test_quantized_conv::<F16x2>();
}
