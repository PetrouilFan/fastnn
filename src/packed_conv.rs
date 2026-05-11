use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::backends::cpu;
use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;
use crate::tensor::Tensor;

fn compute_out_dim(
    in_dim: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let dk = (kernel - 1) * dilation + 1;
    if in_dim + 2 * padding < dk {
        0
    } else {
        (in_dim + 2 * padding - dk) / stride + 1
    }
}

#[allow(clippy::too_many_arguments)]
fn im2col_flat(
    x: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    dilation: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let col_h = n * oh * ow;
    let col_w = c * kh * kw;
    let mut col = vec![0.0; col_h * col_w];
    for batch in 0..n {
        for out_y in 0..oh {
            for out_x in 0..ow {
                let col_row = (batch * oh * ow + out_y * ow + out_x) * col_w;
                for ic in 0..c {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let in_y = (out_y as i64 * stride as i64 + ky as i64 * dilation as i64)
                                - pad as i64;
                            let in_x = (out_x as i64 * stride as i64 + kx as i64 * dilation as i64)
                                - pad as i64;
                            if in_y >= 0 && in_y < h as i64 && in_x >= 0 && in_x < w as i64 {
                                let src_idx = batch * (c * h * w)
                                    + ic * (h * w)
                                    + in_y as usize * w
                                    + in_x as usize;
                                let dst_idx = col_row + ic * (kh * kw) + ky * kw + kx;
                                col[dst_idx] = x[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    col
}

/// A 2D convolution layer with packed precision weights.
///
/// Weights are stored in packed format (U4x8, U8x4, F16x2, or F32x1) with shape
/// [out_channels, in_channels * kernel_size * kernel_size] (im2col-compatible layout).
/// Bias remains in f32. Master weights (f32) are kept for training.
///
/// v1: CPU-only, no backward pass, no groups support yet.
pub struct PackedConv2d<T: PackedWord> {
    /// Packed weights [out_channels, in_channels * kh * kw]
    pub weight: PackedTensor<T>,
    /// Bias in f32 with shape `[out_channels]`
    pub bias: Option<Vec<f32>>,
    /// Master weights in f32 for optimizer updates (training only)
    pub master_weight: Option<Vec<f32>>,
    /// Cached input from the last forward pass (for backward computation)
    pub cached_input: Mutex<Option<Vec<f32>>>,
    /// Gradient accumulator for master weights
    pub master_grad: Mutex<Vec<f32>>,
    /// Training state (train/eval)
    pub training: AtomicBool,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl<T: PackedWord> PackedConv2d<T> {
    /// Create a new packed 2D convolution layer with Kaiming initialization.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        use_bias: bool,
    ) -> Self {
        let numel = in_channels * out_channels * kernel_size * kernel_size;
        // Kaiming initialization
        let std = (2.0 / (in_channels * kernel_size * kernel_size) as f32).sqrt();
        let mut rng = rand::thread_rng();
        let master: Vec<f32> = (0..numel)
            .map(|_| {
                let u1: f32 = rand::Rng::gen::<f32>(&mut rng).max(1e-10);
                let u2: f32 = rand::Rng::gen::<f32>(&mut rng);
                std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();

        let weight = PackedTensor::from_f32_auto(
            &master,
            &[out_channels, in_channels * kernel_size * kernel_size],
        );
        let bias = if use_bias {
            Some(vec![0.0; out_channels])
        } else {
            None
        };

        PackedConv2d {
            weight,
            bias,
            master_weight: Some(master),
            cached_input: Mutex::new(None),
            master_grad: Mutex::new(vec![0.0; numel]),
            training: AtomicBool::new(true),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Create a packed conv2d from pre-existing packed weights.
    #[allow(clippy::too_many_arguments)]
    pub fn from_packed(
        weight: PackedTensor<T>,
        bias: Option<Vec<f32>>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        PackedConv2d {
            weight,
            bias,
            master_weight: None,
            cached_input: Mutex::new(None),
            master_grad: Mutex::new(vec![
                0.0;
                in_channels * out_channels * kernel_size * kernel_size
            ]),
            training: AtomicBool::new(true),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Forward pass on CPU. Input tensor must be 4D [N, C, H, W] in F32.
    /// Returns 4D tensor [N, OC, OH, OW].
    pub fn forward_cpu(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            self.groups, 1,
            "PackedConv2d: groups={} is not supported in v1 (groups=1 only)",
            self.groups
        );
        let x = input.to_numpy();
        let shape = input.shape();
        let n = shape[0] as usize;
        let c = shape[1] as usize;
        let h = shape[2] as usize;
        let w = shape[3] as usize;

        let oh = compute_out_dim(
            h,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );
        let ow = compute_out_dim(
            w,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );

        let col = im2col_flat(
            &x,
            n,
            c,
            h,
            w,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            oh,
            ow,
        );

        let col_w = c * self.kernel_size * self.kernel_size;
        let num_out = n * oh * ow;
        let oc = self.out_channels;
        let mut output = vec![0.0; num_out * oc];

        for row in 0..num_out {
            let col_start = row * col_w;
            let out_start = row * oc;
            cpu::gemv_cpu(
                &self.weight,
                &col[col_start..col_start + col_w],
                &mut output[out_start..out_start + oc],
            );
            if let Some(ref bias) = self.bias {
                for j in 0..oc {
                    output[out_start + j] += bias[j];
                }
            }
        }

        let out_shape = vec![n as i64, oc as i64, oh as i64, ow as i64];
        Tensor::from_vec(output, out_shape)
    }

    /// Get the number of parameters.
    pub fn num_params(&self) -> usize {
        let w = self.out_channels * self.in_channels * self.kernel_size * self.kernel_size;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Repack the master weights into the packed representation.
    /// Call this before each forward pass during training.
    pub fn repack(&mut self) {
        if let Some(ref master) = self.master_weight {
            let scale = PackedTensor::<T>::compute_scale(master);
            self.weight = PackedTensor::from_f32_slice(
                master,
                &[
                    self.out_channels,
                    self.in_channels * self.kernel_size * self.kernel_size,
                ],
                scale,
                0.0,
            );
        }
    }

    /// Zero the master gradient.
    pub fn zero_grad(&self) {
        self.master_grad.lock().unwrap().fill(0.0);
    }

    /// Set training mode.
    pub fn train_mode(&self) {
        self.training.store(true, Ordering::Relaxed);
    }

    /// Set evaluation mode.
    pub fn eval_mode(&self) {
        self.training.store(false, Ordering::Relaxed);
    }

    /// Check if in training mode.
    pub fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    /// Memory usage in bytes for the packed weights.
    pub fn packed_weight_bytes(&self) -> usize {
        self.weight.packed_len() * std::mem::size_of::<T>()
    }

    /// Memory savings ratio vs f32.
    pub fn memory_savings(&self) -> f32 {
        let f32_bytes =
            self.out_channels * self.in_channels * self.kernel_size * self.kernel_size * 4;
        let packed_bytes = self.packed_weight_bytes();
        f32_bytes as f32 / packed_bytes as f32
    }

    /// Fused forward: conv2d → ReLU, no intermediate tensor allocation.
    pub fn forward_relu(&self, x: &Tensor) -> Tensor {
        let mut out = self.forward_cpu(x);
        let data = out.as_f32_slice_mut();
        for v in data.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        out
    }
}

impl<T: PackedWord> Clone for PackedConv2d<T> {
    fn clone(&self) -> Self {
        PackedConv2d {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            master_weight: self.master_weight.clone(),
            cached_input: Mutex::new(None),
            master_grad: Mutex::new(self.master_grad.lock().unwrap().clone()),
            training: AtomicBool::new(self.training.load(Ordering::Relaxed)),
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        }
    }
}

/// Convenience type aliases
pub type Conv2d4 = PackedConv2d<crate::dtypes::U4x8>;
pub type Conv2d8 = PackedConv2d<crate::dtypes::U8x4>;
pub type Conv2d16 = PackedConv2d<crate::dtypes::F16x2>;
pub type Conv2d32 = PackedConv2d<crate::dtypes::F32x1>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F32x1, U4x8};

    #[test]
    fn test_packed_conv2d_creation() {
        let layer = PackedConv2d::<U4x8>::new(3, 16, 3, 1, 0, 1, 1, true);
        assert_eq!(layer.in_channels, 3);
        assert_eq!(layer.out_channels, 16);
        assert_eq!(layer.kernel_size, 3);
        assert!(layer.bias.is_some());
        assert!(layer.master_weight.is_some());
    }

    #[test]
    fn test_packed_conv2d_forward_cpu() {
        let layer = PackedConv2d::<F32x1>::new(1, 2, 3, 1, 0, 1, 1, true);
        let input = Tensor::from_vec(
            vec![0.5; 25],
            vec![1, 1, 5, 5],
        );
        let output = layer.forward_cpu(&input);
        let out_shape = output.shape();
        assert_eq!(out_shape, vec![1, 2, 3, 3]);
    }

    #[test]
    fn test_compute_out_dim() {
        assert_eq!(compute_out_dim(5, 3, 1, 0, 1), 3);
        assert_eq!(compute_out_dim(5, 3, 2, 0, 1), 2);
        assert_eq!(compute_out_dim(5, 3, 1, 1, 1), 5);
    }

    #[test]
    fn test_packed_conv2d_dilation_gt_1() {
        let layer = PackedConv2d::<F32x1>::new(1, 2, 3, 1, 0, 2, 1, false);
        let input = Tensor::from_vec(vec![0.5; 49], vec![1, 1, 7, 7]);
        let output = layer.forward_cpu(&input);
        let out_shape = output.shape();
        assert_eq!(out_shape, vec![1, 2, 3, 3]);
        assert!(output.as_f32_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_packed_conv2d_groups_gt_1_panics() {
        let layer = PackedConv2d::<F32x1>::new(4, 4, 3, 1, 1, 1, 2, false);
        let input = Tensor::from_vec(vec![0.5; 36], vec![1, 4, 3, 3]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            layer.forward_cpu(&input);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_out_dim_zero() {
        assert_eq!(compute_out_dim(0, 3, 1, 0, 1), 0);
        assert_eq!(compute_out_dim(2, 3, 1, 0, 1), 0);
    }

    #[test]
    fn test_packed_conv2d_u4_non_multiple_feature_dim() {
        let layer = PackedConv2d::<U4x8>::new(5, 16, 3, 1, 0, 1, 1, false);
        assert_eq!(layer.in_channels, 5);
        assert_eq!(layer.out_channels, 16);
        let input = Tensor::from_vec(vec![0.3; 5 * 7 * 7], vec![1, 5, 7, 7]);
        let output = layer.forward_cpu(&input);
        let out_shape = output.shape();
        assert_eq!(out_shape, vec![1, 16, 5, 5]);
        assert!(output.as_f32_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_packed_conv2d_minimal_input() {
        let layer = PackedConv2d::<F32x1>::new(1, 1, 1, 1, 0, 1, 1, false);
        let input = Tensor::from_vec(vec![0.5f32], vec![1, 1, 1, 1]);
        let output = layer.forward_cpu(&input);
        let out_shape = output.shape();
        assert_eq!(out_shape, vec![1, 1, 1, 1]);
        assert!(output.as_f32_slice().iter().all(|v| v.is_finite()));
    }
}
