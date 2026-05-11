use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use crate::backends::cpu;
use crate::dtypes::PackedWord;
use crate::kernels::gpu::{get_context, GpuBuffer, GpuContext};
use crate::packed_tensor::PackedTensor;

/// Global backend selector
static BACKEND: AtomicU8 = AtomicU8::new(0); // 0 = CPU, 1 = wgpu

/// Switch to wgpu backend (if available).
pub fn use_wgpu() {
    BACKEND.store(1, Ordering::Relaxed);
}

/// Switch to CPU backend.
pub fn use_cpu() {
    BACKEND.store(0, Ordering::Relaxed);
}

/// Check if wgpu backend is active.
pub fn is_wgpu() -> bool {
    BACKEND.load(Ordering::Relaxed) == 1
}

/// A linear layer with packed precision weights.
///
/// Weights are stored in packed format (U4x8, U8x4, F16x2, or F32x1).
/// Bias remains in f32 (tiny memory overhead).
/// Master weights (f32) are kept for training.
pub struct PackedLinear<T: PackedWord> {
    /// Packed weights [out_features, in_features]
    pub weight: PackedTensor<T>,
    /// Bias in f32 with shape `[out_features]`
    pub bias: Option<Vec<f32>>,
    /// Master weights in f32 for optimizer updates (training only)
    pub master_weight: Option<Vec<f32>>,
    /// Cached input from the last forward pass (for backward computation)
    pub cached_input: Mutex<Option<Vec<f32>>>,
    /// Gradient accumulator for master weights
    pub master_grad: Mutex<Vec<f32>>,
    /// Training state (train/eval)
    pub training: std::sync::atomic::AtomicBool,
    pub in_features: usize,
    pub out_features: usize,
    /// Cached GPU buffer for weights — None until first GPU forward call
    pub gpu_weight_buf: Arc<Mutex<Option<GpuBuffer>>>,
    /// Cached GPU buffer for output — None until first GPU forward call
    pub gpu_output_buf: Arc<Mutex<Option<GpuBuffer>>>,
    /// Cached GPU buffer for uniform params — None until first GPU forward call
    pub gpu_params_buf: Arc<Mutex<Option<GpuBuffer>>>,
    /// Cached GPU buffer for activations — None until first GPU forward call
    pub gpu_activation_buf: Arc<Mutex<Option<GpuBuffer>>>,
    /// Cached GPU bind group — None until first GPU forward call
    pub gpu_bind_group: Arc<Mutex<Option<wgpu::BindGroup>>>,
}

impl<T: PackedWord> PackedLinear<T> {
    /// Create a new packed linear layer with Kaiming initialization.
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        assert!(
            in_features > 0,
            "PackedLinear: in_features must be > 0 (got {})",
            in_features
        );
        assert!(
            out_features > 0,
            "PackedLinear: out_features must be > 0 (got {})",
            out_features
        );
        let numel = in_features * out_features;
        // Kaiming initialization
        let std = (2.0 / in_features as f32).sqrt();
        let mut rng = rand::thread_rng();
        let master: Vec<f32> = (0..numel)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rand::Rng::gen::<f32>(&mut rng).max(1e-10);
                let u2: f32 = rand::Rng::gen::<f32>(&mut rng);
                std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();

        let weight = PackedTensor::from_f32_auto(&master, &[out_features, in_features]);
        let bias = if use_bias {
            Some(vec![0.0; out_features])
        } else {
            None
        };

        let master_grad_size = in_features * out_features;
        PackedLinear {
            weight,
            bias,
            master_weight: Some(master),
            cached_input: Mutex::new(None),
            master_grad: Mutex::new(vec![0.0; master_grad_size]),
            training: std::sync::atomic::AtomicBool::new(true),
            in_features,
            out_features,
            gpu_weight_buf: Arc::new(Mutex::new(None)),
            gpu_output_buf: Arc::new(Mutex::new(None)),
            gpu_params_buf: Arc::new(Mutex::new(None)),
            gpu_activation_buf: Arc::new(Mutex::new(None)),
            gpu_bind_group: Arc::new(Mutex::new(None)),
        }
    }

    /// Create from pre-existing packed weights.
    pub fn from_packed(
        weight: PackedTensor<T>,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        PackedLinear {
            weight,
            bias,
            master_weight: None,
            cached_input: Mutex::new(None),
            master_grad: Mutex::new(vec![0.0; in_features * out_features]),
            training: std::sync::atomic::AtomicBool::new(true),
            in_features,
            out_features,
            gpu_weight_buf: Arc::new(Mutex::new(None)),
            gpu_output_buf: Arc::new(Mutex::new(None)),
            gpu_params_buf: Arc::new(Mutex::new(None)),
            gpu_activation_buf: Arc::new(Mutex::new(None)),
            gpu_bind_group: Arc::new(Mutex::new(None)),
        }
    }

    /// Forward pass with caller-provided output buffer (avoids allocation).
    pub fn forward_into(&self, input: &[f32], output: &mut [f32]) {
        // Cache input for backward pass (only in training mode)
        if self.training.load(std::sync::atomic::Ordering::Relaxed) {
            *self.cached_input.lock().unwrap() = Some(input.to_vec());
        }
        assert_eq!(
            input.len(),
            self.in_features,
            "PackedLinear::forward_into: input length {} != in_features {}",
            input.len(),
            self.in_features
        );
        assert_eq!(
            output.len(),
            self.out_features,
            "PackedLinear::forward_into: output length {} != out_features {}",
            output.len(),
            self.out_features
        );
        output.fill(0.0);
        cpu::gemv_cpu(&self.weight, input, output);
        if let Some(ref bias) = self.bias {
            for (o, b) in output.iter_mut().zip(bias.iter()) {
                *o += *b;
            }
        }
    }

    /// Forward pass on CPU.
    pub fn forward_cpu(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.out_features];
        self.forward_into(input, &mut output);
        output
    }

    /// Forward pass on GPU (requires wgpu context).
    pub fn forward_wgpu(&self, input: &[f32]) -> Vec<f32> {
        use crate::backends::wgpu::gemv_wgpu_persistent;
        let ctx = get_context(0);
        let (weight_buf, output_buf, params_buf, activation_buf) = self.ensure_gpu_bufs(&ctx);

        let mut output = gemv_wgpu_persistent::<T>(
            &self.gpu_bind_group,
            weight_buf,
            output_buf,
            params_buf,
            activation_buf,
            input,
            self.out_features as u32,
            self.in_features as u32,
            self.weight.packed_len() as u32,
            self.weight.scale(),
            self.weight.zero(),
        );

        if let Some(ref bias) = self.bias {
            for (o, b) in output.iter_mut().zip(bias.iter()) {
                *o += b;
            }
        }
        output
    }

    /// Forward pass with automatic backend selection.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        if is_wgpu() {
            self.forward_wgpu(input)
        } else {
            self.forward_cpu(input)
        }
    }

    /// Compute backward pass for input gradient.
    /// grad_output: [out_features] gradient flowing back from loss
    /// Returns: [in_features] gradient for input
    pub fn backward(&self, grad_output: &[f32]) -> Vec<f32> {
        let out_f = self.out_features;
        let in_f = self.in_features;
        let cached = self.cached_input.lock().unwrap();
        let input = cached
            .as_ref()
            .expect("backward called without forward pass");

        // Compute input gradient: W^T @ grad_output
        let grad_input =
            cpu::packed_linear_backward_input_cpu(&self.weight, grad_output, input, out_f, in_f);

        // Accumulate weight gradient: dL/dW = dL/dout ⊗ input
        // grad_output[j] * input[i] → master_grad[j * in_f + i] += ...
        let mut mg = self.master_grad.lock().unwrap();
        cpu::packed_linear_backward_weight_cpu(
            &self.weight,
            grad_output,
            input,
            &mut mg,
            out_f,
            in_f,
        );

        grad_input
    }

    /// Zero the master gradient
    pub fn zero_grad(&self) {
        self.master_grad.lock().unwrap().fill(0.0);
    }

    /// Set training mode
    pub fn train_mode(&self) {
        self.training
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set evaluation mode
    pub fn eval_mode(&self) {
        self.training
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Check if in training mode
    pub fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Ensure GPU buffers are created and cached.
    /// Returns (weight_buf, output_buf, params_buf, activation_buf).
    fn ensure_gpu_bufs(
        &self,
        ctx: &GpuContext,
    ) -> (
        Arc<wgpu::Buffer>,
        Arc<wgpu::Buffer>,
        Arc<wgpu::Buffer>,
        Arc<wgpu::Buffer>,
    ) {
        {
            let mut guard = self.gpu_weight_buf.lock().unwrap();
            if guard.is_none() {
                let buf = ctx.create_gpu_buffer_from_bytes(self.weight.as_bytes(), "packed_weight");
                *guard = Some(buf);
            }
        }
        {
            let mut guard = self.gpu_output_buf.lock().unwrap();
            if guard.is_none() {
                let buf =
                    ctx.create_buffer(self.out_features * std::mem::size_of::<f32>(), "output");
                *guard = Some(buf);
            }
        }
        {
            let mut guard = self.gpu_params_buf.lock().unwrap();
            if guard.is_none() {
                // GemvParams is bound as uniform — needs UNIFORM | COPY_DST usage
                let buf = ctx.create_buffer_with_usage(
                    20,
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    "gemv_params",
                );
                *guard = Some(buf);
            }
        }
        {
            let mut guard = self.gpu_activation_buf.lock().unwrap();
            if guard.is_none() {
                let buf =
                    ctx.create_buffer(self.in_features * std::mem::size_of::<f32>(), "activations");
                *guard = Some(buf);
            }
        }
        let weight = self
            .gpu_weight_buf
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .buffer
            .clone();
        let output = self
            .gpu_output_buf
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .buffer
            .clone();
        let params = self
            .gpu_params_buf
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .buffer
            .clone();
        let activation = self
            .gpu_activation_buf
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .buffer
            .clone();
        (weight, output, params, activation)
    }

    /// Repack the master weights into the packed representation.
    /// Call this before each forward pass during training.
    pub fn repack(&mut self) {
        if let Some(ref master) = self.master_weight {
            let scale = PackedTensor::<T>::compute_scale(master);
            self.weight = PackedTensor::from_f32_slice(
                master,
                &[self.out_features, self.in_features],
                scale,
                0.0,
            );
            // Invalidate GPU cache because weights changed
            *self.gpu_weight_buf.lock().unwrap() = None;
            *self.gpu_params_buf.lock().unwrap() = None;
            *self.gpu_bind_group.lock().unwrap() = None;
        }
    }

    /// Get the number of parameters.
    pub fn num_params(&self) -> usize {
        let w = self.out_features * self.in_features;
        let b = if self.bias.is_some() {
            self.out_features
        } else {
            0
        };
        w + b
    }

    /// Memory usage in bytes for the packed weights.
    pub fn packed_weight_bytes(&self) -> usize {
        self.weight.packed_len() * std::mem::size_of::<T>()
    }

    /// Memory savings ratio vs f32.
    pub fn memory_savings(&self) -> f32 {
        let f32_bytes = self.out_features * self.in_features * 4;
        let packed_bytes = self.packed_weight_bytes();
        f32_bytes as f32 / packed_bytes as f32
    }

    /// Fused forward: linear → GELU, fused in one pass.
    pub fn forward_gelu(&self, input: &[f32]) -> Vec<f32> {
        let mut out = self.forward_cpu(input);
        for v in out.iter_mut() {
            let x = *v;
            // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            // erf approximation via Abramowitz & Stegun
            let z = x * 0.707_106_77;
            let t = 1.0 / (1.0 + 0.3275911 * z.abs());
            let poly = ((((1.061_405_4 * t + -1.453_152_1) * t) + 1.421_413_8) * t + -0.284_496_72)
                * t
                + 0.254_829_6;
            let erf = 1.0 - poly * (-z * z).exp();
            let erf_sgn = if z >= 0.0 { erf } else { -erf };
            *v = 0.5 * x * (1.0 + erf_sgn);
        }
        out
    }
}

impl<T: PackedWord> Clone for PackedLinear<T> {
    fn clone(&self) -> Self {
        PackedLinear {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            master_weight: self.master_weight.clone(),
            cached_input: Mutex::new(None),
            master_grad: Mutex::new(self.master_grad.lock().unwrap().clone()),
            training: std::sync::atomic::AtomicBool::new(
                self.training.load(std::sync::atomic::Ordering::Relaxed),
            ),
            in_features: self.in_features,
            out_features: self.out_features,
            gpu_weight_buf: Arc::new(Mutex::new(None)),
            gpu_output_buf: Arc::new(Mutex::new(None)),
            gpu_params_buf: Arc::new(Mutex::new(None)),
            gpu_activation_buf: Arc::new(Mutex::new(None)),
            gpu_bind_group: Arc::new(Mutex::new(None)),
        }
    }
}

/// Convenience type aliases
pub type Linear4 = PackedLinear<crate::dtypes::U4x8>;
pub type Linear8 = PackedLinear<crate::dtypes::U8x4>;
pub type Linear16 = PackedLinear<crate::dtypes::F16x2>;
pub type Linear32 = PackedLinear<crate::dtypes::F32x1>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F32x1, U4x8, U8x4};

    #[test]
    fn test_packed_linear_creation() {
        let layer = PackedLinear::<U4x8>::new(128, 64, true);
        assert_eq!(layer.in_features, 128);
        assert_eq!(layer.out_features, 64);
        assert!(layer.bias.is_some());
        assert!(layer.master_weight.is_some());
    }

    #[test]
    fn test_packed_linear_forward_cpu() {
        let layer = PackedLinear::<F32x1>::new(4, 2, true);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = layer.forward_cpu(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_memory_savings() {
        let l4 = PackedLinear::<U4x8>::new(512, 512, false);
        let l8 = PackedLinear::<U8x4>::new(512, 512, false);
        let l32 = PackedLinear::<F32x1>::new(512, 512, false);

        // U4x8 should save ~8x, U8x4 ~4x, F32x1 ~1x
        assert!(l4.memory_savings() > 7.0);
        assert!(l8.memory_savings() > 3.5);
        assert!((l32.memory_savings() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_repack() {
        let mut layer = PackedLinear::<U8x4>::new(4, 2, false);
        let orig = layer.weight.to_f32_vec();
        layer.repack();
        let repacked = layer.weight.to_f32_vec();
        // Values should be close after repack
        for (a, b) in orig.iter().zip(repacked.iter()) {
            assert!((a - b).abs() <= layer.weight.scale() + 0.01);
        }
    }

    #[test]
    fn test_packed_linear_u4_non_multiple_feature_dim() {
        let layer = PackedLinear::<U4x8>::new(7, 16, true);
        assert_eq!(layer.in_features, 7);
        assert_eq!(layer.out_features, 16);
        let input = vec![1.0; 7];
        let output = layer.forward_cpu(&input);
        assert_eq!(output.len(), 16);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_packed_linear_u8_non_multiple_feature_dim() {
        let layer = PackedLinear::<U8x4>::new(10, 8, false);
        let input = vec![0.5; 10];
        let output = layer.forward_cpu(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    #[should_panic(expected = "in_features must be > 0")]
    fn test_packed_linear_zero_in_features_panics() {
        let _layer = PackedLinear::<F32x1>::new(0, 4, true);
    }

    #[test]
    #[should_panic(expected = "out_features must be > 0")]
    fn test_packed_linear_zero_out_features_panics() {
        let _layer = PackedLinear::<F32x1>::new(4, 0, true);
    }
}
