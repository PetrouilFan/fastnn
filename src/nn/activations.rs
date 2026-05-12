#![allow(dead_code)]
use crate::autograd::{self, AdaptiveAvgPool2dBackward, AutogradMeta};
use crate::dispatcher::{DispatchKey, dispatch};
use crate::nn::Module;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Stateless activation macro — routes through the AOT pipeline via Tensor methods.
macro_rules! impl_stateless_activation {
    ($name:ident, "relu") => {
        impl_activation_via_tensor!($name, relu);
    };
    ($name:ident, "gelu") => {
        impl_activation_via_tensor!($name, gelu);
    };
    ($name:ident, "sigmoid") => {
        impl_activation_via_tensor!($name, sigmoid);
    };
    ($name:ident, "tanh") => {
        impl_activation_via_tensor!($name, tanh);
    };
    ($name:ident, "silu") => {
        impl_activation_via_tensor!($name, silu);
    };
    ($name:ident, "hardswish") => {
        impl_activation_via_tensor!($name, hardswish);
    };
    ($name:ident, "mish") => {
        impl_activation_via_tensor!($name, mish);
    };
}

/// Dispatch-safe activation via Tensor method (→ AOT pipeline).
macro_rules! impl_activation_via_tensor {
    ($name:ident, $method:ident) => {
        #[allow(dead_code)]
        pub struct $name;

        impl $name {
            pub fn new() -> Self { $name }
        }

        impl Default for $name {
            fn default() -> Self { Self::new() }
        }

        impl Module for $name {
            fn forward(&self, x: &Tensor) -> Tensor {
                x.$method()
            }
            impl_stateless_activation_methods!();
        }
    };
}

/// Macro to implement the standard empty methods for stateless modules.
macro_rules! impl_stateless_activation_methods {
    () => {
        fn parameters(&self) -> Vec<Tensor> {
            vec![]
        }

        fn named_parameters(&self) -> Vec<(String, Tensor)> {
            vec![]
        }

        fn zero_grad(&self) {}

        fn train_mode(&self) {}

        fn eval_mode(&self) {}

        fn is_training(&self) -> bool {
            false
        }
    };
}

impl_stateless_activation!(ReLU, "relu");
impl_stateless_activation!(Gelu, "gelu");
impl_stateless_activation!(Sigmoid, "sigmoid");
impl_stateless_activation!(Tanh, "tanh");
impl_stateless_activation!(SiLU, "silu");
impl_stateless_activation!(Hardswish, "hardswish");

impl_stateless_activation!(Mish, "mish");

pub struct LeakyReLU {
    negative_slope: f64,
}

impl LeakyReLU {
    pub fn new(negative_slope: f64) -> Self {
        LeakyReLU { negative_slope }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.leaky_relu(self.negative_slope as f32)
    }

    impl_stateless_activation_methods!();
}

pub struct PReLU {
    pub weight: Tensor,
}

impl PReLU {
    pub fn new(num_parameters: i64) -> Self {
        let weight = Tensor::full(
            vec![num_parameters],
            0.25,
            crate::storage::DType::F32,
            crate::storage::Device::Cpu,
        );
        let w = weight.clone();
        let w = w.requires_grad_(true);
        PReLU { weight: w }
    }
}

impl Module for PReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        use crate::storage::Device;
        let result = if x.device() == Device::Cpu {
            Tensor::exec_aot(&[x, &self.weight], |g, ins| vec![g.prelu(&ins[0], &ins[1])])
                .expect("PReLU::forward: AOT failed")
        } else {
            dispatch("prelu", DispatchKey::Wgpu, &[x, &self.weight])
                .expect("PReLU::forward: dispatch failed")
        };
        result.into_iter().next().unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }

    fn zero_grad(&self) {}

    fn train_mode(&self) {}

    fn eval_mode(&self) {}

    fn is_training(&self) -> bool {
        false
    }
}

pub struct Softmax {
    dim: i64,
}

impl Softmax {
    pub fn new(dim: i64) -> Self {
        Softmax { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl Module for Softmax {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.softmax(self.dim as i32)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {}

    fn train_mode(&self) {}

    fn eval_mode(&self) {}

    fn is_training(&self) -> bool {
        false
    }
}

pub struct Softplus {
    beta: f64,
    threshold: f64,
}

impl Softplus {
    pub fn new(beta: f64, threshold: f64) -> Self {
        Softplus { beta, threshold }
    }
}

impl Module for Softplus {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.softplus(self.beta as f32, self.threshold as f32)
    }

    impl_stateless_activation_methods!();
}

pub struct Elu {
    pub alpha: f64,
}

impl Elu {
    pub fn new(alpha: f64) -> Self {
        Elu { alpha }
    }
}

impl Module for Elu {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.elu(self.alpha as f32)
    }

    impl_stateless_activation_methods!();
}

pub struct AdaptiveAvgPool2d {
    pub output_size: (i64, i64),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (i64, i64)) -> Self {
        AdaptiveAvgPool2d { output_size }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_shape = x.shape_ref();
        let batch = x_shape[0];
        let channels = x_shape[1];
        let in_h = x_shape[2];
        let in_w = x_shape[3];
        let out_h = self.output_size.0;
        let out_w = self.output_size.1;

        let mut output_data = vec![0.0f32; (batch * channels * out_h * out_w) as usize];
        let x_data = x.as_f32_slice();

        for b in 0..batch as usize {
            for c in 0..channels as usize {
                for oh in 0..out_h as usize {
                    for ow in 0..out_w as usize {
                        let h_start = (oh * in_h as usize) / out_h as usize;
                        let h_end = ((oh + 1) * in_h as usize) / out_h as usize;
                        let w_start = (ow * in_w as usize) / out_w as usize;
                        let w_end = ((ow + 1) * in_w as usize) / out_w as usize;

                        let mut sum = 0.0f32;
                        let mut count = 0;
                        for ih in h_start..h_end {
                            for iw in w_start..w_end {
                                sum +=
                                    x_data[b * channels as usize * in_h as usize * in_w as usize
                                        + c * in_h as usize * in_w as usize
                                        + ih * in_w as usize
                                        + iw];
                                count += 1;
                            }
                        }
                        output_data[b * channels as usize * out_h as usize * out_w as usize
                            + c * out_h as usize * out_w as usize
                            + oh * out_w as usize
                            + ow] = sum / count as f32;
                    }
                }
            }
        }

        let mut output = Tensor::from_vec(output_data, vec![batch, channels, out_h, out_w]);

        if x.requires_grad() {
            let backward = AdaptiveAvgPool2dBackward::new();
            let mut meta = AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(Arc::new(backward));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {}

    fn train_mode(&self) {}

    fn eval_mode(&self) {}

    fn is_training(&self) -> bool {
        false
    }
}
