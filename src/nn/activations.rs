#![allow(dead_code)]
use crate::nn::Module;
use crate::tensor::Tensor;

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
            pub fn new() -> Self {
                $name
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
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

#[derive(Clone)]
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

#[derive(Clone)]
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
        Tensor::exec_aot(&[x, &self.weight], |g, ins| vec![g.prelu(&ins[0], &ins[1])])
            .expect("PReLU::forward: AOT failed")
            .into_iter()
            .next()
            .unwrap()
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

#[derive(Clone)]
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

#[derive(Clone)]
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

#[derive(Clone)]
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

#[derive(Clone)]
pub struct AdaptiveAvgPool2d {
    pub output_h: usize,
    pub output_w: usize,
}

impl AdaptiveAvgPool2d {
    pub fn new(output_h: usize, output_w: usize) -> Self {
        AdaptiveAvgPool2d { output_h, output_w }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::exec_aot(&[x], |g, ins| {
            vec![g.adaptive_avg_pool2d(&ins[0], self.output_h, self.output_w)]
        })
        .expect("AdaptiveAvgPool2d::forward: AOT execution failed")
        .into_iter()
        .next()
        .unwrap()
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
