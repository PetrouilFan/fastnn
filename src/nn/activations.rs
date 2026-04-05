#![allow(dead_code)]
use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::tensor::Tensor;

#[allow(dead_code)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Module for ReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = dispatch("relu", DispatchKey::Cpu, &[x]);
        result[0].clone()
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

#[allow(dead_code)]
pub struct Gelu;

impl Gelu {
    pub fn new() -> Self {
        Gelu
    }
}

impl Module for Gelu {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = dispatch("gelu", DispatchKey::Cpu, &[x]);
        result[0].clone()
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

#[allow(dead_code)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Module for Sigmoid {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = dispatch("sigmoid", DispatchKey::Cpu, &[x]);
        result[0].clone()
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

#[allow(dead_code)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Module for Tanh {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = dispatch("tanh", DispatchKey::Cpu, &[x]);
        result[0].clone()
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

#[allow(dead_code)]
pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        SiLU
    }
}

impl Module for SiLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = dispatch("silu", DispatchKey::Cpu, &[x]);
        result[0].clone()
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
        let slope_tensor = Tensor::from_scalar(self.negative_slope as f32);
        let result = dispatch("leaky_relu", DispatchKey::Cpu, &[x, &slope_tensor]);
        result[0].clone()
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

pub struct PReLU {
    weight: Tensor,
}

impl PReLU {
    pub fn new(num_parameters: i64) -> Self {
        let weight = Tensor::full(vec![num_parameters], 0.25, crate::storage::DType::F32, crate::storage::Device::Cpu);
        let w = weight.clone();
        w.requires_grad_(true);
        PReLU { weight: w }
    }
}

impl Module for PReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = dispatch("prelu", DispatchKey::Cpu, &[x, &self.weight]);
        result[0].clone()
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
        let beta_t = Tensor::from_scalar(self.beta as f32);
        let threshold_t = Tensor::from_scalar(self.threshold as f32);
        let result = dispatch("softplus", DispatchKey::Cpu, &[x, &beta_t, &threshold_t]);
        result[0].clone()
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

pub struct Hardswish;

impl Hardswish {
    pub fn new() -> Self {
        Hardswish
    }
}

impl Module for Hardswish {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = dispatch("hardswish", DispatchKey::Cpu, &[x]);
        result[0].clone()
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
