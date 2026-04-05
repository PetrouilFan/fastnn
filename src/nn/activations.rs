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
        let weight = Tensor::full(
            vec![num_parameters],
            0.25,
            crate::storage::DType::F32,
            crate::storage::Device::Cpu,
        );
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

pub struct ELU {
    pub alpha: f64,
}

impl ELU {
    pub fn new(alpha: f64) -> Self {
        ELU { alpha }
    }
}

impl Module for ELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let alpha_tensor = Tensor::from_scalar(self.alpha as f32);
        let result = dispatch("elu", DispatchKey::Cpu, &[x, &alpha_tensor]);
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

pub struct Mish;

impl Mish {
    pub fn new() -> Self {
        Mish
    }
}

impl Module for Mish {
    fn forward(&self, x: &Tensor) -> Tensor {
        // mish(x) = x * tanh(softplus(x))
        // softplus(x) = ln(1 + exp(x))
        let sp = x.add_scalar(1.0).exp().ln();
        let tanh_sp = sp.tanh();
        x.mul(&tanh_sp)
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
        let x_shape = x.shape();
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

        Tensor::from_vec(output_data, vec![batch, channels, out_h, out_w])
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
