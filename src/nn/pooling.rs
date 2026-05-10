use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::tensor::Tensor;

pub struct MaxPool2d {
    #[allow(dead_code)]
    kernel_size: i64,
    #[allow(dead_code)]
    stride: i64,
    #[allow(dead_code)]
    padding: i64,
    #[allow(dead_code)]
    dilation: i64,
    // Pre-allocated scalar tensors
    kernel_size_scalar: Tensor,
    stride_scalar: Tensor,
    padding_scalar: Tensor,
    dilation_scalar: Tensor,
}

impl MaxPool2d {
    pub fn new(kernel_size: i64, stride: i64, padding: i64, dilation: i64) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
            padding,
            dilation,
            kernel_size_scalar: Tensor::from_scalar(kernel_size as f32),
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
            dilation_scalar: Tensor::from_scalar(dilation as f32),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = dispatch(
            "max_pool2d",
            DispatchKey::Cpu,
            &[
                x,
                &self.kernel_size_scalar,
                &self.stride_scalar,
                &self.padding_scalar,
                &self.dilation_scalar,
            ],
        )
        .expect("MaxPool2d::forward: dispatch failed");
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

pub struct AvgPool2d {
    #[allow(dead_code)]
    kernel_size: i64,
    #[allow(dead_code)]
    stride: i64,
    #[allow(dead_code)]
    padding: i64,
    // Pre-allocated scalar tensors
    kernel_size_scalar: Tensor,
    stride_scalar: Tensor,
    padding_scalar: Tensor,
}

impl AvgPool2d {
    pub fn new(kernel_size: i64, stride: i64, padding: i64) -> Self {
        AvgPool2d {
            kernel_size,
            stride,
            padding,
            kernel_size_scalar: Tensor::from_scalar(kernel_size as f32),
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = dispatch(
            "avg_pool2d",
            DispatchKey::Cpu,
            &[
                x,
                &self.kernel_size_scalar,
                &self.stride_scalar,
                &self.padding_scalar,
            ],
        )
        .expect("AvgPool2d::forward: dispatch failed");
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
