use crate::autograd::{self, AutogradMeta, AvgPool2dBackward, MaxPool2dBackward};
use crate::dispatcher::{DispatchKey, dispatch};
use crate::nn::Module;
use crate::tensor::Tensor;
use std::sync::Arc;

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
        let mut output = result[0].clone();

        if x.requires_grad() {
            let argmax_data = result[1].as_f32_slice();
            let argmax_indices: Vec<usize> = argmax_data.iter().map(|&v| v as usize).collect();

            let backward = MaxPool2dBackward::new();
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

pub struct AvgPool1d {
    #[allow(dead_code)]
    kernel_size: i64,
    #[allow(dead_code)]
    stride: i64,
    #[allow(dead_code)]
    padding: i64,
    kernel_size_scalar: Tensor,
    stride_scalar: Tensor,
    padding_scalar: Tensor,
}

impl AvgPool1d {
    pub fn new(kernel_size: i64, stride: i64, padding: i64) -> Self {
        AvgPool1d {
            kernel_size,
            stride,
            padding,
            kernel_size_scalar: Tensor::from_scalar(kernel_size as f32),
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
        }
    }
}

impl Module for AvgPool1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_shape = x.shape_ref();
        // Add a dummy H dimension: [N, C, W] -> [N, C, 1, W]
        let x_4d = x.reshape(vec![x_shape[0], x_shape[1], 1, x_shape[2]]);
        let result = dispatch(
            "avg_pool2d",
            DispatchKey::Cpu,
            &[
                &x_4d,
                &self.kernel_size_scalar,
                &self.stride_scalar,
                &self.padding_scalar,
            ],
        )
        .expect("AvgPool1d::forward: dispatch failed");
        let out_4d = result[0].clone();
        let out_shape = out_4d.shape_ref();
        // Remove dummy H: [N, C, 1, W] -> [N, C, W]
        out_4d.reshape(vec![out_shape[0], out_shape[1], out_shape[3]])
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

pub struct MaxPool1d {
    #[allow(dead_code)]
    kernel_size: i64,
    #[allow(dead_code)]
    stride: i64,
    #[allow(dead_code)]
    padding: i64,
    #[allow(dead_code)]
    dilation: i64,
    kernel_size_scalar: Tensor,
    stride_scalar: Tensor,
    padding_scalar: Tensor,
    dilation_scalar: Tensor,
}

impl MaxPool1d {
    pub fn new(kernel_size: i64, stride: i64, padding: i64, dilation: i64) -> Self {
        MaxPool1d {
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

impl Module for MaxPool1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_shape = x.shape_ref();
        let x_4d = x.reshape(vec![x_shape[0], x_shape[1], 1, x_shape[2]]);
        let result = dispatch(
            "max_pool2d",
            DispatchKey::Cpu,
            &[
                &x_4d,
                &self.kernel_size_scalar,
                &self.stride_scalar,
                &self.padding_scalar,
                &self.dilation_scalar,
            ],
        )
        .expect("MaxPool1d::forward: dispatch failed");
        let out_4d = result[0].clone();
        let out_shape = out_4d.shape_ref();
        out_4d.reshape(vec![out_shape[0], out_shape[1], out_shape[3]])
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
        let mut output = result[0].clone();

        if x.requires_grad() {
            let backward = AvgPool2dBackward::new();
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
