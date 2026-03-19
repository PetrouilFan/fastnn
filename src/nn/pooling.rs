use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::tensor::Tensor;

pub struct MaxPool2d {
    kernel_size: i64,
    stride: i64,
    padding: i64,
    dilation: i64,
}

impl MaxPool2d {
    pub fn new(kernel_size: i64, stride: i64, padding: i64, dilation: i64) -> Self {
        MaxPool2d {
            kernel_size,
            stride,
            padding,
            dilation,
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
                &Tensor::from_scalar(self.kernel_size as f32),
                &Tensor::from_scalar(self.stride as f32),
                &Tensor::from_scalar(self.padding as f32),
                &Tensor::from_scalar(self.dilation as f32),
            ],
        );
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
