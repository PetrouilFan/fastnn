use crate::autograd::{self, AutogradMeta, Conv2dBackward};
use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::tensor::Tensor;
use std::sync::Arc;

pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    #[allow(dead_code)]
    pub in_channels: i64,
    #[allow(dead_code)]
    pub out_channels: i64,
    #[allow(dead_code)]
    pub kernel_size: i64,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub groups: i64,
    training: std::sync::atomic::AtomicBool,
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
        bias: bool,
    ) -> Self {
        let k = kernel_size * kernel_size * in_channels / groups;
        let scale = (2.0 / k as f32).sqrt();

        let weight_data: Vec<f32> =
            (0..out_channels * in_channels / groups * kernel_size * kernel_size)
                .map(|_| (rand::random::<f32>() - 0.5) * 2.0 * scale)
                .collect();
        let weight = Tensor::from_vec(
            weight_data,
            vec![out_channels, in_channels / groups, kernel_size, kernel_size],
        );
        let weight = weight.requires_grad_(true);

        let bias = if bias {
            let bias_data: Vec<f32> = (0..out_channels).map(|_| 0.0).collect();
            Some(Tensor::from_vec(bias_data, vec![out_channels]).requires_grad_(true))
        } else {
            None
        };

        Conv2d {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            training: std::sync::atomic::AtomicBool::new(true),
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let bias_tensor = self
            .bias
            .clone()
            .unwrap_or_else(|| Tensor::from_scalar(0.0));

        let result = dispatch(
            "conv2d",
            DispatchKey::Cpu,
            &[
                x,
                &self.weight,
                &bias_tensor,
                &Tensor::from_scalar(self.stride as f32),
                &Tensor::from_scalar(self.padding as f32),
                &Tensor::from_scalar(self.dilation as f32),
                &Tensor::from_scalar(self.groups as f32),
            ],
        );

        let output = result[0].clone();

        if x.requires_grad() || self.weight.requires_grad() {
            let edges = {
                let mut edges = autograd::make_edge(x);
                edges.extend(autograd::make_edge(&self.weight));
                edges
            };
            let backward = Conv2dBackward::new(
                x.clone(),
                self.weight.clone(),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                edges,
            );
            let mut meta = AutogradMeta::new(false);
            meta.grad_fn = Some(Arc::new(backward));
            let mut output = output.clone();
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(b) = &self.bias {
            params.push(("bias".to_string(), b.clone()));
        }
        params
    }

    fn zero_grad(&self) {
        if let Some(meta) = &self.weight.inner.autograd_meta {
            if let Ok(mut lock) = meta.lock() {
                lock.grad = None;
            }
        }

        if let Some(b) = &self.bias {
            if let Some(meta) = &b.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn train_mode(&self) {
        self.training
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn eval_mode(&self) {
        self.training
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::SeqCst)
    }
}
