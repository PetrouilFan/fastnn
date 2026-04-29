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
    // Pre-allocated scalar tensors to avoid per-forward allocation
    stride_scalar: Tensor,
    padding_scalar: Tensor,
    dilation_scalar: Tensor,
    groups_scalar: Tensor,
    default_bias: Tensor,
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
                .map(|_| (crate::random_f32() - 0.5) * 2.0 * scale)
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
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
            dilation_scalar: Tensor::from_scalar(dilation as f32),
            groups_scalar: Tensor::from_scalar(groups as f32),
            default_bias: Tensor::from_scalar(0.0),
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let bias_ref = self.bias.as_ref().unwrap_or(&self.default_bias);

        let result = dispatch(
            "conv2d",
            DispatchKey::Cpu,
            &[
                x,
                &self.weight,
                bias_ref,
                &self.stride_scalar,
                &self.padding_scalar,
                &self.dilation_scalar,
                &self.groups_scalar,
            ],
        );

        let output = result.into_iter().next().unwrap();

        if x.requires_grad() || self.weight.requires_grad() {
            let edges = {
                let mut edges = autograd::make_edge(x);
                edges.extend(autograd::make_edge(&self.weight));
                edges
            };
            let backward = Conv2dBackward::new(
                x.clone(),
                self.weight.clone(),
                self.bias.is_some(),
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
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn eval_mode(&self) {
        self.training
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[allow(dead_code)]
pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: i64,
    pub padding: i64,
    pub out_channels: i64,
    pub in_channels: i64,
    pub kernel_size: i64,
}

impl ConvTranspose2d {
    pub fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        bias: bool,
    ) -> Self {
        let k = kernel_size * kernel_size * in_channels;
        let scale = (2.0 / k as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_channels * out_channels * kernel_size * kernel_size)
            .map(|_| (crate::random_f32() - 0.5) * 2.0 * scale)
            .collect();
        let weight = Tensor::from_vec(
            weight_data,
            vec![in_channels, out_channels, kernel_size, kernel_size],
        )
        .requires_grad_(true);
        let bias = if bias {
            let b = Tensor::zeros(
                vec![out_channels],
                crate::storage::DType::F32,
                crate::storage::Device::Cpu,
            );
            let b = b;
            b.requires_grad_(true);
            Some(b)
        } else {
            None
        };
        ConvTranspose2d {
            weight,
            bias,
            stride,
            padding,
            out_channels,
            in_channels,
            kernel_size,
        }
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_shape = x.shape();
        let _batch = x_shape[0];
        let _in_channels = x_shape[1];
        let h_in = x_shape[2];
        let w_in = x_shape[3];

        let _h_out = (h_in - 1) * self.stride - 2 * self.padding + self.kernel_size;
        let _w_out = (w_in - 1) * self.stride - 2 * self.padding + self.kernel_size;

        let dispatch_key = crate::dispatcher::device_to_dispatch_key(x.device());
        let result = crate::dispatcher::dispatch(
            "conv_transpose2d",
            dispatch_key,
            &[
                x,
                &self.weight,
                &Tensor::from_scalar(self.stride as f32),
                &Tensor::from_scalar(self.padding as f32),
            ],
        );
        let mut output = result[0].clone();

        if let Some(ref bias) = self.bias {
            let mut bias_shape: smallvec::SmallVec<[i64; 8]> = smallvec::SmallVec::new();
            bias_shape.push(1);
            bias_shape.push(self.out_channels);
            bias_shape.push(1);
            bias_shape.push(1);
            let bias_reshaped = bias.reshape(bias_shape.into_vec());
            output = output.add(&bias_reshaped);
        }

        // Set up autograd for ConvTranspose2d
        if x.requires_grad() || self.weight.requires_grad() {
            let mut edges = autograd::make_edge(x);
            edges.extend(autograd::make_edge(&self.weight));
            if let Some(ref b) = self.bias {
                if b.requires_grad() {
                    edges.extend(autograd::make_edge(b));
                }
            }
            let backward = autograd::ConvTranspose2dBackward::new(
                x.clone(),
                self.weight.clone(),
                self.bias.clone(),
                self.stride,
                self.padding,
                self.kernel_size,
                edges,
            );
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(Arc::new(backward));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b.clone()));
        }
        params
    }

    fn zero_grad(&self) {
        for t in [&self.weight] {
            if let Some(meta) = &t.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
        if let Some(ref b) = self.bias {
            if let Some(meta) = &b.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn train_mode(&self) {}
    fn eval_mode(&self) {}
    fn is_training(&self) -> bool {
        false
    }
}

#[allow(dead_code)]
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub in_channels: i64,
    pub out_channels: i64,
    pub kernel_size: i64,
}

impl Conv1d {
    pub fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        bias: bool,
    ) -> Self {
        let k = kernel_size * in_channels;
        let scale = (2.0 / k as f32).sqrt();
        let weight_data: Vec<f32> = (0..out_channels * in_channels * kernel_size)
            .map(|_| (crate::random_f32() - 0.5) * 2.0 * scale)
            .collect();
        let weight = Tensor::from_vec(weight_data, vec![out_channels, in_channels, kernel_size])
            .requires_grad_(true);
        let bias = if bias {
            let b = Tensor::zeros(
                vec![out_channels],
                crate::storage::DType::F32,
                crate::storage::Device::Cpu,
            );
            let b = b;
            b.requires_grad_(true);
            Some(b)
        } else {
            None
        };
        Conv1d {
            weight,
            bias,
            stride,
            padding,
            dilation,
            in_channels,
            out_channels,
            kernel_size,
        }
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Conv1d = Conv2d with height=1, so reshape [B, C, L] -> [B, C, 1, L]
        let x_shape = x.shape();
        let batch = x_shape[0];
        let channels = x_shape[1];
        let length = x_shape[2];
        let x_4d = x.reshape(vec![batch, channels, 1, length]);

        // Use Conv2d forward which has autograd support
        let dispatch_key = crate::dispatcher::device_to_dispatch_key(x.device());
        let result = crate::dispatcher::dispatch(
            "conv2d",
            dispatch_key,
            &[
                &x_4d,
                &self.weight,
                &Tensor::from_scalar(self.stride as f32),
                &Tensor::from_scalar(self.padding as f32),
                &Tensor::from_scalar(self.dilation as f32),
                &Tensor::from_scalar(1.0), // groups
            ],
        );
        let mut output = result[0].clone();

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_shape = vec![1, self.out_channels, 1, 1];
            let bias_reshaped = bias.reshape(bias_shape);
            output = output.add(&bias_reshaped);
        }

        // Reshape back to 3D: [B, C_out, 1, L_out] -> [B, C_out, L_out]
        let out_shape = output.shape();
        let out_channels = out_shape[1];
        let out_length = out_shape[3];
        output.reshape(vec![batch, out_channels, out_length])
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b.clone()));
        }
        params
    }

    fn zero_grad(&self) {
        for t in [&self.weight] {
            if let Some(meta) = &t.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
        if let Some(ref b) = self.bias {
            if let Some(meta) = &b.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn train_mode(&self) {}
    fn eval_mode(&self) {}
    fn is_training(&self) -> bool {
        false
    }
}

#[allow(dead_code)]
pub struct Conv3d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub in_channels: i64,
    pub out_channels: i64,
    pub kernel_size: i64,
}

impl Conv3d {
    pub fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        bias: bool,
    ) -> Self {
        let k = kernel_size * kernel_size * kernel_size * in_channels;
        let scale = (2.0 / k as f32).sqrt();
        let weight_data: Vec<f32> =
            (0..out_channels * in_channels * kernel_size * kernel_size * kernel_size)
                .map(|_| (crate::random_f32() - 0.5) * 2.0 * scale)
                .collect();
        let weight = Tensor::from_vec(
            weight_data,
            vec![
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
                kernel_size,
            ],
        )
        .requires_grad_(true);
        let bias = if bias {
            let b = Tensor::zeros(
                vec![out_channels],
                crate::storage::DType::F32,
                crate::storage::Device::Cpu,
            );
            let b = b;
            b.requires_grad_(true);
            Some(b)
        } else {
            None
        };
        Conv3d {
            weight,
            bias,
            stride,
            padding,
            dilation,
            in_channels,
            out_channels,
            kernel_size,
        }
    }
}

impl Module for Conv3d {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Conv3d = Conv2d with depth as batch, so reshape [B, C, D, H, W] -> [B*D, C, H, W]
        let x_shape = x.shape();
        let batch = x_shape[0];
        let channels = x_shape[1];
        let depth = x_shape[2];
        let height = x_shape[3];
        let width = x_shape[4];

        // Reshape to treat depth as batch: [B*D, C, H, W]
        let x_4d = x.reshape(vec![batch * depth, channels, height, width]);

        // Use Conv2d forward which has autograd support
        let dispatch_key = crate::dispatcher::device_to_dispatch_key(x.device());
        let result = crate::dispatcher::dispatch(
            "conv2d",
            dispatch_key,
            &[
                &x_4d,
                &self.weight,
                &Tensor::from_scalar(self.stride as f32),
                &Tensor::from_scalar(self.padding as f32),
                &Tensor::from_scalar(self.dilation as f32),
                &Tensor::from_scalar(1.0), // groups
            ],
        );
        let mut output = result[0].clone();

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_shape = vec![1, self.out_channels, 1, 1];
            let bias_reshaped = bias.reshape(bias_shape);
            output = output.add(&bias_reshaped);
        }

        // Reshape back to 5D: [B*D, C_out, H_out, W_out] -> [B, C_out, D, H_out, W_out]
        let out_shape = output.shape();
        let out_channels = out_shape[1];
        let out_height = out_shape[2];
        let out_width = out_shape[3];
        output.reshape(vec![batch, out_channels, depth, out_height, out_width])
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b.clone()));
        }
        params
    }

    fn zero_grad(&self) {
        for t in [&self.weight] {
            if let Some(meta) = &t.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
        if let Some(ref b) = self.bias {
            if let Some(meta) = &b.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn train_mode(&self) {}
    fn eval_mode(&self) {}
    fn is_training(&self) -> bool {
        false
    }
}
