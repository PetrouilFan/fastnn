use crate::autograd::{self, AutogradMeta};
use crate::tensor::Tensor;
use crate::{
    impl_nn_named_params, impl_nn_params, impl_training_state, impl_zero_grad,
    nn::{Module, TrainingState},
};
use std::sync::Arc;

#[derive(Clone)]
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
    pub padding_mode: String,
    training: TrainingState,
    // Pre-allocated scalar tensors to avoid per-forward allocation
    #[allow(dead_code)]
    stride_scalar: Tensor,
    #[allow(dead_code)]
    padding_scalar: Tensor,
    #[allow(dead_code)]
    dilation_scalar: Tensor,
    #[allow(dead_code)]
    groups_scalar: Tensor,
    #[allow(dead_code)]
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
            padding_mode: "zeros".to_string(),
            training: TrainingState::new(),
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
        let (input, _pad_scalar) = if self.padding_mode == "same" {
            let x_shape = x.shape_ref();
            let h_in = x_shape[2];
            let w_in = x_shape[3];
            let h_out = ((h_in as f64) / self.stride as f64).ceil() as i64;
            let w_out = ((w_in as f64) / self.stride as f64).ceil() as i64;
            let pad_h = ((h_out - 1) * self.stride + self.kernel_size - h_in).max(0);
            let pad_w = ((w_out - 1) * self.stride + self.kernel_size - w_in).max(0);
            let pad_top = pad_h / 2;
            let pad_bottom = pad_h - pad_top;
            let pad_left = pad_w / 2;
            let pad_right = pad_w - pad_left;
            let new_h = x_shape[2] + pad_top + pad_bottom;
            let new_w = x_shape[3] + pad_left + pad_right;
            let mut padded_data = vec![0.0f32; (x_shape[0] * x_shape[1] * new_h * new_w) as usize];
            let x_data = x.as_f32_slice();
            for b in 0..x_shape[0] {
                for c in 0..x_shape[1] {
                    for h in 0..x_shape[2] {
                        let src_off = ((b * x_shape[1] + c) * x_shape[2] + h) * x_shape[3];
                        let dst_off =
                            ((b * x_shape[1] + c) * new_h + (h + pad_top)) * new_w + pad_left;
                        let src = &x_data[src_off as usize..(src_off + x_shape[3]) as usize];
                        let dst =
                            &mut padded_data[dst_off as usize..(dst_off + x_shape[3]) as usize];
                        dst.copy_from_slice(src);
                    }
                }
            }
            (
                Tensor::from_vec(padded_data, vec![x_shape[0], x_shape[1], new_h, new_w]),
                Tensor::from_scalar(0.0f32),
            )
        } else {
            (x.clone(), self.padding_scalar.clone())
        };
        let mut output = {
            let conv = Tensor::exec_aot(&[&input, &self.weight], |g, ins| {
                vec![g.conv2d_with_params(
                    &ins[0],
                    &ins[1],
                    self.stride as usize,
                    self.padding as usize,
                    self.dilation as usize,
                    self.groups as usize,
                )]
            })
            .expect("Conv2d::forward: AOT failed")
            .into_iter()
            .next()
            .unwrap();
            if let Some(bias) = &self.bias {
                let bias_shape = vec![1, self.out_channels, 1, 1];
                conv.add(&bias.reshape(bias_shape))
            } else {
                conv
            }
        };

        if x.requires_grad() || self.weight.requires_grad() {
            let inputs = vec![x.clone(), self.weight.clone()];
            let mut meta = AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(autograd::make_node_info("Conv2dBackward", inputs));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
            output
        } else {
            output
        }
    }

    impl_nn_params!(weight, bias);
    impl_nn_named_params!(weight, bias, "weight", "bias");
    impl_zero_grad!(self, self.weight, self.bias);
    impl_training_state!(self, self.training);
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ConvTranspose2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: i64,
    pub padding: i64,
    pub out_channels: i64,
    pub in_channels: i64,
    pub kernel_size: i64,
    training: TrainingState,
    // Pre-allocated scalars
    stride_scalar: Tensor,
    padding_scalar: Tensor,
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
            Some(b.requires_grad_(true))
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
            training: TrainingState::new(),
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
        }
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_shape = x.shape_ref();
        let _batch = x_shape[0];
        let _in_channels = x_shape[1];
        let h_in = x_shape[2];
        let w_in = x_shape[3];

        let _h_out = (h_in - 1) * self.stride - 2 * self.padding + self.kernel_size;
        let _w_out = (w_in - 1) * self.stride - 2 * self.padding + self.kernel_size;

        let mut output = Tensor::exec_aot(&[x, &self.weight], |g, ins| {
            vec![g.conv_transpose2d(
                &ins[0],
                &ins[1],
                self.stride as usize,
                self.padding as usize,
            )]
        })
        .expect("ConvTranspose2d::forward: AOT failed")
        .into_iter()
        .next()
        .unwrap();

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
            let mut inputs = vec![x.clone(), self.weight.clone()];
            if let Some(ref b) = self.bias {
                inputs.push(b.clone());
            }
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(autograd::make_node_info("ConvTranspose2dBackward", inputs));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        output
    }

    impl_nn_params!(weight, bias);
    impl_nn_named_params!(weight, bias, "weight", "bias");
    impl_zero_grad!(self, self.weight, self.bias);
    impl_training_state!(self, self.training);
}

#[derive(Clone)]
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
    training: TrainingState,
    // Pre-allocated scalars
    stride_scalar: Tensor,
    padding_scalar: Tensor,
    dilation_scalar: Tensor,
    #[allow(dead_code)]
    default_bias: Tensor,
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
            Some(b.requires_grad_(true))
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
            training: TrainingState::new(),
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
            dilation_scalar: Tensor::from_scalar(dilation as f32),
            default_bias: Tensor::from_scalar(0.0),
        }
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Route through conv2d for autograd support.
        //
        // Conv1d = Conv2d with height=1. We reshape [B, C, L] -> [B, C, 1, L] and
        // the weight [out_C, in_C, kL] -> [out_C, in_C, 1, kL].
        //
        // Since conv2d applies the same padding to both H and W, but Conv1d only
        // pads along L (width), we pre-pad the input along L and use padding=0 for conv2d.
        let x_shape = x.shape_ref();
        let batch = x_shape[0];
        let channels = x_shape[1];
        let length = x_shape[2];
        let pad = self.padding;

        // Pad input along the length dimension
        let x_padded = if pad > 0 {
            let padded_len = length + 2 * pad;
            let mut padded_data = vec![0.0f32; (batch * channels * padded_len) as usize];
            let x_slice = x.as_f32_slice();
            for b in 0..batch {
                for c in 0..channels {
                    let src_off = (b * channels + c) * length;
                    let dst_off = (b * channels + c) * padded_len + pad;
                    let src = &x_slice[src_off as usize..(src_off + length) as usize];
                    let dst = &mut padded_data[dst_off as usize..(dst_off + length) as usize];
                    dst.copy_from_slice(src);
                }
            }
            Tensor::from_vec(padded_data, vec![batch, channels, padded_len])
        } else {
            x.clone()
        };

        let x_4d = x_padded.reshape(vec![batch, channels, 1, length + 2 * pad]);

        // Reshape weight from 3D to 4D
        let w_shape = self.weight.shape_ref();
        let w_4d = self
            .weight
            .reshape(vec![w_shape[0], w_shape[1], 1, w_shape[2]]);

        // Use conv2d with padding=0 (since we pre-padded) and the user's stride/dilation
        let output = {
            let conv = Tensor::exec_aot(&[&x_4d, &w_4d], |g, ins| {
                vec![g.conv2d_with_params(
                    &ins[0],
                    &ins[1],
                    self.stride as usize,
                    0,
                    self.dilation as usize,
                    1,
                )]
            })
            .expect("Conv1d::forward: AOT failed")
            .into_iter()
            .next()
            .unwrap();
            if let Some(bias) = &self.bias {
                let bias_shape = vec![1, self.out_channels, 1, 1];
                conv.add(&bias.reshape(bias_shape))
            } else {
                conv
            }
        };

        // Reshape back to 3D: [B, C_out, 1, L_out] -> [B, C_out, L_out]
        let out_shape = output.shape_ref();
        let out_channels = out_shape[1];
        let out_length = out_shape[3];
        output.reshape(vec![batch, out_channels, out_length])
    }

    impl_nn_params!(weight, bias);
    impl_nn_named_params!(weight, bias, "weight", "bias");
    impl_zero_grad!(self, self.weight, self.bias);
    impl_training_state!(self, self.training);
}

#[derive(Clone)]
pub struct Conv3d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub in_channels: i64,
    pub out_channels: i64,
    pub kernel_size: i64,
    training: TrainingState,
    // Pre-allocated scalars
    #[allow(dead_code)]
    stride_scalar: Tensor,
    #[allow(dead_code)]
    padding_scalar: Tensor,
    #[allow(dead_code)]
    dilation_scalar: Tensor,
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
            Some(b.requires_grad_(true))
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
            training: TrainingState::new(),
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
            dilation_scalar: Tensor::from_scalar(dilation as f32),
        }
    }
}

impl Module for Conv3d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut output = Tensor::exec_aot(&[x, &self.weight], |g, ins| {
            vec![g.conv3d(
                &ins[0],
                &ins[1],
                self.stride as usize,
                self.padding as usize,
            )]
        })
        .expect("Conv3d::forward: AOT failed")
        .into_iter()
        .next()
        .unwrap();

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_shape = vec![1, self.out_channels, 1, 1, 1];
            let bias_reshaped = bias.reshape(bias_shape);
            output = output.add(&bias_reshaped);
        }

        output
    }

    impl_nn_params!(weight, bias);
    impl_nn_named_params!(weight, bias, "weight", "bias");
    impl_zero_grad!(self, self.weight, self.bias);
    impl_training_state!(self, self.training);
}
