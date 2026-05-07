use crate::autograd::{AutogradMeta, Edge, Node};
use crate::dispatcher::{dispatch, DispatchKey};
use crate::nn::Module;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Trait for fused activation types, providing dispatch names and backward metadata.
pub trait Activation {
    const DISPATCH_NAME: &'static str;
    const BACKWARD_NAME: &'static str;
    const BACKWARD_MSG: &'static str;
}

/// No activation (fused Conv2d + BatchNorm2d only)
pub struct NoAct;
impl Activation for NoAct {
    const DISPATCH_NAME: &'static str = "fused_conv_bn";
    const BACKWARD_NAME: &'static str = "FusedConvBnBackward";
    const BACKWARD_MSG: &'static str = "Conv2d + BatchNorm2d";
}

/// ReLU activation (fused Conv2d + BatchNorm2d + ReLU)
pub struct ReluAct;
impl Activation for ReluAct {
    const DISPATCH_NAME: &'static str = "fused_conv_bn_relu";
    const BACKWARD_NAME: &'static str = "FusedConvBnReluBackward";
    const BACKWARD_MSG: &'static str = "Conv2d + BatchNorm2d + ReLU";
}

/// GELU activation (fused Conv2d + BatchNorm2d + GELU)
pub struct GeluAct;
impl Activation for GeluAct {
    const DISPATCH_NAME: &'static str = "fused_conv_bn_gelu";
    const BACKWARD_NAME: &'static str = "FusedConvBnGeluBackward";
    const BACKWARD_MSG: &'static str = "Conv2d + BatchNorm2d + GELU";
}

/// SiLU activation (fused Conv2d + BatchNorm2d + SiLU)
pub struct SiluAct;
impl Activation for SiluAct {
    const DISPATCH_NAME: &'static str = "fused_conv_bn_silu";
    const BACKWARD_NAME: &'static str = "FusedConvBnSiluBackward";
    const BACKWARD_MSG: &'static str = "Conv2d + BatchNorm2d + SiLU";
}

/// Generic fused Conv2d + BatchNorm2d + optional activation layer (inference only).
/// Parameterized by `A: Activation` to support different activation functions.
pub struct FusedConvBn<A: Activation = NoAct> {
    pub conv_weight: Tensor,
    pub conv_bias: Option<Tensor>,
    pub bn_weight: Tensor,
    pub bn_bias: Tensor,
    pub bn_running_mean: Tensor,
    pub bn_running_var: Tensor,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub groups: i64,
    pub eps: f64,
    pub in_channels: i64,
    pub out_channels: i64,
    pub kernel_size: i64,
    eps_scalar: Tensor,
    stride_scalar: Tensor,
    padding_scalar: Tensor,
    dilation_scalar: Tensor,
    groups_scalar: Tensor,
    _act: std::marker::PhantomData<A>,
}

impl<A: Activation> FusedConvBn<A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
        eps: f64,
        bias: bool,
    ) -> Self {
        let k = kernel_size * kernel_size * in_channels / groups;
        let scale = (2.0 / k as f32).sqrt();

        let conv_weight_data: Vec<f32> = (0..out_channels * in_channels / groups * kernel_size * kernel_size)
            .map(|_| (crate::random_f32() - 0.5) * 2.0 * scale)
            .collect();
        let conv_weight = Tensor::from_vec(
            conv_weight_data,
            vec![out_channels, in_channels / groups, kernel_size, kernel_size],
        );
        let conv_weight = conv_weight.requires_grad_(true);

        let conv_bias = if bias {
            let bias_data: Vec<f32> = (0..out_channels).map(|_| 0.0).collect();
            Some(Tensor::from_vec(bias_data, vec![out_channels]).requires_grad_(true))
        } else {
            None
        };

        let bn_weight_data: Vec<f32> = (0..out_channels).map(|_| 1.0).collect();
        let bn_weight = Tensor::from_vec(bn_weight_data, vec![out_channels]).requires_grad_(true);

        let bn_bias_data: Vec<f32> = (0..out_channels).map(|_| 0.0).collect();
        let bn_bias = Tensor::from_vec(bn_bias_data, vec![out_channels]).requires_grad_(true);

        let bn_running_mean_data: Vec<f32> = (0..out_channels).map(|_| 0.0).collect();
        let bn_running_mean = Tensor::from_vec(bn_running_mean_data, vec![out_channels]);

        let bn_running_var_data: Vec<f32> = (0..out_channels).map(|_| 1.0).collect();
        let bn_running_var = Tensor::from_vec(bn_running_var_data, vec![out_channels]);

        FusedConvBn {
            conv_weight,
            conv_bias,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            stride,
            padding,
            dilation,
            groups,
            eps,
            in_channels,
            out_channels,
            kernel_size,
            eps_scalar: Tensor::from_scalar(eps as f32),
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
            dilation_scalar: Tensor::from_scalar(dilation as f32),
            groups_scalar: Tensor::from_scalar(groups as f32),
            _act: std::marker::PhantomData,
        }
    }

    pub fn set_conv_weight(&mut self, weight: Tensor) {
        self.conv_weight = weight;
    }

    pub fn set_conv_bias(&mut self, bias: Tensor) {
        self.conv_bias = Some(bias);
    }

    pub fn set_bn_weight(&mut self, weight: Tensor) {
        self.bn_weight = weight;
    }

    pub fn set_bn_bias(&mut self, bias: Tensor) {
        self.bn_bias = bias;
    }

    pub fn set_bn_running_mean(&mut self, running_mean: Tensor) {
        self.bn_running_mean = running_mean;
    }

    pub fn set_bn_running_var(&mut self, running_var: Tensor) {
        self.bn_running_var = running_var;
    }
}

/// Backward node for fused layers (inference-only, panics on gradient request)
struct FusedConvBnBackward {
    backward_name: &'static str,
    msg: &'static str,
}

impl Node for FusedConvBnBackward {
    fn apply(&self, _: Vec<Option<Tensor>>) -> Vec<Option<Tensor>> {
        panic!(
            "FusedConvBn with {} does not support autograd; use separate layers for training",
            self.msg
        );
    }
    fn next_edges(&self) -> &[Edge] { &[] }
    fn num_inputs(&self) -> usize { 0 }
    fn name(&self) -> &str { self.backward_name }
    fn inputs(&self) -> &[Tensor] { &[] }
}

impl<A: Activation + Send + Sync> Module for FusedConvBn<A> {
    fn forward(&self, x: &Tensor) -> Tensor {
        let default_bias = Tensor::from_scalar(0.0);
        let bias_ref = self.conv_bias.as_ref().unwrap_or(&default_bias);

        let result = dispatch(
            A::DISPATCH_NAME,
            DispatchKey::Cpu,
            &[
                x,
                &self.conv_weight,
                bias_ref,
                &self.bn_weight,
                &self.bn_bias,
                &self.bn_running_mean,
                &self.bn_running_var,
                &self.stride_scalar,
                &self.padding_scalar,
                &self.dilation_scalar,
                &self.groups_scalar,
                &self.eps_scalar,
            ],
        );

        let mut output = result[0].clone();

        if self.conv_weight.requires_grad()
            || self.bn_weight.requires_grad()
            || self.bn_bias.requires_grad()
            || self.conv_bias.as_ref().is_some_and(|b| b.requires_grad())
        {
            let backward = Arc::new(FusedConvBnBackward {
                backward_name: A::BACKWARD_NAME,
                msg: A::BACKWARD_MSG,
            });
            let mut meta = AutogradMeta::new_non_leaf(false);
            meta.grad_fn = Some(backward);
            Arc::make_mut(&mut output.inner).autograd_meta = Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.conv_weight.clone(), self.bn_weight.clone(), self.bn_bias.clone()];
        if let Some(b) = &self.conv_bias {
            params.push(b.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![
            ("conv_weight".to_string(), self.conv_weight.clone()),
            ("bn_weight".to_string(), self.bn_weight.clone()),
            ("bn_bias".to_string(), self.bn_bias.clone()),
        ];
        if let Some(b) = &self.conv_bias {
            params.push(("conv_bias".to_string(), b.clone()));
        }
        params
    }

    fn zero_grad(&self) {
        for tensor in [&self.conv_weight, &self.bn_weight, &self.bn_bias] {
            if let Some(meta) = &tensor.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
        if let Some(b) = &self.conv_bias {
            if let Some(meta) = &b.inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn train_mode(&self) {}
    fn eval_mode(&self) {}
    fn is_training(&self) -> bool { false }
}

/// Fused Conv2d + BatchNorm2d + SiLU layer (inference only).
pub type FusedConvBnSilu = FusedConvBn<SiluAct>;

/// Fused Conv2d + BatchNorm2d + ReLU layer (inference only).
pub type FusedConvBnRelu = FusedConvBn<ReluAct>;

/// Fused Conv2d + BatchNorm2d + GELU layer (inference only).
pub type FusedConvBnGelu = FusedConvBn<GeluAct>;

/// FusedConvBn (no activation) specific methods
impl FusedConvBn<NoAct> {
    /// Fuse parameters from separate Conv2d and BatchNorm2d layers
    pub fn from_conv_bn(
        conv: &crate::nn::conv::Conv2d,
        bn: &crate::nn::norm::BatchNorm2d,
    ) -> Self {
        let stride = conv.stride;
        let padding = conv.padding;
        let dilation = conv.dilation;
        let groups = conv.groups;
        let kernel_size = conv.kernel_size;
        let in_channels = conv.in_channels;
        let out_channels = conv.out_channels;
        let eps = 1e-5;

        let conv_weight = conv.weight.clone();
        let conv_bias = conv.bias.clone();

        let bn_weight = bn.weight.clone();
        let bn_bias = bn.bias.clone();

        let bn_running_mean = bn.running_mean.read().clone();
        let bn_running_var = bn.running_var.read().clone();

        FusedConvBn {
            conv_weight,
            conv_bias,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            stride,
            padding,
            dilation,
            groups,
            eps,
            in_channels,
            out_channels,
            kernel_size,
            eps_scalar: Tensor::from_scalar(eps as f32),
            stride_scalar: Tensor::from_scalar(stride as f32),
            padding_scalar: Tensor::from_scalar(padding as f32),
            dilation_scalar: Tensor::from_scalar(dilation as f32),
            groups_scalar: Tensor::from_scalar(groups as f32),
            _act: std::marker::PhantomData,
        }
    }
}
