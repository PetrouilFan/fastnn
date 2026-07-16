macro_rules! impl_nn_module {
    ($struct_name:ident { $($methods:tt)* }) => {
        #[pymethods]
        impl $struct_name {
            fn __call__(&self, x: &PyTensor) -> PyTensor {
                PyTensor::from_tensor(self.inner.forward(&x.inner))
            }

            fn parameters(&self) -> Vec<PyTensor> {
                self.inner
                    .parameters()
                    .into_iter()
                    .map(PyTensor::from_tensor)
                    .collect()
            }

            fn named_parameters(&self) -> Vec<(&str, PyTensor)> {
                self.inner
                    .named_parameters_ref()
                    .into_iter()
                    .map(|(n, t)| (n, PyTensor::from_tensor(t.clone())))
                    .collect()
            }

            fn zero_grad(&self) {
                self.inner.zero_grad();
            }

            fn train(&self) {
                self.inner.train_mode();
            }

            fn eval(&self) {
                self.inner.eval_mode();
            }

            fn is_training(&self) -> bool {
                self.inner.is_training()
            }

            $($methods)*
        }
    };
}

macro_rules! impl_nn_module_with_gpu {
    ($struct_name:ident { $($methods:tt)* }) => {
        impl_nn_module!($struct_name {
            #[cfg(feature = "gpu")]
            #[pyo3(signature = (device_id))]
            #[allow(clippy::wrong_self_convention)]
            fn to_gpu(&mut self, device_id: usize) {
                self.inner.weight = self.inner.weight.to_gpu(device_id);
                self.inner.bias = self.inner.bias.as_ref().map(|b| b.to_gpu(device_id));
            }

            $($methods)*
        });
    };
}

macro_rules! impl_fused_conv_bn {
    ($name:ident, $inner_type:ty) => {
        impl_fused_conv_bn!($name, $inner_type, {});
    };
    ($name:ident, $inner_type:ty, { $($extra_methods:tt)* }) => {
        #[pyclass]
        struct $name {
            inner: $inner_type,
        }

        impl_nn_module!($name {
            #[new]
            #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, eps=1e-5, bias=true))]
            fn new(
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
                $name {
                    inner: <$inner_type>::new(
                        in_channels, out_channels, kernel_size, stride, padding, dilation, groups, eps, bias,
                    ),
                }
            }

            fn forward(&self, x: &PyTensor) -> PyTensor {
                PyTensor::from_tensor(self.inner.forward(&x.inner))
            }

            fn set_conv_weight(&mut self, weight: PyTensor) {
                self.inner.set_conv_weight(weight.inner);
            }

            fn set_conv_bias(&mut self, bias: PyTensor) {
                self.inner.set_conv_bias(bias.inner);
            }

            fn set_bn_weight(&mut self, weight: PyTensor) {
                self.inner.set_bn_weight(weight.inner);
            }

            fn set_bn_bias(&mut self, bias: PyTensor) {
                self.inner.set_bn_bias(bias.inner);
            }

            fn set_bn_running_mean(&mut self, mean: PyTensor) {
                self.inner.set_bn_running_mean(mean.inner);
            }

            fn set_bn_running_var(&mut self, var: PyTensor) {
                self.inner.set_bn_running_var(var.inner);
            }

            $($extra_methods)*
        });
    };
}

macro_rules! impl_activation {
    // Parameterless activation
    ($name:ident, $method:ident) => {
        #[pyclass]
        struct $name;

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> Self {
                $name
            }

            fn __call__(&self, x: &PyTensor) -> PyTensor {
                PyTensor::from_tensor(x.inner.$method())
            }

            fn parameters(&self) -> Vec<PyTensor> {
                vec![]
            }

            fn named_parameters(&self) -> Vec<(String, PyTensor)> {
                vec![]
            }

            fn zero_grad(&self) {}

            fn train(&self) {}

            fn eval(&self) {}

            fn is_training(&self) -> bool {
                false
            }
        }
    };
    // Activation with parameters (f64 converted to f32)
    ($name:ident, $method:ident, $($param_name:ident : $param_type:ty = $default:expr),*) => {
        #[pyclass]
        struct $name {
            $( $param_name: $param_type, )*
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = ( $($param_name = $default),* ))]
            fn new($( $param_name: $param_type, )*) -> Self {
                $name { $( $param_name, )* }
            }

            fn __call__(&self, x: &PyTensor) -> PyTensor {
                PyTensor::from_tensor(x.inner.$method($(self.$param_name as f32),*))
            }

            fn parameters(&self) -> Vec<PyTensor> {
                vec![]
            }

            fn named_parameters(&self) -> Vec<(String, PyTensor)> {
                vec![]
            }

            fn zero_grad(&self) {}

            fn train(&self) {}

            fn eval(&self) {}

            fn is_training(&self) -> bool {
                false
            }
        }
    };
}

#[pyclass]
struct Linear {
    inner: core_nn::linear::Linear,
}

impl_nn_module_with_gpu!(Linear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias = true))]
    fn new(in_features: i64, out_features: i64, bias: bool) -> Self {
        Linear {
            inner: core_nn::linear::Linear::new(in_features, out_features, bias),
        }
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = weight.inner;
    }

    fn set_bias(&mut self, bias: Option<PyTensor>) {
        self.inner.bias = bias.map(|t| t.inner);
    }

    #[classmethod]
    fn from_weights(_cls: &Bound<'_, PyType>, weight: PyTensor, bias: Option<PyTensor>) -> Self {
        let weight_shape = weight.inner.shape();
        let out_features = weight_shape[0];
        let in_features = weight_shape[1];
        let has_bias = bias.is_some();
        let mut inner = core_nn::linear::Linear::new(in_features, out_features, has_bias);
        // Incoming weight is [out_features, in_features] (PyTorch convention);
        // transpose to [in_features, out_features] for FastNN's direct matmul.
        inner.weight = weight.inner.transpose(0, 1).contiguous();
        if let Some(b) = bias {
            inner.bias = Some(b.inner);
        }
        Linear { inner }
    }
});

#[pyclass]
struct Conv2d {
    inner: core_nn::conv::Conv2d,
}

impl_nn_module_with_gpu!(Conv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
        bias: bool,
    ) -> Self {
        Conv2d {
            inner: core_nn::conv::Conv2d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
        }
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = weight.inner;
    }

    fn set_bias(&mut self, bias: Option<PyTensor>) {
        self.inner.bias = bias.map(|t| t.inner);
    }

    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        _cls: &Bound<'_, PyType>,
        weight: PyTensor,
        bias: Option<PyTensor>,
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
    ) -> Self {
        let mut inner = core_nn::conv::Conv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias.is_some(),
        );
        inner.weight = weight.inner;
        if let Some(b) = bias {
            inner.bias = Some(b.inner);
        }
        Conv2d { inner }
    }
});

#[pyclass]
struct MaxPool2d {
    inner: core_nn::pooling::MaxPool2d,
}

impl_nn_module!(MaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=2, padding=0, dilation=1))]
    fn new(kernel_size: i64, stride: i64, padding: i64, dilation: i64) -> Self {
        MaxPool2d {
            inner: core_nn::pooling::MaxPool2d::new(kernel_size, stride, padding, dilation),
        }
    }
});

#[pyclass]
struct AvgPool2d {
    inner: core_nn::pooling::AvgPool2d,
}

impl_nn_module!(AvgPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = 2, padding = 0))]
    fn new(kernel_size: i64, stride: i64, padding: i64) -> Self {
        AvgPool2d {
            inner: core_nn::pooling::AvgPool2d::new(kernel_size, stride, padding),
        }
    }
});

#[pyclass]
struct AvgPool1d {
    inner: core_nn::pooling::AvgPool1d,
}

impl_nn_module!(AvgPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = 2, padding = 0))]
    fn new(kernel_size: i64, stride: i64, padding: i64) -> Self {
        AvgPool1d {
            inner: core_nn::pooling::AvgPool1d::new(kernel_size, stride, padding),
        }
    }
});

#[pyclass]
struct MaxPool1d {
    inner: core_nn::pooling::MaxPool1d,
}

impl_nn_module!(MaxPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = 2, padding = 0, dilation = 1))]
    fn new(kernel_size: i64, stride: i64, padding: i64, dilation: i64) -> Self {
        MaxPool1d {
            inner: core_nn::pooling::MaxPool1d::new(kernel_size, stride, padding, dilation),
        }
    }
});

#[pyclass]
struct ConvTranspose2d {
    inner: core_nn::conv::ConvTranspose2d,
}

impl_nn_module!(ConvTranspose2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = true))]
    fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        bias: bool,
    ) -> Self {
        ConvTranspose2d {
            inner: core_nn::conv::ConvTranspose2d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                1,  // dilation (ignored by ConvTranspose2d)
                1,  // groups (ignored by ConvTranspose2d)
                bias,
            ),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }
});

#[pyclass]
struct Conv1d {
    inner: core_nn::conv::Conv1d,
}

impl_nn_module!(Conv1d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = true))]
    fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
        bias: bool,
    ) -> Self {
        Conv1d {
            inner: core_nn::conv::Conv1d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }
});

#[pyclass]
struct Conv3d {
    inner: core_nn::conv::Conv3d,
}

impl_nn_module!(Conv3d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = true))]
    fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
        groups: i64,
        bias: bool,
    ) -> Self {
        Conv3d {
            inner: core_nn::conv::Conv3d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            ),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }
});

#[pyclass]
struct ResidualBlock {
    inner: residual::ResidualBlock,
}

impl_nn_module!(ResidualBlock {
    #[new]
    #[pyo3(signature = (
        conv1_in, conv1_out, conv1_kernel, conv1_stride, conv1_padding,
        bn1_features,
        conv2_in, conv2_out, conv2_kernel, conv2_stride, conv2_padding,
        bn2_features,
        downsample = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        conv1_in: i64,
        conv1_out: i64,
        conv1_kernel: i64,
        conv1_stride: i64,
        conv1_padding: i64,
        bn1_features: i64,
        conv2_in: i64,
        conv2_out: i64,
        conv2_kernel: i64,
        conv2_stride: i64,
        conv2_padding: i64,
        bn2_features: i64,
        downsample: Option<(i64, i64, i64, i64, i64, i64)>,
    ) -> Self {
        ResidualBlock {
            inner: residual::ResidualBlock::new(
                conv1_in,
                conv1_out,
                conv1_kernel,
                conv1_stride,
                conv1_padding,
                bn1_features,
                conv2_in,
                conv2_out,
                conv2_kernel,
                conv2_stride,
                conv2_padding,
                bn2_features,
                downsample,
            ),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }
});

impl_fused_conv_bn!(FusedConvBn, core_nn::fused::FusedConvBn, {
    #[staticmethod]
    fn from_conv_bn(conv: &Conv2d, bn: &BatchNorm2d) -> Self {
        FusedConvBn {
            inner: core_nn::fused::FusedConvBn::from_conv_bn(&conv.inner, &bn.inner),
        }
    }
});

impl_fused_conv_bn!(FusedConvBnRelu, core_nn::fused::FusedConvBnRelu);
impl_fused_conv_bn!(FusedConvBnGelu, core_nn::fused::FusedConvBnGelu);

#[pyclass]
struct LayerNorm {
    inner: core_nn::norm::LayerNorm,
}

impl_nn_module!(LayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-5))]
    fn new(normalized_shape: i64, eps: f64) -> Self {
        LayerNorm {
            inner: core_nn::norm::LayerNorm::new(normalized_shape, eps),
        }
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = weight.inner;
    }

    fn set_bias(&mut self, bias: PyTensor) {
        self.inner.bias = bias.inner;
    }

    #[classmethod]
    fn from_weights(
        _cls: &Bound<'_, PyType>,
        weight: PyTensor,
        bias: PyTensor,
        normalized_shape: i64,
        eps: f64,
    ) -> Self {
        let mut inner = core_nn::norm::LayerNorm::new(normalized_shape, eps);
        inner.weight = weight.inner;
        inner.bias = bias.inner;
        LayerNorm { inner }
    }
});

#[pyclass]
struct BatchNorm1d {
    inner: core_nn::norm::BatchNorm1d,
}

impl_nn_module!(BatchNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1))]
    fn new(num_features: i64, eps: f64, momentum: f64) -> Self {
        BatchNorm1d {
            inner: core_nn::norm::BatchNorm1d::new(num_features, eps, momentum),
        }
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = Some(weight.inner);
    }

    fn set_bias(&mut self, bias: Option<PyTensor>) {
        self.inner.bias = bias.map(|t| t.inner);
    }

    fn get_running_mean(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.running_mean.read().clone())
    }

    fn set_running_mean(&mut self, running_mean: PyTensor) {
        self.inner.running_mean = Arc::new(RwLock::new(running_mean.inner));
    }

    fn get_running_var(&self) -> PyTensor {
        PyTensor::from_tensor(self.inner.running_var.read().clone())
    }

    fn set_running_var(&mut self, running_var: PyTensor) {
        self.inner.running_var = Arc::new(RwLock::new(running_var.inner));
    }

    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        _cls: &Bound<'_, PyType>,
        weight: PyTensor,
        bias: PyTensor,
        running_mean: PyTensor,
        running_var: PyTensor,
        num_features: i64,
        eps: f64,
        momentum: f64,
    ) -> Self {
        let mut inner = core_nn::norm::BatchNorm1d::new(num_features, eps, momentum);
        inner.weight = Some(weight.inner);
        inner.bias = Some(bias.inner);
        inner.running_mean = Arc::new(RwLock::new(running_mean.inner));
        inner.running_var = Arc::new(RwLock::new(running_var.inner));
        BatchNorm1d { inner }
    }
});

#[pyclass]
struct Dropout {
    inner: core_nn::dropout::Dropout,
}

impl_nn_module!(Dropout {
    #[new]
    fn new(p: f64) -> Self {
        Dropout {
            inner: core_nn::dropout::Dropout::new(p),
        }
    }
});

#[pyclass]
struct Dropout2d {
    inner: core_nn::dropout::Dropout2d,
}

impl_nn_module!(Dropout2d {
    #[new]
    fn new(p: f64) -> Self {
        Dropout2d {
            inner: core_nn::dropout::Dropout2d::new(p),
        }
    }
});

#[pyclass]
struct Upsample {
    inner: core_nn::upsample::Upsample,
}

impl_nn_module!(Upsample {
    #[new]
    fn new(scale_factor: f64, mode: String) -> Self {
        Upsample {
            inner: core_nn::upsample::Upsample::new(scale_factor as f32, mode),
        }
    }
});

#[pyclass]
struct Embedding {
    inner: core_nn::embedding::Embedding,
}

impl_nn_module!(Embedding {
    #[new]
    fn new(num_embeddings: i64, embedding_dim: i64) -> Self {
        Embedding {
            inner: core_nn::embedding::Embedding::new(num_embeddings, embedding_dim),
        }
    }

    fn forward(&self, indices: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&indices.inner))
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = weight.inner;
    }

    #[classmethod]
    fn from_weights(
        _cls: &Bound<'_, PyType>,
        weight: PyTensor,
        num_embeddings: i64,
        embedding_dim: i64,
    ) -> Self {
        let mut inner = core_nn::embedding::Embedding::new(num_embeddings, embedding_dim);
        inner.weight = weight.inner;
        Embedding { inner }
    }
});

#[pyclass]
struct RMSNorm {
    inner: core_nn::norm::RMSNorm,
}

impl_nn_module!(RMSNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-5))]
    fn new(normalized_shape: i64, eps: f32) -> Self {
        RMSNorm {
            inner: core_nn::norm::RMSNorm::new(normalized_shape, eps),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }
});

#[pyclass]
struct GroupNorm {
    inner: core_nn::norm::GroupNorm,
}

impl_nn_module!(GroupNorm {
    #[new]
    #[pyo3(signature = (num_groups, num_channels, eps = 1e-5))]
    fn new(num_groups: i64, num_channels: i64, eps: f32) -> Self {
        GroupNorm {
            inner: core_nn::norm::GroupNorm::new(num_groups, num_channels, eps),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }
});

#[pyclass]
struct BatchNorm2d {
    inner: core_nn::norm::BatchNorm2d,
}

impl_nn_module!(BatchNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5, momentum = 0.1))]
    fn new(num_features: i64, eps: f32, momentum: f32) -> Self {
        BatchNorm2d {
            inner: core_nn::norm::BatchNorm2d::new(num_features, eps, momentum),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = weight.inner;
    }

    fn set_bias(&mut self, bias: PyTensor) {
        self.inner.bias = bias.inner;
    }

    fn set_running_mean(&mut self, mean: PyTensor) {
        *self.inner.running_mean.write() = mean.inner;
    }

    fn set_running_var(&mut self, var: PyTensor) {
        *self.inner.running_var.write() = var.inner;
    }
});

impl_activation!(ReLU, relu);
impl_activation!(Gelu, gelu);
impl_activation!(Sigmoid, sigmoid);
impl_activation!(Tanh, tanh);
impl_activation!(SiLU, silu);
impl_activation!(Hardswish, hardswish);

impl_activation!(LeakyReLU, leaky_relu, negative_slope: f64 = 0.01);
impl_activation!(Softplus, softplus, beta: f64 = 1.0, threshold: f64 = 20.0);
impl_activation!(Elu, elu, alpha: f64 = 1.0);

#[pyclass]
struct PReLU {
    inner: core_nn::activations::PReLU,
}

impl_nn_module!(PReLU {
    #[new]
    fn new(num_parameters: i64) -> Self {
        PReLU {
            inner: core_nn::activations::PReLU::new(num_parameters),
        }
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = weight.inner;
    }
});

#[pyclass]
struct Softmax {
    inner: core_nn::activations::Softmax,
}

impl_nn_module!(Softmax {
    #[new]
    fn new(dim: i64) -> Self {
        Softmax {
            inner: core_nn::activations::Softmax::new(dim),
        }
    }
});

#[pyclass]
struct Mish;

#[pymethods]
impl Mish {
    #[new]
    fn new() -> Self {
        Mish
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        // mish(x) = x * tanh(softplus(x))
        // softplus(x) = ln(1 + exp(x))
        let sp = x.inner.exp().add_scalar(1.0).ln();
        let tanh_sp = sp.tanh();
        PyTensor::from_tensor(x.inner.mul(&tanh_sp))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, PyTensor)> {
        vec![]
    }

    fn zero_grad(&self) {}

    fn train(&self) {}

    fn eval(&self) {}

    fn is_training(&self) -> bool {
        false
    }
}

#[pyclass]
struct AdaptiveAvgPool2d {
    inner: core_nn::activations::AdaptiveAvgPool2d,
}

impl_nn_module!(AdaptiveAvgPool2d {
    #[new]
    fn new(output_h: i64, output_w: i64) -> Self {
        AdaptiveAvgPool2d {
            inner: core_nn::activations::AdaptiveAvgPool2d::new(output_h as usize, output_w as usize),
        }
    }
});

// ---------------------------------------------------------------------------
// Native module extraction helpers
// ---------------------------------------------------------------------------

/// Try to extract a native `Arc<dyn Module>` from a Python object that wraps
/// a Rust-backed module via its `inner` field.
macro_rules! try_extract_inner {
    ($b:expr, $py_type:ty, $native_type:ty) => {
        if let Ok(layer) = $b.extract::<pyo3::PyRef<'_, $py_type>>() {
            let cloned: $native_type = layer.inner.clone();
            return Some(Arc::new(cloned) as Arc<dyn Module>);
        }
    };
}

/// Try to extract a native `Arc<dyn Module>` from a parameterless Python
/// activation class.
macro_rules! try_extract_activation {
    ($b:expr, $py_type:ty, $native_type:ty) => {
        if $b.extract::<pyo3::PyRef<'_, $py_type>>().is_ok() {
            return Some(Arc::new(<$native_type>::new()) as Arc<dyn Module>);
        }
    };
}

/// Attempt to obtain a native `Arc<dyn Module>` for a Python-bound layer.
/// Returns `None` when the layer is a Python-only module that cannot be
/// converted.
fn try_extract_native_layer(b: &Bound<'_, PyAny>) -> Option<Arc<dyn Module>> {
    // ---- modules backed by an inner field (cloneable Rust Module) ----
    try_extract_inner!(b, Linear, core_nn::linear::Linear);
    try_extract_inner!(b, Conv2d, core_nn::conv::Conv2d);
    try_extract_inner!(b, Conv1d, core_nn::conv::Conv1d);
    try_extract_inner!(b, Conv3d, core_nn::conv::Conv3d);
    try_extract_inner!(b, ConvTranspose2d, core_nn::conv::ConvTranspose2d);
    try_extract_inner!(b, MaxPool2d, core_nn::pooling::MaxPool2d);
    try_extract_inner!(b, MaxPool1d, core_nn::pooling::MaxPool1d);
    try_extract_inner!(b, AvgPool2d, core_nn::pooling::AvgPool2d);
    try_extract_inner!(b, AvgPool1d, core_nn::pooling::AvgPool1d);
    try_extract_inner!(b, ResidualBlock, residual::ResidualBlock);
    try_extract_inner!(b, LayerNorm, core_nn::norm::LayerNorm);
    try_extract_inner!(b, BatchNorm1d, core_nn::norm::BatchNorm1d);
    try_extract_inner!(b, BatchNorm2d, core_nn::norm::BatchNorm2d);
    try_extract_inner!(b, RMSNorm, core_nn::norm::RMSNorm);
    try_extract_inner!(b, GroupNorm, core_nn::norm::GroupNorm);
    try_extract_inner!(b, Dropout, core_nn::dropout::Dropout);
    try_extract_inner!(b, Dropout2d, core_nn::dropout::Dropout2d);
    try_extract_inner!(b, Upsample, core_nn::upsample::Upsample);
    try_extract_inner!(b, Embedding, core_nn::embedding::Embedding);
    try_extract_inner!(b, PReLU, core_nn::activations::PReLU);
    try_extract_inner!(b, Softmax, core_nn::activations::Softmax);
    try_extract_inner!(
        b,
        AdaptiveAvgPool2d,
        core_nn::activations::AdaptiveAvgPool2d
    );
    try_extract_inner!(
        b,
        FusedConvBn,
        core_nn::fused::FusedConvBn<core_nn::fused::NoAct>
    );
    try_extract_inner!(
        b,
        FusedConvBnRelu,
        core_nn::fused::FusedConvBn<core_nn::fused::ReluAct>
    );
    try_extract_inner!(
        b,
        FusedConvBnGelu,
        core_nn::fused::FusedConvBn<core_nn::fused::GeluAct>
    );
    try_extract_inner!(
        b,
        PyTransformerEncoder,
        core_nn::transformer::TransformerEncoder
    );

    // ---- parameterless activations (no inner) ----
    try_extract_activation!(b, ReLU, core_nn::activations::ReLU);
    try_extract_activation!(b, Gelu, core_nn::activations::Gelu);
    try_extract_activation!(b, Sigmoid, core_nn::activations::Sigmoid);
    try_extract_activation!(b, Tanh, core_nn::activations::Tanh);
    try_extract_activation!(b, SiLU, core_nn::activations::SiLU);
    try_extract_activation!(b, Hardswish, core_nn::activations::Hardswish);
    try_extract_activation!(b, Mish, core_nn::activations::Mish);

    // ---- parameterised activations (extract fields directly) ----
    if let Ok(layer) = b.extract::<pyo3::PyRef<'_, LeakyReLU>>() {
        return Some(
            Arc::new(core_nn::activations::LeakyReLU::new(layer.negative_slope)) as Arc<dyn Module>,
        );
    }
    if let Ok(layer) = b.extract::<pyo3::PyRef<'_, Softplus>>() {
        return Some(Arc::new(core_nn::activations::Softplus::new(
            layer.beta,
            layer.threshold,
        )) as Arc<dyn Module>);
    }
    if let Ok(layer) = b.extract::<pyo3::PyRef<'_, Elu>>() {
        return Some(Arc::new(core_nn::activations::Elu::new(layer.alpha)) as Arc<dyn Module>);
    }

    None // Python-only module – will fall back to Python dispatch
}

/// Build native layer list for a `Sequential`.  Returns `None` when any layer
/// is a Python-only module (triggers the Python fallback path).
fn build_native_layers(py: Python<'_>, layers: &[Py<PyAny>]) -> Option<Vec<Arc<dyn Module>>> {
    let mut native = Vec::with_capacity(layers.len());
    for layer in layers {
        native.push(try_extract_native_layer(layer.bind(py))?);
    }
    Some(native)
}

// ---------------------------------------------------------------------------
// Sequential container
// ---------------------------------------------------------------------------

#[pyclass(name = "Sequential_")]
struct Sequential {
    layers: Vec<Py<PyAny>>,
    native_layers: Vec<Arc<dyn Module>>,
}

#[pymethods]
impl Sequential {
    #[new]
    fn new(py: Python<'_>, layers: Vec<Py<PyAny>>) -> Self {
        let native_layers = build_native_layers(py, &layers).unwrap_or_default();
        Sequential {
            layers,
            native_layers,
        }
    }

    #[getter]
    fn layers<'py>(&self, py: Python<'py>) -> Vec<Py<PyAny>> {
        self.layers.iter().map(|l| l.clone_ref(py)).collect()
    }

    fn __call__(&self, py: Python<'_>, x: PyTensor) -> PyResult<PyTensor> {
        if self.native_layers.len() == self.layers.len() {
            // Fast native path – zero Python C API calls per layer
            let mut result = x.inner;
            for layer in &self.native_layers {
                result = layer.forward(&result);
            }
            Ok(PyTensor::from_tensor(result))
        } else {
            // Python fallback for mixed / Python-only layers
            let mut result = x;
            for layer in &self.layers {
                let new_result = layer.call1(py, (result,))?;
                result = new_result.extract::<PyTensor>(py)?;
            }
            Ok(result)
        }
    }

    fn parameters(&self, py: Python<'_>) -> PyResult<Vec<PyTensor>> {
        if self.native_layers.len() == self.layers.len() {
            let mut params = vec![];
            for layer in &self.native_layers {
                for t in layer.parameters() {
                    params.push(PyTensor::from_tensor(t));
                }
            }
            Ok(params)
        } else {
            let mut params = vec![];
            for layer in &self.layers {
                let params_method: Py<PyAny> = layer.getattr(py, "parameters")?;
                let layer_params: Vec<PyTensor> = params_method.call0(py)?.extract(py)?;
                params.extend(layer_params);
            }
            Ok(params)
        }
    }

    fn named_parameters(&self, py: Python<'_>) -> PyResult<Vec<(String, PyTensor)>> {
        if self.native_layers.len() == self.layers.len() {
            let mut params = vec![];
            for (i, layer) in self.native_layers.iter().enumerate() {
                for (name, t) in layer.named_parameters() {
                    params.push((format!("{}.{}", i, name), PyTensor::from_tensor(t)));
                }
            }
            Ok(params)
        } else {
            let mut params = vec![];
            for (i, layer) in self.layers.iter().enumerate() {
                if let Ok(m) = layer.getattr(py, "named_parameters") {
                    let layer_params: Vec<(String, PyTensor)> = m.call0(py)?.extract(py)?;
                    for (name, t) in layer_params {
                        params.push((format!("{}.{}", i, name), t));
                    }
                }
            }
            Ok(params)
        }
    }

    fn train(&self, py: Python<'_>) {
        for layer in &self.native_layers {
            layer.train_mode();
        }
        for layer in &self.layers {
            if let Ok(m) = layer.getattr(py, "train") {
                let _ = m.call0(py);
            }
        }
    }

    fn eval(&self, py: Python<'_>) {
        for layer in &self.native_layers {
            layer.eval_mode();
        }
        for layer in &self.layers {
            if let Ok(m) = layer.getattr(py, "eval") {
                let _ = m.call0(py);
            }
        }
    }

    fn is_training(&self, py: Python<'_>) -> bool {
        if let Some(layer) = self.native_layers.first() {
            return layer.is_training();
        }
        if let Some(layer) = self.layers.first() {
            if let Ok(m) = layer.getattr(py, "is_training") {
                return m
                    .call0(py)
                    .ok()
                    .and_then(|v| v.extract::<bool>(py).ok())
                    .unwrap_or(false);
            }
        }
        false
    }

    fn zero_grad(&self, py: Python<'_>) {
        for layer in &self.native_layers {
            layer.zero_grad();
        }
        for layer in &self.layers {
            if let Ok(m) = layer.getattr(py, "zero_grad") {
                let _ = m.call0(py);
            }
        }
    }
}

#[pyclass]
struct ModuleList {
    modules: Vec<Py<PyAny>>,
}

#[pymethods]
impl ModuleList {
    #[new]
    fn new(modules: Vec<Py<PyAny>>) -> Self {
        ModuleList { modules }
    }

    fn __len__(&self) -> usize {
        self.modules.len()
    }

    fn __getitem__(&self, idx: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if idx >= self.modules.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "ModuleList index {} out of range (len={})",
                idx,
                self.modules.len()
            )));
        }
        Ok(self.modules[idx].clone_ref(py))
    }
}

#[pyclass]
struct PyTransformerEncoder {
    inner: core_nn::transformer::TransformerEncoder,
}

impl_nn_module!(PyTransformerEncoder {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocab_size: i64,
        max_seq_len: i64,
        d_model: i64,
        num_heads: i64,
        num_layers: i64,
        ff_dim: i64,
        num_classes: i64,
        dropout_p: f32,
    ) -> Self {
        PyTransformerEncoder {
            inner: core_nn::transformer::TransformerEncoder::new(
                vocab_size,
                max_seq_len,
                d_model,
                num_heads,
                num_layers,
                ff_dim,
                num_classes,
                dropout_p,
            ),
        }
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }
});

// ---- AotExecutor (ONNX graph execution via AOT compiler pipeline) ----

#[pyclass]
pub struct AotExecutor {
    plan: crate::backend::ExecutablePlan,
    memory_plan: crate::compiler::plan::MemoryPlan,
    graph: crate::ir::ComputeGraph,
    executor: crate::backend::executor::GraphExecutor<crate::backend::cpu::CpuBackend>,
    input_names: Vec<String>,
    output_map: Vec<(String, usize)>,
    prepared_plan: crate::backend::prepared::PreparedExecutablePlan,
}

#[pymethods]
impl AotExecutor {
    #[new]
    #[pyo3(signature = (nodes, params, input_names, output_names, input_shapes=None, quantize=None))]
    fn new(
        _py: pyo3::Python<'_>,
        nodes: Vec<std::collections::HashMap<String, String>>,
        params: std::collections::HashMap<String, PyTensor>,
        input_names: Vec<String>,
        output_names: Vec<String>,
        input_shapes: Option<std::collections::HashMap<String, Vec<i64>>>,
        quantize: Option<pyo3::Bound<'_, pyo3::PyAny>>,
    ) -> pyo3::PyResult<Self> {
        // Clear the global f32 weight cache to prevent unbounded memory
        // accumulation across executor instances (e.g., when the benchmark
        // script creates a new AotExecutor for a different quantization dtype).
        // Without this, dequantized f32 weights from all dtypes persist
        // in the global cache for the lifetime of the Python process.
        crate::backend::cpu::clear_f32_weight_cache();
        // Convert Python node dicts to OnnxNodes
        let onnx_nodes: Vec<crate::onnx::converter::OnnxNode> = nodes
            .into_iter()
            .map(|m| {
                let name = m.get("name").cloned().unwrap_or_default();
                let op_type = m.get("op_type").cloned().unwrap_or_default();
                let inputs: Vec<String> = m
                    .get("inputs")
                    .map(|s| {
                        s.trim_matches(|c| c == '[' || c == ']')
                            .split(',')
                            .map(|x| x.trim().to_string())
                            .filter(|x| !x.is_empty())
                            .collect()
                    })
                    .unwrap_or_default();
                let outputs: Vec<String> = m
                    .get("outputs")
                    .map(|s| {
                        s.trim_matches(|c| c == '[' || c == ']')
                            .split(',')
                            .map(|x| x.trim().to_string())
                            .filter(|x| !x.is_empty())
                            .collect()
                    })
                    .unwrap_or_default();
                let attrs: std::collections::HashMap<String, String> = m
                    .into_iter()
                    .filter(|(k, _)| {
                        *k != "name"
                            && *k != "op_type"
                            && *k != "inputs"
                            && *k != "outputs"
                            && *k != "output_shape"
                    })
                    .collect();
                crate::onnx::converter::OnnxNode {
                    name,
                    op_type,
                    inputs,
                    outputs,
                    attrs,
                }
            })
            .collect();

        let rust_params: std::collections::HashMap<String, Tensor> =
            params.into_iter().map(|(k, v)| (k, v.inner)).collect();

        // Build input shapes map if provided.
        let mut rust_input_shapes: std::collections::HashMap<
            String,
            Vec<crate::ir::DimExpr>,
        > = std::collections::HashMap::new();
        if let Some(shapes) = input_shapes {
            for (name, dims) in shapes {
                let ir_dims: Vec<crate::ir::DimExpr> = dims
                    .into_iter()
                    .map(|d| -> PyResult<crate::ir::DimExpr> {
                        if d == i64::MIN {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "input {name} contains unsupported dimension {d}"
                            )));
                        }
                        if d < 0 {
                            Ok(crate::ir::DimExpr::Symbol(format!("d{}", -d)))
                        } else {
                            Ok(crate::ir::DimExpr::Known(d as u64))
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                rust_input_shapes.insert(name, ir_dims);
            }
        }

        let converter = crate::onnx::converter::OnnxConverter::new(
            &onnx_nodes,
            &rust_params,
            &input_names,
            &output_names,
        )
        .with_input_shapes(&rust_input_shapes);
        let graph = converter
            .to_compute_graph()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        use crate::types::{CompileTarget, QuantTarget};
        let target = match quantize {
            None => CompileTarget::Native,
            Some(obj) => {
                if let Ok(val) = obj.extract::<u8>() {
                    match val {
                        4 => CompileTarget::WeightOnly(QuantTarget::I4),
                        8 => CompileTarget::WeightOnly(QuantTarget::I8),
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "unsupported quantize value for integer: {} (expected 4 or 8)",
                                val
                            )))
                        }
                    }
                } else if let Ok(s) = obj.extract::<String>() {
                    match s.as_str() {
                        "f32" => CompileTarget::Native,
                        "i4" => CompileTarget::WeightOnly(QuantTarget::I4),
                        "i8" => CompileTarget::WeightOnly(QuantTarget::I8),
                        "u4" => CompileTarget::WeightOnly(QuantTarget::U4),
                        "u8" => CompileTarget::WeightOnly(QuantTarget::U8),
                        "f8" => CompileTarget::WeightOnly(QuantTarget::Fp8E4M3),
                        "f8r" => CompileTarget::WeightOnly(QuantTarget::Fp8E5M2),
                        "f4" => CompileTarget::WeightOnly(QuantTarget::Fp4E2M1),
                        "i4cb" => CompileTarget::WeightOnly(QuantTarget::I4Codebook),
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "unsupported quantize string: '{}' (expected f32, i4, i8, u4, u8, f8, f8r, f4, i4cb, 4, or 8)",
                                s
                            )))
                        }
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "quantize must be int (4, 8) or str (f32, f8, f8r, f4) or None",
                    ));
                }
            }
        };

        let executor =
            crate::backend::executor::GraphExecutor::new(crate::backend::cpu::CpuBackend);
        let (plan, memory_plan, compiled_graph) = executor
            .compile_with_target(graph, target, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let output_map: Vec<(String, usize)> = output_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        let prepared_plan = crate::backend::prepared::prepare_executable_plan(&plan);

        Ok(AotExecutor {
            plan,
            memory_plan,
            graph: compiled_graph,
            executor,
            input_names,
            output_map,
            prepared_plan,
        })
    }

    fn forward(
        &mut self,
        inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<std::collections::HashMap<String, PyTensor>> {
        let input_refs: Vec<&[u8]> = self
            .input_names
            .iter()
            .map(|name| {
                inputs
                    .get(name.as_str())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "required input '{}' not found",
                            name
                        ))
                    })
                    .and_then(|tensor| {
                        tensor.inner.try_as_bytes().map_err(|error| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "input {name} cannot be passed to AOT execution: {error}"
                            ))
                        })
                    })
            })
            .collect::<pyo3::PyResult<Vec<&[u8]>>>()?;

        #[cfg(feature = "prepared-plan")]
        let output_data = {
            if self.prepared_plan.static_weight_binding_count() > 0 {
                self.executor
                    .execute_prepared_no_copy(
                        &self.graph,
                        &mut self.plan,
                        &self.memory_plan,
                        &input_refs,
                        &self.prepared_plan,
                    )
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            } else {
                self.executor
                    .execute(&self.graph, &mut self.plan, &self.memory_plan, &input_refs)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
        };

        #[cfg(not(feature = "prepared-plan"))]
        let output_data = self
            .executor
            .execute(&self.graph, &mut self.plan, &self.memory_plan, &input_refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        self.decode_outputs(output_data)
    }

    /// Apply calibration scales from a JSON file to recompile the model with
    /// optimized activation quantization parameters.
    ///
    /// The JSON should be in the format produced by `calibrate_yolo.py` or
    /// `CalibrationData::to_quant_config()`:
    /// ```json
    /// {
    ///   "tensor_name": {
    ///     "scale": 0.01,
    ///     "zero_point": 128.0,
    ///     "bit_width": 8,
    ///     "min": -1.0,
    ///     "max": 1.0
    ///   },
    ///   ...
    /// }
    /// ```
    ///
    /// This recompiles the entire model with the calibration data, so it may
    /// take some time. The model must have been originally created with
    /// `quantize=4` or `quantize=8`.
    fn apply_calibration(&mut self, scales_json: String) -> pyo3::PyResult<()> {
        // Parse calibration data from JSON
        let calib = crate::compiler::passes::calibration::CalibrationData::from_json(&scales_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        // Compile a clone so failure leaves the live model and all of its
        // execution state untouched. Commit the replacement artifacts only
        // after every derived representation has been rebuilt successfully.
        let (plan, memory_plan, graph) = self
            .executor
            .compile_with_plan_and_quantize(self.graph.clone(), None, Some(calib))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let prepared_plan = crate::backend::prepared::prepare_executable_plan(&plan);

        self.executor.invalidate_runtime_cache();
        self.plan = plan;
        self.memory_plan = memory_plan;
        self.graph = graph;
        self.prepared_plan = prepared_plan;

        Ok(())
    }

    /// Opt-in prepared-execution fallback. Accepts the same input dict
    /// shape as [`Self::forward`] and returns a `dict` of output tensors
    /// that is **byte-identical** to the corresponding `forward` call.
    ///
    /// This method is the user-facing entry point on top of the
    /// `prepared-plan` Rust feature. It validates that the prepared
    /// plan attached at construction time stays in lock-step with the
    /// underlying [`ExecutablePlan`] and then delegates to the existing
    /// executor path, so its observable behaviour is exactly the same
    /// as `forward` on the same inputs. Future lanes can layer
    /// specialised prepared-instruction execution on top of this entry
    /// point without changing the contract.
    ///
    /// Falls back to a clear runtime error when the `prepared-plan`
    /// cargo feature is not enabled at build time.
    #[cfg(feature = "prepared-plan")]
    fn forward_prepared_fallback(
        &mut self,
        inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<std::collections::HashMap<String, PyTensor>> {
        let input_refs: Vec<&[u8]> = self
            .input_names
            .iter()
            .map(|name| {
                inputs
                    .get(name.as_str())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "required input '{}' not found",
                            name
                        ))
                    })
                    .and_then(|tensor| {
                        tensor.inner.try_as_bytes().map_err(|error| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "input {name} cannot be passed to AOT execution: {error}"
                            ))
                        })
                    })
            })
            .collect::<pyo3::PyResult<Vec<&[u8]>>>()?;

        let output_data = self
            .executor
            .execute_prepared_fallback(
                &self.graph,
                &mut self.plan,
                &self.memory_plan,
                &input_refs,
                &self.prepared_plan,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        self.decode_outputs(output_data)
    }

    #[cfg(not(feature = "prepared-plan"))]
    fn forward_prepared_fallback(
        &self,
        _inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<std::collections::HashMap<String, PyTensor>> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "forward_prepared_fallback requires the 'prepared-plan' feature",
        ))
    }

    /// Opt-in prepared fallback that preloads fp32 constants from the
    /// PreparedConstantArena into the runtime arena before normal
    /// dispatch. WriteConst instructions still run, so this is a
    /// behaviour-identical plumbing check rather than an optimisation.
    #[cfg(feature = "prepared-plan")]
    fn forward_prepared_arena_fallback(
        &mut self,
        inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<std::collections::HashMap<String, PyTensor>> {
        let input_refs: Vec<&[u8]> = self
            .input_names
            .iter()
            .map(|name| {
                inputs
                    .get(name.as_str())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "required input '{}' not found",
                            name
                        ))
                    })
                    .and_then(|tensor| {
                        tensor.inner.try_as_bytes().map_err(|error| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "input {name} cannot be passed to AOT execution: {error}"
                            ))
                        })
                    })
            })
            .collect::<pyo3::PyResult<Vec<&[u8]>>>()?;

        let output_data = self
            .executor
            .execute_prepared_arena_fallback(
                &self.graph,
                &mut self.plan,
                &self.memory_plan,
                &input_refs,
                &self.prepared_plan,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        self.decode_outputs(output_data)
    }

    #[cfg(not(feature = "prepared-plan"))]
    fn forward_prepared_arena_fallback(
        &self,
        _inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<std::collections::HashMap<String, PyTensor>> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "forward_prepared_arena_fallback requires the 'prepared-plan' feature",
        ))
    }

    /// Opt-in prepared no-copy path.  Built on top of a persistent
    /// immutable view of the fp32 weight / bias constants stored in
    /// [`PreparedConstantArena`].  The runtime arena is not
    /// pre-populated for those slots, and the corresponding
    /// `WriteConst` instructions are filtered out of the plan; the
    /// dispatch kernel borrows the weight / bias bytes directly from
    /// the persistent view instead of pulling them out of the mutable
    /// arena.  Behaviour is identical to [`AotExecutor::forward`] when
    /// the model has no fp32 Conv2d / MatMul slot to override, and
    /// matches it bit-exactly for the models covered by the view.
    #[cfg(feature = "prepared-plan")]
    fn forward_prepared_no_copy(
        &mut self,
        inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<std::collections::HashMap<String, PyTensor>> {
        let input_refs: Vec<&[u8]> = self
            .input_names
            .iter()
            .map(|name| {
                inputs
                    .get(name.as_str())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "required input '{}' not found",
                            name
                        ))
                    })
                    .and_then(|tensor| {
                        tensor.inner.try_as_bytes().map_err(|error| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "input {name} cannot be passed to AOT execution: {error}"
                            ))
                        })
                    })
            })
            .collect::<pyo3::PyResult<Vec<&[u8]>>>()?;

        let output_data = self
            .executor
            .execute_prepared_no_copy(
                &self.graph,
                &mut self.plan,
                &self.memory_plan,
                &input_refs,
                &self.prepared_plan,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        self.decode_outputs(output_data)
    }

    #[cfg(not(feature = "prepared-plan"))]
    fn forward_prepared_no_copy(
        &self,
        _inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<std::collections::HashMap<String, PyTensor>> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "forward_prepared_no_copy requires the 'prepared-plan' feature",
        ))
    }

    fn profile(
        &mut self,
        py: pyo3::Python<'_>,
        inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        let input_refs: Vec<&[u8]> = self
            .input_names
            .iter()
            .map(|name| {
                inputs
                    .get(name.as_str())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "required input '{}' not found",
                            name
                        ))
                    })
                    .and_then(|tensor| {
                        tensor.inner.try_as_bytes().map_err(|error| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "input {name} cannot be passed to AOT execution: {error}"
                            ))
                        })
                    })
            })
            .collect::<pyo3::PyResult<Vec<&[u8]>>>()?;

        let (output_data, profile_entries) = self
            .executor
            .execute_profile(&self.graph, &mut self.plan, &self.memory_plan, &input_refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        self.encode_profile_result(py, output_data, profile_entries)
    }

    #[cfg(feature = "prepared-plan")]
    fn profile_prepared_arena_fallback(
        &mut self,
        py: pyo3::Python<'_>,
        inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        let input_refs: Vec<&[u8]> = self
            .input_names
            .iter()
            .map(|name| {
                inputs
                    .get(name.as_str())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "required input '{}' not found",
                            name
                        ))
                    })
                    .and_then(|tensor| {
                        tensor.inner.try_as_bytes().map_err(|error| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "input {name} cannot be passed to AOT execution: {error}"
                            ))
                        })
                    })
            })
            .collect::<pyo3::PyResult<Vec<&[u8]>>>()?;

        let (output_data, profile_entries) = self
            .executor
            .execute_profile_prepared_arena_fallback(
                &self.graph,
                &mut self.plan,
                &self.memory_plan,
                &input_refs,
                &self.prepared_plan,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        self.encode_profile_result(py, output_data, profile_entries)
    }

    #[cfg(not(feature = "prepared-plan"))]
    fn profile_prepared_arena_fallback(
        &self,
        _py: pyo3::Python<'_>,
        _inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "profile_prepared_arena_fallback requires the 'prepared-plan' feature",
        ))
    }

    /// Return static memory-efficiency statistics for the compiled graph/plan.
    ///
    /// This is model-agnostic compiler/runtime introspection: it reports arena
    /// pressure, physical-copy/write-constant bytes, instruction mix, and alias
    /// reuse groups without executing the graph. Use it to find broad memory and
    /// layout efficiency targets before adding kernel-specific optimisations.
    fn memory_stats(&self, py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        use crate::backend::Instruction;
        use pyo3::types::{PyDict, PyList};
        use std::collections::{BTreeMap, HashMap};

        let stats = PyDict::new(py);
        let plan = &self.plan;
        let memory_plan = &self.memory_plan;

        let mut logical_slot_bytes = 0usize;
        let mut physical_by_offset: BTreeMap<usize, usize> = BTreeMap::new();
        let mut nodes_by_offset: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (node_id, slot) in &memory_plan.slots {
            logical_slot_bytes += slot.size;
            physical_by_offset
                .entry(slot.offset)
                .and_modify(|size| *size = (*size).max(slot.size))
                .or_insert(slot.size);
            nodes_by_offset
                .entry(slot.offset)
                .or_default()
                .push(*node_id);
        }
        for ((node_id, _output_index), slot) in &memory_plan.secondary_slots {
            logical_slot_bytes += slot.size;
            physical_by_offset
                .entry(slot.offset)
                .and_modify(|size| *size = (*size).max(slot.size))
                .or_insert(slot.size);
            nodes_by_offset
                .entry(slot.offset)
                .or_default()
                .push(*node_id);
        }
        let physical_slot_bytes: usize = physical_by_offset.values().copied().sum();
        let alias_groups = nodes_by_offset
            .values()
            .filter(|nodes| nodes.len() > 1)
            .count();
        let aliased_nodes: usize = nodes_by_offset
            .values()
            .filter(|nodes| nodes.len() > 1)
            .map(|nodes| nodes.len())
            .sum();

        let mut call_kernel_count = 0usize;
        let mut memcpy_count = 0usize;
        let mut fill_count = 0usize;
        let mut write_const_count = 0usize;
        let mut kernel_read_bytes = 0usize;
        let mut kernel_write_bytes = 0usize;
        let mut memcpy_bytes = 0usize;
        let mut fill_bytes = 0usize;
        let mut write_const_bytes = 0usize;
        let mut kernel_counts: HashMap<String, (usize, usize, usize)> = HashMap::new();
        #[derive(Clone)]
        struct InstructionTrafficRow {
            instruction_index: usize,
            kind: String,
            kernel_name: String,
            node_id: Option<usize>,
            read_bytes: usize,
            write_bytes: usize,
            node_name: Option<String>,
            op_type: Option<String>,
            input_nodes: Vec<usize>,
            input_shapes: Vec<Vec<Option<u64>>>,
            output_shape: Option<Vec<Option<u64>>>,
        }
        let mut instruction_traffic: Vec<InstructionTrafficRow> = Vec::new();
        let mut write_const_traffic: Vec<(usize, usize, usize, usize)> = Vec::new();

        #[derive(Clone)]
        struct PreparedStaticSlotInfo {
            consumer_instruction_index: usize,
            input_index: usize,
            role: &'static str,
            constant_index: usize,
            constant_name: String,
        }

        #[cfg(feature = "prepared-plan")]
        let prepared_static_slots: HashMap<(usize, usize), PreparedStaticSlotInfo> = {
            use crate::backend::prepared::PreparedInstruction;

            let mut slots = HashMap::new();
            if let Some(arena) = self.prepared_plan.constant_arena() {
                for prepared in &self.prepared_plan.instructions {
                    match prepared {
                        PreparedInstruction::Conv2d(conv) => {
                            if let Some(id) = conv.packed_weight {
                                if let Some(entry) = arena.entry(id) {
                                    slots.insert(
                                        (conv.weight.offset, conv.weight.size),
                                        PreparedStaticSlotInfo {
                                            consumer_instruction_index: conv.instruction_index,
                                            input_index: 1,
                                            role: "conv_weight",
                                            constant_index: id.index,
                                            constant_name: entry.name.clone(),
                                        },
                                    );
                                }
                            }
                            if let (Some(bias), Some(id)) = (conv.bias, conv.packed_bias) {
                                if let Some(entry) = arena.entry(id) {
                                    slots.insert(
                                        (bias.offset, bias.size),
                                        PreparedStaticSlotInfo {
                                            consumer_instruction_index: conv.instruction_index,
                                            input_index: 2,
                                            role: "conv_bias",
                                            constant_index: id.index,
                                            constant_name: entry.name.clone(),
                                        },
                                    );
                                }
                            }
                        }
                        PreparedInstruction::MatMul(matmul) => {
                            if let Some(id) = matmul.packed_weight {
                                if let Some(entry) = arena.entry(id) {
                                    slots.insert(
                                        (matmul.b.offset, matmul.b.size),
                                        PreparedStaticSlotInfo {
                                            consumer_instruction_index: matmul.instruction_index,
                                            input_index: 1,
                                            role: "matmul_weight",
                                            constant_index: id.index,
                                            constant_name: entry.name.clone(),
                                        },
                                    );
                                }
                            }
                            if let (Some(bias), Some(id)) = (matmul.bias, matmul.packed_bias) {
                                if let Some(entry) = arena.entry(id) {
                                    slots.insert(
                                        (bias.offset, bias.size),
                                        PreparedStaticSlotInfo {
                                            consumer_instruction_index: matmul.instruction_index,
                                            input_index: 2,
                                            role: "matmul_bias",
                                            constant_index: id.index,
                                            constant_name: entry.name.clone(),
                                        },
                                    );
                                }
                            }
                        }
                        PreparedInstruction::Generic { .. } => {}
                    }
                }
            }
            slots
        };
        #[cfg(not(feature = "prepared-plan"))]
        let prepared_static_slots: HashMap<(usize, usize), PreparedStaticSlotInfo> = HashMap::new();

        for (instruction_index, instr) in plan.instructions.iter().enumerate() {
            match instr {
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    secondary_output_slice,
                    node_id,
                    ..
                } => {
                    call_kernel_count += 1;
                    let entry = kernel_counts
                        .entry(kernel_name.clone())
                        .or_insert((0, 0, 0));
                    entry.0 += 1;
                    let read_bytes = input_slices.iter().map(|s| s.size).sum::<usize>();
                    let mut write_bytes = output_slice.size;
                    if let Some(sec) = secondary_output_slice {
                        write_bytes += sec.size;
                    }
                    entry.1 += read_bytes;
                    entry.2 += write_bytes;
                    kernel_read_bytes += read_bytes;
                    kernel_write_bytes += write_bytes;

                    let node = node_id.and_then(|id| self.graph.get_node(id));
                    let input_nodes = node.map(|n| n.inputs.clone()).unwrap_or_default();
                    let input_shapes = input_nodes
                        .iter()
                        .filter_map(|id| self.graph.get_node(*id))
                        .map(|input| {
                            input
                                .output_type
                                .shape
                                .iter()
                                .map(|d| d.evaluate())
                                .collect()
                        })
                        .collect();
                    let output_shape =
                        node.map(|n| n.output_type.shape.iter().map(|d| d.evaluate()).collect());
                    instruction_traffic.push(InstructionTrafficRow {
                        instruction_index,
                        kind: "call_kernel".to_string(),
                        kernel_name: kernel_name.clone(),
                        node_id: *node_id,
                        read_bytes,
                        write_bytes,
                        node_name: node.map(|n| n.name.clone()),
                        op_type: node.map(|n| format!("{:?}", n.opcode)),
                        input_nodes,
                        input_shapes,
                        output_shape,
                    });
                }
                Instruction::MemCopy { dst, src } => {
                    memcpy_count += 1;
                    let bytes = dst.size.min(src.size);
                    memcpy_bytes += bytes;
                    instruction_traffic.push(InstructionTrafficRow {
                        instruction_index,
                        kind: "memcopy".to_string(),
                        kernel_name: "memcopy".to_string(),
                        node_id: None,
                        read_bytes: bytes,
                        write_bytes: bytes,
                        node_name: None,
                        op_type: None,
                        input_nodes: Vec::new(),
                        input_shapes: Vec::new(),
                        output_shape: None,
                    });
                }
                Instruction::Fill { dst, .. } => {
                    fill_count += 1;
                    fill_bytes += dst.size;
                    instruction_traffic.push(InstructionTrafficRow {
                        instruction_index,
                        kind: "fill".to_string(),
                        kernel_name: "fill".to_string(),
                        node_id: None,
                        read_bytes: 0,
                        write_bytes: dst.size,
                        node_name: None,
                        op_type: None,
                        input_nodes: Vec::new(),
                        input_shapes: Vec::new(),
                        output_shape: None,
                    });
                }
                Instruction::WriteConst { dst, data } => {
                    write_const_count += 1;
                    let bytes = dst.size.min(data.len());
                    write_const_bytes += bytes;
                    write_const_traffic.push((instruction_index, dst.offset, dst.size, data.len()));
                    instruction_traffic.push(InstructionTrafficRow {
                        instruction_index,
                        kind: "write_const".to_string(),
                        kernel_name: "write_const".to_string(),
                        node_id: None,
                        read_bytes: 0,
                        write_bytes: bytes,
                        node_name: None,
                        op_type: None,
                        input_nodes: Vec::new(),
                        input_shapes: Vec::new(),
                        output_shape: None,
                    });
                }
            }
        }

        stats.set_item("arena_size", plan.arena_size)?;
        stats.set_item("memory_plan_total_size", memory_plan.total_size)?;
        stats.set_item("logical_slot_bytes", logical_slot_bytes)?;
        stats.set_item("physical_slot_bytes", physical_slot_bytes)?;
        stats.set_item(
            "slot_reuse_saved_bytes",
            logical_slot_bytes.saturating_sub(physical_slot_bytes),
        )?;
        stats.set_item("alias_groups", alias_groups)?;
        stats.set_item("aliased_nodes", aliased_nodes)?;
        stats.set_item("primary_slots", memory_plan.slots.len())?;
        stats.set_item("secondary_slots", memory_plan.secondary_slots.len())?;
        stats.set_item("instructions", plan.instructions.len())?;
        stats.set_item("call_kernel_count", call_kernel_count)?;
        stats.set_item("memcpy_count", memcpy_count)?;
        stats.set_item("fill_count", fill_count)?;
        stats.set_item("write_const_count", write_const_count)?;
        stats.set_item("kernel_read_bytes", kernel_read_bytes)?;
        stats.set_item("kernel_write_bytes", kernel_write_bytes)?;
        stats.set_item("memcpy_bytes", memcpy_bytes)?;
        stats.set_item("fill_bytes", fill_bytes)?;
        stats.set_item("write_const_bytes", write_const_bytes)?;
        stats.set_item(
            "estimated_static_traffic_bytes",
            kernel_read_bytes + kernel_write_bytes + memcpy_bytes + fill_bytes + write_const_bytes,
        )?;

        let top_kernels = PyList::empty(py);
        let mut kernel_counts: Vec<(String, (usize, usize, usize))> =
            kernel_counts.into_iter().collect();
        kernel_counts.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then_with(|| a.0.cmp(&b.0)));
        for (kernel, (count, read_bytes, write_bytes)) in kernel_counts.into_iter().take(20) {
            let item = PyDict::new(py);
            item.set_item("kernel", kernel)?;
            item.set_item("count", count)?;
            item.set_item("read_bytes", read_bytes)?;
            item.set_item("write_bytes", write_bytes)?;
            item.set_item("static_bytes", read_bytes + write_bytes)?;
            top_kernels.append(item)?;
        }
        stats.set_item("top_kernels_by_count", top_kernels)?;

        let top_instructions = PyList::empty(py);
        instruction_traffic.sort_by(|a, b| {
            let a_total = a.read_bytes + a.write_bytes;
            let b_total = b.read_bytes + b.write_bytes;
            b_total
                .cmp(&a_total)
                .then_with(|| a.instruction_index.cmp(&b.instruction_index))
        });
        for row in instruction_traffic.into_iter().take(50) {
            let item = PyDict::new(py);
            item.set_item("instruction_index", row.instruction_index)?;
            item.set_item("kind", row.kind)?;
            item.set_item("kernel_name", row.kernel_name)?;
            item.set_item("node_id", row.node_id)?;
            item.set_item("read_bytes", row.read_bytes)?;
            item.set_item("write_bytes", row.write_bytes)?;
            item.set_item("static_bytes", row.read_bytes + row.write_bytes)?;
            if let Some(node_name) = row.node_name {
                item.set_item("node_name", node_name)?;
            }
            if let Some(op_type) = row.op_type {
                item.set_item("op_type", op_type)?;
            }
            if !row.input_nodes.is_empty() {
                item.set_item("input_nodes", row.input_nodes.clone())?;
                item.set_item("input_node_ids", row.input_nodes)?;
                item.set_item("input_shapes", row.input_shapes)?;
            }
            if let Some(output_shape) = row.output_shape {
                item.set_item("output_shape", output_shape)?;
            }
            top_instructions.append(item)?;
        }
        stats.set_item("top_instructions_by_static_bytes", top_instructions)?;

        let top_write_consts = PyList::empty(py);
        write_const_traffic.sort_by(|a, b| {
            let a_bytes = a.2.min(a.3);
            let b_bytes = b.2.min(b.3);
            b_bytes.cmp(&a_bytes).then_with(|| a.0.cmp(&b.0))
        });
        for (instruction_index, dst_offset, dst_size, data_len) in
            write_const_traffic.into_iter().take(50)
        {
            let item = PyDict::new(py);
            item.set_item("instruction_index", instruction_index)?;
            item.set_item("dst_offset", dst_offset)?;
            item.set_item("dst_size", dst_size)?;
            item.set_item("data_len", data_len)?;
            item.set_item("write_bytes", dst_size.min(data_len))?;
            if let Some(prepared) = prepared_static_slots.get(&(dst_offset, dst_size)) {
                item.set_item(
                    "prepared_consumer_instruction_index",
                    prepared.consumer_instruction_index,
                )?;
                item.set_item("prepared_input_index", prepared.input_index)?;
                item.set_item("prepared_static_role", prepared.role)?;
                item.set_item("prepared_constant_index", prepared.constant_index)?;
                item.set_item("prepared_constant_name", prepared.constant_name.as_str())?;
            }
            top_write_consts.append(item)?;
        }
        stats.set_item("top_write_consts_by_size", top_write_consts)?;

        let top_alias_groups = PyList::empty(py);
        let mut alias_vec: Vec<(usize, Vec<usize>, usize)> = nodes_by_offset
            .into_iter()
            .filter_map(|(offset, mut nodes)| {
                if nodes.len() <= 1 {
                    return None;
                }
                nodes.sort_unstable();
                let size = physical_by_offset.get(&offset).copied().unwrap_or(0);
                Some((offset, nodes, size))
            })
            .collect();
        alias_vec.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.1.len().cmp(&a.1.len())));
        for (offset, nodes, size) in alias_vec.into_iter().take(20) {
            let item = PyDict::new(py);
            item.set_item("offset", offset)?;
            item.set_item("size", size)?;
            item.set_item("nodes", nodes)?;
            top_alias_groups.append(item)?;
        }
        stats.set_item("top_alias_groups", top_alias_groups)?;

        Ok(stats.into())
    }

    #[cfg(feature = "prepared-plan")]
    fn prepared_stats(&self) -> pyo3::PyResult<std::collections::HashMap<String, usize>> {
        use crate::backend::prepared::PreparedInstruction;
        let instructions = &self.prepared_plan.instructions;
        let mut stats = std::collections::HashMap::new();
        stats.insert("total".to_string(), instructions.len());
        stats.insert(
            "generic".to_string(),
            instructions
                .iter()
                .filter(|i| matches!(i, PreparedInstruction::Generic { .. }))
                .count(),
        );
        stats.insert(
            "conv2d".to_string(),
            instructions
                .iter()
                .filter(|i| matches!(i, PreparedInstruction::Conv2d(_)))
                .count(),
        );
        stats.insert(
            "matmul".to_string(),
            instructions
                .iter()
                .filter(|i| matches!(i, PreparedInstruction::MatMul(_)))
                .count(),
        );
        stats.insert(
            "static_weight_bindings".to_string(),
            self.prepared_plan.static_weight_binding_count(),
        );
        stats.insert(
            "constant_arena_entries".to_string(),
            self.prepared_plan.constant_arena_entry_count(),
        );
        stats.insert(
            "constant_arena_bytes".to_string(),
            self.prepared_plan.constant_arena_total_bytes(),
        );
        stats.insert(
            "packed_fp32_conv_candidates".to_string(),
            self.prepared_plan.packed_fp32_conv_candidate_count(),
        );
        stats.insert(
            "packed_fp32_conv_candidate_flops".to_string(),
            self.prepared_plan.packed_fp32_conv_candidate_flops(),
        );
        stats.insert(
            "transposed_fp32_conv_entries".to_string(),
            self.prepared_plan.transposed_fp32_conv_entry_count(),
        );
        stats.insert(
            "transposed_fp32_conv_bytes".to_string(),
            self.prepared_plan.transposed_fp32_conv_total_bytes(),
        );
        stats.insert(
            "transposed_fp32_conv_bindings".to_string(),
            self.prepared_plan.transposed_fp32_conv_binding_count(),
        );
        stats.insert(
            "transposed_fp32_conv_binding_flops".to_string(),
            self.prepared_plan.transposed_fp32_conv_binding_flops(),
        );
        Ok(stats)
    }

    #[cfg(not(feature = "prepared-plan"))]
    fn prepared_stats(&self) -> pyo3::PyResult<std::collections::HashMap<String, usize>> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "prepared_stats requires the 'prepared-plan' feature",
        ))
    }

    /// Debug method to print the compiled graph structure.
    #[cfg(feature = "prepared-plan")]
    fn debug_graph(&self) -> pyo3::PyResult<String> {
        let mut out = String::new();
        out.push_str("=== COMPILED GRAPH ===\n");
        for node_id in self.graph.try_topological_sort().map_err(PyErr::from)? {
            if let Some(node) = self.graph.get_node(node_id) {
                out.push_str(&format!(
                    "  Node {}: {} / {:?} / inputs={:?}\n",
                    node_id, node.name, node.opcode, node.inputs
                ));
                out.push_str(&format!(
                    "    output_type: shape={:?}, dtype={:?}\n",
                    node.output_type.shape, node.output_type.dtype
                ));
                if !node.attrs.is_empty() {
                    out.push_str(&format!("    attrs: {:?}\n", node.attrs));
                }
            }
        }
        out.push_str(&format!("Inputs: {:?}\n", self.graph.inputs));
        out.push_str(&format!("Outputs: {:?}\n", self.graph.outputs));
        Ok(out)
    }

    #[cfg(not(feature = "prepared-plan"))]
    fn debug_graph(&self) -> pyo3::PyResult<String> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "debug_graph requires 'prepared-plan' feature",
        ))
    }
}

// ── AotExecutor helpers (not exposed to Python) ────────────────────────

impl AotExecutor {
    fn encode_profile_result(
        &self,
        py: pyo3::Python<'_>,
        output_data: Vec<Vec<u8>>,
        profile_entries: Vec<crate::backend::ProfileEntry>,
    ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
        use pyo3::types::{PyDict, PyList};

        let outputs = PyDict::new(py);
        for (name, idx) in &self.output_map {
            if let Some(data) = output_data.get(*idx) {
                let output_node_id = self.graph.outputs[*idx];
                let output_node = self.graph.get_node(output_node_id).ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        "AotExecutor: output node not found in graph",
                    )
                })?;
                let shape: Vec<i64> = output_node
                    .output_type
                    .shape
                    .iter()
                    .filter_map(|d| match d {
                        crate::ir::DimExpr::Known(v) => Some(*v as i64),
                        _ => None,
                    })
                    .collect();
                let f32_vals: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                outputs.set_item(
                    name,
                    PyTensor::from_tensor(Tensor::from_vec(f32_vals, shape)),
                )?;
            }
        }

        let profile_list = PyList::empty(py);
        for entry in profile_entries {
            let item = PyDict::new(py);
            item.set_item("instruction_index", entry.instruction_index)?;
            item.set_item("node_id", entry.node_id)?;
            item.set_item("kernel_name", entry.kernel_name)?;
            item.set_item("elapsed_ns", entry.elapsed_ns)?;
            let node_name = entry
                .node_id
                .and_then(|node_id| self.graph.get_node(node_id))
                .map(|node| node.name.clone())
                .unwrap_or_default();
            item.set_item("node_name", node_name)?;
            profile_list.append(item)?;
        }

        let result = PyDict::new(py);
        result.set_item("outputs", outputs)?;
        result.set_item("profile", profile_list)?;
        Ok(result.into())
    }

    /// Decode the raw `Vec<Vec<u8>>` returned by `GraphExecutor::execute`
    /// into a `HashMap<output_name, PyTensor>` keyed by the executor's
    /// `output_map`. Shared by [`AotExecutor::forward`] and
    /// [`AotExecutor::forward_prepared_fallback`] so both code paths
    /// produce identical Python-visible outputs for the same input
    /// bytes.
    fn decode_outputs(
        &self,
        output_data: Vec<Vec<u8>>,
    ) -> pyo3::PyResult<std::collections::HashMap<String, PyTensor>> {
        let mut result = std::collections::HashMap::new();
        for (name, idx) in &self.output_map {
            if let Some(data) = output_data.get(*idx) {
                // Resolve the output node's dtype and shape from the graph.
                let output_node_id = self.graph.outputs[*idx];
                let output_node = self.graph.get_node(output_node_id).ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        "AotExecutor: output node not found in graph",
                    )
                })?;
                let ir_dtype = output_node.output_type.dtype.clone();
                // Extract quantization metadata before ir_to_dtype strips it
                let (q_scales, q_dequant_offsets) = match &ir_dtype {
                    crate::ir::IrDType::I4 {
                        scales,
                        dequant_offsets,
                        ..
                    }
                    | crate::ir::IrDType::I8Scaled {
                        scales,
                        dequant_offsets,
                    } => (scales.clone(), dequant_offsets.clone()),
                    _ => (vec![], vec![]),
                };
                let dtype: crate::storage::DType =
                    crate::tensor::ir_to_dtype(ir_dtype).map_err(PyErr::from)?;
                // Resolve shape from DimExpr (all should be Known after compilation).
                let shape: Vec<i64> = output_node
                    .output_type
                    .shape
                    .iter()
                    .filter_map(|d| match d {
                        crate::ir::DimExpr::Known(v) => Some(*v as i64),
                        _ => None,
                    })
                    .collect();

                let tensor = match dtype {
                    crate::storage::DType::F32 => {
                        let f32_vals: Vec<f32> = data
                            .chunks_exact(4)
                            .map(|chunk| {
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::I32 => {
                        let i32_vals: Vec<i32> = data
                            .chunks_exact(4)
                            .map(|chunk| {
                                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        let f32_vals: Vec<f32> = i32_vals.iter().map(|&v| v as f32).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::I64 => {
                        let i64_vals: Vec<i64> = data
                            .chunks_exact(8)
                            .map(|chunk| {
                                i64::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                    chunk[6], chunk[7],
                                ])
                            })
                            .collect();
                        let f32_vals: Vec<f32> = i64_vals.iter().map(|&v| v as f32).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::Bool => {
                        let vals: Vec<f32> = data
                            .iter()
                            .map(|&b| if b != 0 { 1.0f32 } else { 0.0f32 })
                            .collect();
                        Tensor::from_vec(vals, shape)
                    }
                    crate::storage::DType::F16 => {
                        let f16_vals: Vec<half::f16> = data
                            .chunks_exact(2)
                            .map(|chunk| {
                                half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]))
                            })
                            .collect();
                        let f32_vals: Vec<f32> = f16_vals.iter().map(|v| v.to_f32()).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::BF16 => {
                        let bf16_vals: Vec<half::bf16> = data
                            .chunks_exact(2)
                            .map(|chunk| {
                                half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]))
                            })
                            .collect();
                        let f32_vals: Vec<f32> = bf16_vals.iter().map(|v| v.to_f32()).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    // Packed U4/U8 outputs: unpack nibbles/bytes and dequantize per-channel.
                    // In the normal pipeline, MatMul/Conv2d outputs remain F32 (dequant happens
                    // inside the SIMD kernel). This path is a safety net for ops whose IR output
                    // type is U4/U8.
                    crate::storage::DType::I4 => {
                        let words: Vec<u32> = data
                            .chunks_exact(4)
                            .map(|chunk| {
                                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        let num_elements = words.len() * 8; // 8 nibbles per u32
                        let mut f32_vals = Vec::with_capacity(num_elements);
                        for (word_idx, &word) in words.iter().enumerate() {
                            for nibble in 0..8 {
                                let val = (word >> (nibble * 4)) & 0xF;
                                let ch = word_idx * 8 + nibble;
                                let s = q_scales
                                    .get(ch % q_scales.len().max(1))
                                    .copied()
                                    .unwrap_or(1.0);
                                let zp = q_dequant_offsets
                                    .get(ch % q_dequant_offsets.len().max(1))
                                    .copied()
                                    .unwrap_or(0.0);
                                f32_vals.push(val as f32 * s + zp);
                            }
                        }
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::I8Scaled => {
                        let words: Vec<u32> = data
                            .chunks_exact(4)
                            .map(|chunk| {
                                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        let num_elements = words.len() * 4; // 4 bytes per u32
                        let mut f32_vals = Vec::with_capacity(num_elements);
                        for (word_idx, &word) in words.iter().enumerate() {
                            for byte in 0..4 {
                                let val = (word >> (byte * 8)) & 0xFF;
                                let ch = word_idx * 4 + byte;
                                let s = q_scales
                                    .get(ch % q_scales.len().max(1))
                                    .copied()
                                    .unwrap_or(1.0);
                                let zp = q_dequant_offsets
                                    .get(ch % q_dequant_offsets.len().max(1))
                                    .copied()
                                    .unwrap_or(0.0);
                                f32_vals.push(val as f32 * s + zp);
                            }
                        }
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::F64 => {
                        let f64_vals: Vec<f64> = data
                            .chunks_exact(8)
                            .map(|chunk| {
                                f64::from_le_bytes([
                                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                    chunk[6], chunk[7],
                                ])
                            })
                            .collect();
                        let f32_vals: Vec<f32> = f64_vals.iter().map(|&v| v as f32).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::F8 => {
                        let words: Vec<u32> = data
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let mut f32_vals = Vec::with_capacity(words.len() * 4);
                        for &word in &words {
                            for byte in 0..4 {
                                let val = ((word >> (byte * 8)) & 0xFF) as u8;
                                f32_vals.push(crate::dtypes::f8x4::e4m3_to_f32(val));
                            }
                        }
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::F8R => {
                        let words: Vec<u32> = data
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let mut f32_vals = Vec::with_capacity(words.len() * 4);
                        for &word in &words {
                            for byte in 0..4 {
                                let val = ((word >> (byte * 8)) & 0xFF) as u8;
                                f32_vals.push(crate::dtypes::f8x4r::e5m2_to_f32(val));
                            }
                        }
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::U4Scaled => {
                        let words: Vec<u32> = data
                            .chunks_exact(4)
                            .map(|chunk| {
                                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        let num_elements = words.len() * 8; // 8 nibbles per u32
                        let mut f32_vals = Vec::with_capacity(num_elements);
                        for (word_idx, &word) in words.iter().enumerate() {
                            for nibble in 0..8 {
                                let val = (word >> (nibble * 4)) & 0xF;
                                let ch = word_idx * 8 + nibble;
                                let s = q_scales
                                    .get(ch % q_scales.len().max(1))
                                    .copied()
                                    .unwrap_or(1.0);
                                let zp = q_dequant_offsets
                                    .get(ch % q_dequant_offsets.len().max(1))
                                    .copied()
                                    .unwrap_or(0.0);
                                f32_vals.push(val as f32 * s + zp);
                            }
                        }
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::U8Scaled => {
                        let words: Vec<u32> = data
                            .chunks_exact(4)
                            .map(|chunk| {
                                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();
                        let num_elements = words.len() * 4; // 4 bytes per u32
                        let mut f32_vals = Vec::with_capacity(num_elements);
                        for (word_idx, &word) in words.iter().enumerate() {
                            for byte in 0..4 {
                                let val = (word >> (byte * 8)) & 0xFF;
                                let ch = word_idx * 4 + byte;
                                let s = q_scales
                                    .get(ch % q_scales.len().max(1))
                                    .copied()
                                    .unwrap_or(1.0);
                                let zp = q_dequant_offsets
                                    .get(ch % q_dequant_offsets.len().max(1))
                                    .copied()
                                    .unwrap_or(0.0);
                                f32_vals.push(val as f32 * s + zp);
                            }
                        }
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::F4 => {
                        let words: Vec<u32> = data
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let mut f32_vals = Vec::with_capacity(words.len() * 8);
                        for &word in &words {
                            for nibble in 0..8 {
                                let val = ((word >> (nibble * 4)) & 0xF) as u8;
                                f32_vals.push(crate::dtypes::f4x8::fp4_to_f32(val));
                            }
                        }
                        Tensor::from_vec(f32_vals, shape)
                    }
                };
                result.insert(name.clone(), PyTensor::from_tensor(tensor));
            }
        }
        Ok(result)
    }
}
