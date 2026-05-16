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
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, bias = true))]
    fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
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
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, bias = true))]
    fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        dilation: i64,
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
            inner: core_nn::upsample::Upsample::new(scale_factor, mode),
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
        let sp = x.inner.add_scalar(1.0).exp().ln();
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
    try_extract_inner!(b, AdaptiveAvgPool2d, core_nn::activations::AdaptiveAvgPool2d);
    try_extract_inner!(b, FusedConvBn, core_nn::fused::FusedConvBn<core_nn::fused::NoAct>);
    try_extract_inner!(b, FusedConvBnRelu, core_nn::fused::FusedConvBn<core_nn::fused::ReluAct>);
    try_extract_inner!(b, FusedConvBnGelu, core_nn::fused::FusedConvBn<core_nn::fused::GeluAct>);
    try_extract_inner!(b, PyTransformerEncoder, core_nn::transformer::TransformerEncoder);

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
        return Some(
            Arc::new(core_nn::activations::Softplus::new(layer.beta, layer.threshold))
                as Arc<dyn Module>,
        );
    }
    if let Ok(layer) = b.extract::<pyo3::PyRef<'_, Elu>>() {
        return Some(
            Arc::new(core_nn::activations::Elu::new(layer.alpha)) as Arc<dyn Module>,
        );
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
        Sequential { layers, native_layers }
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

    fn __getitem__(&self, idx: usize, py: Python<'_>) -> Py<PyAny> {
        self.modules[idx].clone_ref(py)
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
    memory_plan: crate::compiler::passes::memory_planning::MemoryPlan,
    graph: crate::ir::node::ComputeGraph,
    executor: crate::backend::executor::GraphExecutor<crate::backend::cpu::CpuBackend>,
    input_names: Vec<String>,
    output_map: Vec<(String, usize)>,
}

#[pymethods]
impl AotExecutor {
    #[new]
    #[pyo3(signature = (nodes, params, input_names, output_names, input_shapes=None, quantize=None))]
    fn new(
        nodes: Vec<std::collections::HashMap<String, String>>,
        params: std::collections::HashMap<String, PyTensor>,
        input_names: Vec<String>,
        output_names: Vec<String>,
        input_shapes: Option<std::collections::HashMap<String, Vec<i64>>>,
        quantize: Option<u8>,
    ) -> pyo3::PyResult<Self> {
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
                        *k != "name" && *k != "op_type" && *k != "inputs" && *k != "outputs"
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

        let rust_params: std::collections::HashMap<String, Tensor> = params
            .into_iter()
            .map(|(k, v)| (k, v.inner))
            .collect();

        // Build input shapes map if provided.
        let mut rust_input_shapes: std::collections::HashMap<String, Vec<crate::ir::node::DimExpr>> =
            std::collections::HashMap::new();
        if let Some(shapes) = input_shapes {
            for (name, dims) in shapes {
                let ir_dims: Vec<crate::ir::node::DimExpr> = dims
                    .into_iter()
                    .map(|d| {
                        if d < 0 {
                            // Negative dim = symbolic (batch, height, width, etc.)
                            crate::ir::node::DimExpr::Symbol(format!("d{}", -d))
                        } else {
                            crate::ir::node::DimExpr::Known(d as u64)
                        }
                    })
                    .collect();
                rust_input_shapes.insert(name, ir_dims);
            }
        }

        let converter = crate::onnx::converter::OnnxConverter::new(
            &onnx_nodes, &rust_params, &input_names, &output_names,
        )
        .with_input_shapes(&rust_input_shapes);
        let graph = converter
            .to_compute_graph()
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let executor = crate::backend::executor::GraphExecutor::new(crate::backend::cpu::CpuBackend);
        let (plan, memory_plan, compiled_graph) = executor
            .compile_with_plan_and_quantize(&graph, quantize)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let output_map: Vec<(String, usize)> = output_names
            .iter().enumerate().map(|(i, name)| (name.clone(), i)).collect();

        Ok(AotExecutor { plan, memory_plan, graph: compiled_graph, executor, input_names, output_map })
    }

    fn forward(
        &mut self,
        inputs: std::collections::HashMap<String, PyTensor>,
    ) -> pyo3::PyResult<std::collections::HashMap<String, PyTensor>> {
        let input_refs: Vec<&[u8]> = self.input_names.iter().map(|name| {
            inputs.get(name.as_str())
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                    format!("required input '{}' not found", name),
                ))
                .map(|t| t.inner.as_bytes())
        }).collect::<pyo3::PyResult<Vec<&[u8]>>>()?;

        let output_data = self.executor
            .execute(&self.graph, &mut self.plan, &self.memory_plan, &input_refs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut result = std::collections::HashMap::new();
        for (name, idx) in &self.output_map {
            if let Some(data) = output_data.get(*idx) {
                // Resolve the output node's dtype and shape from the graph.
                let output_node_id = self.graph.outputs[*idx];
                let output_node = self.graph.get_node(output_node_id)
                    .expect("AotExecutor: output node not found in graph");
                let ir_dtype = output_node.output_type.dtype.clone();
                // Extract quantization metadata before ir_to_dtype strips it
                let (q_scales, q_zero_points) = match &ir_dtype {
                    crate::ir::node::IrDType::U4 { scales, zero_points }
                    | crate::ir::node::IrDType::U8 { scales, zero_points } => {
                        (scales.clone(), zero_points.clone())
                    }
                    _ => (vec![], vec![]),
                };
                let dtype: crate::storage::DType = crate::tensor::ir_to_dtype(ir_dtype);
                // Resolve shape from DimExpr (all should be Known after compilation).
                let shape: Vec<i64> = output_node.output_type.shape.iter()
                    .filter_map(|d| match d {
                        crate::ir::node::DimExpr::Known(v) => Some(*v as i64),
                        _ => None,
                    })
                    .collect();

                let tensor = match dtype {
                    crate::storage::DType::F32 => {
                        let f32_vals: Vec<f32> = data.chunks_exact(4)
                            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                            .collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::I32 => {
                        let i32_vals: Vec<i32> = data.chunks_exact(4)
                            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                            .collect();
                        let f32_vals: Vec<f32> = i32_vals.iter().map(|&v| v as f32).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::I64 => {
                        let i64_vals: Vec<i64> = data.chunks_exact(8)
                            .map(|chunk| i64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3],
                                chunk[4], chunk[5], chunk[6], chunk[7],
                            ]))
                            .collect();
                        let f32_vals: Vec<f32> = i64_vals.iter().map(|&v| v as f32).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::Bool => {
                        let vals: Vec<f32> = data.iter().map(|&b| if b != 0 { 1.0f32 } else { 0.0f32 }).collect();
                        Tensor::from_vec(vals, shape)
                    }
                    crate::storage::DType::F16 => {
                        let f16_vals: Vec<half::f16> = data.chunks_exact(2)
                            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                            .collect();
                        let f32_vals: Vec<f32> = f16_vals.iter().map(|v| v.to_f32()).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::BF16 => {
                        let bf16_vals: Vec<half::bf16> = data.chunks_exact(2)
                            .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                            .collect();
                        let f32_vals: Vec<f32> = bf16_vals.iter().map(|v| v.to_f32()).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                    // Packed U4/U8 outputs: unpack nibbles/bytes and dequantize per-channel.
                    // In the normal pipeline, MatMul/Conv2d outputs remain F32 (dequant happens
                    // inside the SIMD kernel). This path is a safety net for ops whose IR output
                    // type is U4/U8.
                    crate::storage::DType::U4 => {
                        let words: Vec<u32> = data.chunks_exact(4)
                            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                            .collect();
                        let num_elements = words.len() * 8; // 8 nibbles per u32
                        let mut f32_vals = Vec::with_capacity(num_elements);
                        for (word_idx, &word) in words.iter().enumerate() {
                            for nibble in 0..8 {
                                let val = (word >> (nibble * 4)) & 0xF;
                                let ch = word_idx * 8 + nibble;
                                let s = q_scales.get(ch % q_scales.len().max(1)).copied().unwrap_or(1.0);
                                let zp = q_zero_points.get(ch % q_zero_points.len().max(1)).copied().unwrap_or(0.0);
                                f32_vals.push(val as f32 * s + zp);
                            }
                        }
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::U8 => {
                        let words: Vec<u32> = data.chunks_exact(4)
                            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                            .collect();
                        let num_elements = words.len() * 4; // 4 bytes per u32
                        let mut f32_vals = Vec::with_capacity(num_elements);
                        for (word_idx, &word) in words.iter().enumerate() {
                            for byte in 0..4 {
                                let val = (word >> (byte * 8)) & 0xFF;
                                let ch = word_idx * 4 + byte;
                                let s = q_scales.get(ch % q_scales.len().max(1)).copied().unwrap_or(1.0);
                                let zp = q_zero_points.get(ch % q_zero_points.len().max(1)).copied().unwrap_or(0.0);
                                f32_vals.push(val as f32 * s + zp);
                            }
                        }
                        Tensor::from_vec(f32_vals, shape)
                    }
                    crate::storage::DType::F64 => {
                        let f64_vals: Vec<f64> = data.chunks_exact(8)
                            .map(|chunk| f64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3],
                                chunk[4], chunk[5], chunk[6], chunk[7],
                            ]))
                            .collect();
                        let f32_vals: Vec<f32> = f64_vals.iter().map(|&v| v as f32).collect();
                        Tensor::from_vec(f32_vals, shape)
                    }
                };
                result.insert(name.clone(), PyTensor::from_tensor(tensor));
            }
        }
        Ok(result)
    }
}

