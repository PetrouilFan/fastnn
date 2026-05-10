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

            fn named_parameters(&self) -> Vec<(String, PyTensor)> {
                self.inner
                    .named_parameters()
                    .into_iter()
                    .map(|(n, t)| (n, PyTensor::from_tensor(t)))
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
            inner: core_nn::activations::AdaptiveAvgPool2d::new((output_h, output_w)),
        }
    }
});

#[pyclass(name = "Sequential_")]
struct Sequential {
    layers: Vec<Py<PyAny>>,
}

#[pymethods]
impl Sequential {
    #[new]
    fn new(layers: Vec<Py<PyAny>>) -> Self {
        Sequential { layers }
    }

    fn __call__(&self, py: Python<'_>, x: PyTensor) -> PyResult<PyTensor> {
        let mut result = x;
        for layer in &self.layers {
            // Call the layer using __call__ (which internally calls forward)
            let new_result = layer.call1(py, (result,))?;
            result = new_result.extract::<PyTensor>(py)?;
        }
        Ok(result)
    }

    fn parameters(&self, py: Python<'_>) -> PyResult<Vec<PyTensor>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            let params_method: Py<PyAny> = layer.getattr(py, "parameters")?;
            let layer_params: Vec<PyTensor> = params_method.call0(py)?.extract(py)?;
            params.extend(layer_params);
        }
        Ok(params)
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

// ---- DAGExecutor (ONNX graph execution) ----

#[pyclass]
pub struct DAGExecutor {
    inner: core_nn::dag::DAGExecutor,
}

#[pymethods]
impl DAGExecutor {
    #[new]
    #[pyo3(signature = (nodes, params, input_names, output_names))]
    fn new(
        nodes: Vec<HashMap<String, String>>,
        params: HashMap<String, PyTensor>,
        input_names: Vec<String>,
        output_names: Vec<String>,
    ) -> Self {
        let dag_nodes: Vec<core_nn::dag::DAGNode> = nodes
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
                let attrs: HashMap<String, String> = m
                    .into_iter()
                    .filter(|(k, _)| *k != "name" && *k != "op_type" && *k != "inputs" && *k != "outputs")
                    .collect();
                core_nn::dag::DAGNode {
                    name,
                    op_type,
                    inputs,
                    outputs,
                    attrs,
                }
            })
            .collect();
        let rust_params: HashMap<String, Tensor> = params
            .into_iter()
            .map(|(k, v)| (k, v.inner))
            .collect();
        DAGExecutor {
            inner: core_nn::dag::DAGExecutor::new(dag_nodes, rust_params, input_names, output_names),
        }
    }

    fn forward(&self, inputs: HashMap<String, PyTensor>) -> HashMap<String, PyTensor> {
        let rust_inputs: HashMap<String, Tensor> = inputs
            .into_iter()
            .map(|(k, v)| (k, v.inner))
            .collect();
        let outputs = self.inner.forward(&rust_inputs);
        outputs
            .into_iter()
            .map(|(k, v)| (k, PyTensor::from_tensor(v)))
            .collect()
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, PyTensor)> {
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (n, PyTensor::from_tensor(t)))
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
}

// ---- PackedMultiHeadAttention (4-bit, U4x8) ----

#[pyclass]
pub struct PyPackedMultiHeadAttention4 {
    inner: crate::nn::attention::PackedMultiHeadAttention<crate::dtypes::U4x8>,
}

impl_nn_module!(PyPackedMultiHeadAttention4 {
    #[new]
    #[pyo3(signature = (d_model, num_heads, dropout_p = 0.0, causal = false))]
    fn new(d_model: i64, num_heads: i64, dropout_p: f64, causal: bool) -> Self {
        PyPackedMultiHeadAttention4 {
            inner: crate::nn::attention::PackedMultiHeadAttention::new(
                d_model, num_heads, dropout_p as f32, causal
            )
        }
    }

    fn set_kv_cache(&mut self, k: &PyTensor, v: &PyTensor) {
        let k_data = k.numpy();
        let v_data = v.numpy();
        self.inner.set_kv_cache(k_data, v_data);
    }

    fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }
});

// ---- PackedMultiHeadAttention (8-bit, U8x4) ----

#[pyclass]
pub struct PyPackedMultiHeadAttention8 {
    inner: crate::nn::attention::PackedMultiHeadAttention<crate::dtypes::U8x4>,
}

impl_nn_module!(PyPackedMultiHeadAttention8 {
    #[new]
    #[pyo3(signature = (d_model, num_heads, dropout_p = 0.0, causal = false))]
    fn new(d_model: i64, num_heads: i64, dropout_p: f64, causal: bool) -> Self {
        PyPackedMultiHeadAttention8 {
            inner: crate::nn::attention::PackedMultiHeadAttention::new(
                d_model, num_heads, dropout_p as f32, causal
            )
        }
    }

    fn set_kv_cache(&mut self, k: &PyTensor, v: &PyTensor) {
        let k_data = k.numpy();
        let v_data = v.numpy();
        self.inner.set_kv_cache(k_data, v_data);
    }

    fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }
});

// ---- PackedTransformerEncoder (4-bit, U4x8) ----

#[pyclass]
pub struct PyPackedTransformerEncoder4 {
    inner: crate::nn::transformer::PackedTransformerEncoder<crate::dtypes::U4x8>,
}

impl_nn_module!(PyPackedTransformerEncoder4 {
    #[new]
    #[pyo3(signature = (vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim, num_classes, dropout_p = 0.0))]
    fn new(vocab_size: i64, max_seq_len: i64, d_model: i64, num_heads: i64,
           num_layers: i64, ff_dim: i64, num_classes: i64, dropout_p: f64) -> Self {
        PyPackedTransformerEncoder4 {
            inner: crate::nn::transformer::PackedTransformerEncoder::new(
                vocab_size,
                max_seq_len,
                d_model,
                num_heads,
                num_layers,
                ff_dim,
                num_classes,
                dropout_p as f32,
            )
        }
    }
});

// ---- PackedTransformerEncoder (8-bit, U8x4) ----

#[pyclass]
pub struct PyPackedTransformerEncoder8 {
    inner: crate::nn::transformer::PackedTransformerEncoder<crate::dtypes::U8x4>,
}

impl_nn_module!(PyPackedTransformerEncoder8 {
    #[new]
    #[pyo3(signature = (vocab_size, max_seq_len, d_model, num_heads, num_layers, ff_dim, num_classes, dropout_p = 0.0))]
    fn new(vocab_size: i64, max_seq_len: i64, d_model: i64, num_heads: i64,
           num_layers: i64, ff_dim: i64, num_classes: i64, dropout_p: f64) -> Self {
        PyPackedTransformerEncoder8 {
            inner: crate::nn::transformer::PackedTransformerEncoder::new(
                vocab_size,
                max_seq_len,
                d_model,
                num_heads,
                num_layers,
                ff_dim,
                num_classes,
                dropout_p as f32,
            )
        }
    }
});

