#[pyclass]
struct Linear {
    inner: core_nn::linear::Linear,
}

#[pymethods]
impl Linear {
    #[new]
    #[pyo3(signature = (in_features, out_features, bias = true))]
    fn new(in_features: i64, out_features: i64, bias: bool) -> Self {
        Linear {
            inner: core_nn::linear::Linear::new(in_features, out_features, bias),
        }
    }

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

    #[pyo3(signature = (device_id))]
    #[allow(clippy::wrong_self_convention)]
    fn to_gpu(&mut self, device_id: usize) {
        self.inner.weight = self.inner.weight.to_gpu(device_id);
        self.inner.bias = self.inner.bias.as_ref().map(|b| b.to_gpu(device_id));
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
        inner.weight = weight.inner;
        if let Some(b) = bias {
            inner.bias = Some(b.inner);
        }
        Linear { inner }
    }
}

#[pyclass]
struct Conv2d {
    inner: core_nn::conv::Conv2d,
}

#[pymethods]
impl Conv2d {
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

    #[pyo3(signature = (device_id))]
    #[allow(clippy::wrong_self_convention)]
    fn to_gpu(&mut self, device_id: usize) {
        self.inner.weight = self.inner.weight.to_gpu(device_id);
        self.inner.bias = self.inner.bias.as_ref().map(|b| b.to_gpu(device_id));
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
}

#[pyclass]
struct MaxPool2d {
    inner: core_nn::pooling::MaxPool2d,
}

#[pymethods]
impl MaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=2, padding=1, dilation=1))]
    fn new(kernel_size: i64, stride: i64, padding: i64, dilation: i64) -> Self {
        MaxPool2d {
            inner: core_nn::pooling::MaxPool2d::new(kernel_size, stride, padding, dilation),
        }
    }

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
}

#[pyclass]
struct ConvTranspose2d {
    inner: core_nn::conv::ConvTranspose2d,
}

#[pymethods]
impl ConvTranspose2d {
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

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
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

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct Conv1d {
    inner: core_nn::conv::Conv1d,
}

#[pymethods]
impl Conv1d {
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

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
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

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct Conv3d {
    inner: core_nn::conv::Conv3d,
}

#[pymethods]
impl Conv3d {
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

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
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

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct ResidualBlock {
    inner: residual::ResidualBlock,
}

#[pymethods]
impl ResidualBlock {
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

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
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

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct LayerNorm {
    inner: core_nn::norm::LayerNorm,
}

#[pymethods]
impl LayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-5))]
    fn new(normalized_shape: i64, eps: f64) -> Self {
        LayerNorm {
            inner: core_nn::norm::LayerNorm::new(normalized_shape, eps),
        }
    }

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

    fn zero_grad(&self) {
        self.inner.zero_grad();
    }

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = Some(weight.inner);
    }

    fn set_bias(&mut self, bias: Option<PyTensor>) {
        self.inner.bias = bias.map(|t| t.inner);
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
        inner.weight = Some(weight.inner);
        inner.bias = Some(bias.inner);
        LayerNorm { inner }
    }
}

#[pyclass]
struct BatchNorm1d {
    inner: core_nn::norm::BatchNorm1d,
}

#[pymethods]
impl BatchNorm1d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1))]
    fn new(num_features: i64, eps: f64, momentum: f64) -> Self {
        BatchNorm1d {
            inner: core_nn::norm::BatchNorm1d::new(num_features, eps, momentum),
        }
    }

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

    fn set_weight(&mut self, weight: PyTensor) {
        self.inner.weight = Some(weight.inner);
    }

    fn set_bias(&mut self, bias: Option<PyTensor>) {
        self.inner.bias = bias.map(|t| t.inner);
    }

    fn set_running_mean(&mut self, running_mean: PyTensor) {
        self.inner.running_mean = Arc::new(RwLock::new(running_mean.inner));
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
}

#[pyclass]
struct Dropout {
    inner: core_nn::dropout::Dropout,
}

#[pymethods]
impl Dropout {
    #[new]
    fn new(p: f64) -> Self {
        Dropout {
            inner: core_nn::dropout::Dropout::new(p),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn train(&self) {
        self.inner.train_mode();
    }

    fn eval(&self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct Dropout2d {
    inner: core_nn::dropout::Dropout2d,
}

#[pymethods]
impl Dropout2d {
    #[new]
    fn new(p: f64) -> Self {
        Dropout2d {
            inner: core_nn::dropout::Dropout2d::new(p),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn train(&self) {
        self.inner.train_mode();
    }

    fn eval(&self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct Upsample {
    inner: core_nn::upsample::Upsample,
}

#[pymethods]
impl Upsample {
    #[new]
    fn new(scale_factor: f64, mode: String) -> Self {
        Upsample {
            inner: core_nn::upsample::Upsample::new(scale_factor, mode),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }
}

#[pyclass]
struct Embedding {
    inner: core_nn::embedding::Embedding,
}

#[pymethods]
impl Embedding {
    #[new]
    fn new(num_embeddings: i64, embedding_dim: i64) -> Self {
        Embedding {
            inner: core_nn::embedding::Embedding::new(num_embeddings, embedding_dim),
        }
    }

    fn __call__(&self, indices: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&indices.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
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
}

#[pyclass]
struct RMSNorm {
    inner: core_nn::norm::RMSNorm,
}

#[pymethods]
impl RMSNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps = 1e-5))]
    fn new(normalized_shape: i64, eps: f32) -> Self {
        RMSNorm {
            inner: core_nn::norm::RMSNorm::new(normalized_shape, eps),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
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

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct GroupNorm {
    inner: core_nn::norm::GroupNorm,
}

#[pymethods]
impl GroupNorm {
    #[new]
    #[pyo3(signature = (num_groups, num_channels, eps = 1e-5))]
    fn new(num_groups: i64, num_channels: i64, eps: f32) -> Self {
        GroupNorm {
            inner: core_nn::norm::GroupNorm::new(num_groups, num_channels, eps),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
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

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

#[pyclass]
struct BatchNorm2d {
    inner: core_nn::norm::BatchNorm2d,
}

#[pymethods]
impl BatchNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps = 1e-5, momentum = 0.1))]
    fn new(num_features: i64, eps: f32, momentum: f32) -> Self {
        BatchNorm2d {
            inner: core_nn::norm::BatchNorm2d::new(num_features, eps, momentum),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
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

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
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
}

#[pyclass]
struct ReLU;

#[pymethods]
impl ReLU {
    #[new]
    fn new() -> Self {
        ReLU
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.relu())
    }
}

#[pyclass]
struct Gelu;

#[pymethods]
impl Gelu {
    #[new]
    fn new() -> Self {
        Gelu
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.gelu())
    }
}

#[pyclass]
struct Sigmoid;

#[pymethods]
impl Sigmoid {
    #[new]
    fn new() -> Self {
        Sigmoid
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.sigmoid())
    }
}

#[pyclass]
struct Tanh;

#[pymethods]
impl Tanh {
    #[new]
    fn new() -> Self {
        Tanh
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.tanh())
    }
}

#[pyclass]
struct SiLU;

#[pymethods]
impl SiLU {
    #[new]
    fn new() -> Self {
        SiLU
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.silu())
    }
}

#[pyclass]
struct LeakyReLU {
    negative_slope: f64,
}

#[pymethods]
impl LeakyReLU {
    #[new]
    #[pyo3(signature = (negative_slope = 0.01))]
    fn new(negative_slope: f64) -> Self {
        LeakyReLU { negative_slope }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.leaky_relu(self.negative_slope as f32))
    }
}

#[pyclass]
struct Softplus {
    beta: f64,
    threshold: f64,
}

#[pymethods]
impl Softplus {
    #[new]
    #[pyo3(signature = (beta = 1.0, threshold = 20.0))]
    fn new(beta: f64, threshold: f64) -> Self {
        Softplus { beta, threshold }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.softplus(self.beta as f32, self.threshold as f32))
    }
}

#[pyclass]
struct Hardswish;

#[pymethods]
impl Hardswish {
    #[new]
    fn new() -> Self {
        Hardswish
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.hardswish())
    }
}

#[pyclass]
struct Elu {
    alpha: f64,
}

#[pymethods]
impl Elu {
    #[new]
    #[pyo3(signature = (alpha = 1.0))]
    fn new(alpha: f64) -> Self {
        Elu { alpha }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(x.inner.elu(self.alpha as f32))
    }
}

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
}

#[pyclass]
struct AdaptiveAvgPool2d {
    inner: core_nn::activations::AdaptiveAvgPool2d,
}

#[pymethods]
impl AdaptiveAvgPool2d {
    #[new]
    fn new(output_h: i64, output_w: i64) -> Self {
        AdaptiveAvgPool2d {
            inner: core_nn::activations::AdaptiveAvgPool2d::new((output_h, output_w)),
        }
    }

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }
}

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

#[pymethods]
impl PyTransformerEncoder {
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

    fn __call__(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn forward(&self, x: &PyTensor) -> PyTensor {
        PyTensor::from_tensor(self.inner.forward(&x.inner))
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(PyTensor::from_tensor)
            .collect()
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn train(&mut self) {
        self.inner.train_mode();
    }

    fn eval(&mut self) {
        self.inner.eval_mode();
    }
}

