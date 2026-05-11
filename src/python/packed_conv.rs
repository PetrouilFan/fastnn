use crate::packed_conv::PackedConv2d;

macro_rules! impl_pyconv2d_pymethods {
    ($conv:ident, $word:ty) => {
        #[pymethods]
        impl $conv {
            #[new]
            #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true))]
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
                $conv {
                    inner: PackedConv2d::new(
                        in_channels as usize,
                        out_channels as usize,
                        kernel_size as usize,
                        stride as usize,
                        padding as usize,
                        dilation as usize,
                        groups as usize,
                        bias,
                    ),
                }
            }

            fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
                let output = self.inner.forward_cpu(&input.inner);
                Ok(PyTensor::from_tensor(output))
            }

            fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
                Ok(PyTensor::from_tensor(self.inner.forward_cpu(&input.inner)))
            }

            #[getter]
            fn in_channels(&self) -> i64 {
                self.inner.in_channels as i64
            }

            #[getter]
            fn out_channels(&self) -> i64 {
                self.inner.out_channels as i64
            }

            #[getter]
            fn kernel_size(&self) -> i64 {
                self.inner.kernel_size as i64
            }

            #[getter]
            fn stride(&self) -> i64 {
                self.inner.stride as i64
            }

            #[getter]
            fn padding(&self) -> i64 {
                self.inner.padding as i64
            }

            #[getter]
            fn dilation(&self) -> i64 {
                self.inner.dilation as i64
            }

            #[getter]
            fn groups(&self) -> i64 {
                self.inner.groups as i64
            }

            fn to_tensor(&self) -> PyTensor {
                let data = self.inner.weight.to_f32_vec();
                let shape: Vec<i64> = self.inner.weight.shape().iter().map(|&s| s as i64).collect();
                let t = Tensor::from_vec(data, shape);
                PyTensor::from_tensor(t)
            }

            #[getter]
            fn num_params(&self) -> i64 {
                self.inner.num_params() as i64
            }

            #[getter]
            fn memory_savings(&self) -> f64 {
                self.inner.memory_savings() as f64
            }

            #[getter]
            fn bias(&self) -> Option<Vec<f64>> {
                self.inner
                    .bias
                    .as_ref()
                    .map(|v| v.iter().map(|&x| x as f64).collect())
            }

            fn train(&self) {
                self.inner.train_mode();
            }

            fn eval(&self) {
                self.inner.eval_mode();
            }

            #[getter]
            fn is_training(&self) -> bool {
                self.inner.is_training()
            }
        }
    };
}

// ---- PyPackedConv2d4 (4-bit, U4x8) ----

#[pyclass]
pub struct PyPackedConv2d4 {
    inner: PackedConv2d<U4x8>,
}

impl_pyconv2d_pymethods!(PyPackedConv2d4, U4x8);

// ---- PyPackedConv2d8 (8-bit, U8x4) ----

#[pyclass]
pub struct PyPackedConv2d8 {
    inner: PackedConv2d<U8x4>,
}

impl_pyconv2d_pymethods!(PyPackedConv2d8, U8x4);

// ---- PyPackedConv2d16 (16-bit, F16x2) ----

#[pyclass]
pub struct PyPackedConv2d16 {
    inner: PackedConv2d<F16x2>,
}

impl_pyconv2d_pymethods!(PyPackedConv2d16, F16x2);

// ---- PyPackedConv2d32 (32-bit, F32x1) ----

#[pyclass]
pub struct PyPackedConv2d32 {
    inner: PackedConv2d<F32x1>,
}

impl_pyconv2d_pymethods!(PyPackedConv2d32, F32x1);

// ---- Fused PackedConvRelu wrappers ----

macro_rules! impl_pyconv_relu_pymethods {
    ($conv_relu:ident) => {
        #[pymethods]
        impl $conv_relu {
            #[new]
            #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true))]
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
                $conv_relu {
                    inner: PackedConv2d::new(
                        in_channels as usize,
                        out_channels as usize,
                        kernel_size as usize,
                        stride as usize,
                        padding as usize,
                        dilation as usize,
                        groups as usize,
                        bias,
                    ),
                }
            }

            fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
                let output = self.inner.forward_relu(&input.inner);
                Ok(PyTensor::from_tensor(output))
            }

            fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
                Ok(PyTensor::from_tensor(self.inner.forward_relu(&input.inner)))
            }

            #[getter]
            fn in_channels(&self) -> i64 { self.inner.in_channels as i64 }

            #[getter]
            fn out_channels(&self) -> i64 { self.inner.out_channels as i64 }

            #[getter]
            fn kernel_size(&self) -> i64 { self.inner.kernel_size as i64 }

            #[getter]
            fn stride(&self) -> i64 { self.inner.stride as i64 }

            #[getter]
            fn padding(&self) -> i64 { self.inner.padding as i64 }

            #[getter]
            fn dilation(&self) -> i64 { self.inner.dilation as i64 }

            #[getter]
            fn groups(&self) -> i64 { self.inner.groups as i64 }

            fn to_tensor(&self) -> PyTensor {
                let data = self.inner.weight.to_f32_vec();
                let shape: Vec<i64> = self.inner.weight.shape().iter().map(|&s| s as i64).collect();
                let t = Tensor::from_vec(data, shape);
                PyTensor::from_tensor(t)
            }

            #[getter]
            fn num_params(&self) -> i64 { self.inner.num_params() as i64 }

            #[getter]
            fn memory_savings(&self) -> f64 { self.inner.memory_savings() as f64 }

            fn train(&self) { self.inner.train_mode(); }

            fn eval(&self) { self.inner.eval_mode(); }

            #[getter]
            fn is_training(&self) -> bool { self.inner.is_training() }
        }
    };
}

#[pyclass]
pub struct PyPackedConvRelu4 {
    inner: PackedConv2d<U4x8>,
}

impl_pyconv_relu_pymethods!(PyPackedConvRelu4);

#[pyclass]
pub struct PyPackedConvRelu8 {
    inner: PackedConv2d<U8x4>,
}

impl_pyconv_relu_pymethods!(PyPackedConvRelu8);

#[pyclass]
pub struct PyPackedConvRelu16 {
    inner: PackedConv2d<F16x2>,
}

impl_pyconv_relu_pymethods!(PyPackedConvRelu16);

#[pyclass]
pub struct PyPackedConvRelu32 {
    inner: PackedConv2d<F32x1>,
}

impl_pyconv_relu_pymethods!(PyPackedConvRelu32);
