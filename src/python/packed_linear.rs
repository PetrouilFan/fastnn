use std::sync::Mutex;
use crate::packed_layer::PackedLinear;
use crate::packed_train::MasterWeightOptimizer;

macro_rules! impl_pylinear_pymethods {
    ($linear:ident) => {
        #[pymethods]
        impl $linear {
            #[new]
            #[pyo3(signature = (in_features, out_features, bias = true))]
            fn new(in_features: i64, out_features: i64, bias: bool) -> Self {
                $linear {
                    inner: PackedLinear::new(in_features as usize, out_features as usize, bias),
                    optimizer: Mutex::new(None),
                }
            }

            fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
                let input_vec = input.inner.to_numpy();
                let output = self.inner.forward(&input_vec);
                let n = output.len() as i64;
                let t = Tensor::from_vec(output, vec![n]);
                Ok(PyTensor::from_tensor(t))
            }

            fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
                self.__call__(input)
            }

            fn backward(&self, grad_output: &PyTensor) -> PyResult<PyTensor> {
                let go = grad_output.inner.to_numpy();
                let grad_input = self.inner.backward(&go);
                let n = grad_input.len() as i64;
                let t = Tensor::from_vec(grad_input, vec![n]);
                Ok(PyTensor::from_tensor(t))
            }

            #[pyo3(signature = (input, target, loss_fn = "cross_entropy", lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.0, _scale_update_freq = 100))]
            fn train_step(
                &mut self,
                input: &PyTensor,
                target: &PyTensor,
                loss_fn: &str,
                lr: f64,
                betas: (f64, f64),
                eps: f64,
                weight_decay: f64,
                _scale_update_freq: i64,
            ) -> PyResult<PyTensor> {
                let input_vec = input.inner.to_numpy();
                let output = self.inner.forward(&input_vec);
                let num_classes = output.len();

                let target_vec = target.inner.to_numpy();

                let grad_output: Vec<f32> = match loss_fn {
                    "cross_entropy" => {
                        let inv_nc = 1.0 / num_classes as f32;
                        output.iter().zip(target_vec.iter())
                            .map(|(o, t)| (o - t) * inv_nc)
                            .collect()
                    }
                    "mse" => {
                        let n = num_classes as f32;
                        output.iter().zip(target_vec.iter())
                            .map(|(o, t)| 2.0 * (o - t) / n)
                            .collect()
                    }
                    _ => return Err(PyRuntimeError::new_err(
                        format!("Unknown loss function: {}", loss_fn)
                    )),
                };

                let loss_value: f32 = match loss_fn {
                    "cross_entropy" => {
                        let inv_nc = 1.0 / num_classes as f32;
                        output.iter().zip(target_vec.iter())
                            .map(|(o, t)| -(t * o.ln().max(-100.0)) * inv_nc)
                            .sum::<f32>()
                    }
                    "mse" => {
                        let n = num_classes as f32;
                        output.iter().zip(target_vec.iter())
                            .map(|(o, t)| (o - t).powi(2) / n)
                            .sum::<f32>()
                    }
                    _ => unreachable!(),
                };

                self.inner.backward(&grad_output);

                let master = self.inner.master_weight.as_ref()
                    .ok_or_else(|| PyRuntimeError::new_err(
                        "No master weights — layer was created without training support"
                    ))?;

                let mut opt_lock = self.optimizer.lock().unwrap();
                if opt_lock.is_none() {
                    *opt_lock = Some(MasterWeightOptimizer::new(
                        master,
                        lr as f32,
                        (betas.0 as f32, betas.1 as f32),
                        eps as f32,
                        weight_decay as f32,
                    ));
                }
                let opt = opt_lock.as_mut().unwrap();

                let new_weight = {
                    let mg = self.inner.master_grad.lock().unwrap();
                    opt.step(&*mg, self.inner.out_features, self.inner.in_features)
                };
                self.inner.weight = new_weight;

                *self.inner.gpu_weight_buf.lock().unwrap() = None;
                *self.inner.gpu_params_buf.lock().unwrap() = None;
                *self.inner.gpu_bind_group.lock().unwrap() = None;

                self.inner.zero_grad();

                let loss_t = Tensor::from_vec(vec![loss_value], vec![1]);
                Ok(PyTensor::from_tensor(loss_t))
            }

            fn repack(&mut self) { self.inner.repack(); }

            fn zero_grad(&self) { self.inner.zero_grad(); }

            fn train(&self) { self.inner.train_mode(); }

            fn eval(&self) { self.inner.eval_mode(); }

            #[getter]
            fn is_training(&self) -> bool { self.inner.is_training() }

            #[getter]
            fn in_features(&self) -> i64 { self.inner.in_features as i64 }

            #[getter]
            fn out_features(&self) -> i64 { self.inner.out_features as i64 }

            #[getter]
            fn num_params(&self) -> i64 { self.inner.num_params() as i64 }

            #[getter]
            fn memory_savings(&self) -> f64 { self.inner.memory_savings() as f64 }

            #[getter]
            fn master_weight(&self) -> Option<Vec<f64>> {
                self.inner.master_weight.as_ref()
                    .map(|v| v.iter().map(|&x| x as f64).collect())
            }

            #[getter]
            fn bias(&self) -> Option<Vec<f64>> {
                self.inner.bias.as_ref()
                    .map(|v| v.iter().map(|&x| x as f64).collect())
            }
        }
    };
}

// ---- PyLinear4 (4-bit, U4x8) ----

#[pyclass]
pub struct PyLinear4 {
    inner: PackedLinear<U4x8>,
    optimizer: Mutex<Option<MasterWeightOptimizer<U4x8>>>,
}

impl_pylinear_pymethods!(PyLinear4);

// ---- PyLinear8 (8-bit, U8x4) ----

#[pyclass]
pub struct PyLinear8 {
    inner: PackedLinear<U8x4>,
    optimizer: Mutex<Option<MasterWeightOptimizer<U8x4>>>,
}

impl_pylinear_pymethods!(PyLinear8);

// ---- PyLinear16 (16-bit, F16x2) ----

#[pyclass]
pub struct PyLinear16 {
    inner: PackedLinear<F16x2>,
    optimizer: Mutex<Option<MasterWeightOptimizer<F16x2>>>,
}

impl_pylinear_pymethods!(PyLinear16);

// ---- PyLinear32 (32-bit, F32x1) ----

#[pyclass]
pub struct PyLinear32 {
    inner: PackedLinear<F32x1>,
    optimizer: Mutex<Option<MasterWeightOptimizer<F32x1>>>,
}

impl_pylinear_pymethods!(PyLinear32);
