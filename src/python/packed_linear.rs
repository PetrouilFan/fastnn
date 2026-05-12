use std::sync::Mutex;
use crate::packed_layer::PackedLinear;
use crate::packed_train::MasterWeightOptimizer;

macro_rules! impl_pylinear_pymethods {
    ($linear:ident, $word:ty) => {
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
                let shape = input.inner.shape();
                match shape.len() {
                    1 => {
                        let in_f = shape[0] as usize;
                        assert_eq!(in_f, self.inner.in_features,
                            "Input length {} != in_features {}", in_f, self.inner.in_features);
                        let output = self.inner.forward(&input_vec);
                        let t = Tensor::from_vec(output, vec![self.inner.out_features as i64]);
                        Ok(PyTensor::from_tensor(t))
                    }
                    2 => {
                        let batch = shape[0] as usize;
                        let in_f = shape[1] as usize;
                        assert_eq!(in_f, self.inner.in_features,
                            "Input features {} != in_features {}", in_f, self.inner.in_features);
                        let mut outputs = Vec::with_capacity(batch * self.inner.out_features);
                        for b in 0..batch {
                            let row = &input_vec[b * in_f .. (b + 1) * in_f];
                            let out = self.inner.forward(row);
                            outputs.extend_from_slice(&out);
                        }
                        let t = Tensor::from_vec(outputs, vec![batch as i64, self.inner.out_features as i64]);
                        Ok(PyTensor::from_tensor(t))
                    }
                    _ => Err(PyRuntimeError::new_err(format!(
                        "Expected 1D or 2D input, got {}D tensor with shape {:?}", shape.len(), shape
                    ))),
                }
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
                let target_vec = target.inner.to_numpy();
                let shape = input.inner.shape();
                let out_f = self.inner.out_features;

                let loss_value: f32 = match shape.len() {
                    1 => {
                        let in_f = shape[0] as usize;
                        assert_eq!(in_f, self.inner.in_features);
                        let output = self.inner.forward(&input_vec);

                        let grad_output: Vec<f32> = match loss_fn {
                            "cross_entropy" => {
                                let inv_nc = 1.0 / out_f as f32;
                                output.iter().zip(target_vec.iter())
                                    .map(|(o, t)| (o - t) * inv_nc)
                                    .collect()
                            }
                            "mse" => {
                                let n = out_f as f32;
                                output.iter().zip(target_vec.iter())
                                    .map(|(o, t)| 2.0 * (o - t) / n)
                                    .collect()
                            }
                            _ => return Err(PyRuntimeError::new_err(
                                format!("Unknown loss function: {}", loss_fn)
                            )),
                        };

                        let loss: f32 = match loss_fn {
                            "cross_entropy" => {
                                let inv_nc = 1.0 / out_f as f32;
                                output.iter().zip(target_vec.iter())
                                    .map(|(o, t)| -(t * o.ln().max(-100.0)) * inv_nc)
                                    .sum::<f32>()
                            }
                            "mse" => {
                                let n = out_f as f32;
                                output.iter().zip(target_vec.iter())
                                    .map(|(o, t)| (o - t).powi(2) / n)
                                    .sum::<f32>()
                            }
                            _ => unreachable!(),
                        };

                        self.inner.backward(&grad_output);
                        loss
                    }
                    2 => {
                        let batch = shape[0] as usize;
                        let in_f = shape[1] as usize;
                        assert_eq!(in_f, self.inner.in_features,
                            "Input features {} != in_features {}", in_f, self.inner.in_features);

                        self.inner.zero_grad();
                        let inv_batch = 1.0 / batch as f32;
                        let mut total_loss = 0.0;

                        for b in 0..batch {
                            let row = &input_vec[b * in_f .. (b + 1) * in_f];
                            let row_target = &target_vec[b * out_f .. (b + 1) * out_f];
                            let output = self.inner.forward(row);

                            let (row_loss, grad_output): (f32, Vec<f32>) = match loss_fn {
                                "cross_entropy" => {
                                    let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    let exps: Vec<f32> = output.iter().map(|x| (x - max_val).exp()).collect();
                                    let sum_exp: f32 = exps.iter().sum();
                                    let softmax: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
                                    let mut loss = 0.0;
                                    for (s, t_val) in softmax.iter().zip(row_target.iter()) {
                                        if *t_val > 0.0 {
                                            loss -= t_val * s.ln().max(-1e10);
                                        }
                                    }
                                    let grad: Vec<f32> = softmax.iter().zip(row_target.iter())
                                        .map(|(s, t)| (s - t) * inv_batch).collect();
                                    (loss, grad)
                                }
                                "mse" => {
                                    let loss: f32 = output.iter().zip(row_target.iter())
                                        .map(|(o, t)| (o - t).powi(2)).sum();
                                    let grad: Vec<f32> = output.iter().zip(row_target.iter())
                                        .map(|(o, t)| 2.0 * (o - t) * inv_batch / out_f as f32).collect();
                                    (loss, grad)
                                }
                                _ => return Err(PyRuntimeError::new_err(
                                    format!("Unknown loss function: {}", loss_fn)
                                )),
                            };
                            total_loss += row_loss;
                            self.inner.backward(&grad_output);
                        }

                        match loss_fn {
                            "cross_entropy" => total_loss * inv_batch,
                            "mse" => total_loss * inv_batch / out_f as f32,
                            _ => unreachable!(),
                        }
                    }
                    _ => return Err(PyRuntimeError::new_err(format!(
                        "Expected 1D or 2D input, got {}D tensor with shape {:?}", shape.len(), shape
                    ))),
                };

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

                let (new_weight, new_master) = {
                    let mg = self.inner.master_grad.lock().unwrap();
                    let packed = opt.step(&*mg, self.inner.out_features, self.inner.in_features);
                    (packed, opt.master.clone())
                };
                self.inner.weight = new_weight;
                if let Some(ref mut m) = self.inner.master_weight {
                    m.copy_from_slice(&new_master);
                }

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

            fn save(&self, path: &str) -> PyResult<()> {
                use std::io::Write;
                let file = std::fs::File::create(path)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create: {}", e)))?;
                let mut w = std::io::BufWriter::new(file);

                w.write_all(b"FPKL").map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                w.write_all(&1u32.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                w.write_all(&(self.inner.in_features as u64).to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                w.write_all(&(self.inner.out_features as u64).to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                w.write_all(&[if self.inner.bias.is_some() { 1u8 } else { 0u8 }]).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                let weight_words = self.inner.weight.as_u32();
                w.write_all(&(weight_words.len() as u64).to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                for &word in weight_words {
                    w.write_all(&word.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }

                w.write_all(&self.inner.weight.scale().to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                w.write_all(&self.inner.weight.zero().to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                w.write_all(&[if self.inner.weight.is_per_channel() { 1u8 } else { 0u8 }]).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                if let Some(ref mw) = self.inner.master_weight {
                    w.write_all(&(mw.len() as u64).to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    for &v in mw {
                        w.write_all(&v.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    }
                } else {
                    w.write_all(&0u64.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }

                if let Some(ref bias) = self.inner.bias {
                    w.write_all(&(bias.len() as u64).to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    for &v in bias {
                        w.write_all(&v.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    }
                } else {
                    w.write_all(&0u64.to_le_bytes()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                }

                Ok(())
            }

            #[staticmethod]
            fn load(path: &str) -> PyResult<Self> {
                use std::io::Read;
                let file = std::fs::File::open(path)
                    .map_err(|e| PyRuntimeError::new_err(format!("Failed to open: {}", e)))?;
                let mut r = std::io::BufReader::new(file);

                let mut buf4 = [0u8; 4];
                let mut buf8 = [0u8; 8];

                r.read_exact(&mut buf4).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                if &buf4 != b"FPKL" {
                    return Err(PyRuntimeError::new_err("Invalid file format: missing FPKL magic"));
                }

                r.read_exact(&mut buf4).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let _version = u32::from_le_bytes(buf4);

                r.read_exact(&mut buf8).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let in_f = u64::from_le_bytes(buf8) as usize;
                r.read_exact(&mut buf8).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let out_f = u64::from_le_bytes(buf8) as usize;

                let mut hb = [0u8; 1];
                r.read_exact(&mut hb).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                r.read_exact(&mut buf8).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let wlen = u64::from_le_bytes(buf8) as usize;
                let mut weight_u32 = vec![0u32; wlen];
                for word in &mut weight_u32 {
                    r.read_exact(&mut buf4).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    *word = u32::from_le_bytes(buf4);
                }

                r.read_exact(&mut buf4).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let scale = f32::from_le_bytes(buf4);
                r.read_exact(&mut buf4).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let zero = f32::from_le_bytes(buf4);
                r.read_exact(&mut hb).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                let t_data: Vec<$word> = bytemuck::cast_slice(&weight_u32).to_vec();
                let weight = PackedTensor::from_raw(t_data, vec![out_f, in_f], vec![scale], vec![zero]);

                r.read_exact(&mut buf8).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let mw_len = u64::from_le_bytes(buf8) as usize;
                let mut master_weight = None;
                if mw_len > 0 {
                    let mut mw = vec![0.0f32; mw_len];
                    for v in &mut mw {
                        r.read_exact(&mut buf4).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        *v = f32::from_le_bytes(buf4);
                    }
                    master_weight = Some(mw);
                }

                r.read_exact(&mut buf8).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let b_len = u64::from_le_bytes(buf8) as usize;
                let bias = if b_len > 0 {
                    let mut b = vec![0.0f32; b_len];
                    for v in &mut b {
                        r.read_exact(&mut buf4).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        *v = f32::from_le_bytes(buf4);
                    }
                    Some(b)
                } else {
                    None
                };

                let mut layer = PackedLinear::from_packed(weight, bias, in_f, out_f);
                layer.master_weight = master_weight;

                Ok($linear {
                    inner: layer,
                    optimizer: Mutex::new(None),
                })
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

impl_pylinear_pymethods!(PyLinear4, U4x8);

// ---- PyLinear8 (8-bit, U8x4) ----

#[pyclass]
pub struct PyLinear8 {
    inner: PackedLinear<U8x4>,
    optimizer: Mutex<Option<MasterWeightOptimizer<U8x4>>>,
}

impl_pylinear_pymethods!(PyLinear8, U8x4);

// ---- PyLinear16 (16-bit, F16x2) ----

#[pyclass]
pub struct PyLinear16 {
    inner: PackedLinear<F16x2>,
    optimizer: Mutex<Option<MasterWeightOptimizer<F16x2>>>,
}

impl_pylinear_pymethods!(PyLinear16, F16x2);

// ---- PyLinear32 (32-bit, F32x1) ----

#[pyclass]
pub struct PyLinear32 {
    inner: PackedLinear<F32x1>,
    optimizer: Mutex<Option<MasterWeightOptimizer<F32x1>>>,
}

impl_pylinear_pymethods!(PyLinear32, F32x1);

// ---- Fused PackedLinearGelu wrappers ----

macro_rules! impl_pylinear_gelu_pymethods {
    ($linear_gelu:ident) => {
        #[pymethods]
        impl $linear_gelu {
            #[new]
            #[pyo3(signature = (in_features, out_features, bias = true))]
            fn new(in_features: i64, out_features: i64, bias: bool) -> Self {
                $linear_gelu {
                    inner: PackedLinear::new(in_features as usize, out_features as usize, bias),
                }
            }

            fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
                let input_vec = input.inner.to_numpy();
                let shape = input.inner.shape();
                match shape.len() {
                    1 => {
                        let in_f = shape[0] as usize;
                        assert_eq!(in_f, self.inner.in_features,
                            "Input length {} != in_features {}", in_f, self.inner.in_features);
                        let output = self.inner.forward_gelu(&input_vec);
                        let t = Tensor::from_vec(output, vec![self.inner.out_features as i64]);
                        Ok(PyTensor::from_tensor(t))
                    }
                    2 => {
                        let batch = shape[0] as usize;
                        let in_f = shape[1] as usize;
                        assert_eq!(in_f, self.inner.in_features,
                            "Input features {} != in_features {}", in_f, self.inner.in_features);
                        let mut outputs = Vec::with_capacity(batch * self.inner.out_features);
                        for b in 0..batch {
                            let row = &input_vec[b * in_f .. (b + 1) * in_f];
                            let out = self.inner.forward_gelu(row);
                            outputs.extend_from_slice(&out);
                        }
                        let t = Tensor::from_vec(outputs, vec![batch as i64, self.inner.out_features as i64]);
                        Ok(PyTensor::from_tensor(t))
                    }
                    _ => Err(PyRuntimeError::new_err(format!(
                        "Expected 1D or 2D input, got {}D tensor with shape {:?}", shape.len(), shape
                    ))),
                }
            }

            fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
                self.__call__(input)
            }

            #[getter]
            fn in_features(&self) -> i64 { self.inner.in_features as i64 }

            #[getter]
            fn out_features(&self) -> i64 { self.inner.out_features as i64 }

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
pub struct PyPackedLinearGelu4 {
    inner: PackedLinear<U4x8>,
}

impl_pylinear_gelu_pymethods!(PyPackedLinearGelu4);

#[pyclass]
pub struct PyPackedLinearGelu8 {
    inner: PackedLinear<U8x4>,
}

impl_pylinear_gelu_pymethods!(PyPackedLinearGelu8);

#[pyclass]
pub struct PyPackedLinearGelu16 {
    inner: PackedLinear<F16x2>,
}

impl_pylinear_gelu_pymethods!(PyPackedLinearGelu16);

#[pyclass]
pub struct PyPackedLinearGelu32 {
    inner: PackedLinear<F32x1>,
}

impl_pylinear_gelu_pymethods!(PyPackedLinearGelu32);
