// PyO3 bindings for packed-precision MasterWeightOptimizer (4/8/16/32 bit).

// Backend control wrappers
#[pyfunction]
pub fn use_wgpu() {
    crate::packed_layer::use_wgpu();
}

#[pyfunction]
pub fn use_cpu() {
    crate::packed_layer::use_cpu();
}

#[pyfunction]
pub fn is_wgpu() -> bool {
    crate::packed_layer::is_wgpu()
}

// ---- PyMasterWeightOptimizer4 (4-bit, U4x8) ----

#[pyclass]
pub struct PyMasterWeightOptimizer4 {
    inner: crate::packed_train::MasterWeightOptimizer<crate::dtypes::U4x8>,
    rows: usize,
    cols: usize,
}

#[pymethods]
impl PyMasterWeightOptimizer4 {
    #[new]
    #[pyo3(signature = (master_weights, rows, cols, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.0))]
    fn new(
        master_weights: Vec<f64>,
        rows: i64,
        cols: i64,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let master: Vec<f32> = master_weights.iter().map(|&x| x as f32).collect();
        let inner = MasterWeightOptimizer::new(
            &master,
            lr as f32,
            (betas.0 as f32, betas.1 as f32),
            eps as f32,
            weight_decay as f32,
        );
        PyMasterWeightOptimizer4 {
            inner,
            rows: rows as usize,
            cols: cols as usize,
        }
    }

    fn step(&mut self, gradients: Vec<f64>) -> PyResult<()> {
        let grad: Vec<f32> = gradients.iter().map(|&x| x as f32).collect();
        let _ = self.inner.step(&grad, self.rows, self.cols);
        Ok(())
    }

    fn get_master_weights(&self) -> Vec<f64> {
        self.inner.master.iter().map(|&x| x as f64).collect()
    }

    #[getter]
    fn lr(&self) -> f64 {
        self.inner.lr as f64
    }

    fn set_lr(&mut self, lr: f64) {
        self.inner.set_lr(lr as f32);
    }

    #[getter]
    fn step_count(&self) -> i64 {
        self.inner.step as i64
    }

    fn set_scale_update_freq(&mut self, freq: i64) {
        self.inner.set_scale_update_freq(freq as usize);
    }
}

// ---- PyMasterWeightOptimizer8 (8-bit, U8x4) ----

#[pyclass]
pub struct PyMasterWeightOptimizer8 {
    inner: crate::packed_train::MasterWeightOptimizer<crate::dtypes::U8x4>,
    rows: usize,
    cols: usize,
}

#[pymethods]
impl PyMasterWeightOptimizer8 {
    #[new]
    #[pyo3(signature = (master_weights, rows, cols, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.0))]
    fn new(
        master_weights: Vec<f64>,
        rows: i64,
        cols: i64,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let master: Vec<f32> = master_weights.iter().map(|&x| x as f32).collect();
        let inner = MasterWeightOptimizer::new(
            &master,
            lr as f32,
            (betas.0 as f32, betas.1 as f32),
            eps as f32,
            weight_decay as f32,
        );
        PyMasterWeightOptimizer8 {
            inner,
            rows: rows as usize,
            cols: cols as usize,
        }
    }

    fn step(&mut self, gradients: Vec<f64>) -> PyResult<()> {
        let grad: Vec<f32> = gradients.iter().map(|&x| x as f32).collect();
        let _ = self.inner.step(&grad, self.rows, self.cols);
        Ok(())
    }

    fn get_master_weights(&self) -> Vec<f64> {
        self.inner.master.iter().map(|&x| x as f64).collect()
    }

    #[getter]
    fn lr(&self) -> f64 {
        self.inner.lr as f64
    }

    fn set_lr(&mut self, lr: f64) {
        self.inner.set_lr(lr as f32);
    }

    #[getter]
    fn step_count(&self) -> i64 {
        self.inner.step as i64
    }

    fn set_scale_update_freq(&mut self, freq: i64) {
        self.inner.set_scale_update_freq(freq as usize);
    }
}

// ---- PyMasterWeightOptimizer16 (16-bit, F16x2) ----

#[pyclass]
pub struct PyMasterWeightOptimizer16 {
    inner: crate::packed_train::MasterWeightOptimizer<crate::dtypes::F16x2>,
    rows: usize,
    cols: usize,
}

#[pymethods]
impl PyMasterWeightOptimizer16 {
    #[new]
    #[pyo3(signature = (master_weights, rows, cols, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.0))]
    fn new(
        master_weights: Vec<f64>,
        rows: i64,
        cols: i64,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let master: Vec<f32> = master_weights.iter().map(|&x| x as f32).collect();
        let inner = MasterWeightOptimizer::new(
            &master,
            lr as f32,
            (betas.0 as f32, betas.1 as f32),
            eps as f32,
            weight_decay as f32,
        );
        PyMasterWeightOptimizer16 {
            inner,
            rows: rows as usize,
            cols: cols as usize,
        }
    }

    fn step(&mut self, gradients: Vec<f64>) -> PyResult<()> {
        let grad: Vec<f32> = gradients.iter().map(|&x| x as f32).collect();
        let _ = self.inner.step(&grad, self.rows, self.cols);
        Ok(())
    }

    fn get_master_weights(&self) -> Vec<f64> {
        self.inner.master.iter().map(|&x| x as f64).collect()
    }

    #[getter]
    fn lr(&self) -> f64 {
        self.inner.lr as f64
    }

    fn set_lr(&mut self, lr: f64) {
        self.inner.set_lr(lr as f32);
    }

    #[getter]
    fn step_count(&self) -> i64 {
        self.inner.step as i64
    }

    fn set_scale_update_freq(&mut self, freq: i64) {
        self.inner.set_scale_update_freq(freq as usize);
    }
}

// ---- PyMasterWeightOptimizer32 (32-bit, F32x1) ----

#[pyclass]
pub struct PyMasterWeightOptimizer32 {
    inner: crate::packed_train::MasterWeightOptimizer<crate::dtypes::F32x1>,
    rows: usize,
    cols: usize,
}

#[pymethods]
impl PyMasterWeightOptimizer32 {
    #[new]
    #[pyo3(signature = (master_weights, rows, cols, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0.0))]
    fn new(
        master_weights: Vec<f64>,
        rows: i64,
        cols: i64,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let master: Vec<f32> = master_weights.iter().map(|&x| x as f32).collect();
        let inner = MasterWeightOptimizer::new(
            &master,
            lr as f32,
            (betas.0 as f32, betas.1 as f32),
            eps as f32,
            weight_decay as f32,
        );
        PyMasterWeightOptimizer32 {
            inner,
            rows: rows as usize,
            cols: cols as usize,
        }
    }

    fn step(&mut self, gradients: Vec<f64>) -> PyResult<()> {
        let grad: Vec<f32> = gradients.iter().map(|&x| x as f32).collect();
        let _ = self.inner.step(&grad, self.rows, self.cols);
        Ok(())
    }

    fn get_master_weights(&self) -> Vec<f64> {
        self.inner.master.iter().map(|&x| x as f64).collect()
    }

    #[getter]
    fn lr(&self) -> f64 {
        self.inner.lr as f64
    }

    fn set_lr(&mut self, lr: f64) {
        self.inner.set_lr(lr as f32);
    }

    #[getter]
    fn step_count(&self) -> i64 {
        self.inner.step as i64
    }

    fn set_scale_update_freq(&mut self, freq: i64) {
        self.inner.set_scale_update_freq(freq as usize);
    }
}
