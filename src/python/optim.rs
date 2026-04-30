#[pyclass]
struct PySGD {
    inner: core_optim::sgd::SGD,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr = 0.01, momentum = 0.0, weight_decay = 0.0))]
    fn new(params: Vec<PyTensor>, lr: f64, momentum: f64, weight_decay: f64) -> Self {
        let tensors: Vec<core_tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        PySGD {
            inner: core_optim::sgd::SGD::new(tensors, lr, momentum, 0.0, weight_decay, false),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("momentum", self.inner.momentum)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        let v_list = PyList::new(
            py,
            self.inner
                .velocity
                .iter()
                .map(|t| PyTensor::from_tensor(t.clone())),
        )?;
        dict.set_item("velocity", v_list)?;
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.lr = state.get_item("lr")?.extract()?;
        if let Ok(m) = state.get_item("momentum")?.extract::<f64>() {
            self.inner.momentum = m;
        }
        if let Ok(wd) = state.get_item("weight_decay")?.extract::<f64>() {
            self.inner.weight_decay = wd;
        }
        if let Ok(v_list) = state.get_item("velocity")?.extract::<Vec<PyTensor>>() {
            self.inner.velocity = v_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

#[pyclass]
struct PyAdam {
    inner: core_optim::adam::Adam,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (params, lr = 0.001, betas = None, eps = 1e-8, weight_decay = 0.0))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        betas: Option<(f64, f64)>,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let tensors: Vec<core_tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        let betas = betas.unwrap_or((0.9, 0.999));
        PyAdam {
            inner: core_optim::adam::Adam::new(tensors, lr, betas, eps, weight_decay, false),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("betas", self.inner.betas)?;
        dict.set_item("eps", self.inner.eps)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        dict.set_item("amsgrad", self.inner.amsgrad)?;
        let steps = PyList::new(py, self.inner.step.iter().copied())?;
        dict.set_item("step", steps)?;
        let m_list = PyList::new(
            py,
            self.inner
                .m
                .iter()
                .map(|t| PyTensor::from_tensor(t.clone())),
        )?;
        dict.set_item("m", m_list)?;
        let v_list = PyList::new(
            py,
            self.inner
                .v
                .iter()
                .map(|t| PyTensor::from_tensor(t.clone())),
        )?;
        dict.set_item("v", v_list)?;
        let v_hat_list = PyList::new(
            py,
            self.inner
                .v_hat
                .iter()
                .map(|t| PyTensor::from_tensor(t.clone())),
        )?;
        dict.set_item("v_hat", v_hat_list)?;
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, _py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.lr = state.get_item("lr")?.extract()?;
        self.inner.betas = state.get_item("betas")?.extract()?;
        self.inner.eps = state.get_item("eps")?.extract()?;
        self.inner.weight_decay = state.get_item("weight_decay")?.extract()?;
        if let Ok(ams) = state.get_item("amsgrad")?.extract::<bool>() {
            self.inner.amsgrad = ams;
        }
        if let Ok(steps) = state.get_item("step")?.extract::<Vec<u64>>() {
            self.inner.step = steps;
        }
        if let Ok(m_list) = state.get_item("m")?.extract::<Vec<PyTensor>>() {
            self.inner.m = m_list.into_iter().map(|p| p.inner).collect();
        }
        if let Ok(v_list) = state.get_item("v")?.extract::<Vec<PyTensor>>() {
            self.inner.v = v_list.into_iter().map(|p| p.inner).collect();
        }
        if let Ok(v_hat_list) = state.get_item("v_hat")?.extract::<Vec<PyTensor>>() {
            self.inner.v_hat = v_hat_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

#[pyclass]
struct PyAdamW {
    inner: core_optim::adamw::AdamW,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (params, lr = 0.001, betas = None, eps = 1e-8, weight_decay = 0.0))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        betas: Option<(f64, f64)>,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let tensors: Vec<core_tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        let betas = betas.unwrap_or((0.9, 0.999));
        PyAdamW {
            inner: core_optim::adamw::AdamW::new(tensors, lr, betas, eps, weight_decay, false),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("betas", self.inner.betas)?;
        dict.set_item("eps", self.inner.eps)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        let steps = PyList::new(py, self.inner.step.iter().copied())?;
        dict.set_item("step", steps)?;
        let m_list = PyList::new(
            py,
            self.inner
                .m
                .iter()
                .map(|t| PyTensor::from_tensor(t.clone())),
        )?;
        dict.set_item("m", m_list)?;
        let v_list = PyList::new(
            py,
            self.inner
                .v
                .iter()
                .map(|t| PyTensor::from_tensor(t.clone())),
        )?;
        dict.set_item("v", v_list)?;
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.lr = state.get_item("lr")?.extract()?;
        self.inner.betas = state.get_item("betas")?.extract()?;
        self.inner.eps = state.get_item("eps")?.extract()?;
        self.inner.weight_decay = state.get_item("weight_decay")?.extract()?;
        if let Ok(steps) = state.get_item("step")?.extract::<Vec<u64>>() {
            self.inner.step = steps;
        }
        if let Ok(m_list) = state.get_item("m")?.extract::<Vec<PyTensor>>() {
            self.inner.m = m_list.into_iter().map(|p| p.inner).collect();
        }
        if let Ok(v_list) = state.get_item("v")?.extract::<Vec<PyTensor>>() {
            self.inner.v = v_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

#[pyclass]
struct PyMuon {
    inner: core_optim::muon::Muon,
}

#[pymethods]
impl PyMuon {
    #[new]
    #[pyo3(signature = (params, lr = 0.025, momentum = 0.95, weight_decay = 0.0, nesterov = true))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> Self {
        let tensors: Vec<core_tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        PyMuon {
            inner: core_optim::muon::Muon::new(tensors, lr, momentum, weight_decay, nesterov),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("momentum", self.inner.momentum)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        dict.set_item("nesterov", self.inner.nesterov)?;
        let m_list = PyList::new(
            py,
            self.inner
                .m
                .iter()
                .map(|t| PyTensor::from_tensor(t.clone())),
        )?;
        dict.set_item("m", m_list)?;
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.lr = state.get_item("lr")?.extract()?;
        if let Ok(m) = state.get_item("momentum")?.extract::<f64>() {
            self.inner.momentum = m;
        }
        if let Ok(wd) = state.get_item("weight_decay")?.extract::<f64>() {
            self.inner.weight_decay = wd;
        }
        if let Ok(n) = state.get_item("nesterov")?.extract::<bool>() {
            self.inner.nesterov = n;
        }
        if let Ok(m_list) = state.get_item("m")?.extract::<Vec<PyTensor>>() {
            self.inner.m = m_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

#[pyclass]
struct PyLion {
    inner: core_optim::lion::Lion,
}

#[pymethods]
impl PyLion {
    #[new]
    #[pyo3(signature = (params, lr = 0.0001, betas = None, weight_decay = 0.0))]
    fn new(params: Vec<PyTensor>, lr: f64, betas: Option<(f64, f64)>, weight_decay: f64) -> Self {
        let tensors: Vec<core_tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        let betas = betas.unwrap_or((0.95, 0.98));
        PyLion {
            inner: core_optim::lion::Lion::new(tensors, lr, betas, weight_decay),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("betas", self.inner.betas)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        let m_list = PyList::new(
            py,
            self.inner
                .m
                .iter()
                .map(|t| PyTensor::from_tensor(t.clone())),
        )?;
        dict.set_item("m", m_list)?;
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.lr = state.get_item("lr")?.extract()?;
        self.inner.betas = state.get_item("betas")?.extract()?;
        if let Ok(wd) = state.get_item("weight_decay")?.extract::<f64>() {
            self.inner.weight_decay = wd;
        }
        if let Ok(m_list) = state.get_item("m")?.extract::<Vec<PyTensor>>() {
            self.inner.m = m_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

#[pyclass]
struct PyRMSprop {
    inner: core_optim::rmsprop::RMSprop,
}

#[pymethods]
impl PyRMSprop {
    #[new]
    #[pyo3(signature = (params, lr = 0.01, alpha = 0.99, eps = 1e-8, weight_decay = 0.0, momentum = 0.0, centered = false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        alpha: f64,
        eps: f64,
        weight_decay: f64,
        momentum: f64,
        centered: bool,
    ) -> Self {
        let tensors: Vec<core_tensor::Tensor> = params.into_iter().map(|p| p.inner).collect();
        PyRMSprop {
            inner: core_optim::rmsprop::RMSprop::new(
                tensors,
                lr,
                alpha,
                eps,
                weight_decay,
                momentum,
                centered,
            ),
        }
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let dict = PyDict::new(py);
        dict.set_item("lr", self.inner.lr)?;
        dict.set_item("alpha", self.inner.alpha)?;
        dict.set_item("eps", self.inner.eps)?;
        dict.set_item("weight_decay", self.inner.weight_decay)?;
        dict.set_item("momentum", self.inner.momentum)?;
        dict.set_item("centered", self.inner.centered)?;
        let sq_list = PyList::new(
            py,
            self.inner
                .square_avg
                .iter()
                .map(|t| PyTensor::from_tensor(t.clone())),
        )?;
        dict.set_item("square_avg", sq_list)?;
        if self.inner.centered {
            let ga_list = PyList::new(
                py,
                self.inner
                    .grad_avg
                    .iter()
                    .map(|t| PyTensor::from_tensor(t.clone())),
            )?;
            dict.set_item("grad_avg", ga_list)?;
        }
        if self.inner.momentum != 0.0 {
            let mb_list = PyList::new(
                py,
                self.inner
                    .momentum_buf
                    .iter()
                    .map(|t| PyTensor::from_tensor(t.clone())),
            )?;
            dict.set_item("momentum_buf", mb_list)?;
        }
        Ok(dict.into())
    }

    fn load_state_dict(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.lr = state.get_item("lr")?.extract()?;
        self.inner.alpha = state.get_item("alpha")?.extract()?;
        self.inner.eps = state.get_item("eps")?.extract()?;
        if let Ok(wd) = state.get_item("weight_decay")?.extract::<f64>() {
            self.inner.weight_decay = wd;
        }
        if let Ok(m) = state.get_item("momentum")?.extract::<f64>() {
            self.inner.momentum = m;
        }
        if let Ok(sq_list) = state.get_item("square_avg")?.extract::<Vec<PyTensor>>() {
            self.inner.square_avg = sq_list.into_iter().map(|p| p.inner).collect();
        }
        if let Ok(ga_list) = state.get_item("grad_avg")?.extract::<Vec<PyTensor>>() {
            self.inner.grad_avg = ga_list.into_iter().map(|p| p.inner).collect();
        }
        if let Ok(mb_list) = state.get_item("momentum_buf")?.extract::<Vec<PyTensor>>() {
            self.inner.momentum_buf = mb_list.into_iter().map(|p| p.inner).collect();
        }
        Ok(())
    }
}

