//! Sessions and tensors for Python.
//!
//! A [`PySession`] owns one [`RuntimeSession`]: its plan, its arena and its
//! worker pool. The Python object holds a `std::sync::Mutex`, so two Python
//! threads calling the same session serialise instead of corrupting state; two
//! *different* sessions run in parallel and share only the read-only mapping.
//!
//! The GIL is released around the forward. Inputs are copied out of Python
//! objects before the detached call and outputs are copied into `bytes`
//! afterwards, so no Python object is touched while the GIL is not held.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex as StdMutex;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};

use crate::inference::error::Fn3Error as CoreError;
use crate::inference::runtime::{HostTensor, Session as RuntimeSession};

use super::errors;

/// Process-local session identity, for reprs and diagnostics.
static SESSION_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// One output tensor, copied out of the session arena.
#[pyclass(module = "fastnn._core.inference", name = "Tensor", frozen)]
pub struct PyTensor {
    name: String,
    shape: Vec<u64>,
    dtype: String,
    bytes: Py<PyBytes>,
}

impl PyTensor {
    fn of(
        py: Python<'_>,
        name: String,
        shape: Vec<u64>,
        tensor: &HostTensor,
    ) -> PyResult<PyTensor> {
        let (dtype, raw): (&str, Vec<u8>) = match tensor {
            HostTensor::F32(values) => (
                "float32",
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            HostTensor::I64(values) => (
                "int64",
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            HostTensor::I32(values) => (
                "int32",
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            HostTensor::Bool(values) => (
                "bool",
                values.iter().map(|&value| u8::from(value)).collect(),
            ),
        };
        Ok(PyTensor {
            name,
            shape,
            dtype: dtype.to_string(),
            bytes: PyBytes::new(py, &raw).unbind(),
        })
    }
}

#[pymethods]
impl PyTensor {
    /// Output name exactly as the model declared it.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Shape as a tuple.
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        Ok(pyo3::types::PyTuple::new(py, self.shape.iter())?.unbind())
    }

    /// `"float32"`, `"int64"`, `"int32"` or `"bool"`.
    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }

    /// Number of elements.
    #[getter]
    fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Payload size in bytes.
    #[getter]
    fn nbytes(&self, py: Python<'_>) -> u64 {
        self.bytes.bind(py).as_bytes().len() as u64
    }

    /// The raw little-endian payload.
    #[getter]
    fn data(&self, py: Python<'_>) -> Py<PyBytes> {
        Py::clone_ref(&self.bytes, py)
    }

    /// A read-only NumPy view over [`PyTensor::data`].
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let numpy = py.import("numpy")?;
        let dtype = match self.dtype.as_str() {
            "float32" => numpy.getattr("float32")?,
            "int64" => numpy.getattr("int64")?,
            "int32" => numpy.getattr("int32")?,
            _ => numpy.getattr("bool_")?,
        };
        let array = numpy.call_method1("frombuffer", (self.bytes.bind(py), dtype))?;
        let shape: Vec<u64> = self.shape.clone();
        array.call_method1("reshape", (shape,))
    }

    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut array = self.to_numpy(py)?;
        if let Some(dtype) = dtype {
            array = array.call_method1("astype", (dtype,))?;
        }
        if copy == Some(true) {
            array = array.call_method0("copy")?;
        }
        Ok(array)
    }

    /// The values as a nested Python list (copies).
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let bytes = self.bytes.bind(py).as_bytes();
        let flat: Vec<f64> = match self.dtype.as_str() {
            "float32" => bytes
                .chunks_exact(4)
                .map(|chunk| {
                    f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                })
                .collect(),
            "int64" => bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f64
                })
                .collect(),
            "int32" => bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64)
                .collect(),
            _ => bytes.iter().map(|&value| f64::from(value)).collect(),
        };
        nest(py, &flat, &self.shape)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(name={:?}, shape={:?}, dtype={:?})",
            self.name, self.shape, self.dtype
        )
    }
}

fn nest(py: Python<'_>, values: &[f64], shape: &[u64]) -> PyResult<Py<PyAny>> {
    if shape.len() <= 1 {
        return Ok(pyo3::types::PyList::new(py, values.iter().copied())?
            .into_any()
            .unbind());
    }
    let stride: usize = shape[1..].iter().map(|dim| *dim as usize).product();
    let mut rows: Vec<Py<PyAny>> = Vec::new();
    if stride == 0 {
        return Ok(pyo3::types::PyList::empty(py).into_any().unbind());
    }
    for chunk in values.chunks(stride) {
        rows.push(nest(py, chunk, &shape[1..])?);
    }
    Ok(pyo3::types::PyList::new(py, rows)?.into_any().unbind())
}

/// A session: one plan, one arena, one worker pool.
#[pyclass(module = "fastnn._core.inference", name = "Session")]
pub struct PySession {
    session: StdMutex<RuntimeSession>,
    identity: u64,
    model: u64,
    worker_count: usize,
}

impl PySession {
    pub(crate) fn open(
        py: Python<'_>,
        session: RuntimeSession,
        model: u64,
    ) -> PyResult<Py<PySession>> {
        let worker_count = session.worker_count();
        let state = PySession {
            session: StdMutex::new(session),
            identity: SESSION_IDENTITY.fetch_add(1, Ordering::Relaxed),
            model,
            worker_count,
        };
        Py::new(py, state)
    }

    fn read(&self) -> Result<std::sync::MutexGuard<'_, RuntimeSession>, PyErr> {
        self.session
            .lock()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("session poisoned"))
    }
}

#[pymethods]
impl PySession {
    /// Workers this session owns.
    #[getter]
    fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Arena bytes the plan reserved.
    #[getter]
    fn arena_bytes(&self) -> u64 {
        self.read()
            .map(|session| session.arena().bytes() as u64)
            .unwrap_or(0)
    }

    /// Segments in the plan.
    #[getter]
    fn segments(&self) -> usize {
        self.read()
            .map(|session| session.plan().segments.len())
            .unwrap_or(0)
    }

    /// Recognised fusions.
    #[getter]
    fn fusions(&self) -> usize {
        self.read()
            .map(|session| session.fusions().len())
            .unwrap_or(0)
    }

    /// A compact plan report as a dictionary.
    fn plan(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let session = self.read()?;
        let plan = session.plan();
        let dict = PyDict::new(py);
        dict.set_item("segments", plan.segments.len())?;
        dict.set_item("fusions", plan.fusions.len())?;
        dict.set_item("reorders", plan.reorders.len())?;
        dict.set_item("reorder_bytes", plan.reorder_bytes)?;
        dict.set_item("arena_bytes", plan.arena_bytes)?;
        dict.set_item("peak_bytes", plan.peak_bytes)?;
        dict.set_item("layout", plan.layout.name())?;
        dict.set_item("inputs", plan.inputs.len())?;
        dict.set_item("outputs", plan.outputs.len())?;
        Ok(dict.unbind())
    }

    /// Run one forward.
    ///
    /// `inputs` maps every graph input name to a sequence of floats, a
    /// `bytes`/`bytearray` of little-endian f32 values, or an object with a
    /// `tolist()` method (a NumPy array). The result maps output names to
    /// [`Tensor`] objects, in the model's declared output order.
    fn run(&self, py: Python<'_>, inputs: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
        let mut owned: Vec<(String, Vec<f32>)> = Vec::with_capacity(inputs.len());
        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;
            if owned.iter().any(|(existing, _)| existing == &name) {
                return Err(PyTypeError::new_err(format!(
                    "the input {name:?} was supplied twice"
                )));
            }
            let values = extract_f32(&value).map_err(|error| {
                PyTypeError::new_err(format!("input {name:?} is not a float sequence: {error}"))
            })?;
            owned.push((name, values));
        }
        let borrowed: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(name, values)| (name.as_str(), values.as_slice()))
            .collect();
        let outputs = py.detach(|| {
            let mut session = self.session.lock().map_err(|_| CoreError::NotImplemented {
                what: "a Python session whose mutex was poisoned",
                owner: "python binding",
            })?;
            session.execute(&borrowed)?;
            let outputs = session.outputs()?;
            let shapes: Vec<Vec<u64>> = outputs
                .iter()
                .map(|(name, _)| {
                    session
                        .plan()
                        .output_slot(name)
                        .and_then(|id| session.plan().slot(id).ok())
                        .and_then(|slot| slot.ty.static_shape())
                        .map(|shape| shape.dims().to_vec())
                        .unwrap_or_default()
                })
                .collect();
            Ok::<_, CoreError>(
                outputs
                    .into_iter()
                    .zip(shapes)
                    .map(|((name, tensor), shape)| (name, shape, tensor))
                    .collect::<Vec<_>>(),
            )
        });
        let outputs = outputs.map_err(|error| errors::from_core(py, error))?;
        let dict = PyDict::new(py);
        for (name, shape, tensor) in &outputs {
            let tensor = PyTensor::of(py, name.clone(), shape.clone(), tensor)?;
            dict.set_item(name, Py::new(py, tensor)?)?;
        }
        Ok(dict.unbind())
    }

    /// Copy one output out by name.
    fn read_output(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyTensor>> {
        let session = self.read()?;
        let tensor = session
            .read_output(name)
            .map_err(|error| errors::from_core(py, error))?;
        let slot = session
            .plan()
            .output_slot(name)
            .and_then(|id| session.plan().slot(id).ok());
        let shape = slot
            .and_then(|slot| slot.ty.static_shape())
            .map(|shape| shape.dims().to_vec())
            .unwrap_or_default();
        drop(session);
        Py::new(py, PyTensor::of(py, name.to_string(), shape, &tensor)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "Session(#{} model=#{} workers={} segments={})",
            self.identity,
            self.model,
            self.worker_count,
            self.segments()
        )
    }
}

fn extract_f32(value: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(bytes) = value.cast::<PyBytes>() {
        let bytes = bytes.as_bytes();
        if bytes.len() % 4 != 0 {
            return Err(PyTypeError::new_err(format!(
                "the byte payload is {} bytes, not a whole number of f32 elements",
                bytes.len()
            )));
        }
        return Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect());
    }
    if let Ok(values) = value.extract::<Vec<f32>>() {
        return Ok(values);
    }
    if value.hasattr("tolist")? {
        return value.call_method0("tolist")?.extract::<Vec<f32>>();
    }
    Err(PyTypeError::new_err(format!(
        "expected a sequence, bytes or a NumPy array, found {}",
        value
            .get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_else(|_| "an object".to_string())
    )))
}

/// Register the session and tensor surface.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensor>()?;
    module.add_class::<PySession>()?;
    Ok(())
}
