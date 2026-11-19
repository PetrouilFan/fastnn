//! Device enumeration and request parsing for Python.
//!
//! # The rule this module implements
//!
//! A caller who names a device gets exactly that device or an exception. Only
//! `"auto"` (or `None`) may choose, and it chooses what
//! [`crate::inference::device::available_devices`] lists first, which is always the
//! CPU in this alpha. There is no "Vulkan was requested, CPU will do" path:
//! [`super::errors::from_device`] renders the request, the reason and the list
//! of devices this process can actually execute on.
//!
//! # Accepted spellings
//!
//! ```text
//! None, "auto"                 the build chooses (CPU today)
//! "cpu"                        the host CPU
//! "vulkan", "vulkan(0)"        the first Vulkan adapter (explicit!)
//! "vulkan:1"                   Vulkan adapter index 1
//! Device(...)                  an object returned by devices(), matched by name
//! ```
//!
//! A `Device` object is matched by adapter *name* rather than by index, so a
//! device list captured on one machine cannot silently select a different
//! adapter on another.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::inference::device::{available_devices, DeviceId, DeviceInfo, DeviceRequest};

use super::errors;

/// One device this process can execute on.
#[pyclass(module = "fastnn._core.inference", name = "Device", frozen)]
pub struct PyDevice {
    id: String,
    kind: String,
    index: Option<u32>,
    name: String,
}

impl PyDevice {
    fn of(info: &DeviceInfo) -> PyDevice {
        PyDevice {
            id: info.id.diagnostic(),
            kind: info.id.kind_name().to_string(),
            index: info.id.vulkan_index(),
            name: info.name.clone(),
        }
    }
}

#[pymethods]
impl PyDevice {
    /// Stable identifier, `"cpu"` or `"vulkan(0)"`.
    #[getter]
    fn id(&self) -> &str {
        &self.id
    }

    /// Device kind, `"cpu"` or `"vulkan"`.
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// Vulkan physical-device index, or `None` for the CPU.
    #[getter]
    fn index(&self) -> Option<u32> {
        self.index
    }

    /// Human-readable adapter name; `"cpu"` for the host CPU.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __str__(&self) -> &str {
        &self.id
    }

    fn __repr__(&self) -> String {
        if self.kind == "cpu" {
            format!("Device(id={:?})", self.id)
        } else {
            format!("Device(id={:?}, name={:?})", self.id, self.name)
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyRef<'_, PyDevice>>() {
            Ok(other) => self.id == other.id && self.name == other.name,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        // Stable across runs; `__eq__` is the authority.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in self.id.as_bytes().iter().chain(self.name.as_bytes()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }
}

/// Convert a resolved device into its Python description.
pub(crate) fn describe(py: Python<'_>, device: DeviceId) -> PyResult<Py<PyDevice>> {
    let available = available_devices();
    let info = available
        .iter()
        .find(|info| info.id == device)
        .cloned()
        .unwrap_or_else(|| DeviceInfo::new(device, device.kind_name().to_string()));
    Py::new(py, PyDevice::of(&info))
}

/// Devices this process can execute on, in preference order.
#[pyfunction]
pub fn devices(py: Python<'_>) -> PyResult<Vec<Py<PyDevice>>> {
    available_devices()
        .iter()
        .map(|info| Py::new(py, PyDevice::of(info)))
        .collect()
}

/// A device request the caller wrote, kept verbatim for error messages.
fn device_error(py: Python<'_>, requested: &str, reason: &str) -> PyErr {
    let available: Vec<String> = available_devices().iter().map(DeviceInfo::label).collect();
    let message = format!(
        "unrecognised device request {requested:?}: {reason}; accepted forms are \"auto\", \
         \"cpu\", \"vulkan\", \"vulkan:<index>\" and a Device object from fastnn.inference.devices(); \
         available devices: [{}]",
        available.join(", ")
    );
    let requested = requested.to_string();
    errors::typed_device(py, message, requested, available, reason.to_string())
}

/// Parse `"vulkan:<index>"` or `"vulkan(<index>)"`.
fn parse_vulkan_index(text: &str) -> Option<Option<u32>> {
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("vulkan") {
        return Some(None);
    }
    for rest in [
        trimmed.strip_prefix("vulkan:"),
        trimmed
            .strip_prefix("vulkan(")
            .and_then(|rest| rest.strip_suffix(')')),
    ]
    .into_iter()
    .flatten()
    {
        if rest.is_empty() {
            return Some(None);
        }
        if let Ok(index) = rest.parse::<u32>() {
            return Some(Some(index));
        }
    }
    None
}

/// Turn a Python value into a device request.
///
/// `None` is `Auto`. A string is one of the documented spellings. A `Device`
/// object requests exactly that adapter, by name.
pub(crate) fn parse_request(
    py: Python<'_>,
    request: Option<&Bound<'_, PyAny>>,
) -> PyResult<DeviceRequest> {
    let Some(request) = request else {
        return Ok(DeviceRequest::Auto);
    };
    if request.is_none() {
        return Ok(DeviceRequest::Auto);
    }
    if let Ok(device) = request.extract::<PyRef<'_, PyDevice>>() {
        if device.kind == "cpu" {
            return Ok(DeviceRequest::Cpu);
        }
        return Ok(match device.index {
            // An adapter with a real name is requested by that name: an index
            // captured on another machine must not silently pick a different
            // adapter here.
            Some(index) if device.name == "vulkan" || device.name.is_empty() => {
                DeviceRequest::Vulkan { index: Some(index) }
            }
            _ => DeviceRequest::VulkanNamed {
                name: device.name.clone(),
            },
        });
    }
    if let Ok(text) = request.extract::<String>() {
        let trimmed = text.trim();
        let keyword = trimmed.to_ascii_lowercase();
        if keyword == "auto" || keyword.is_empty() {
            return Ok(DeviceRequest::Auto);
        }
        if keyword == "cpu" {
            return Ok(DeviceRequest::Cpu);
        }
        if let Some(index) = parse_vulkan_index(&keyword) {
            return Ok(DeviceRequest::Vulkan { index });
        }
        return Err(device_error(
            py,
            &text,
            "it is not one of the accepted spellings",
        ));
    }
    Err(device_error(
        py,
        &request.repr()?.to_string(),
        "a device is named by a string or by a Device object",
    ))
}

/// Resolve a request on this host, or raise `Fn3DeviceError`.
///
/// This is where "never silently fall back" is enforced: the only value that can
/// come out is the one that was asked for.
pub(crate) fn resolve(py: Python<'_>, request: Option<&Bound<'_, PyAny>>) -> PyResult<DeviceId> {
    let request = parse_request(py, request)?;
    request
        .resolve_on_host()
        .map_err(|error| errors::from_device(py, error))
}

/// Register the device surface.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDevice>()?;
    module.add_function(wrap_pyfunction!(devices, module)?)?;
    Ok(())
}
