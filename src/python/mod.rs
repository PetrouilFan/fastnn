//! Python bindings for the inference API.
//!
//! The extension module is the only thing this crate exposes to Python. Its
//! contents are registered by [`inference`], which owns the session, model and
//! tensor classes; there is no eager-op or legacy graph surface here.

use pyo3::prelude::*;

#[path = "../fn3/python/mod.rs"]
pub mod inference;

/// `fastnn._core` — the compiled part of the `fastnn` package.
#[pymodule]
pub fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    inference::register(module)?;
    Ok(())
}
