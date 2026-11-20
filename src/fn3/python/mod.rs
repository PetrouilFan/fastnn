//! The 3.0 inference surface for Python (FN3-18 / #64).
//!
//! # What this module is
//!
//! `fastnn.inference` is the inference API that survives the 2.x removal (#65): one
//! versioned artifact, one loaded model, an explicit device, and sessions that
//! own their cache. It is a thin adaptation of `crate::inference`, not a second
//! implementation: every number a Python caller sees is produced by the same
//! `Session::prefill` / `Session::decode` the CLI and the Rust tests call.
//!
//! ```text
//! prepare(onnx, artifact)          import ──► PreparedArtifact ──► file
//! inspect(artifact)                file ──► geometry / size / checksums
//! load(artifact, source=…, device=…)   artifact + source ──► Model
//! Model.session()                  one independent KV cache + plan cache
//! Session.prefill(tokens)          logits [batch, 1, vocab]
//! Session.decode(tokens)           logits [batch, 1, vocab], cache grows by n
//! ```
//!
//! # Where the code lives
//!
//! `src/python/mod.rs` declares this file with `#[path]` and calls
//! [`register`], so the binding is compiled only under the existing `python`
//! feature and `src/fn3/` itself stays free of PyO3. The module is registered as
//! the submodule `fastnn._core.inference` (and in `sys.modules` under that name, so
//! `from fastnn._core.inference import load` works); the user-facing package is
//! `fastnn/inference/`.
//!
//! # Contracts this module keeps
//!
//! * **No silent device fallback.** The requested device is resolved before any
//!   file is read, and only `"auto"` may choose.
//! * **Typed errors.** Every core failure is mapped to a class a caller can
//!   branch on (see `errors`), and no panic is allowed to cross the FFI
//!   boundary: the binding contains no `unwrap`/`expect`/index outside checked
//!   arithmetic on user input.
//! * **Copies are explicit.** Logits leave the session as one immutable `bytes`
//!   copy; NumPy views that copy instead of copying it again. A task result can
//!   therefore outlive the session.
//! * **The GIL is released** around prepare, load, prefill and decode. Python
//!   objects are only touched with the GIL held.

mod device;
mod errors;
mod model;
mod session;

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Module docstring for `fastnn._core.inference`.
const DOC: &str = "\
FastNN 3.0 inference primitives.

Use the `fastnn.inference` package rather than this module: it adds path handling,
docstrings and re-exports these objects. Everything here is the same Rust
execution path the CLI uses.

prepare(onnx, artifact, ...) -> PrepareReport
inspect(artifact) -> ArtifactInfo
load(artifact, source=..., device=..., capacity=..., weight_int8=...) -> Model
devices() -> list[Device]
";

/// Attach the `inference` submodule to `_core` and register it for `import`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let module = PyModule::new(py, "inference")?;
    module.add("__doc__", DOC)?;

    errors::register(&module)?;
    device::register(&module)?;
    model::register(&module)?;
    session::register(&module)?;

    let mut names: Vec<&str> = vec![
        "prepare",
        "inspect",
        "load",
        "devices",
        "Device",
        "Geometry",
        "Envelope",
        "Metadata",
        "PrepareReport",
        "ArtifactInfo",
        "Model",
        "Session",
        "SessionStats",
        "WeightInt8Report",
        "WeightInt8Route",
        "Logits",
    ];
    names.extend_from_slice(errors::NAMES);
    module.add("__all__", names)?;

    parent.add_submodule(&module)?;
    // `add_submodule` only sets the attribute; a dotted import also needs the
    // entry in `sys.modules` because `_core` is not a package.
    let qualified = format!("{}.inference", parent.name()?);
    py.import("sys")?
        .getattr("modules")?
        .set_item(qualified, &module)?;
    Ok(())
}
