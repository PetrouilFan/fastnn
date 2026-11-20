//! One operand resolver for every Vulkan kernel (FN3-26 / #72).
//!
//! # Why this exists
//!
//! Before #72 each kernel family had its own staging code and every operand
//! travelled host → device (inputs) or device → host (outputs). Residency makes
//! the *source* of an operand a property of the tensor, not of the kernel, so
//! the kernels ask here and get one of two answers:
//!
//! ```text
//! Operand::Resident { buffer }   bind this device buffer, move nothing
//! Operand::Host { bytes }        upload these host bytes (staged path)
//! ```
//!
//! `OperandMut` is the same for outputs: a resident output is written in place
//! and never read back, a host output is uploaded to a fresh device buffer,
//! dispatched, and copied back — exactly the alpha behaviour.
//!
//! # What is checked before a bind
//!
//! 1. **dtype** — the kernel's declared element type;
//! 2. **contiguity** — every Vulkan kernel indexes a contiguous window;
//! 3. **whole-buffer residency** — a resident operand must start at element 0
//!    and be contiguous, because the shader indexes the bound buffer from zero
//!    (`resident::require_whole_meta`);
//! 4. **no silent re-upload** — a persistent (weight) tensor reaching the host
//!    path is a hard error (`VulkanBackend::check_staged_input`), which is what
//!    makes "zero weight bytes uploaded after prepare" an invariant.
//!
//! A resident tensor is never read through the host byte API: the `Tensor` and
//! `TensorView` accessors return a typed error first, so this module cannot
//! accidentally stage an empty buffer.

use std::sync::Arc;

use super::super::context::Buffer;
use super::super::error::Error;
use super::super::VulkanBackend;
use crate::inference::dtype::DType;
use crate::inference::error::{Fn3Error, Fn3Result};
use crate::inference::ops::OpId;
use crate::inference::tensor::{TensorMut, TensorView};

/// A read-only kernel operand: a device buffer or host bytes.
pub(super) enum Operand<'a> {
    /// A device buffer that already holds the operand.
    Resident {
        /// The bound buffer.
        buffer: Arc<Buffer>,
    },
    /// Host bytes the caller must upload (the staged path).
    Host {
        /// The contiguous window the kernel reads.
        bytes: &'a [u8],
    },
}

/// A writable kernel operand: a device buffer or a host window.
pub(super) enum OperandMut<'a> {
    /// A device buffer the kernel writes in place; nothing is read back.
    Resident {
        /// The bound buffer.
        buffer: Arc<Buffer>,
    },
    /// A host window the kernel's result is copied into.
    Host {
        /// The destination window.
        bytes: &'a mut [u8],
    },
}

/// The contiguous window of a host view, as raw bytes.
pub(super) fn host_window<'a>(view: &TensorView<'a>) -> Fn3Result<&'a [u8]> {
    let start = view.meta().byte_offset()?;
    let len = view.meta().byte_len()?;
    let bytes = view.host_bytes()?;
    let end = start.checked_add(len).ok_or(Fn3Error::ShapeOverflow {
        context: "vulkan kernel input window",
        axis: None,
        detail: format!("byte offset {start} plus length {len} exceeds usize::MAX"),
    })?;
    bytes.get(start..end).ok_or(Fn3Error::OutOfBounds {
        what: "vulkan kernel input window",
        index: end as u64,
        len: bytes.len() as u64,
    })
}

/// The contiguous window of a host output view, as mutable raw bytes.
pub(super) fn host_window_mut<'a>(out: &'a mut TensorMut<'_>) -> Fn3Result<&'a mut [u8]> {
    let start = out.meta().byte_offset()?;
    let len = out.meta().byte_len()?;
    let end = start.checked_add(len).ok_or(Fn3Error::ShapeOverflow {
        context: "vulkan kernel output window",
        axis: None,
        detail: format!("byte offset {start} plus length {len} exceeds usize::MAX"),
    })?;
    let bytes = out.host_bytes_mut()?;
    if end > bytes.len() {
        return Err(Fn3Error::OutOfBounds {
            what: "vulkan kernel output window",
            index: end as u64,
            len: bytes.len() as u64,
        });
    }
    Ok(&mut bytes[start..end])
}

/// Require a contiguous row-major operand, naming the op, the view and its owner.
fn require_contiguous(
    op: OpId,
    view: &TensorView<'_>,
    what: &'static str,
    owner: &'static str,
) -> Fn3Result<()> {
    if view.is_contiguous() {
        return Ok(());
    }
    Err(Fn3Error::UnsupportedOp {
        op,
        backend: super::super::VULKAN_BACKEND_NAME,
        reason: format!(
            "the Vulkan kernels execute contiguous row-major windows only and \
             `supports_strided_views` is false for this backend; the {what} is {} with strides {}. \
             Materialise it with `TensorView::to_contiguous` before executing (owner: {owner})",
            view.meta(),
            view.strides()
        ),
    })
}

/// Resolve one read-only operand of `dtype`.
pub(super) fn input<'a>(
    owner: &'static str,
    backend: &VulkanBackend,
    op: OpId,
    name: &'static str,
    view: &TensorView<'a>,
    dtype: DType,
) -> Fn3Result<Operand<'a>> {
    view.meta().require_dtype(Some(op), dtype)?;
    require_contiguous(op, view, name, owner)?;
    // Hold the gate across lookup and check so a concurrent prepare cannot
    // publish this operand between the two and turn it into a late upload.
    let _gate = backend.prepare_gate_read();
    if let Some(buffer) = backend.resident_input(view)? {
        let bytes = view.meta().byte_len()? as u64;
        if bytes == 0 {
            return Err(Error::internal(format!(
                "op {op}: {name} is device-resident but its logical length is zero; handle \
                 zero-element operands before binding"
            ))
            .to_fn3());
        }
        let _ = bytes;
        return Ok(Operand::Resident { buffer });
    }
    let bytes = host_window(view)?;
    backend.check_staged_input(Some(view), bytes.len() as u64)?;
    Ok(Operand::Host { bytes })
}

/// Resolve one writable operand of `dtype`.
pub(super) fn output<'a>(
    owner: &'static str,
    backend: &VulkanBackend,
    op: OpId,
    name: &'static str,
    out: &'a mut TensorMut<'_>,
    dtype: DType,
) -> Fn3Result<OperandMut<'a>> {
    out.meta().require_dtype(Some(op), dtype)?;
    if !out.is_contiguous() {
        return Err(Fn3Error::UnsupportedOp {
            op,
            backend: super::super::VULKAN_BACKEND_NAME,
            reason: format!(
                "the Vulkan kernels write contiguous row-major windows only; the {name} output is \
                 {} with strides {} (owner: {owner})",
                out.meta(),
                out.strides()
            ),
        });
    }
    if let Some(buffer) = backend.resident_output(out)? {
        let bytes = out.meta().byte_len()? as u64;
        let _ = bytes;
        return Ok(OperandMut::Resident { buffer });
    }
    let bytes = host_window_mut(out)?;
    Ok(OperandMut::Host { bytes })
}
