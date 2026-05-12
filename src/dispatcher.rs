//! Runtime dispatch table (v1.x backward compatibility).
//!
//! In v2.0.0 the dispatch table is replaced by the AOT compiler
//! (`ComputeGraph` → compiler passes → `Backend::compile` → `Backend::dispatch`).
//! This module is kept so that v1.x nn/ and tensor/ code compiles during
//! the migration.  New code should use the v2.0 pipeline directly.

#![allow(dead_code)]

use crate::error::{FastnnError, FastnnResult};
use crate::storage::Device;
use crate::tensor::Tensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchKey {
    Cpu,
    Wgpu,
}

pub fn device_to_dispatch_key(device: Device) -> DispatchKey {
    match device {
        Device::Cpu => DispatchKey::Cpu,
        Device::Wgpu(_) => DispatchKey::Wgpu,
    }
}

/// A kernel takes tensor arguments and returns either a result vector or an error.
/// Previously returned `Vec<Tensor>` and panicked on failure.
/// A kernel takes tensor arguments and returns either a result vector or an error.
/// Previously returned `Vec<Tensor>` and panicked on failure.
///
/// Uses `Box<dyn Fn>` so both safe function pointers and wrapping closures
/// can be stored in the dispatch table.
pub type KernelFn = Box<dyn Fn(&[&Tensor]) -> FastnnResult<Vec<Tensor>> + Send + Sync>;

struct DispatcherInner {
    ops: HashMap<(&'static str, DispatchKey), KernelFn>,
}

impl DispatcherInner {
    fn new() -> Self {
        Self {
            ops: HashMap::new(),
        }
    }
}

static DISPATCHER: OnceLock<RwLock<DispatcherInner>> = OnceLock::new();

fn get_dispatcher() -> &'static RwLock<DispatcherInner> {
    DISPATCHER.get_or_init(|| RwLock::new(DispatcherInner::new()))
}

/// Register an infallible kernel (returns `Vec<Tensor>`).
/// The kernel is automatically wrapped in `Ok(...)`.
///
/// Most kernels are infallible (they return `Vec<Tensor>` and don't have
/// error paths).  For fallible kernels use [`register_fallible`].
pub fn register(op: &'static str, key: DispatchKey, kernel: unsafe fn(&[&Tensor]) -> Vec<Tensor>) {
    let dispatcher = get_dispatcher();
    let mut guard = dispatcher.write();
    guard.ops.insert((op, key), Box::new(move |args| Ok(unsafe { kernel(args) })));
}

/// Register a fallible kernel (returns `FastnnResult<Vec<Tensor>>`).
/// Use this for kernels that have proper error paths instead of panicking.
pub fn register_fallible(op: &'static str, key: DispatchKey, kernel: fn(&[&Tensor]) -> FastnnResult<Vec<Tensor>>) {
    let dispatcher = get_dispatcher();
    let mut guard = dispatcher.write();
    guard.ops.insert((op, key), Box::new(kernel));
}

pub fn try_dispatch(op: &str, key: DispatchKey, args: &[&Tensor]) -> FastnnResult<Vec<Tensor>> {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();

    let kernel = guard.ops.get(&(op, key)).ok_or_else(|| {
        FastnnError::Computation(format!(
            "No kernel registered for op '{}' with key {:?}",
            op, key
        ))
    })?;

    kernel(args)
}

pub fn dispatch(op: &str, key: DispatchKey, args: &[&Tensor]) -> FastnnResult<Vec<Tensor>> {
    try_dispatch(op, key, args)
}

pub fn list_registered_ops() -> Vec<String> {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();

    let mut ops: Vec<String> = guard.ops.keys().map(|(name, _)| name.to_string()).collect();
    ops.sort();
    ops.dedup();
    ops
}

#[allow(dead_code)]
pub fn is_registered(op: &str, key: DispatchKey) -> bool {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();
    guard.ops.contains_key(&(op, key))
}
