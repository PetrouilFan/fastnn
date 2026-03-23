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

pub type KernelFn = fn(&[&Tensor]) -> Vec<Tensor>;

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

pub fn register(op: &'static str, key: DispatchKey, kernel: KernelFn) {
    let dispatcher = get_dispatcher();
    let mut guard = dispatcher.write();
    guard.ops.insert((op, key), kernel);
}

pub fn dispatch(op: &str, key: DispatchKey, args: &[&Tensor]) -> Vec<Tensor> {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();

    // Use &str directly as key - no allocation needed since HashMap stores &'static str
    // and we look up by borrowed &str which is compatible via Borrow
    #[allow(clippy::expect_fun_call)]
    let kernel = guard.ops.get(&(op, key)).expect(&format!(
        "No kernel registered for op '{}' with key {:?}",
        op, key
    ));

    kernel(args)
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
