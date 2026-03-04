use crate::tensor::Tensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DispatchKey {
    Cpu,
    // Autograd,     // TODO(gpu): add autograd dispatch
    // AutocastCpu,  // TODO(gpu): add mixed precision dispatch
}

pub type KernelFn = fn(&[&Tensor]) -> Vec<Tensor>;

struct DispatcherInner {
    ops: HashMap<(String, DispatchKey), KernelFn>,
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
    guard.ops.insert((op.to_string(), key), kernel);
}

pub fn dispatch(op: &str, key: DispatchKey, args: &[&Tensor]) -> Vec<Tensor> {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();

    let kernel = guard.ops.get(&(op.to_string(), key)).expect(&format!(
        "No kernel registered for op '{}' with key {:?}",
        op, key
    ));

    kernel(args)
}

pub fn list_registered_ops() -> Vec<String> {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();

    let mut ops: Vec<String> = guard.ops.keys().map(|(name, _)| name.clone()).collect();
    ops.sort();
    ops.dedup();
    ops
}

pub fn is_registered(op: &str, key: DispatchKey) -> bool {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();
    guard.ops.contains_key(&(op.to_string(), key))
}
