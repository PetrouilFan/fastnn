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

// Kernel ID enum for fast dispatch (avoids string hashing)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum KernelId {
    Conv2d = 0,
    Conv1d = 1,
    Conv3d = 2,
    ConvTranspose2d = 3,
    MaxPool2d = 4,
    AvgPool2d = 5,
    BatchNorm2d = 6,
    ReLU = 7,
    GELU = 8,
    SiLU = 9,
    Sigmoid = 10,
    Tanh = 11,
    LeakyReLU = 12,
    Elu = 13,
    Softmax = 14,
    LogSoftmax = 15,
    MatMul = 16,
    Add = 17,
    Sub = 18,
    Mul = 19,
    Div = 20,
    Reshape = 21,
    Transpose = 22,
    Permute = 23,
    Cat = 24,
    Slice = 25,
    // Add more as needed
}

pub type KernelFn = fn(&[&Tensor]) -> Vec<Tensor>;

struct DispatcherInner {
    // Fast path: direct array lookup by KernelId (no hashmap)
    kernel_table: [Option<KernelFn>; 256],
    // Slow path: fallback for dynamic ops
    dynamic_ops: HashMap<(&'static str, DispatchKey), KernelFn>,
}

impl DispatcherInner {
    fn new() -> Self {
        Self {
            kernel_table: [None; 256],
            dynamic_ops: HashMap::new(),
        }
    }
    
    fn register_static(&mut self, id: KernelId, kernel: KernelFn) {
        self.kernel_table[id as usize] = Some(kernel);
    }
    
    fn register_dynamic(&mut self, op: &'static str, key: DispatchKey, kernel: KernelFn) {
        self.dynamic_ops.insert((op, key), kernel);
    }
    
    fn dispatch(&self, id: KernelId, key: DispatchKey, args: &[&Tensor]) -> Vec<Tensor> {
        // Fast path: direct array lookup
        if let Some(kernel) = self.kernel_table[id as usize] {
            return kernel(args);
        }
        // Slow path: dynamic lookup (shouldn't happen for registered static ops)
        self.dispatch_dynamic("", key, args)
    }
    
    fn dispatch_dynamic(&self, op: &str, key: DispatchKey, args: &[&Tensor]) -> Vec<Tensor> {
        let kernel = self.dynamic_ops
            .get(&(op, key))
            .unwrap_or_else(|| panic!("No kernel registered for op '{}' with key {:?}", op, key));
        kernel(args)
    }
}

pub fn device_to_dispatch_key(device: Device) -> DispatchKey {
    match device {
        Device::Cpu => DispatchKey::Cpu,
        Device::Wgpu(_) => DispatchKey::Wgpu,
    }
}

static DISPATCHER: OnceLock<RwLock<DispatcherInner>> = OnceLock::new();

pub fn register_static(op_id: KernelId, key: DispatchKey, kernel: KernelFn) {
    let dispatcher = get_dispatcher();
    let mut guard = dispatcher.write();
    if key == DispatchKey::Cpu {
        guard.register_static(op_id, kernel);
    } else {
        let name = match op_id {
            KernelId::Conv2d => "conv2d",
            KernelId::ReLU => "relu",
            _ => "unknown",
        };
        guard.register_dynamic(name, key, kernel);
    }
}

fn get_dispatcher() -> &'static RwLock<DispatcherInner> {
    DISPATCHER.get_or_init(|| RwLock::new(DispatcherInner::new()))
}

pub fn register(op: &'static str, key: DispatchKey, kernel: KernelFn) {
    let dispatcher = get_dispatcher();
    let mut guard = dispatcher.write();
    guard.dynamic_ops.insert((op, key), kernel);
}

pub fn try_dispatch(op: &str, key: DispatchKey, args: &[&Tensor]) -> Result<Vec<Tensor>, String> {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();

    let kernel = guard
        .dynamic_ops
        .get(&(op, key))
        .ok_or_else(|| format!("No kernel registered for op '{}' with key {:?}", op, key))?;

    Ok(kernel(args))
}

pub fn dispatch(op: &str, key: DispatchKey, args: &[&Tensor]) -> Vec<Tensor> {
    try_dispatch(op, key, args).expect("dispatch failed")
}

pub fn list_registered_ops() -> Vec<String> {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();

    let mut ops: Vec<String> = guard.dynamic_ops.keys().map(|(name, _)| name.to_string()).collect();
    ops.sort();
    ops.dedup();
    ops
}

#[allow(dead_code)]
pub fn is_registered(op: &str, key: DispatchKey) -> bool {
    let dispatcher = get_dispatcher();
    let guard = dispatcher.read();
    guard.dynamic_ops.contains_key(&(op, key))
}
