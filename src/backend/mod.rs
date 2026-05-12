#![allow(dead_code)]

use crate::ir::node::{ComputeGraph, DimExpr, ShapeEnv};
use std::fmt;

#[derive(Debug)]
pub enum BackendError {
    Compilation(String),
    Dispatch(String),
    Memory(String),
    UnsupportedOp(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::Compilation(msg) => write!(f, "Compilation error: {msg}"),
            BackendError::Dispatch(msg) => write!(f, "Dispatch error: {msg}"),
            BackendError::Memory(msg) => write!(f, "Memory error: {msg}"),
            BackendError::UnsupportedOp(msg) => write!(f, "Unsupported operation: {msg}"),
        }
    }
}

impl std::error::Error for BackendError {}

#[derive(Debug, Clone, Copy)]
pub struct BufferSlice {
    pub offset: usize,
    pub size: usize,
}

impl BufferSlice {
    pub fn new(offset: usize, size: usize) -> Self {
        BufferSlice { offset, size }
    }
}

#[derive(Debug, Clone)]
pub enum Instruction {
    CallKernel {
        kernel_name: String,
        input_slices: Vec<BufferSlice>,
        output_slice: BufferSlice,
        /// Optional shape/dimension parameters for kernels that need them
        /// (e.g., matmul stores [M, K, N], transpose stores [M, N]).
        params: Vec<usize>,
        /// Symbolic dim expressions corresponding to `params`.
        /// When `Some`, the backend can re-evaluate these at dispatch time
        /// using a [`ShapeEnv`](crate::ir::node::ShapeEnv) to resolve
        /// any [`DimExpr::Symbol`](crate::ir::node::DimExpr::Symbol) dims
        /// that were not known at compile time.
        param_dims: Option<Vec<DimExpr>>,
    },
    MemCopy {
        dst: BufferSlice,
        src: BufferSlice,
    },
    Fill {
        dst: BufferSlice,
        value: f32,
    },
}

pub struct ExecutablePlan {
    pub instructions: Vec<Instruction>,
    pub arena_size: usize,
}

/// AOT graph executor (ties together IR → compiler passes → backend)
pub mod executor;

/// CPU backend implementation (inline f32 kernels)
pub mod cpu;

/// Re-export the MemoryPlan from the compiler passes module
pub use crate::compiler::passes::memory_planning::MemoryPlan;

pub trait Backend {
    type Buffer;

    fn name(&self) -> &str;

    fn allocate_arena(&self, total_bytes: usize) -> Self::Buffer;

    fn compile(
        &self,
        graph: &ComputeGraph,
        memory_plan: &MemoryPlan,
    ) -> Result<ExecutablePlan, BackendError>;

    /// Execute a compiled plan against the memory arena.
    ///
    /// `shape_env` carries runtime concrete values for symbolic dimension names
    /// (e.g. batch size "N"), allowing kernels to resolve their dimension-dependent
    /// parameters at dispatch time.  The environment is built from input byte sizes
    /// in [`GraphExecutor::execute`](crate::backend::executor::GraphExecutor::execute).
    fn dispatch(
        &self,
        plan: &ExecutablePlan,
        arena: &Self::Buffer,
        shape_env: &ShapeEnv,
    ) -> Result<(), BackendError>;

    /// Write `data` into the arena at byte `offset`.
    /// Used by the executor to populate graph inputs.
    /// Default implementation panics — backends that support host-visible
    /// buffers should override this.
    fn write_arena(&self, _arena: &Self::Buffer, _offset: usize, _data: &[u8]) {
        panic!("write_arena not implemented for backend '{}'", self.name());
    }

    /// Read `size` bytes from the arena starting at `offset`.
    /// Used by the executor to extract graph outputs.
    fn read_arena(&self, _arena: &Self::Buffer, _offset: usize, _size: usize) -> Vec<u8> {
        panic!("read_arena not implemented for backend '{}'", self.name());
    }
}
