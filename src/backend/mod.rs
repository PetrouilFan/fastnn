use crate::ir::node::{ComputeGraph, DimExpr, NodeId, ShapeEnv};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BufferSlice {
    pub offset: usize,
    pub size: usize,
}

impl BufferSlice {
    pub fn new(offset: usize, size: usize) -> Self {
        BufferSlice { offset, size }
    }
}

/// Quantized weight metadata for packed precision kernels.
/// Carries per-channel scales, zero-points, shape, and bit-width
/// so the backend dispatch can construct a [`PackedTensor`] for SIMD
/// or GPU compute shaders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedWeightMeta {
    /// Bit width of the packed type: 4 or 8.
    pub bit_width: usize,
    /// Per-output-channel scale factors.
    pub scales: Vec<f32>,
    /// Per-output-channel zero points.
    pub zero_points: Vec<f32>,
    /// Logical shape of the weight tensor.
    pub shape: Vec<usize>,
    /// Per-element quantization block size within each row. 0 = per-channel.
    pub quant_block_size: usize,
    /// Per-block codebooks for I4 codebook quantization (16 centroids per block, normalized to [-1, 1]).
    /// Empty when not using codebook quantization.
    pub codebooks: Vec<[f32; 16]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Instruction {
    CallKernel {
        kernel_name: String,
        input_slices: Vec<BufferSlice>,
        output_slice: BufferSlice,
        /// Optional secondary output slice for multi-output kernels (e.g. MaxPool indices).
        secondary_output_slice: Option<BufferSlice>,
        /// Optional shape/dimension parameters for kernels that need them
        /// (e.g., matmul stores [M, K, N], transpose stores [M, N]).
        params: Vec<usize>,
        /// Node ID in the compute graph (used for tightening params at runtime).
        node_id: Option<NodeId>,
        /// Symbolic dim expressions corresponding to `params`.
        /// When `Some`, the backend can re-evaluate these at dispatch time
        /// using a [`ShapeEnv`](crate::ir::node::ShapeEnv) to resolve
        /// any [`DimExpr::Symbol`](crate::ir::node::DimExpr::Symbol) dims
        /// that were not known at compile time.
        param_dims: Option<Vec<DimExpr>>,
        /// Optional weight metadata for quantized matmul kernels.
        weight_meta: Option<Arc<QuantizedWeightMeta>>,
    },
    MemCopy {
        dst: BufferSlice,
        src: BufferSlice,
    },
    Fill {
        dst: BufferSlice,
        value: f32,
    },
    /// Materialise arbitrary byte payloads into the arena at dispatch time.
    /// Used for [`Opcode::Constant`](crate::ir::node::Opcode::Constant) nodes
    /// that carry [`TensorValue::Data`](crate::ir::node::TensorValue::Data).
    WriteConst {
        dst: BufferSlice,
        data: Vec<u8>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutablePlan {
    pub instructions: Vec<Instruction>,
    pub arena_size: usize,
    /// Topological level for each instruction (same index as `instructions`).
    /// Instructions with the same level have no data dependencies and can
    /// execute in parallel. Levels increase monotonically.
    pub levels: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEntry {
    pub instruction_index: usize,
    pub node_id: Option<NodeId>,
    pub kernel_name: String,
    pub elapsed_ns: u128,
}

impl ExecutablePlan {
    /// Save the plan to a binary file using bincode serialization.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a plan from a binary file created by [`save`](Self::save).
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let plan: ExecutablePlan = bincode::deserialize(&bytes)?;
        Ok(plan)
    }
}

/// AOT graph executor (ties together IR → compiler passes → backend)
pub mod executor;

/// CPU backend implementation (inline f32 kernels)
pub mod cpu;

/// Wgpu (GPU) backend implementation (WGSL compute shaders)
#[cfg(feature = "gpu")]
pub mod wgpu;

/// Prepared executable plan for AOT inference compilation.
pub mod prepared;

/// Standalone runtime for executing pre-compiled plans
pub mod runtime;

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

    fn dispatch_profile(
        &self,
        plan: &ExecutablePlan,
        arena: &Self::Buffer,
        shape_env: &ShapeEnv,
    ) -> Result<Vec<ProfileEntry>, BackendError> {
        let mut entries = Vec::with_capacity(plan.instructions.len());
        for (instruction_index, instruction) in plan.instructions.iter().cloned().enumerate() {
            let (node_id, kernel_name) = match &instruction {
                Instruction::CallKernel {
                    node_id,
                    kernel_name,
                    ..
                } => (*node_id, kernel_name.clone()),
                Instruction::MemCopy { .. } => (None, "memcopy".to_string()),
                Instruction::Fill { .. } => (None, "fill".to_string()),
                Instruction::WriteConst { .. } => (None, "write_const".to_string()),
            };
            let single = ExecutablePlan {
                instructions: vec![instruction],
                arena_size: plan.arena_size,
                levels: vec![0],
            };
            let start = std::time::Instant::now();
            self.dispatch(&single, arena, shape_env)?;
            entries.push(ProfileEntry {
                instruction_index,
                node_id,
                kernel_name,
                elapsed_ns: start.elapsed().as_nanos(),
            });
        }
        Ok(entries)
    }

    /// Execute a compiled plan with an optional persistent immutable
    /// weight view.
    ///
    /// `persistent_view` is the opt-in no-copy path consumed by
    /// [`crate::backend::prepared::PersistentPreparedWeights`]: when
    /// a `(offset, size)` slot is present in the view, the dispatch
    /// loop reads the weight bytes directly from the persistent
    /// payload instead of from the mutable arena.  When the view is
    /// `None` (or empty), the behaviour is identical to
    /// [`Self::dispatch`].
    ///
    /// Default implementation: ignore the view and call
    /// [`Self::dispatch`]. Backends that support the no-copy path
    /// (currently only `CpuBackend` with the `prepared-plan` cargo
    /// feature) override this to consult the view at the relevant
    /// `CallKernel` sites.
    fn dispatch_with_persistent_view(
        &self,
        plan: &ExecutablePlan,
        arena: &Self::Buffer,
        shape_env: &ShapeEnv,
        #[cfg_attr(not(feature = "prepared-plan"), allow(unused_variables))]
        persistent_view: Option<&crate::backend::prepared::PersistentPreparedWeights>,
    ) -> Result<(), BackendError> {
        let _ = persistent_view;
        self.dispatch(plan, arena, shape_env)
    }

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
