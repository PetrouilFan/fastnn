use crate::ir::{ComputeGraph, DimExpr, NodeId, ShapeEnv};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
/// Carries per-channel scales, dequantization offsets, shape, and bit-width
/// so the backend dispatch can construct a [`PackedTensor`] for SIMD
/// or GPU compute shaders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedWeightMeta {
    /// Bit width of the packed type: 4 or 8.
    pub bit_width: usize,
    /// Per-output-channel scale factors.
    pub scales: Vec<f32>,
    /// Per-output-channel floating offsets in `real = q * scale + offset`.
    pub dequant_offsets: Vec<f32>,
    /// Logical shape of the weight tensor.
    pub shape: Vec<usize>,
    /// Per-element quantization block size within each row. 0 = per-channel.
    pub quant_block_size: usize,
    /// Per-block codebooks for I4 codebook quantization (16 centroids per block, normalized to [-1, 1]).
    /// Empty when not using codebook quantization.
    pub codebooks: Vec<[f32; 16]>,
}

impl QuantizedWeightMeta {
    fn validate(&self, instruction_index: usize) -> Result<(), BackendError> {
        if !matches!(self.bit_width, 4 | 8) {
            return Err(BackendError::Dispatch(format!(
                "instruction {instruction_index} has unsupported quantized weight bit width {}",
                self.bit_width
            )));
        }
        let Some((&rows, inner_shape)) = self.shape.split_first() else {
            return Err(BackendError::Dispatch(format!(
                "instruction {instruction_index} quantized weight shape must not be empty"
            )));
        };
        let columns = inner_shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension).ok_or_else(|| {
                BackendError::Dispatch(format!(
                    "instruction {instruction_index} quantized weight shape overflows"
                ))
            })
        })?;
        let blocks_per_row = if self.quant_block_size == 0 {
            1
        } else {
            columns.div_ceil(self.quant_block_size)
        };
        let metadata_len = rows.checked_mul(blocks_per_row).ok_or_else(|| {
            BackendError::Dispatch(format!(
                "instruction {instruction_index} quantized metadata size overflows"
            ))
        })?;
        let offsets_valid =
            self.dequant_offsets.len() == metadata_len || self.dequant_offsets.is_empty();
        if self.scales.len() != metadata_len || !offsets_valid {
            return Err(BackendError::Dispatch(format!(
                "instruction {instruction_index} quantized metadata length mismatch: expected {metadata_len}, scales {}, offsets {}",
                self.scales.len(), self.dequant_offsets.len()
            )));
        }
        if self
            .scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
            || self
                .dequant_offsets
                .iter()
                .any(|offset| !offset.is_finite())
        {
            return Err(BackendError::Dispatch(format!(
                "instruction {instruction_index} quantized metadata contains invalid affine values"
            )));
        }
        if !self.codebooks.is_empty() {
            if self.bit_width != 4
                || !(self.codebooks.len() == 1 || self.codebooks.len() == metadata_len)
            {
                return Err(BackendError::Dispatch(format!(
                    "instruction {instruction_index} quantized codebook metadata is inconsistent"
                )));
            }
            if self
                .codebooks
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
            {
                return Err(BackendError::Dispatch(format!(
                    "instruction {instruction_index} quantized codebook contains non-finite values"
                )));
            }
        }
        Ok(())
    }
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
        /// using a [`ShapeEnv`](crate::ir::ShapeEnv) to resolve
        /// any [`DimExpr::Symbol`](crate::ir::DimExpr::Symbol) dims
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
    /// Used for [`Opcode::Constant`](crate::ir::Opcode::Constant) nodes
    /// that carry [`TensorValue::Data`](crate::ir::TensorValue::Data).
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
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        if self.levels.windows(2).any(|levels| levels[0] > levels[1]) {
            return Err(BackendError::Dispatch(
                "executable plan scheduling levels are not monotonic".into(),
            ));
        }

        let validate_slice = |slice: BufferSlice, instruction_index: usize, label: &str| {
            let end = slice.offset.checked_add(slice.size).ok_or_else(|| {
                BackendError::Dispatch(format!(
                    "instruction {instruction_index} {label} range overflows"
                ))
            })?;
            if end > self.arena_size {
                return Err(BackendError::Dispatch(format!(
                    "instruction {instruction_index} {label} range {}..{end} exceeds arena size {}",
                    slice.offset, self.arena_size
                )));
            }
            Ok(())
        };

        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            match instruction {
                Instruction::CallKernel {
                    input_slices,
                    output_slice,
                    secondary_output_slice,
                    weight_meta,
                    ..
                } => {
                    for (input_index, &slice) in input_slices.iter().enumerate() {
                        validate_slice(slice, instruction_index, &format!("input {input_index}"))?;
                    }
                    validate_slice(*output_slice, instruction_index, "output")?;
                    if let Some(slice) = secondary_output_slice {
                        validate_slice(*slice, instruction_index, "secondary output")?;
                    }
                    if let Some(weight_meta) = weight_meta {
                        weight_meta.validate(instruction_index)?;
                    }
                }
                Instruction::MemCopy { dst, src } => {
                    validate_slice(*dst, instruction_index, "memcopy destination")?;
                    validate_slice(*src, instruction_index, "memcopy source")?;
                    if dst.size != src.size {
                        return Err(BackendError::Dispatch(format!(
                            "instruction {instruction_index} memcopy size mismatch: destination {}, source {}",
                            dst.size, src.size
                        )));
                    }
                }
                Instruction::Fill { dst, .. } => {
                    validate_slice(*dst, instruction_index, "fill destination")?;
                }
                Instruction::WriteConst { dst, data } => {
                    validate_slice(*dst, instruction_index, "constant destination")?;
                    let data_end = dst.offset.checked_add(data.len()).ok_or_else(|| {
                        BackendError::Dispatch(format!(
                            "instruction {instruction_index} constant data range overflows"
                        ))
                    })?;
                    if data_end > self.arena_size {
                        return Err(BackendError::Dispatch(format!(
                            "instruction {instruction_index} constant data range {}..{data_end} exceeds arena size {}",
                            dst.offset, self.arena_size
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod executable_plan_validation_tests {
    use super::*;

    #[test]
    fn rejects_malformed_quantized_weight_metadata() {
        let invalid = QuantizedWeightMeta {
            bit_width: 4,
            scales: vec![1.0],
            dequant_offsets: vec![],
            shape: vec![2, 4],
            quant_block_size: 0,
            codebooks: vec![],
        };
        assert!(invalid.validate(0).is_err());

        let non_finite = QuantizedWeightMeta {
            bit_width: 8,
            scales: vec![f32::NAN],
            dequant_offsets: vec![0.0],
            shape: vec![1, 4],
            quant_block_size: 0,
            codebooks: vec![],
        };
        assert!(non_finite.validate(0).is_err());
    }

    #[test]
    fn rejects_malformed_executable_plan_metadata() {
        let nonmonotonic_levels = ExecutablePlan {
            instructions: vec![],
            arena_size: 4,
            levels: vec![1, 0],
        };
        assert!(nonmonotonic_levels.validate().is_err());

        let out_of_bounds = ExecutablePlan {
            instructions: vec![Instruction::WriteConst {
                dst: BufferSlice::new(usize::MAX, 4),
                data: vec![0; 4],
            }],
            arena_size: 4,
            levels: vec![0],
        };
        assert!(out_of_bounds.validate().is_err());

        let mismatched_copy = ExecutablePlan {
            instructions: vec![Instruction::MemCopy {
                dst: BufferSlice::new(0, 4),
                src: BufferSlice::new(4, 8),
            }],
            arena_size: 12,
            levels: vec![0],
        };
        assert!(mismatched_copy.validate().is_err());
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

use crate::compiler::MemoryPlan;

pub trait Backend {
    type Buffer;

    fn name(&self) -> &str;

    fn allocate_arena(&self, total_bytes: usize) -> Self::Buffer;

    fn try_allocate_arena(&self, total_bytes: usize) -> Result<Self::Buffer, BackendError> {
        Ok(self.allocate_arena(total_bytes))
    }

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
    /// Backends must define their host-transfer behavior explicitly.
    fn write_arena(&self, arena: &Self::Buffer, offset: usize, data: &[u8]);

    /// Read `size` bytes from the arena starting at `offset`.
    /// Used by the executor to extract graph outputs.
    fn read_arena(&self, arena: &Self::Buffer, offset: usize, size: usize) -> Vec<u8>;
}
