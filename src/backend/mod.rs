use crate::ir::{ComputeGraph, DimExpr, NodeId, ShapeEnv};
use crate::types::ScalarType;
use bincode::Options;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiasDomain {
    None,
    FloatOutput,
    I32Accumulator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ZeroPointCorrection {
    None,
    AffineInputAndWeight,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequantizationMode {
    None,
    FixedPoint {
        multiplier: i32,
        right_shift: u8,
        output_zero_point: i32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoundingMode {
    NotApplicable,
    NearestTiesToEven,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SaturationMode {
    None,
    Clamp { min: i32, max: i32 },
}

/// Numerical semantics selected for a quantized kernel invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedExecutionContract {
    pub activation_storage: ScalarType,
    pub weight_storage: ScalarType,
    pub accumulator: ScalarType,
    pub bias_domain: BiasDomain,
    pub zero_point_correction: ZeroPointCorrection,
    pub requantization: RequantizationMode,
    pub rounding: RoundingMode,
    pub saturation: SaturationMode,
    pub output_storage: ScalarType,
}

impl QuantizedExecutionContract {
    pub fn fixed_point_output(
        activation_storage: ScalarType,
        weight_storage: ScalarType,
        bias_domain: BiasDomain,
        multiplier: i32,
        right_shift: u8,
        output_zero_point: i32,
        output_storage: ScalarType,
    ) -> Self {
        let (min, max) = match output_storage {
            ScalarType::I8 => (i8::MIN as i32, i8::MAX as i32),
            ScalarType::U8 => (u8::MIN as i32, u8::MAX as i32),
            _ => (i32::MIN, i32::MAX),
        };
        Self {
            activation_storage,
            weight_storage,
            accumulator: ScalarType::I32,
            bias_domain,
            zero_point_correction: ZeroPointCorrection::AffineInputAndWeight,
            requantization: RequantizationMode::FixedPoint {
                multiplier,
                right_shift,
                output_zero_point,
            },
            rounding: RoundingMode::NearestTiesToEven,
            saturation: SaturationMode::Clamp { min, max },
            output_storage,
        }
    }

    pub fn requantize_i32(&self, accumulator: i32) -> Result<i32, BackendError> {
        let RequantizationMode::FixedPoint {
            multiplier,
            right_shift,
            output_zero_point,
        } = self.requantization
        else {
            return Err(BackendError::Dispatch(
                "quantized execution contract has no integer requantization".into(),
            ));
        };
        let SaturationMode::Clamp { min, max } = self.saturation else {
            return Err(BackendError::Dispatch(
                "integer requantization requires an explicit saturation range".into(),
            ));
        };
        if !matches!(self.rounding, RoundingMode::NearestTiesToEven) {
            return Err(BackendError::Dispatch(
                "integer requantization requires an explicit rounding mode".into(),
            ));
        }
        if multiplier <= 0 || right_shift > 31 || min > max {
            return Err(BackendError::Dispatch(
                "integer requantization parameters are invalid".into(),
            ));
        }

        let divisor = 1i64 << right_shift;
        let product = i64::from(accumulator) * i64::from(multiplier);
        let quotient = product.div_euclid(divisor);
        let remainder = product.rem_euclid(divisor);
        let half = divisor / 2;
        let rounded = if remainder > half || (remainder == half && quotient & 1 != 0) {
            quotient + 1
        } else {
            quotient
        };
        let shifted = rounded
            .checked_add(i64::from(output_zero_point))
            .ok_or_else(|| BackendError::Dispatch("requantized output overflow".into()))?;
        Ok(shifted.clamp(i64::from(min), i64::from(max)) as i32)
    }

    pub fn current_for_kernel(kernel_name: &str, bit_width: usize, has_bias: bool) -> Self {
        let activation_storage = if kernel_name.ends_with("_i8") {
            ScalarType::I8
        } else {
            ScalarType::F32
        };
        let unsigned = kernel_name.contains("_u4") || kernel_name.contains("_u8");
        let weight_storage = match (bit_width, unsigned) {
            (4, true) => ScalarType::U4,
            (4, false) => ScalarType::I4,
            (8, true) => ScalarType::U8,
            (8, false) => ScalarType::I8,
            _ => ScalarType::I8,
        };
        Self::current_f32_output(activation_storage, weight_storage, has_bias)
    }

    pub fn current_f32_output(
        activation_storage: ScalarType,
        weight_storage: ScalarType,
        has_bias: bool,
    ) -> Self {
        let integer_activation = activation_storage == ScalarType::I8;
        Self {
            activation_storage,
            weight_storage,
            accumulator: if integer_activation {
                ScalarType::I32
            } else {
                ScalarType::F32
            },
            bias_domain: if has_bias {
                BiasDomain::FloatOutput
            } else {
                BiasDomain::None
            },
            zero_point_correction: if integer_activation {
                ZeroPointCorrection::AffineInputAndWeight
            } else {
                ZeroPointCorrection::None
            },
            requantization: RequantizationMode::None,
            rounding: RoundingMode::NotApplicable,
            saturation: SaturationMode::None,
            output_storage: ScalarType::F32,
        }
    }

    fn validate(&self, instruction_index: usize) -> Result<(), BackendError> {
        if !matches!(
            self.weight_storage,
            ScalarType::I4 | ScalarType::I8 | ScalarType::U4 | ScalarType::U8
        ) {
            return Err(BackendError::Dispatch(format!(
                "instruction {instruction_index} has unsupported quantized weight storage {:?}",
                self.weight_storage
            )));
        }
        match self.activation_storage {
            ScalarType::F32 if self.accumulator == ScalarType::F32 => {}
            ScalarType::I8 if self.accumulator == ScalarType::I32 => {}
            _ => {
                return Err(BackendError::Dispatch(format!(
                    "instruction {instruction_index} has incompatible quantized activation/accumulator contract"
                )))
            }
        }
        match self.output_storage {
            ScalarType::F32
                if matches!(self.requantization, RequantizationMode::None)
                    && matches!(self.rounding, RoundingMode::NotApplicable)
                    && matches!(self.saturation, SaturationMode::None) => {}
            ScalarType::I8 | ScalarType::U8 if self.accumulator == ScalarType::I32 => {
                let RequantizationMode::FixedPoint {
                    multiplier,
                    right_shift,
                    output_zero_point,
                } = self.requantization
                else {
                    return Err(BackendError::Dispatch(format!(
                        "instruction {instruction_index} integer output requires fixed-point requantization"
                    )));
                };
                let expected_range = if self.output_storage == ScalarType::I8 {
                    (i8::MIN as i32, i8::MAX as i32)
                } else {
                    (u8::MIN as i32, u8::MAX as i32)
                };
                if multiplier <= 0
                    || right_shift > 31
                    || !matches!(self.rounding, RoundingMode::NearestTiesToEven)
                    || !matches!(
                        self.saturation,
                        SaturationMode::Clamp { min, max } if (min, max) == expected_range
                    )
                    || !expected_range.0.le(&output_zero_point)
                    || !output_zero_point.le(&expected_range.1)
                    || matches!(self.bias_domain, BiasDomain::FloatOutput)
                {
                    return Err(BackendError::Dispatch(format!(
                        "instruction {instruction_index} has invalid integer requantization semantics"
                    )));
                }
            }
            _ => return Err(BackendError::Dispatch(format!(
                "instruction {instruction_index} requests an unsupported quantized output contract"
            ))),
        }
        Ok(())
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
    /// Explicit numerical contract for the consuming quantized kernel.
    pub execution: QuantizedExecutionContract,
}

impl QuantizedWeightMeta {
    fn validate(&self, instruction_index: usize) -> Result<(), BackendError> {
        self.execution.validate(instruction_index)?;
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

    fn validate_for_kernel(
        &self,
        instruction_index: usize,
        kernel_name: &str,
    ) -> Result<(), BackendError> {
        self.validate(instruction_index)?;
        let expected = QuantizedExecutionContract::current_for_kernel(
            kernel_name,
            self.bit_width,
            !matches!(self.execution.bias_domain, BiasDomain::None),
        );
        let correction_matches = std::mem::discriminant(&self.execution.zero_point_correction)
            == std::mem::discriminant(&expected.zero_point_correction);
        if self.execution.activation_storage != expected.activation_storage
            || self.execution.weight_storage != expected.weight_storage
            || self.execution.accumulator != expected.accumulator
            || !correction_matches
            || self.execution.output_storage != expected.output_storage
        {
            return Err(BackendError::Dispatch(format!(
                "instruction {instruction_index} quantized execution contract does not match kernel {kernel_name}"
            )));
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PlanResourceLimits {
    pub max_serialized_bytes: u64,
    pub max_arena_bytes: usize,
    pub max_instructions: usize,
    pub max_inputs_per_instruction: usize,
    pub max_total_input_slices: usize,
    pub max_total_params: usize,
    pub max_total_param_dims: usize,
    pub max_total_kernel_name_bytes: usize,
    pub max_total_constant_bytes: usize,
    pub max_total_quant_metadata: usize,
}

impl Default for PlanResourceLimits {
    fn default() -> Self {
        const GIB: u64 = 1024 * 1024 * 1024;
        Self {
            max_serialized_bytes: 8 * GIB,
            max_arena_bytes: usize::try_from(8 * GIB).unwrap_or(usize::MAX),
            max_instructions: 1_000_000,
            max_inputs_per_instruction: 1_024,
            max_total_input_slices: 16_000_000,
            max_total_params: 16_000_000,
            max_total_param_dims: 16_000_000,
            max_total_kernel_name_bytes: 64_000_000,
            max_total_constant_bytes: usize::try_from(4 * GIB).unwrap_or(usize::MAX),
            max_total_quant_metadata: 16_000_000,
        }
    }
}

impl ExecutablePlan {
    /// Save the plan to a binary file using bincode serialization.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.save_with_limits(path, &PlanResourceLimits::default())
    }

    pub fn save_with_limits(
        &self,
        path: &str,
        limits: &PlanResourceLimits,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = self.to_bytes_with_limits(limits)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn to_bytes_with_limits(
        &self,
        limits: &PlanResourceLimits,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        self.validate_with_limits(limits)?;
        let serialized_size = bincode::serialized_size(self)?;
        if serialized_size > limits.max_serialized_bytes {
            return Err(format!(
                "serialized executable plan would have {serialized_size} bytes, exceeding limit {}",
                limits.max_serialized_bytes
            )
            .into());
        }
        Ok(bincode::serialize(self)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BackendError> {
        Self::from_bytes_with_limits(bytes, &PlanResourceLimits::default())
    }

    pub fn from_bytes_with_limits(
        bytes: &[u8],
        limits: &PlanResourceLimits,
    ) -> Result<Self, BackendError> {
        decode_executable_plan(bytes, limits)
    }

    /// Load a plan from a binary file created by [`save`](Self::save).
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let limits = PlanResourceLimits::default();
        let file_size = std::fs::metadata(path)?.len();
        if file_size > limits.max_serialized_bytes {
            return Err(format!(
                "serialized executable plan has {file_size} bytes, exceeding limit {}",
                limits.max_serialized_bytes
            )
            .into());
        }
        let bytes = std::fs::read(path)?;
        let plan = Self::from_bytes_with_limits(&bytes, &limits)?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        self.validate_with_limits(&PlanResourceLimits::default())
    }

    pub fn validate_with_limits(&self, limits: &PlanResourceLimits) -> Result<(), BackendError> {
        if self.instructions.len() != self.levels.len() {
            return Err(BackendError::Dispatch(format!(
                "executable plan has {} instructions but {} scheduling levels",
                self.instructions.len(),
                self.levels.len()
            )));
        }
        if self.instructions.len() > limits.max_instructions {
            return Err(BackendError::Dispatch(format!(
                "executable plan has {} instructions, exceeding limit {}",
                self.instructions.len(),
                limits.max_instructions
            )));
        }
        if self.arena_size > limits.max_arena_bytes {
            return Err(BackendError::Dispatch(format!(
                "executable plan arena requests {} bytes, exceeding limit {}",
                self.arena_size, limits.max_arena_bytes
            )));
        }
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

        let mut total_input_slices = 0usize;
        let mut total_params = 0usize;
        let mut total_param_dims = 0usize;
        let mut total_kernel_name_bytes = 0usize;
        let mut total_constant_bytes = 0usize;
        let mut total_quant_metadata = 0usize;
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            match instruction {
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    secondary_output_slice,
                    params,
                    param_dims,
                    weight_meta,
                    node_id: _,
                } => {
                    if input_slices.len() > limits.max_inputs_per_instruction {
                        return Err(BackendError::Dispatch(format!(
                            "instruction {instruction_index} has {} inputs, exceeding limit {}",
                            input_slices.len(),
                            limits.max_inputs_per_instruction
                        )));
                    }
                    total_input_slices = total_input_slices
                        .checked_add(input_slices.len())
                        .ok_or_else(|| {
                            BackendError::Dispatch(
                                "executable plan input slice count overflows".into(),
                            )
                        })?;
                    if total_input_slices > limits.max_total_input_slices {
                        return Err(BackendError::Dispatch(format!(
                            "executable plan has {total_input_slices} input slices, exceeding limit {}",
                            limits.max_total_input_slices
                        )));
                    }
                    total_kernel_name_bytes = total_kernel_name_bytes
                        .checked_add(kernel_name.len())
                        .ok_or_else(|| {
                            BackendError::Dispatch(
                                "executable plan kernel name bytes overflow".into(),
                            )
                        })?;
                    if total_kernel_name_bytes > limits.max_total_kernel_name_bytes {
                        return Err(BackendError::Dispatch(format!(
                            "executable plan has {total_kernel_name_bytes} kernel name bytes, exceeding limit {}",
                            limits.max_total_kernel_name_bytes
                        )));
                    }
                    total_params = total_params.checked_add(params.len()).ok_or_else(|| {
                        BackendError::Dispatch("executable plan parameter count overflows".into())
                    })?;
                    if total_params > limits.max_total_params {
                        return Err(BackendError::Dispatch(format!(
                            "executable plan has {total_params} kernel parameters, exceeding limit {}",
                            limits.max_total_params
                        )));
                    }
                    total_param_dims = total_param_dims
                        .checked_add(param_dims.as_ref().map_or(0, Vec::len))
                        .ok_or_else(|| {
                            BackendError::Dispatch(
                                "executable plan parameter dimension count overflows".into(),
                            )
                        })?;
                    if total_param_dims > limits.max_total_param_dims {
                        return Err(BackendError::Dispatch(format!(
                            "executable plan has {total_param_dims} parameter dimensions, exceeding limit {}",
                            limits.max_total_param_dims
                        )));
                    }
                    for (input_index, &slice) in input_slices.iter().enumerate() {
                        validate_slice(slice, instruction_index, &format!("input {input_index}"))?;
                    }
                    validate_slice(*output_slice, instruction_index, "output")?;
                    if let Some(slice) = secondary_output_slice {
                        validate_slice(*slice, instruction_index, "secondary output")?;
                    }
                    if let Some(weight_meta) = weight_meta {
                        let metadata_count =
                            weight_meta
                                .scales
                                .len()
                                .checked_add(weight_meta.dequant_offsets.len())
                                .and_then(|count| count.checked_add(weight_meta.shape.len()))
                                .and_then(|count| {
                                    weight_meta.codebooks.len().checked_mul(16).and_then(
                                        |codebook_values| count.checked_add(codebook_values),
                                    )
                                })
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "executable plan quantized metadata count overflows".into(),
                                    )
                                })?;
                        total_quant_metadata = total_quant_metadata
                            .checked_add(metadata_count)
                            .ok_or_else(|| {
                                BackendError::Dispatch(
                                    "executable plan quantized metadata count overflows".into(),
                                )
                            })?;
                        if total_quant_metadata > limits.max_total_quant_metadata {
                            return Err(BackendError::Dispatch(format!(
                                "executable plan has {total_quant_metadata} quantized metadata entries, exceeding limit {}",
                                limits.max_total_quant_metadata
                            )));
                        }
                        weight_meta.validate_for_kernel(instruction_index, kernel_name)?;
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
                    if dst.offset % std::mem::align_of::<f32>() != 0
                        || dst.size % std::mem::size_of::<f32>() != 0
                    {
                        return Err(BackendError::Dispatch(format!(
                            "instruction {instruction_index} fill destination is not f32-aligned"
                        )));
                    }
                }
                Instruction::WriteConst { dst, data } => {
                    total_constant_bytes = total_constant_bytes
                        .checked_add(data.len())
                        .ok_or_else(|| {
                            BackendError::Dispatch(
                                "executable plan constant payload size overflows".into(),
                            )
                        })?;
                    if total_constant_bytes > limits.max_total_constant_bytes {
                        return Err(BackendError::Dispatch(format!(
                            "executable plan has {total_constant_bytes} constant bytes, exceeding limit {}",
                            limits.max_total_constant_bytes
                        )));
                    }
                    validate_slice(*dst, instruction_index, "constant destination")?;
                    if data.len() > dst.size {
                        return Err(BackendError::Dispatch(format!(
                            "instruction {instruction_index} constant payload has {} bytes but destination slot has {}",
                            data.len(), dst.size
                        )));
                    }
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

pub(crate) fn deserialize_executable_plan(
    bytes: &[u8],
    limits: &PlanResourceLimits,
) -> Result<ExecutablePlan, BackendError> {
    ExecutablePlan::from_bytes_with_limits(bytes, limits)
}

fn decode_executable_plan(
    bytes: &[u8],
    limits: &PlanResourceLimits,
) -> Result<ExecutablePlan, BackendError> {
    let byte_count = u64::try_from(bytes.len()).map_err(|_| {
        BackendError::Dispatch("serialized executable plan size exceeds u64".into())
    })?;
    if byte_count > limits.max_serialized_bytes {
        return Err(BackendError::Dispatch(format!(
            "serialized executable plan has {byte_count} bytes, exceeding limit {}",
            limits.max_serialized_bytes
        )));
    }
    let plan: ExecutablePlan = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .with_limit(limits.max_serialized_bytes)
        .deserialize(bytes)
        .map_err(|error| BackendError::Dispatch(format!("executable plan decode: {error}")))?;
    plan.validate_with_limits(limits)?;
    Ok(plan)
}

#[cfg(test)]
mod executable_plan_validation_tests {
    use super::*;

    fn valid_i4_meta(kernel_name: &str) -> QuantizedWeightMeta {
        QuantizedWeightMeta {
            bit_width: 4,
            scales: vec![1.0],
            dequant_offsets: vec![0.0],
            shape: vec![1, 4],
            quant_block_size: 0,
            codebooks: vec![],
            execution: QuantizedExecutionContract::current_for_kernel(kernel_name, 4, false),
        }
    }

    #[test]
    fn quantized_execution_contract_distinguishes_float_and_integer_accumulation() {
        let float = valid_i4_meta("matmul_i4");
        assert_eq!(float.execution.activation_storage, ScalarType::F32);
        assert_eq!(float.execution.accumulator, ScalarType::F32);
        assert!(matches!(
            float.execution.zero_point_correction,
            ZeroPointCorrection::None
        ));

        let integer = valid_i4_meta("matmul_i4_i8");
        assert_eq!(integer.execution.activation_storage, ScalarType::I8);
        assert_eq!(integer.execution.accumulator, ScalarType::I32);
        assert!(matches!(
            integer.execution.zero_point_correction,
            ZeroPointCorrection::AffineInputAndWeight
        ));
        assert!(matches!(integer.execution.bias_domain, BiasDomain::None));
        assert!(matches!(
            integer.execution.requantization,
            RequantizationMode::None
        ));
        assert!(matches!(
            integer.execution.rounding,
            RoundingMode::NotApplicable
        ));
        assert!(matches!(integer.execution.saturation, SaturationMode::None));
        assert_eq!(integer.execution.output_storage, ScalarType::F32);

        assert!(integer.validate_for_kernel(0, "matmul_i4_i8").is_ok());
        assert!(integer.validate_for_kernel(0, "matmul_i4").is_err());
    }

    #[test]
    fn fixed_point_requantization_rounds_ties_to_even_and_saturates() {
        let contract = QuantizedExecutionContract::fixed_point_output(
            ScalarType::I8,
            ScalarType::U4,
            BiasDomain::I32Accumulator,
            1,
            1,
            0,
            ScalarType::I8,
        );
        assert!(contract.validate(0).is_ok());
        assert_eq!(contract.requantize_i32(1).unwrap(), 0);
        assert_eq!(contract.requantize_i32(3).unwrap(), 2);
        assert_eq!(contract.requantize_i32(-1).unwrap(), 0);
        assert_eq!(contract.requantize_i32(-3).unwrap(), -2);
        assert_eq!(contract.requantize_i32(10_000).unwrap(), i8::MAX as i32);
        assert_eq!(contract.requantize_i32(-10_000).unwrap(), i8::MIN as i32);
    }

    #[test]
    fn fixed_point_requantization_applies_output_zero_point() {
        let contract = QuantizedExecutionContract::fixed_point_output(
            ScalarType::I8,
            ScalarType::U8,
            BiasDomain::None,
            1,
            0,
            128,
            ScalarType::U8,
        );
        assert!(contract.validate(0).is_ok());
        assert_eq!(contract.requantize_i32(-200).unwrap(), u8::MIN as i32);
        assert_eq!(contract.requantize_i32(0).unwrap(), 128);
        assert_eq!(contract.requantize_i32(200).unwrap(), u8::MAX as i32);
    }

    #[test]
    fn rejects_malformed_quantized_weight_metadata() {
        let invalid = QuantizedWeightMeta {
            bit_width: 4,
            scales: vec![1.0],
            dequant_offsets: vec![],
            shape: vec![2, 4],
            quant_block_size: 0,
            codebooks: vec![],
            execution: QuantizedExecutionContract::current_for_kernel("matmul_i4", 4, false),
        };
        assert!(invalid.validate(0).is_err());

        let non_finite = QuantizedWeightMeta {
            bit_width: 8,
            scales: vec![f32::NAN],
            dequant_offsets: vec![0.0],
            shape: vec![1, 4],
            quant_block_size: 0,
            codebooks: vec![],
            execution: QuantizedExecutionContract::current_for_kernel("matmul_i8", 8, false),
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

        let oversized_constant = ExecutablePlan {
            instructions: vec![Instruction::WriteConst {
                dst: BufferSlice::new(0, 4),
                data: vec![0; 8],
            }],
            arena_size: 8,
            levels: vec![0],
        };
        assert!(oversized_constant.validate().is_err());

        let mismatched_copy = ExecutablePlan {
            instructions: vec![Instruction::MemCopy {
                dst: BufferSlice::new(0, 4),
                src: BufferSlice::new(4, 8),
            }],
            arena_size: 12,
            levels: vec![0],
        };
        assert!(mismatched_copy.validate().is_err());

        let missing_level = ExecutablePlan {
            instructions: vec![Instruction::Fill {
                dst: BufferSlice::new(0, 4),
                value: 0.0,
            }],
            arena_size: 4,
            levels: vec![],
        };
        assert!(missing_level.validate().is_err());
    }

    #[test]
    fn enforces_explicit_plan_resource_limits() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::WriteConst {
                dst: BufferSlice::new(0, 4),
                data: vec![0; 4],
            }],
            arena_size: 4,
            levels: vec![0],
        };
        let limits = PlanResourceLimits {
            max_total_constant_bytes: 3,
            ..PlanResourceLimits::default()
        };
        assert!(plan.validate_with_limits(&limits).is_err());
    }

    #[test]
    fn bounded_deserialization_preserves_format_and_rejects_oversized_payloads() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::Fill {
                dst: BufferSlice::new(0, 4),
                value: 1.0,
            }],
            arena_size: 4,
            levels: vec![0],
        };
        let bytes = bincode::serialize(&plan).expect("plan serialization must succeed");
        let decoded = deserialize_executable_plan(&bytes, &PlanResourceLimits::default())
            .expect("bounded decoder must accept the existing format");
        assert_eq!(decoded.instructions.len(), 1);

        let limits = PlanResourceLimits {
            max_serialized_bytes: u64::try_from(bytes.len() - 1).unwrap(),
            ..PlanResourceLimits::default()
        };
        assert!(deserialize_executable_plan(&bytes, &limits).is_err());
        assert!(plan.to_bytes_with_limits(&limits).is_err());
    }

    #[test]
    fn executable_limits_cover_nested_kernel_metadata() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "relu".into(),
                input_slices: vec![BufferSlice::new(0, 4)],
                output_slice: BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![1],
                node_id: None,
                param_dims: Some(vec![DimExpr::Known(1)]),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };

        let input_limits = PlanResourceLimits {
            max_total_input_slices: 0,
            ..PlanResourceLimits::default()
        };
        assert!(plan.validate_with_limits(&input_limits).is_err());

        let dimension_limits = PlanResourceLimits {
            max_total_param_dims: 0,
            ..PlanResourceLimits::default()
        };
        assert!(plan.validate_with_limits(&dimension_limits).is_err());

        let name_limits = PlanResourceLimits {
            max_total_kernel_name_bytes: 3,
            ..PlanResourceLimits::default()
        };
        assert!(plan.validate_with_limits(&name_limits).is_err());
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

    fn try_write_arena(
        &self,
        arena: &Self::Buffer,
        offset: usize,
        data: &[u8],
    ) -> Result<(), BackendError> {
        self.write_arena(arena, offset, data);
        Ok(())
    }

    /// Read `size` bytes from the arena starting at `offset`.
    /// Used by the executor to extract graph outputs.
    fn read_arena(&self, arena: &Self::Buffer, offset: usize, size: usize) -> Vec<u8>;

    fn try_read_arena(
        &self,
        arena: &Self::Buffer,
        offset: usize,
        size: usize,
    ) -> Result<Vec<u8>, BackendError> {
        Ok(self.read_arena(arena, offset, size))
    }
}
