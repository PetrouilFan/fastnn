use crate::backend::{BufferSlice, ExecutablePlan, Instruction};
use crate::ir::node::NodeId;

#[derive(Clone, Debug)]
pub struct PreparedExecutablePlan {
    pub instructions: Vec<PreparedInstruction>,
    pub arena_size: usize,
    pub scratch_size: usize,
}

#[derive(Clone, Debug)]
pub enum PreparedInstruction {
    Generic { instruction_index: usize },
    Conv2d(PreparedConv2d),
    MatMul(PreparedMatMul),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedActivation {
    None,
    Relu,
    Gelu,
    Silu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedConvKernelKind {
    CurrentIm2colGemm,
    CurrentOneByOneGemm,
    FuturePackedFp32,
    FuturePackedI8,
    FuturePackedU4,
}

#[derive(Clone, Debug)]
pub struct PreparedConv2d {
    pub node_id: Option<NodeId>,
    pub input: BufferSlice,
    pub weight: BufferSlice,
    pub bias: Option<BufferSlice>,
    pub output: BufferSlice,
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub f: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub activation: PreparedActivation,
    pub kernel: PreparedConvKernelKind,
    pub packed_weight: Option<PackedWeightId>,
    pub scratch_offset: usize,
    pub scratch_len: usize,
}

#[derive(Clone, Debug)]
pub struct PreparedMatMul {
    pub node_id: Option<NodeId>,
    pub a: BufferSlice,
    pub b: BufferSlice,
    pub bias: Option<BufferSlice>,
    pub output: BufferSlice,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub activation: PreparedActivation,
    pub packed_weight: Option<PackedWeightId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedWeightId(pub usize);

/// Map a kernel name suffix to the corresponding [`PreparedActivation`].
///
/// Recognized names: `"conv2d"` (None), `"conv2d_relu"`, `"conv2d_gelu"`,
/// `"conv2d_silu"`. Everything else returns `None`.
pub fn activation_from_kernel_name(kernel_name: &str) -> PreparedActivation {
    match kernel_name {
        "conv2d_relu" | "matmul_relu" => PreparedActivation::Relu,
        "conv2d_gelu" | "matmul_gelu" => PreparedActivation::Gelu,
        "conv2d_silu" | "matmul_silu" => PreparedActivation::Silu,
        _ => PreparedActivation::None,
    }
}

/// Map a kernel name to the appropriate [`PreparedConvKernelKind`].
///
/// Quantized kernels (`"conv2d_u4"`, `"conv2d_u8"`) map to their future
/// packed variants; float kernels map to the current im2col/gemm path,
/// with a 1x1 specialisation when kernel dimensions are both 1.
pub fn kernel_kind_from_kernel_name(kernel_name: &str) -> PreparedConvKernelKind {
    match kernel_name {
        "conv2d_u4" => PreparedConvKernelKind::FuturePackedU4,
        "conv2d_u8" => PreparedConvKernelKind::FuturePackedI8,
        _ => PreparedConvKernelKind::CurrentIm2colGemm,
    }
}

/// Attempt to promote an [`Instruction`] into a [`PreparedInstruction::Conv2d`].
///
/// Returns `Some(PreparedInstruction::Conv2d(...))` only when:
/// - The instruction is a `CallKernel` whose `kernel_name` starts with `"conv2d"`.
/// - `params` has exactly 9 elements: `[stride, padding, dilation, groups, c, h, w, kh, kw]`.
/// - At least two input slices are present (input tensor + weight tensor).
///
/// Returns `None` for any instruction that does not satisfy these invariants;
/// callers should fall back to `PreparedInstruction::Generic`.
///
/// # Required metadata fields from `CallKernel`
/// - `kernel_name`: must begin with `"conv2d"` to trigger promotion.
/// - `params`: `[stride, padding, dilation, groups, c, h, w, kh, kw]`
///   (all `usize`). These are baked in at compile time by the CPU backend.
/// - `input_slices[0]`: input tensor; `size / 4` gives element count (f32).
/// - `input_slices[1]`: weight tensor; `size / 4` gives element count.
/// - `input_slices[2]` (optional): bias tensor.
/// - `output_slice`: output tensor.
/// - `node_id`: optional node identifier for runtime param tightening.
pub fn try_prepare_conv2d(
    instruction: &Instruction,
    _instruction_index: usize,
) -> Option<PreparedInstruction> {
    let Instruction::CallKernel {
        kernel_name,
        input_slices,
        output_slice,
        params,
        node_id,
        ..
    } = instruction
    else {
        return None;
    };

    if !kernel_name.starts_with("conv2d") {
        return None;
    }

    // Conv2d params: [stride, padding, dilation, groups, c, h, w, kh, kw]
    let slice = params.as_slice();
    if slice.len() != 9 {
        return None;
    }
    let [stride, padding, dilation, groups, c, h, w, kh, kw]: [usize; 9] = slice.try_into().ok()?;

    if input_slices.len() < 2 {
        return None;
    }

    let input = input_slices[0];
    let weight = input_slices[1];
    let bias = input_slices.get(2).copied();

    // Derive n (batch) and f (output channels) from buffer sizes.
    // Buffers store f32, so element count = size / 4.
    let input_elems = input.size / std::mem::size_of::<f32>();
    let n = input_elems / c.saturating_mul(h).saturating_mul(w).max(1);

    let weight_elems = weight.size / std::mem::size_of::<f32>();
    let c_per_group = c / groups.max(1);
    let f = weight_elems / c_per_group.saturating_mul(kh).saturating_mul(kw).max(1);

    let activation = activation_from_kernel_name(kernel_name);
    let kernel = kernel_kind_from_kernel_name(kernel_name);

    Some(PreparedInstruction::Conv2d(PreparedConv2d {
        node_id: *node_id,
        input,
        weight,
        bias,
        output: *output_slice,
        n,
        c,
        h,
        w,
        f,
        kh,
        kw,
        stride,
        padding,
        dilation,
        groups,
        activation,
        kernel,
        packed_weight: None,
        scratch_offset: 0,
        scratch_len: 0,
    }))
}

/// Build a [`PreparedExecutablePlan`] by inspecting each instruction.
///
/// Statically recognizable Conv2d `CallKernel` instructions are promoted to
/// `PreparedInstruction::Conv2d`; everything else becomes
/// `PreparedInstruction::Generic`. This preserves runtime semantics while
/// enabling future kernel-specialization paths.
pub fn prepare_executable_plan(plan: &ExecutablePlan) -> PreparedExecutablePlan {
    PreparedExecutablePlan {
        instructions: plan
            .instructions
            .iter()
            .enumerate()
            .map(|(instruction_index, inst)| {
                try_prepare_conv2d(inst, instruction_index)
                    .unwrap_or_else(|| PreparedInstruction::Generic { instruction_index })
            })
            .collect(),
        arena_size: plan.arena_size,
        scratch_size: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BufferSlice, Instruction};

    fn empty_plan() -> ExecutablePlan {
        ExecutablePlan {
            instructions: vec![],
            arena_size: 1024,
            levels: vec![],
        }
    }

    /// Helper: build a conv2d CallKernel instruction with the canonical param layout.
    fn make_conv2d_instruction(
        kernel_name: &str,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        with_bias: bool,
    ) -> Instruction {
        // f32 element size = 4 bytes
        let input_size = c * h * w * 4; // 1 batch
        let c_per_group = c / groups.max(1);
        let f = 8; // arbitrary output channels
        let weight_size = f * c_per_group * kh * kw * 4;
        let bias_size = if with_bias { f * 4 } else { 0 };

        let mut input_slices = vec![
            BufferSlice::new(0, input_size),           // input
            BufferSlice::new(input_size, weight_size), // weight
        ];
        if with_bias {
            input_slices.push(BufferSlice::new(input_size + weight_size, bias_size));
        }

        Instruction::CallKernel {
            kernel_name: kernel_name.to_string(),
            input_slices,
            output_slice: BufferSlice::new(
                input_size + weight_size + bias_size,
                input_size, // output size doesn't matter for metadata
            ),
            secondary_output_slice: None,
            params: vec![stride, padding, dilation, groups, c, h, w, kh, kw],
            node_id: Some(42usize),
            param_dims: None,
            weight_meta: None,
        }
    }

    // ── prepare_executable_plan ────────────────────────────────

    #[test]
    fn prepare_empty_plan() {
        let plan = empty_plan();
        let prepared = prepare_executable_plan(&plan);
        assert!(prepared.instructions.is_empty());
        assert_eq!(prepared.arena_size, 1024);
        assert_eq!(prepared.scratch_size, 0);
    }

    #[test]
    fn prepare_plan_with_generic_instructions() {
        let plan = ExecutablePlan {
            instructions: vec![
                Instruction::Fill {
                    dst: BufferSlice::new(0, 4),
                    value: 0.0,
                },
                Instruction::MemCopy {
                    dst: BufferSlice::new(0, 4),
                    src: BufferSlice::new(4, 4),
                },
            ],
            arena_size: 2048,
            levels: vec![0, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        assert_eq!(prepared.instructions.len(), 2);
        match &prepared.instructions[0] {
            PreparedInstruction::Generic { instruction_index } => {
                assert_eq!(*instruction_index, 0);
            }
            _ => panic!("expected Generic instruction"),
        }
        match &prepared.instructions[1] {
            PreparedInstruction::Generic { instruction_index } => {
                assert_eq!(*instruction_index, 1);
            }
            _ => panic!("expected Generic instruction"),
        }
        assert_eq!(prepared.arena_size, 2048);
    }

    #[test]
    fn conv2d_promoted_in_full_plan() {
        let conv = make_conv2d_instruction("conv2d", 1, 0, 1, 1, 3, 8, 8, 3, 3, false);
        let fill = Instruction::Fill {
            dst: BufferSlice::new(0, 4),
            value: 1.0,
        };
        let plan = ExecutablePlan {
            instructions: vec![conv, fill],
            arena_size: 4096,
            levels: vec![0, 0],
        };
        let prepared = prepare_executable_plan(&plan);
        assert_eq!(prepared.instructions.len(), 2);
        assert!(matches!(
            &prepared.instructions[0],
            PreparedInstruction::Conv2d(_)
        ));
        assert!(matches!(
            &prepared.instructions[1],
            PreparedInstruction::Generic {
                instruction_index: 1
            }
        ));
    }

    // ── activation_from_kernel_name ────────────────────────────

    #[test]
    fn activation_from_kernel_name_none() {
        assert_eq!(
            activation_from_kernel_name("conv2d"),
            PreparedActivation::None
        );
    }

    #[test]
    fn activation_from_kernel_name_relu() {
        assert_eq!(
            activation_from_kernel_name("conv2d_relu"),
            PreparedActivation::Relu
        );
        assert_eq!(
            activation_from_kernel_name("matmul_relu"),
            PreparedActivation::Relu
        );
    }

    #[test]
    fn activation_from_kernel_name_gelu() {
        assert_eq!(
            activation_from_kernel_name("conv2d_gelu"),
            PreparedActivation::Gelu
        );
    }

    #[test]
    fn activation_from_kernel_name_silu() {
        assert_eq!(
            activation_from_kernel_name("conv2d_silu"),
            PreparedActivation::Silu
        );
    }

    #[test]
    fn activation_from_kernel_name_fallback() {
        assert_eq!(
            activation_from_kernel_name("conv2d_u4"),
            PreparedActivation::None
        );
        assert_eq!(
            activation_from_kernel_name("matmul"),
            PreparedActivation::None
        );
        assert_eq!(
            activation_from_kernel_name("unknown_kernel"),
            PreparedActivation::None
        );
    }

    // ── kernel_kind_from_kernel_name ───────────────────────────

    #[test]
    fn kernel_kind_im2col() {
        assert_eq!(
            kernel_kind_from_kernel_name("conv2d"),
            PreparedConvKernelKind::CurrentIm2colGemm
        );
        assert_eq!(
            kernel_kind_from_kernel_name("conv2d_relu"),
            PreparedConvKernelKind::CurrentIm2colGemm
        );
    }

    #[test]
    fn kernel_kind_quantized() {
        assert_eq!(
            kernel_kind_from_kernel_name("conv2d_u4"),
            PreparedConvKernelKind::FuturePackedU4
        );
        assert_eq!(
            kernel_kind_from_kernel_name("conv2d_u8"),
            PreparedConvKernelKind::FuturePackedI8
        );
    }

    #[test]
    fn kernel_kind_fallback() {
        assert_eq!(
            kernel_kind_from_kernel_name("matmul"),
            PreparedConvKernelKind::CurrentIm2colGemm
        );
    }

    // ── try_prepare_conv2d ─────────────────────────────────────

    #[test]
    fn try_prepare_conv2d_basic() {
        let inst = make_conv2d_instruction("conv2d", 2, 1, 1, 1, 16, 32, 32, 3, 3, false);
        let result = try_prepare_conv2d(&inst, 0).expect("should promote conv2d");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                assert_eq!(c.node_id, Some(42usize));
                assert_eq!(c.stride, 2);
                assert_eq!(c.padding, 1);
                assert_eq!(c.dilation, 1);
                assert_eq!(c.groups, 1);
                assert_eq!(c.c, 16);
                assert_eq!(c.h, 32);
                assert_eq!(c.w, 32);
                assert_eq!(c.kh, 3);
                assert_eq!(c.kw, 3);
                assert_eq!(c.n, 1);
                assert_eq!(c.f, 8);
                assert_eq!(c.activation, PreparedActivation::None);
                assert_eq!(c.kernel, PreparedConvKernelKind::CurrentIm2colGemm);
                assert!(c.bias.is_none());
                assert_eq!(c.scratch_offset, 0);
                assert_eq!(c.scratch_len, 0);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_with_bias() {
        let inst = make_conv2d_instruction("conv2d_relu", 1, 0, 1, 1, 8, 16, 16, 1, 1, true);
        let result = try_prepare_conv2d(&inst, 5).expect("should promote");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                let bias = c.bias.expect("should have bias");
                assert_eq!(bias.offset, 8 * 16 * 16 * 4 + 8 * 8 * 4);
                assert_eq!(bias.size, 8 * 4);
                assert_eq!(c.activation, PreparedActivation::Relu);
                assert_eq!(c.kh, 1);
                assert_eq!(c.kw, 1);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_with_groups() {
        let inst = make_conv2d_instruction("conv2d", 1, 0, 1, 4, 32, 8, 8, 3, 3, false);
        let result = try_prepare_conv2d(&inst, 2).expect("should promote");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                assert_eq!(c.groups, 4);
                let c_per_group = c.c / c.groups;
                assert_eq!(c_per_group, 8);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_quantized_u4() {
        let inst = make_conv2d_instruction("conv2d_u4", 1, 0, 1, 1, 16, 8, 8, 3, 3, false);
        let result = try_prepare_conv2d(&inst, 0).expect("should promote");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                assert_eq!(c.kernel, PreparedConvKernelKind::FuturePackedU4);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_quantized_u8() {
        let inst = make_conv2d_instruction("conv2d_u8", 1, 0, 1, 1, 16, 8, 8, 3, 3, false);
        let result = try_prepare_conv2d(&inst, 0).expect("should promote");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                assert_eq!(c.kernel, PreparedConvKernelKind::FuturePackedI8);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_not_conv2d_kernel() {
        let inst = Instruction::CallKernel {
            kernel_name: "matmul".to_string(),
            input_slices: vec![BufferSlice::new(0, 16), BufferSlice::new(16, 16)],
            output_slice: BufferSlice::new(32, 16),
            secondary_output_slice: None,
            params: vec![4, 4, 4],
            node_id: None,
            param_dims: None,
            weight_meta: None,
        };
        assert!(try_prepare_conv2d(&inst, 0).is_none());
    }

    #[test]
    fn try_prepare_conv2d_not_call_kernel() {
        let inst = Instruction::Fill {
            dst: BufferSlice::new(0, 4),
            value: 0.0,
        };
        assert!(try_prepare_conv2d(&inst, 0).is_none());
    }

    #[test]
    fn try_prepare_conv2d_wrong_param_count() {
        let inst = Instruction::CallKernel {
            kernel_name: "conv2d".to_string(),
            input_slices: vec![BufferSlice::new(0, 64), BufferSlice::new(64, 64)],
            output_slice: BufferSlice::new(128, 64),
            secondary_output_slice: None,
            params: vec![1, 0, 1, 1], // only 4 params, need 9
            node_id: None,
            param_dims: None,
            weight_meta: None,
        };
        assert!(try_prepare_conv2d(&inst, 0).is_none());
    }

    #[test]
    fn try_prepare_conv2d_insufficient_inputs() {
        let inst = Instruction::CallKernel {
            kernel_name: "conv2d".to_string(),
            input_slices: vec![BufferSlice::new(0, 64)], // only 1 input, need >= 2
            output_slice: BufferSlice::new(64, 64),
            secondary_output_slice: None,
            params: vec![1, 0, 1, 1, 3, 8, 8, 3, 3],
            node_id: None,
            param_dims: None,
            weight_meta: None,
        };
        assert!(try_prepare_conv2d(&inst, 0).is_none());
    }

    // ── PreparedActivation ─────────────────────────────────────

    #[test]
    fn prepared_activation_variants() {
        assert_eq!(PreparedActivation::None, PreparedActivation::None);
        assert_eq!(PreparedActivation::Relu, PreparedActivation::Relu);
        assert_ne!(PreparedActivation::Relu, PreparedActivation::Gelu);
    }

    #[test]
    fn prepared_activation_debug() {
        assert_eq!(format!("{:?}", PreparedActivation::Silu), "Silu");
    }

    // ── PackedWeightId ─────────────────────────────────────────

    #[test]
    fn packed_weight_id_equality() {
        let a = PackedWeightId(0);
        let b = PackedWeightId(0);
        let c = PackedWeightId(1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── PreparedConvKernelKind ─────────────────────────────────

    #[test]
    fn kernel_kind_debug() {
        assert_eq!(
            format!("{:?}", PreparedConvKernelKind::FuturePackedU4),
            "FuturePackedU4"
        );
    }

    #[test]
    fn kernel_kind_equality() {
        assert_eq!(
            PreparedConvKernelKind::CurrentIm2colGemm,
            PreparedConvKernelKind::CurrentIm2colGemm
        );
        assert_ne!(
            PreparedConvKernelKind::CurrentIm2colGemm,
            PreparedConvKernelKind::FuturePackedFp32
        );
    }
}
