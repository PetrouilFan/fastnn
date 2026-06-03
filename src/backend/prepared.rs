use crate::backend::{BufferSlice, ExecutablePlan};
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

/// Build a [`PreparedExecutablePlan`] by wrapping every existing instruction
/// as a `PreparedInstruction::Generic`. This is the initial semantics-preserving
/// skeleton: no behavior changes, no kernel specialization yet.
pub fn prepare_executable_plan(plan: &ExecutablePlan) -> PreparedExecutablePlan {
    PreparedExecutablePlan {
        instructions: (0..plan.instructions.len())
            .map(|instruction_index| PreparedInstruction::Generic { instruction_index })
            .collect(),
        arena_size: plan.arena_size,
        scratch_size: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Instruction;

    fn empty_plan() -> ExecutablePlan {
        ExecutablePlan {
            instructions: vec![],
            arena_size: 1024,
            levels: vec![],
        }
    }

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
    fn prepared_activation_variants() {
        assert_eq!(PreparedActivation::None, PreparedActivation::None);
        assert_eq!(PreparedActivation::Relu, PreparedActivation::Relu);
        assert_ne!(PreparedActivation::Relu, PreparedActivation::Gelu);
    }

    #[test]
    fn packed_weight_id_equality() {
        let a = PackedWeightId(0);
        let b = PackedWeightId(0);
        let c = PackedWeightId(1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
