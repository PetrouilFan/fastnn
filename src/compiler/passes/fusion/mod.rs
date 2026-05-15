use crate::ir::node::ComputeGraph;

pub mod backward;
pub mod matmul_add_relu;
pub mod op_relu;
pub mod residual_add_norm;

pub trait FusionPass {
    fn name() -> &'static str;
    fn fuse(graph: &mut ComputeGraph) -> Result<bool, String>;
}

#[cfg(feature = "fusion-op-relu")]
fn apply_op_relu(graph: &mut ComputeGraph) -> Result<bool, String> {
    op_relu::OpRelu::fuse(graph)
}
#[cfg(not(feature = "fusion-op-relu"))]
fn apply_op_relu(_graph: &mut ComputeGraph) -> Result<bool, String> {
    Ok(false)
}

#[cfg(feature = "fusion-op-gelu")]
fn apply_op_gelu(graph: &mut ComputeGraph) -> Result<bool, String> {
    op_relu::OpRelu::fuse(graph)
}
#[cfg(not(feature = "fusion-op-gelu"))]
fn apply_op_gelu(_graph: &mut ComputeGraph) -> Result<bool, String> {
    Ok(false)
}

#[cfg(feature = "fusion-op-silu")]
fn apply_op_silu(graph: &mut ComputeGraph) -> Result<bool, String> {
    op_relu::OpRelu::fuse(graph)
}
#[cfg(not(feature = "fusion-op-silu"))]
fn apply_op_silu(_graph: &mut ComputeGraph) -> Result<bool, String> {
    Ok(false)
}

#[cfg(feature = "fusion-matmul-add-relu")]
fn apply_matmul_add_relu(graph: &mut ComputeGraph) -> Result<bool, String> {
    matmul_add_relu::MatMulAddRelu::fuse(graph)
}
#[cfg(not(feature = "fusion-matmul-add-relu"))]
fn apply_matmul_add_relu(_graph: &mut ComputeGraph) -> Result<bool, String> {
    Ok(false)
}

#[cfg(feature = "fusion-matmul-add-gelu")]
fn apply_matmul_add_gelu(graph: &mut ComputeGraph) -> Result<bool, String> {
    matmul_add_relu::MatMulAddRelu::fuse(graph)
}
#[cfg(not(feature = "fusion-matmul-add-gelu"))]
fn apply_matmul_add_gelu(_graph: &mut ComputeGraph) -> Result<bool, String> {
    Ok(false)
}

#[cfg(feature = "fusion-matmul-add-silu")]
fn apply_matmul_add_silu(graph: &mut ComputeGraph) -> Result<bool, String> {
    matmul_add_relu::MatMulAddRelu::fuse(graph)
}
#[cfg(not(feature = "fusion-matmul-add-silu"))]
fn apply_matmul_add_silu(_graph: &mut ComputeGraph) -> Result<bool, String> {
    Ok(false)
}

#[cfg(feature = "fusion-backward")]
fn apply_backward_relu_matmul(graph: &mut ComputeGraph) -> Result<bool, String> {
    backward::BackwardReluMatMul::fuse(graph)
}
#[cfg(not(feature = "fusion-backward"))]
fn apply_backward_relu_matmul(_graph: &mut ComputeGraph) -> Result<bool, String> {
    Ok(false)
}

#[cfg(feature = "fusion-backward")]
fn apply_backward_matmul_add_relu(graph: &mut ComputeGraph) -> Result<bool, String> {
    backward::BackwardMatMulAddRelu::fuse(graph)
}
#[cfg(not(feature = "fusion-backward"))]
fn apply_backward_matmul_add_relu(_graph: &mut ComputeGraph) -> Result<bool, String> {
    Ok(false)
}

#[cfg(feature = "fusion-residual-add-norm")]
fn apply_residual_add_norm(graph: &mut ComputeGraph) -> Result<bool, String> {
    residual_add_norm::FusedResidualAddNorm::fuse(graph)
}
#[cfg(not(feature = "fusion-residual-add-norm"))]
fn apply_residual_add_norm(_graph: &mut ComputeGraph) -> Result<bool, String> {
    Ok(false)
}

fn apply_pass(graph: &mut ComputeGraph, idx: usize) -> Result<bool, String> {
    match idx {
        0 => apply_matmul_add_relu(graph),
        1 => apply_matmul_add_gelu(graph),
        2 => apply_matmul_add_silu(graph),
        3 => apply_op_relu(graph),
        4 => apply_op_gelu(graph),
        5 => apply_op_silu(graph),
        6 => apply_residual_add_norm(graph),
        7 => apply_backward_relu_matmul(graph),
        8 => apply_backward_matmul_add_relu(graph),
        _ => Ok(false),
    }
}

pub fn fuse_operators(graph: &mut ComputeGraph) -> Result<(), String> {
    for i in 0..9 {
        while apply_pass(graph, i)? {}
    }
    Ok(())
}
