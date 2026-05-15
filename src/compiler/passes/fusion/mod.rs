use crate::ir::node::ComputeGraph;

pub mod op_relu;
pub mod matmul_add_relu;
pub mod backward;

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

#[cfg(feature = "fusion-matmul-add-relu")]
fn apply_matmul_add_relu(graph: &mut ComputeGraph) -> Result<bool, String> {
    matmul_add_relu::MatMulAddRelu::fuse(graph)
}
#[cfg(not(feature = "fusion-matmul-add-relu"))]
fn apply_matmul_add_relu(_graph: &mut ComputeGraph) -> Result<bool, String> {
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

pub fn fuse_operators(graph: &mut ComputeGraph) -> Result<(), String> {
    // Pass ordering: more specific patterns first, general fallbacks last.
    // MatMulAddRelu must precede OpRelu so that BiasAdd→Relu isn't consumed
    // by the general OpRelu pass before MatMulAddRelu can fuse the full chain.
    let mut changed = true;
    while changed {
        changed = false;
        changed |= apply_matmul_add_relu(graph)?;
        changed |= apply_op_relu(graph)?;
        changed |= apply_backward_relu_matmul(graph)?;
        changed |= apply_backward_matmul_add_relu(graph)?;
    }
    Ok(())
}
