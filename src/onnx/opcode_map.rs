use crate::ir::node::Opcode;
use std::collections::HashMap;

/// Returns the ONNX op_type for opcodes that have a simple 1:1 mapping
/// and require no attribute extraction.
pub fn simple_opcode_name(op: &Opcode) -> Option<&'static str> {
    Some(match op {
        Opcode::Add => "Add",
        Opcode::Sub => "Sub",
        Opcode::Mul => "Mul",
        Opcode::Div => "Div",
        Opcode::MatMul => "MatMul",
        Opcode::Relu => "Relu",
        Opcode::Gelu => "Gelu",
        Opcode::Sigmoid => "Sigmoid",
        Opcode::Tanh => "Tanh",
        Opcode::Exp => "Exp",
        Opcode::Log => "Log",
        Opcode::Neg => "Neg",
        Opcode::Sqrt => "Sqrt",
        Opcode::Abs => "Abs",
        Opcode::Reshape => "Reshape",
        Opcode::Flatten => "Flatten",
        Opcode::Expand => "Expand",
        Opcode::Tile => "Tile",
        Opcode::Where => "Where",
        _ => return None,
    })
}

type AttrRule = (&'static str, &'static str);

/// Returns the ONNX op_type and a list of attribute remapping rules for
/// opcodes that need attribute extraction (with possible key renaming).
pub fn opcode_attr_rules(op: &Opcode) -> Option<(&'static str, &'static [AttrRule])> {
    Some(match op {
        Opcode::Conv2d => (
            "Conv2d",
            &[
                ("stride", "stride"),
                ("padding", "padding"),
                ("dilation", "dilation"),
                ("group", "group"),
            ],
        ),
        Opcode::Transpose => ("Transpose", &[("perm", "perm")]),
        Opcode::Squeeze => ("Squeeze", &[("axes", "axes")]),
        Opcode::Unsqueeze => ("Unsqueeze", &[("axes", "axes")]),
        Opcode::Concat => ("Concat", &[("axis", "axis")]),
        Opcode::ReduceSum => ("ReduceSum", &[("axis", "axes")]),
        Opcode::ReduceMean => ("ReduceMean", &[("axis", "axes")]),
        Opcode::ReduceMax => ("ReduceMax", &[("axis", "axes")]),
        Opcode::BatchNorm => ("BatchNormalization", &[("eps", "epsilon")]),
        Opcode::Softmax => ("Softmax", &[("axis", "axis")]),
        Opcode::LeakyRelu => ("LeakyRelu", &[("negative_slope", "alpha")]),
        Opcode::Pad => ("Pad", &[("pads", "pads"), ("mode", "mode")]),
        Opcode::Slice => (
            "Slice",
            &[("axes", "axes"), ("starts", "starts"), ("ends", "ends")],
        ),
        Opcode::Cast => ("Cast", &[("to", "to")]),
        Opcode::Gather => ("Gather", &[("axis", "axis")]),
        _ => return None,
    })
}

/// Extract attributes for a given opcode from the node's attribute map,
/// applying any key remapping rules.
pub fn extract_attrs(op: &Opcode, attrs: &HashMap<String, String>) -> HashMap<String, String> {
    let mut result = HashMap::new();
    if let Some((_name, rules)) = opcode_attr_rules(op) {
        for (internal_key, onnx_key) in rules {
            if let Some(val) = attrs.get(*internal_key) {
                result.insert(onnx_key.to_string(), val.clone());
            }
        }
    }
    result
}

/// Combined lookup: returns the ONNX op_type and extracted attributes for any
/// known opcode. Returns `None` if the opcode is not recognized (caller should
/// handle via fallback or skip).
pub fn opcode_to_onnx(
    op: &Opcode,
    attrs: &HashMap<String, String>,
) -> Option<(String, HashMap<String, String>)> {
    if let Some(name) = simple_opcode_name(op) {
        return Some((name.to_string(), HashMap::new()));
    }
    if let Some((name, _)) = opcode_attr_rules(op) {
        return Some((name.to_string(), extract_attrs(op, attrs)));
    }
    None
}
