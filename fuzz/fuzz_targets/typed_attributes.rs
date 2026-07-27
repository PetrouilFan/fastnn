#![no_main]

use fastnn::ir::{DimExpr, IRNode, IrDType, Opcode, TensorType};
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

fuzz_target!(|data: &[u8]| {
    let Ok(value) = std::str::from_utf8(data) else {
        return;
    };
    let mut attrs = HashMap::new();
    attrs.insert("value".to_owned(), value.to_owned());
    let node = IRNode {
        id: 7,
        opcode: Opcode::Transpose,
        inputs: Vec::new(),
        output_type: TensorType::new(vec![DimExpr::Known(1)], IrDType::F32),
        secondary_output_type: None,
        attrs,
        name: "typed-attribute-fuzz".to_owned(),
    };

    let required = node.required_attr::<i64>("value");
    let optional = node.optional_attr::<i64>("value");
    match (required, optional) {
        (Ok(required), Ok(Some(optional))) => assert_eq!(required, optional),
        (Err(_), Err(_)) => {}
        outcome => panic!("required/optional parser disagreement: {outcome:?}"),
    }

    let _ = node.required_attr::<usize>("value");
    let _ = node.optional_attr::<u64>("value");
    let _ = node.optional_attr::<f32>("value");
    let _ = node.optional_attr_list::<i64>("value");
    let _ = node.optional_attr_list::<usize>("value");
    let _ = node.optional_attr_list::<f32>("value");
    let _ = node.optional_bool_attr("value");
});
