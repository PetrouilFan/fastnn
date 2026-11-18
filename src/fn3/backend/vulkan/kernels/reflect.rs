//! Minimal SPIR-V reflection.
//!
//! The registration seam derives a descriptor-set layout and a pipeline layout
//! from one [`KernelSpec`](super::KernelSpec). To keep that declaration honest,
//! registration reads the committed SPIR-V and compares:
//!
//! - entry-point name and `LocalSize` against the declared workgroup geometry,
//! - `DescriptorSet`/`Binding` decorations against the declared bindings,
//! - the `PushConstant` block size against the declared push-constant bytes.
//!
//! This is deliberately dependency-free: the crate already carries `ash` for
//! Vulkan, and adding `spirv-reflect` for four checks would be a second parser
//! with its own version skew. The subset implemented here covers compute
//! shaders without images, samplers or descriptor arrays; a module that uses
//! those is rejected at registration rather than silently mis-bound.

use std::collections::HashMap;

use super::super::error::{Error, Result};

const MAGIC: u32 = 0x0723_0203;

// Instruction opcodes (SPIR-V 1.3).
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_CONSTANT: u16 = 43;
const OP_VARIABLE: u16 = 59;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;

// Decorations.
const DECORATION_BLOCK: u32 = 2;
const DECORATION_BUFFER_BLOCK: u32 = 3;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;

// Storage classes.
const STORAGE_UNIFORM: u32 = 2;
const STORAGE_PUSH_CONSTANT: u32 = 9;
const STORAGE_STORAGE_BUFFER: u32 = 12;

// Execution modes.
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

/// Descriptor kind derived from the SPIR-V storage class and block decoration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectedKind {
    /// `storage buffer` / `buffer` block.
    StorageBuffer,
    /// `uniform` block.
    UniformBuffer,
}

impl ReflectedKind {
    /// Stable label used in registration errors.
    pub const fn name(self) -> &'static str {
        match self {
            ReflectedKind::StorageBuffer => "storage buffer",
            ReflectedKind::UniformBuffer => "uniform buffer",
        }
    }
}

/// One reflected descriptor binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReflectedBinding {
    /// Descriptor set.
    pub set: u32,
    /// Binding number.
    pub binding: u32,
    /// Descriptor kind.
    pub kind: ReflectedKind,
}

/// One reflected entry point.
#[derive(Debug, Clone)]
pub struct ReflectedEntry {
    /// Entry-point name (`"main"`).
    pub name: String,
    /// Execution model (SPIR-V `ExecutionModel`; 5 = GLCompute).
    pub execution_model: u32,
    /// Declared `LocalSize`.
    pub local_size: [u32; 3],
    /// Descriptors the entry point's module declares.
    pub bindings: Vec<ReflectedBinding>,
    /// Size of the `PushConstant` block, in bytes (0 when absent).
    pub push_constant_bytes: u32,
}

/// The reflected parts of one module.
#[derive(Debug, Clone)]
pub struct ReflectedModule {
    /// SPIR-V version word.
    pub version: u32,
    /// Entry points, in declaration order.
    pub entries: Vec<ReflectedEntry>,
}

impl ReflectedModule {
    /// Find an entry point by name.
    pub fn entry(&self, name: &str) -> Option<&ReflectedEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }
}

#[derive(Debug, Clone)]
enum Type {
    Void,
    Bool,
    Int {
        width: u32,
    },
    Float {
        width: u32,
    },
    Vector {
        component: u32,
        count: u32,
    },
    Matrix {
        column: u32,
        count: u32,
    },
    Array {
        element: u32,
        length: Option<u64>,
        stride: Option<u64>,
    },
    RuntimeArray,
    Struct(Vec<u32>),
    Pointer {
        pointee: u32,
    },
    Opaque,
}

#[derive(Debug, Clone)]
struct Variable {
    id: u32,
    pointer_type: u32,
    storage: u32,
}

struct Module {
    version: u32,
    types: HashMap<u32, Type>,
    constants: HashMap<u32, u64>,
    variables: Vec<Variable>,
    /// `(target, decoration) -> first literal`.
    decorations: HashMap<(u32, u32), u32>,
    /// `(struct, member, decoration) -> first literal`.
    member_decorations: HashMap<(u32, u32, u32), u32>,
    entries: Vec<RawEntry>,
    local_sizes: HashMap<u32, [u32; 3]>,
}

struct RawEntry {
    execution_model: u32,
    function: u32,
    name: String,
}

impl Module {
    fn parse(spirv: &[u8]) -> Result<Module> {
        if spirv.len() < 20 || !spirv.len().is_multiple_of(4) {
            return Err(Error::internal(format!(
                "SPIR-V module has {} bytes; a module needs at least a 5-word header and \
                 4-byte instruction alignment",
                spirv.len()
            )));
        }
        let words: Vec<u32> = spirv
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        if words[0] != MAGIC {
            return Err(Error::internal(format!(
                "SPIR-V module has magic 0x{:08x}, expected 0x{MAGIC:08x}",
                words[0]
            )));
        }
        let version = words[1];

        let mut module = Module {
            version,
            types: HashMap::new(),
            constants: HashMap::new(),
            variables: Vec::new(),
            decorations: HashMap::new(),
            member_decorations: HashMap::new(),
            entries: Vec::new(),
            local_sizes: HashMap::new(),
        };

        let mut index = 5usize;
        while index < words.len() {
            let header = words[index];
            let opcode = (header & 0xffff) as u16;
            let word_count = (header >> 16) as usize;
            if word_count == 0 || index + word_count > words.len() {
                return Err(Error::internal(format!(
                    "SPIR-V instruction at word {index} has invalid word count {word_count}"
                )));
            }
            let operands = &words[index + 1..index + word_count];
            module.instruction(opcode, operands)?;
            index += word_count;
        }
        Ok(module)
    }

    fn instruction(&mut self, opcode: u16, operands: &[u32]) -> Result<()> {
        match opcode {
            OP_ENTRY_POINT => {
                if operands.len() < 3 {
                    return Err(Error::internal("truncated OpEntryPoint"));
                }
                let execution_model = operands[0];
                let function = operands[1];
                let name = literal_string(&operands[2..]).ok_or_else(|| {
                    Error::internal("OpEntryPoint has an unterminated name literal")
                })?;
                self.entries.push(RawEntry {
                    execution_model,
                    function,
                    name,
                });
            }
            OP_EXECUTION_MODE => {
                if operands.len() >= 2 && operands[1] == EXECUTION_MODE_LOCAL_SIZE {
                    if operands.len() < 5 {
                        return Err(Error::internal("truncated LocalSize execution mode"));
                    }
                    self.local_sizes
                        .insert(operands[0], [operands[2], operands[3], operands[4]]);
                }
            }
            OP_TYPE_VOID => {
                if let Some(result) = operands.first() {
                    self.types.insert(*result, Type::Void);
                }
            }
            OP_TYPE_BOOL => {
                if let Some(result) = operands.first() {
                    self.types.insert(*result, Type::Bool);
                }
            }
            OP_TYPE_INT => {
                if operands.len() >= 2 {
                    self.types
                        .insert(operands[0], Type::Int { width: operands[1] });
                }
            }
            OP_TYPE_FLOAT => {
                if operands.len() >= 2 {
                    self.types
                        .insert(operands[0], Type::Float { width: operands[1] });
                }
            }
            OP_TYPE_VECTOR => {
                if operands.len() >= 3 {
                    self.types.insert(
                        operands[0],
                        Type::Vector {
                            component: operands[1],
                            count: operands[2],
                        },
                    );
                }
            }
            OP_TYPE_MATRIX => {
                if operands.len() >= 3 {
                    self.types.insert(
                        operands[0],
                        Type::Matrix {
                            column: operands[1],
                            count: operands[2],
                        },
                    );
                }
            }
            OP_TYPE_ARRAY => {
                if operands.len() >= 3 {
                    let length = self.constants.get(&operands[2]).copied();
                    let stride = self
                        .decorations
                        .get(&(operands[0], DECORATION_ARRAY_STRIDE))
                        .map(|value| u64::from(*value));
                    self.types.insert(
                        operands[0],
                        Type::Array {
                            element: operands[1],
                            length,
                            stride,
                        },
                    );
                }
            }
            OP_TYPE_RUNTIME_ARRAY => {
                if let Some(result) = operands.first() {
                    self.types.insert(*result, Type::RuntimeArray);
                }
            }
            OP_TYPE_STRUCT => {
                if let Some(result) = operands.first() {
                    self.types
                        .insert(*result, Type::Struct(operands[1..].to_vec()));
                }
            }
            OP_TYPE_POINTER => {
                if operands.len() >= 3 {
                    self.types.insert(
                        operands[0],
                        Type::Pointer {
                            pointee: operands[2],
                        },
                    );
                }
            }
            OP_TYPE_IMAGE | OP_TYPE_SAMPLER | OP_TYPE_SAMPLED_IMAGE => {
                if let Some(result) = operands.first() {
                    self.types.insert(*result, Type::Opaque);
                }
            }
            OP_CONSTANT => {
                if operands.len() >= 3 {
                    let value = match operands.len() {
                        3 => u64::from(operands[2]),
                        _ => u64::from(operands[2]) | (u64::from(operands[3]) << 32),
                    };
                    self.constants.insert(operands[1], value);
                }
            }
            OP_VARIABLE => {
                if operands.len() >= 3 {
                    self.variables.push(Variable {
                        pointer_type: operands[0],
                        id: operands[1],
                        storage: operands[2],
                    });
                }
            }
            OP_DECORATE => {
                if operands.len() >= 3 {
                    self.decorations
                        .insert((operands[0], operands[1]), operands[2]);
                }
            }
            OP_MEMBER_DECORATE => {
                if operands.len() >= 4 {
                    self.member_decorations
                        .insert((operands[0], operands[1], operands[2]), operands[3]);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn pointee(&self, pointer_type: u32) -> Option<u32> {
        match self.types.get(&pointer_type)? {
            Type::Pointer { pointee } => Some(*pointee),
            _ => None,
        }
    }

    fn has_decoration(&self, target: u32, decoration: u32) -> bool {
        self.decorations.contains_key(&(target, decoration))
    }

    fn type_size(&self, id: u32, depth: u32) -> Option<u64> {
        if depth > 16 {
            return None;
        }
        match self.types.get(&id)? {
            Type::Void => Some(0),
            Type::Bool => Some(4),
            Type::Int { width } | Type::Float { width } => Some(u64::from(*width) / 8),
            Type::Vector { component, count } => {
                Some(self.type_size(*component, depth + 1)? * u64::from(*count))
            }
            Type::Matrix { column, count } => {
                Some(self.type_size(*column, depth + 1)? * u64::from(*count))
            }
            Type::Array {
                element,
                length,
                stride,
            } => {
                let element_size = self.type_size(*element, depth + 1)?;
                let element_size = stride.unwrap_or(element_size);
                Some(element_size * (*length)?)
            }
            Type::RuntimeArray => None,
            Type::Struct(members) => {
                let mut end = 0u64;
                for (index, member) in members.iter().enumerate() {
                    let offset = u64::from(
                        *self
                            .member_decorations
                            .get(&(id, index as u32, DECORATION_OFFSET))
                            .unwrap_or(&0),
                    );
                    let size = self.type_size(*member, depth + 1)?;
                    end = end.max(offset + size);
                }
                Some(end)
            }
            Type::Pointer { .. } | Type::Opaque => None,
        }
    }

    fn reflect_entry(&self, entry: &RawEntry) -> Result<ReflectedEntry> {
        let local_size = *self.local_sizes.get(&entry.function).ok_or_else(|| {
            Error::internal(format!(
                "entry point \"{}\" has no LocalSize execution mode; it is not a compute kernel",
                entry.name
            ))
        })?;

        let mut bindings = Vec::new();
        let mut push_constant_bytes = 0u32;
        for variable in &self.variables {
            let storage = variable.storage;
            if storage != STORAGE_UNIFORM
                && storage != STORAGE_STORAGE_BUFFER
                && storage != STORAGE_PUSH_CONSTANT
            {
                continue;
            }
            let pointee = self.pointee(variable.pointer_type).ok_or_else(|| {
                Error::internal(format!(
                    "entry point \"{}\": variable {} has a non-pointer type",
                    entry.name, variable.id
                ))
            })?;
            if storage == STORAGE_PUSH_CONSTANT {
                let size = self.type_size(pointee, 0).ok_or_else(|| {
                    Error::internal(format!(
                        "entry point \"{}\": the PushConstant block has a runtime-sized or \
                         opaque member; push constants must be statically sized",
                        entry.name
                    ))
                })?;
                push_constant_bytes = u32::try_from(size)
                    .map_err(|_| Error::internal("push-constant block exceeds 4 GiB"))?;
                continue;
            }

            let kind = match storage {
                STORAGE_STORAGE_BUFFER => ReflectedKind::StorageBuffer,
                STORAGE_UNIFORM if self.has_decoration(pointee, DECORATION_BUFFER_BLOCK) => {
                    ReflectedKind::StorageBuffer
                }
                STORAGE_UNIFORM if self.has_decoration(pointee, DECORATION_BLOCK) => {
                    ReflectedKind::UniformBuffer
                }
                _ => continue,
            };
            let binding = self
                .decorations
                .get(&(variable.id, DECORATION_BINDING))
                .copied()
                .ok_or_else(|| {
                    Error::internal(format!(
                        "entry point \"{}\": descriptor variable {} has no Binding decoration",
                        entry.name, variable.id
                    ))
                })?;
            let set = self
                .decorations
                .get(&(variable.id, DECORATION_DESCRIPTOR_SET))
                .copied()
                .unwrap_or(0);
            bindings.push(ReflectedBinding { set, binding, kind });
        }
        bindings.sort_by_key(|binding| (binding.set, binding.binding));

        Ok(ReflectedEntry {
            name: entry.name.clone(),
            execution_model: entry.execution_model,
            local_size,
            bindings,
            push_constant_bytes,
        })
    }
}

/// Reflect a SPIR-V module.
pub fn reflect(spirv: &[u8]) -> Result<ReflectedModule> {
    let module = Module::parse(spirv)?;
    let entries = module
        .entries
        .iter()
        .map(|entry| module.reflect_entry(entry))
        .collect::<Result<Vec<_>>>()?;
    Ok(ReflectedModule {
        version: module.version,
        entries,
    })
}

/// Decode a packed SPIR-V literal string starting at `words[0]`.
fn literal_string(words: &[u32]) -> Option<String> {
    let mut bytes = Vec::new();
    for word in words {
        for byte in word.to_le_bytes() {
            if byte == 0 {
                return Some(String::from_utf8_lossy(&bytes).into_owned());
            }
            bytes.push(byte);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOKE_ADD: &[u8] = include_bytes!("../shaders/smoke_add.comp.spv");

    #[test]
    fn smoke_module_reflects_entry_bindings_and_push_constants() {
        let module = reflect(SMOKE_ADD).expect("committed SPIR-V reflects");
        let entry = module.entry("main").expect("main is present");
        assert_eq!(entry.local_size, [64, 1, 1]);
        assert_eq!(entry.execution_model, 5, "GLCompute");
        assert_eq!(entry.push_constant_bytes, 4);
        assert_eq!(
            entry.bindings,
            vec![
                ReflectedBinding {
                    set: 0,
                    binding: 0,
                    kind: ReflectedKind::StorageBuffer
                },
                ReflectedBinding {
                    set: 0,
                    binding: 1,
                    kind: ReflectedKind::StorageBuffer
                },
                ReflectedBinding {
                    set: 0,
                    binding: 2,
                    kind: ReflectedKind::StorageBuffer
                },
            ]
        );
    }

    #[test]
    fn rejects_a_bad_magic_and_a_bad_length() {
        assert!(reflect(&[0, 1, 2, 3]).is_err());
        let mut corrupt = SMOKE_ADD.to_vec();
        corrupt[0] = 0;
        assert!(reflect(&corrupt).is_err());
    }

    #[test]
    fn literal_strings_stop_at_the_first_nul() {
        let words = [u32::from_le_bytes(*b"mai\0"), u32::from_le_bytes(*b"ign\0")];
        assert_eq!(literal_string(&words).as_deref(), Some("mai"));
    }
}
