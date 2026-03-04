use crate::storage::DType;
use crate::tensor::Tensor;

#[repr(C)]
pub struct DLDevice {
    pub device_type: u32,
    pub device_id: u32,
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut std::ffi::c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: i64,
}

#[repr(C)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub tensor: DLTensor,
    pub manager_ctx: *mut std::ffi::c_void,
    pub deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
}

pub fn to_dlpack(tensor: &Tensor) -> *mut DLManagedTensor {
    let shape = tensor.shape();
    let ndim = shape.len() as i32;

    let managed = Box::new(DLManagedTensor {
        tensor: DLTensor {
            data: std::ptr::null_mut(),
            device: DLDevice {
                device_type: 1, // kDLCPU
                device_id: 0,
            },
            ndim,
            dtype: match tensor.dtype() {
                DType::F32 => DLDataType {
                    code: 0,
                    bits: 32,
                    lanes: 1,
                },
                DType::F64 => DLDataType {
                    code: 0,
                    bits: 64,
                    lanes: 1,
                },
                DType::I32 => DLDataType {
                    code: 1,
                    bits: 32,
                    lanes: 1,
                },
                DType::I64 => DLDataType {
                    code: 1,
                    bits: 64,
                    lanes: 1,
                },
                _ => DLDataType {
                    code: 0,
                    bits: 32,
                    lanes: 1,
                },
            },
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    });

    Box::into_raw(managed)
}

pub fn from_dlpack(capsule: *mut DLManagedTensor) -> Tensor {
    // Would parse DLPack capsule and create tensor
    Tensor::from_scalar(0.0)
}
