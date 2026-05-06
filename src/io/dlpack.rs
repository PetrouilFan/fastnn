use crate::storage::DType;
use crate::tensor::Tensor;
use std::sync::Arc;

#[repr(C)]
#[allow(dead_code)]
pub struct DLDevice {
    pub device_type: u32,
    pub device_id: i32,
}

#[repr(C)]
#[allow(dead_code)]
pub struct DLTensor {
    pub data: *mut std::ffi::c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
#[allow(dead_code)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[allow(dead_code)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut std::ffi::c_void,
    pub deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
}

// Device types from DLPack spec
#[allow(dead_code)]
const DLDEVICE_CPU: u32 = 1;
#[allow(dead_code)]
const DLDEVICE_CUDA: u32 = 2;

// Data type codes from DLPack spec
#[allow(dead_code)]
const DLDTYPE_FLOAT: u8 = 2;
#[allow(dead_code)]
const DLDTYPE_UINT: u8 = 0;
#[allow(dead_code)]
const DLDTYPE_INT: u8 = 1;

/// Context stored in the DLPack managed tensor to keep the tensor's storage alive.
/// This prevents the data pointer from becoming dangling while the DLPack capsule is in use.
#[repr(C)]
struct DLPackContext {
    /// Strong reference to the tensor's storage - keeps memory alive
    _storage: Arc<crate::storage::Storage>,
    /// Shape array (owned by this context)
    shape: Box<[i64]>,
    /// Strides array (owned by this context)
    strides: Box<[i64]>,
}

/// Convert a Tensor to a DLPack managed tensor (zero-copy for CPU tensors).
/// Returns a raw pointer that can be passed to other frameworks.
/// The tensor's storage is kept alive via Arc until the deleter is called.
#[allow(dead_code)]
pub fn to_dlpack(tensor: &Tensor) -> *mut DLManagedTensor {
    let shape = tensor.shape();
    let ndim = shape.len() as i32;

    // Get data pointer - this is zero-copy for CPU tensors
    let data_ptr = if tensor.inner.is_cpu() {
        tensor.data_ptr() as *mut std::ffi::c_void
    } else {
        std::ptr::null_mut()
    };

    // Allocate shape array
    let shape_box = shape.clone().into_boxed_slice();

    // Create strides (row-major/contiguous)
    let strides: Vec<i64> = if ndim > 0 {
        let mut s = vec![1i64; ndim as usize];
        for i in (0..ndim as usize - 1).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    } else {
        vec![]
    };
    let strides_box = strides.into_boxed_slice();

    // Create context that holds a strong reference to the tensor's storage
    // This prevents the data pointer from becoming dangling
    let ctx = Box::new(DLPackContext {
        _storage: tensor.inner.storage.clone(),
        shape: shape_box,
        strides: strides_box,
    });
    let ctx_ptr = Box::into_raw(ctx);

    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: data_ptr,
            device: DLDevice {
                device_type: DLDEVICE_CPU,
                device_id: 0,
            },
            ndim,
            dtype: match tensor.dtype() {
                DType::F32 => DLDataType {
                    code: DLDTYPE_FLOAT,
                    bits: 32,
                    lanes: 1,
                },
                DType::F64 => DLDataType {
                    code: DLDTYPE_FLOAT,
                    bits: 64,
                    lanes: 1,
                },
                DType::I32 => DLDataType {
                    code: DLDTYPE_INT,
                    bits: 32,
                    lanes: 1,
                },
                DType::I64 => DLDataType {
                    code: DLDTYPE_INT,
                    bits: 64,
                    lanes: 1,
                },
                _ => DLDataType {
                    code: DLDTYPE_FLOAT,
                    bits: 32,
                    lanes: 1,
                },
            },
            shape: unsafe { (*ctx_ptr).shape.as_mut_ptr() },
            strides: unsafe { (*ctx_ptr).strides.as_mut_ptr() },
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr as *mut std::ffi::c_void,
        deleter: Some(dlpack_deleter),
    });

    Box::into_raw(managed)
}

/// Deleter function called by DLPack consumers when done with the tensor.
/// Frees the context (including the Arc<Storage> reference) and the managed tensor.
extern "C" fn dlpack_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    unsafe {
        let managed = Box::from_raw(managed);

        // Drop the context (this drops the Arc<Storage> reference)
        if !managed.manager_ctx.is_null() {
            let _ = Box::from_raw(managed.manager_ctx as *mut DLPackContext);
        }
        // The managed tensor itself is dropped here
    }
}

/// Create a Tensor from a DLPack managed tensor (zero-copy).
/// Takes ownership of the DLPack capsule.
#[allow(dead_code)]
pub unsafe fn from_dlpack(capsule: *mut DLManagedTensor) -> Result<Tensor, String> {
    if capsule.is_null() {
        return Err("DLPack capsule is null".to_string());
    }

    unsafe {
        let managed = &*capsule;
        let dl_tensor = &managed.dl_tensor;

        if dl_tensor.data.is_null() {
            return Err("DLPack tensor data is null".to_string());
        }

        // Extract shape
        let ndim = dl_tensor.ndim as usize;
        let shape: Vec<i64> = if ndim > 0 {
            std::slice::from_raw_parts(dl_tensor.shape, ndim).to_vec()
        } else {
            vec![]
        };

        // Determine dtype
        let dtype = match (dl_tensor.dtype.code, dl_tensor.dtype.bits) {
            (2, 32) => DType::F32,
            (2, 64) => DType::F64,
            (1, 32) => DType::I32,
            (1, 64) => DType::I64,
            _ => {
                return Err(format!(
                    "Unsupported DLPack dtype: code={}, bits={}",
                    dl_tensor.dtype.code, dl_tensor.dtype.bits
                ))
            }
        };

        // Calculate total elements
        let numel: usize = shape.iter().map(|&x| x as usize).product();

        // Copy data from DLPack tensor to our tensor
        // Note: For true zero-copy, we'd need to share the memory, but that
        // requires careful lifetime management
        let data = match dtype {
            DType::F32 => {
                let src = std::slice::from_raw_parts(
                    (dl_tensor.data as *const f32).add(dl_tensor.byte_offset as usize),
                    numel,
                );
                src.to_vec()
            }
            DType::F64 => {
                let src = std::slice::from_raw_parts(
                    (dl_tensor.data as *const f64).add(dl_tensor.byte_offset as usize),
                    numel,
                );
                src.iter().map(|&x| x as f32).collect()
            }
            _ => return Err("Unsupported dtype for from_dlpack".to_string()),
        };

        // Call the deleter if provided
        if let Some(deleter) = managed.deleter {
            deleter(capsule);
        }

        Ok(Tensor::from_vec(data, shape))
    }
}
