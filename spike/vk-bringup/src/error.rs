//! Typed errors for the Vulkan bring-up.
//!
//! Every fallible Vulkan call is funnelled through [`check`] or an explicit
//! match so a failing `VkResult` never reaches `unwrap`/`expect`. Errors carry
//! the call name, the raw `VkResult` and a human-readable detail plus, where
//! known, an actionable hint.

use std::fmt;

use ash::vk;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// Bad command-line usage or conflicting selectors.
    Usage(String),
    /// The Vulkan loader itself could not be loaded (no `libvulkan.so.1`).
    Loader(String),
    /// A Vulkan call returned a non-success result.
    Vk {
        call: &'static str,
        code: vk::Result,
        detail: String,
    },
    /// A logical precondition failed (result mismatch, missing capability, …).
    Internal(String),
}
impl Error {
    pub fn vk(call: &'static str, code: vk::Result, detail: impl Into<String>) -> Self {
        Self::Vk {
            call,
            code,
            detail: detail.into(),
        }
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        Self::Internal(detail.into())
    }

    /// Return an actionable hint for the failure, if one is known.
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Self::Loader(_) => Some(
                "the Vulkan loader library could not be opened; install the loader \
                 (`vulkan-icd-loader` on Arch) or pass --loader-path with its location",
            ),
            Self::Vk {
                code: vk::Result::ERROR_INCOMPATIBLE_DRIVER,
                ..
            } => Some(
                "the loader found no usable Vulkan ICD; check VK_ICD_FILENAMES, \
                 /usr/share/vulkan/icd.d/ and `vulkaninfo --summary`",
            ),
            Self::Vk {
                code: vk::Result::ERROR_LAYER_NOT_PRESENT,
                ..
            } => Some("install the validation layer (`vulkan-validation-layers` on Arch) or drop --validation"),
            Self::Vk {
                code: vk::Result::ERROR_EXTENSION_NOT_PRESENT,
                ..
            } => Some("the driver or loader does not expose this extension; see --list"),
            Self::Vk {
                code: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
                ..
            } => Some("the device is out of memory; reduce --count or free GPU memory"),
            Self::Vk {
                code: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                ..
            } => Some("the host is out of memory; reduce --count"),
            Self::Vk {
                code: vk::Result::ERROR_DEVICE_LOST,
                ..
            } => Some("the GPU was lost (driver reset or hang); check `dmesg` and retry"),
            Self::Vk {
                code: vk::Result::TIMEOUT,
                ..
            } => Some("the GPU did not signal the fence in time; a dispatch may have hung"),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(msg) => write!(f, "usage error: {msg}"),
            Self::Loader(msg) => write!(f, "Vulkan loader unavailable: {msg}"),
            Self::Vk { call, code, detail } => {
                write!(f, "{call} failed: {code:?} ({}): {detail}", code.as_raw())
            }
            Self::Internal(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Accept `VK_SUCCESS` and convert everything else into a typed error.
///
/// ash 0.38 returns `Result<(), vk::Result>` from commands that produce no
/// value; this helper keeps every call site uniform.
pub fn check(
    call: &'static str,
    result: std::result::Result<(), vk::Result>,
    detail: impl Into<String>,
) -> Result<()> {
    match result {
        Ok(()) => Ok(()),
        Err(code) => Err(Error::vk(call, code, detail)),
    }
}
