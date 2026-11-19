//! Typed errors for the Vulkan platform layer.
//!
//! Every fallible Vulkan call is funnelled through [`check`] or an explicit
//! match, so a non-success `VkResult` never reaches `unwrap`/`expect`. The
//! shape is the one proven by the FN3-05 spike: **call name + `VkResult` +
//! detail**, plus a hint where the fix is known. [`Error::hint`] is what turns a
//! driver failure into an actionable message.
//!
//! The platform layer never invents an `Fn3Error`. Conversion happens once, in
//! [`Error::to_fn3`], so the `Backend` contract keeps its single error type
//! while this module keeps driver-level context.

use std::fmt;

use ash::vk;

use crate::inference::error::Fn3Error;

/// Result alias for the Vulkan platform layer.
pub type Result<T> = std::result::Result<T, Error>;

/// A failure in the Vulkan platform layer.
///
/// # Rust source compatibility
///
/// The `Vk` variant now carries `requested_bytes`. Patterns that listed every
/// field must include it or use `..`.
#[derive(Debug)]
pub enum Error {
    /// The Vulkan loader itself could not be loaded, or the loader reports an
    /// unusable API version.
    Loader(String),
    /// A Vulkan call returned a non-success result.
    Vk {
        /// Vulkan entry point that failed (`"vkCreateBuffer"`).
        call: &'static str,
        /// Raw result code.
        code: vk::Result,
        /// Byte count of the operation's request: the Vulkan buffer size, the
        /// full allocation or mapping size, or the buffer memory requirement
        /// for an injected allocation fault before chunk selection. `None`
        /// means no request size is known. This is not a measure of memory
        /// consumed by the driver.
        requested_bytes: Option<u64>,
        /// What was being attempted, with the values that matter.
        detail: String,
    },
    /// A request was refused by a device limit before any Vulkan object was
    /// created. `requested` and `limit` are the numbers the caller must change.
    Limit {
        /// What is bounded (`"buffer size"`, `"device allocation size"`).
        what: String,
        /// Requested amount in bytes.
        requested: u64,
        /// Device or policy limit in bytes.
        limit: u64,
        /// Device name and any policy context.
        detail: String,
    },
    /// The device is no longer usable: it was lost, a submission's completion
    /// is unknown, or the driver failed a fence operation.
    ///
    /// This is the **terminal** platform error. Unlike [`Error::Vk`], it is
    /// remembered by [`super::device::Device`], so every later operation on the
    /// same device returns the same failure instead of reusing a fence or a
    /// command buffer whose state is unknown. A caller must drop the backend
    /// and open a new device; there is no in-place recovery.
    Lost {
        /// Adapter name and UUID, so the message names the device.
        device: String,
        /// Which call failed and the driver's report.
        detail: String,
    },
    /// A kernel declaration did not match its SPIR-V or the device limits.
    Registration {
        /// Declared kernel name.
        kernel: &'static str,
        /// What mismatched, with both values.
        detail: String,
    },
    /// A logical precondition failed (range check, missing capability, ...).
    Internal(String),
}

impl Error {
    /// A Vulkan call failed.
    pub fn vk(call: &'static str, code: vk::Result, detail: impl Into<String>) -> Self {
        Error::Vk {
            call,
            code,
            requested_bytes: None,
            detail: detail.into(),
        }
    }

    /// A Vulkan allocator call failed with a known request size.
    pub(crate) fn vk_allocation(
        call: &'static str,
        code: vk::Result,
        bytes: u64,
        detail: impl Into<String>,
    ) -> Self {
        Error::Vk {
            call,
            code,
            requested_bytes: Some(bytes),
            detail: detail.into(),
        }
    }

    /// A limit refused the request.
    pub fn limit(
        what: impl Into<String>,
        requested: u64,
        limit: u64,
        detail: impl Into<String>,
    ) -> Self {
        Error::Limit {
            what: what.into(),
            requested,
            limit,
            detail: detail.into(),
        }
    }

    /// A logical precondition failed.
    pub fn internal(detail: impl Into<String>) -> Self {
        Error::Internal(detail.into())
    }

    /// A kernel declaration failed validation.
    pub fn registration(kernel: &'static str, detail: impl Into<String>) -> Self {
        Error::Registration {
            kernel,
            detail: detail.into(),
        }
    }

    /// A terminal device failure for `device` (adapter name and UUID).
    pub fn lost(device: impl Into<String>, detail: impl Into<String>) -> Self {
        Error::Lost {
            device: device.into(),
            detail: detail.into(),
        }
    }

    /// `true` when the GPU was lost and the device cannot be used again.
    pub fn is_device_lost(&self) -> bool {
        matches!(
            self,
            Error::Lost { .. }
                | Error::Vk {
                    code: vk::Result::ERROR_DEVICE_LOST,
                    ..
                }
        )
    }

    /// An actionable hint for the failure, when one is known.
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            Error::Loader(_) => Some(
                "the Vulkan loader library could not be opened; install the loader \
                 (`vulkan-icd-loader` on Arch), or point VK_ICD_FILENAMES at a valid ICD",
            ),
            Error::Vk {
                code: vk::Result::ERROR_INCOMPATIBLE_DRIVER,
                ..
            } => Some(
                "the loader found no usable Vulkan ICD; check VK_ICD_FILENAMES, \
                 /usr/share/vulkan/icd.d/ and `vulkaninfo --summary`",
            ),
            Error::Vk {
                code: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
                ..
            } => Some(
                "the device is out of memory; free GPU memory or reduce the buffer sizes. \
                 The 3.0 Vulkan allocator suballocates from chunks, so this is heap pressure, \
                 not the per-allocation cap",
            ),
            Error::Vk {
                code: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                ..
            } => Some("the host is out of memory; reduce the buffer sizes"),
            Error::Vk {
                code: vk::Result::ERROR_DEVICE_LOST,
                ..
            }
            | Error::Lost { .. } => Some(
                "the GPU was lost (driver reset or hang); the device is poisoned and must be \
                 recreated. Check `dmesg` for the driver's report",
            ),
            Error::Vk {
                code: vk::Result::TIMEOUT,
                ..
            } => Some(
                "the GPU did not signal the fence in time; a dispatch may have hung. \
                 Increase the timeout only after ruling out an infinite shader loop",
            ),
            Error::Vk {
                code: vk::Result::ERROR_LAYER_NOT_PRESENT,
                ..
            } => Some(
                "install the validation layer (`vulkan-validation-layers` on Arch) or disable \
                 FN3_VK_VALIDATION",
            ),
            Error::Vk {
                code: vk::Result::ERROR_EXTENSION_NOT_PRESENT,
                ..
            } => Some("the driver or loader does not expose this extension; see the adapter dump"),
            Error::Limit { .. } => Some(
                "the request exceeds a device limit; the message names the limit and the device, \
                 so a plan can be rebounded instead of retried",
            ),
            Error::Registration { .. } => Some(
                "the kernel declaration and its SPIR-V disagree; fix the spec, not the pipeline",
            ),
            Error::Internal(_) => None,
            _ => None,
        }
    }

    /// Convert to the 3.0 core error type.
    ///
    /// # Contract note
    ///
    /// `Fn3Error` (owned by FN3-02 / #48) has no generic "backend execution
    /// failed" variant. Mapping is therefore explicit:
    ///
    /// | Vulkan error | `Fn3Error` |
    /// |---|---|
    /// | loader / device open | `DeviceUnavailable` |
    /// | limit | `CapacityExceeded` |
    /// | out of memory | `Allocation` or `AllocationSizeUnknown` |
    /// | registration | `InvalidShape { context: "vulkan kernel registration" }` |
    /// | other Vulkan call | `InvalidShape { context: "vulkan runtime" }` |
    ///
    /// The full driver message is preserved in the detail string, so no context
    /// is lost. #48 is the owner of a first-class backend-failure variant.
    pub fn to_fn3(&self) -> Fn3Error {
        match self {
            Error::Loader(detail) => {
                Fn3Error::DeviceUnavailable(crate::inference::device::DeviceResolveError::new(
                    "vulkan",
                    &crate::inference::device::available_devices(),
                    format!("the Vulkan loader is unavailable: {detail}"),
                ))
            }
            Error::Limit {
                what,
                requested,
                limit,
                ..
            } => {
                let allocation_limit = matches!(
                    what.as_str(),
                    "vulkan buffer size"
                        | "vulkan device allocation size"
                        | "vulkan storage buffer range"
                        | "vulkan staging block"
                );
                if allocation_limit {
                    // `Fn3Error::CapacityExceeded` has no detail field, and an
                    // allocation-limit message without the device name and both
                    // numbers is not actionable. `Allocation` carries the full
                    // rendered text.
                    Fn3Error::Allocation {
                        bytes: *requested,
                        detail: format!(
                            "{self} (limit {limit} on this device; requested {requested})"
                        ),
                    }
                } else {
                    Fn3Error::CapacityExceeded {
                        what: leak_static(what),
                        requested: *requested,
                        capacity: *limit,
                    }
                }
            }
            Error::Registration { kernel, detail } => Fn3Error::InvalidShape {
                context: "vulkan kernel registration",
                detail: format!("kernel {kernel} registration failed: {detail}"),
            },
            Error::Vk {
                code: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                requested_bytes,
                ..
            } => match requested_bytes {
                Some(bytes) => Fn3Error::Allocation {
                    bytes: *bytes,
                    detail: self.to_string(),
                },
                None => Fn3Error::AllocationSizeUnknown {
                    detail: self.to_string(),
                },
            },
            Error::Vk {
                code: vk::Result::ERROR_DEVICE_LOST,
                detail,
                ..
            } => Fn3Error::DeviceUnavailable(crate::inference::device::DeviceResolveError::new(
                "the Vulkan device",
                &crate::inference::device::available_devices(),
                format!("the device was lost and is unusable: {detail}"),
            )),
            Error::Lost { device, detail } => {
                Fn3Error::DeviceUnavailable(crate::inference::device::DeviceResolveError::new(
                    "the Vulkan device",
                    &crate::inference::device::available_devices(),
                    format!(
                        "adapter \"{device}\" is no longer usable: {detail}. The device is \
                         poisoned; drop the backend and open a new one"
                    ),
                ))
            }
            Error::Vk { .. } => Fn3Error::InvalidShape {
                context: "vulkan runtime",
                detail: self.to_string(),
            },
            Error::Internal(detail) => Fn3Error::InvalidShape {
                context: "vulkan runtime",
                detail: detail.clone(),
            },
        }
    }
}

/// `Fn3Error::CapacityExceeded::what` is `&'static str`; the allocator builds
/// its description dynamically. The small set of names is interned here so a
/// useful label survives the conversion without leaking.
fn leak_static(what: &str) -> &'static str {
    const NAMES: &[&str] = &[
        "vulkan buffer size",
        "vulkan device allocation size",
        "vulkan storage buffer range",
        "vulkan staging block",
        "vulkan descriptor binding",
    ];
    for name in NAMES {
        if *name == what {
            return name;
        }
    }
    "vulkan capacity"
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Loader(message) => write!(f, "Vulkan loader unavailable: {message}"),
            Error::Vk {
                call, code, detail, ..
            } => {
                write!(f, "{call} failed: {code:?} ({}): {detail}", code.as_raw())
            }
            Error::Limit {
                what,
                requested,
                limit,
                detail,
            } => write!(
                f,
                "{what} of {requested} bytes exceeds the limit of {limit} bytes: {detail}"
            ),
            Error::Lost { device, detail } => {
                write!(f, "device {device} is no longer usable: {detail}")
            }
            Error::Registration { kernel, detail } => {
                write!(f, "kernel {kernel} registration failed: {detail}")
            }
            Error::Internal(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

/// Accept `VK_SUCCESS` and convert everything else into a typed error.
///
/// ash returns `Result<(), vk::Result>` from commands that produce no value;
/// this helper keeps every call site uniform.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_keeps_call_result_and_detail() {
        let error = Error::vk(
            "vkCreateBuffer",
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            "8 GiB",
        );
        let text = error.to_string();
        assert!(text.contains("vkCreateBuffer"), "{text}");
        assert!(text.contains("ERROR_OUT_OF_DEVICE_MEMORY"), "{text}");
        assert!(text.contains("8 GiB"), "{text}");
        assert!(error.hint().is_some());
    }

    #[test]
    fn oom_preserves_known_bytes_and_distinguishes_an_unknown_size() {
        for code in [
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        ] {
            let known = Error::vk_allocation("vkAllocateMemory", code, 4096, "test chunk").to_fn3();
            assert!(matches!(known, Fn3Error::Allocation { bytes: 4096, .. }));

            let unknown = Error::vk("vkBindBufferMemory", code, "binding test buffer").to_fn3();
            assert!(matches!(unknown, Fn3Error::AllocationSizeUnknown { .. }));
            let text = unknown.to_string();
            assert!(text.contains("request size unknown"), "{text}");
            assert!(text.contains("vkBindBufferMemory"), "{text}");
            assert!(text.contains(&format!("{code:?}")), "{text}");
        }
    }

    #[test]
    fn a_lost_device_is_terminal_and_names_the_adapter() {
        let error = Error::lost(
            "NVIDIA GeForce GTX 1080 Ti (471b2bda-51e1-6451-e692-ff5d1e146f1e)",
            "vkQueueSubmit: simulated loss",
        );
        assert!(error.is_device_lost(), "Lost is a terminal device failure");
        let text = error.to_string();
        assert!(text.contains("NVIDIA GeForce GTX 1080 Ti"), "{text}");
        assert!(text.contains("vkQueueSubmit"), "{text}");
        let converted = error.to_fn3();
        assert!(
            matches!(converted, Fn3Error::DeviceUnavailable(_)),
            "{converted:?}"
        );
        let text = converted.to_string();
        assert!(text.contains("NVIDIA GeForce GTX 1080 Ti"), "{text}");
        assert!(text.contains("no longer usable"), "{text}");
        assert!(text.contains("available devices:"), "{text}");
    }

    #[test]
    fn limit_display_names_both_numbers() {
        let error = Error::limit("buffer size", 5_000_000_000, 4_292_870_144, "GTX 1080 Ti");
        let text = error.to_string();
        assert!(text.contains("5000000000"), "{text}");
        assert!(text.contains("4292870144"), "{text}");
        assert!(text.contains("GTX 1080 Ti"), "{text}");
    }
}
