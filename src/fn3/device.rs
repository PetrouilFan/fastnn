//! Device identity, requests and resolution.
//!
//! # The alpha rule: no silent fallback
//!
//! [`DeviceRequest::Auto`] is the only request that may pick a different
//! device than the one named by the caller. Every explicit request either
//! resolves to exactly that device or fails with [`DeviceResolveError`], whose
//! message names what was requested and lists what this process can actually
//! execute on.
//!
//! Falling back from `Vulkan` to `Cpu` when a Vulkan device is missing is a
//! release-gate violation for the 3.0 inference core: it turns a measured
//! Vulkan run into an unlabelled CPU run. See
//! `docs/roadmap/fn3-architecture.md`.
//!
//! # Build seam
//!
//! [`available_devices`] currently reports CPU only. FN3-11 (#57) declares the
//! `vulkan` feature together with its dependencies and appends enumerated
//! physical devices there:
//!
//! ```ignore
//! #[cfg(feature = "vulkan")]
//! {
//!     // FN3-11: enumerate adapters and push DeviceInfo::new(DeviceId::Vulkan(id), name).
//! }
//! ```
//!
//! The `vulkan` feature is intentionally *not* declared by the core contract
//! change: an empty feature would make `DeviceRequest::Vulkan` compile while
//! still having no way to execute on it.

use std::fmt;

/// Stable index of a Vulkan physical device, as enumerated by the Vulkan
/// backend. Stable for the lifetime of a process and ordered by the backend's
/// discovery order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalDeviceId(u32);

impl PhysicalDeviceId {
    /// Construct from a physical-device index.
    pub const fn new(index: u32) -> Self {
        PhysicalDeviceId(index)
    }

    /// The underlying index.
    pub const fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for PhysicalDeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The device a tensor lives on and an operation must execute on.
///
/// `DeviceId` is `Copy` and has no resource handle: it names a device, it does
/// not own one. The backend that owns a device is the authority on whether a
/// tensor's storage is valid there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DeviceId {
    /// The host CPU.
    Cpu,
    /// A Vulkan physical device.
    Vulkan(PhysicalDeviceId),
}

impl DeviceId {
    /// Short stable name: `"cpu"` or `"vulkan"`.
    pub const fn kind_name(self) -> &'static str {
        match self {
            DeviceId::Cpu => "cpu",
            DeviceId::Vulkan(_) => "vulkan",
        }
    }

    /// Full stable label used in diagnostics: `"cpu"` or `"vulkan(0)"`.
    pub fn diagnostic(self) -> String {
        match self {
            DeviceId::Cpu => "cpu".to_string(),
            DeviceId::Vulkan(id) => format!("vulkan({})", id.index()),
        }
    }

    /// `true` for the host CPU.
    pub const fn is_cpu(self) -> bool {
        matches!(self, DeviceId::Cpu)
    }

    /// The physical-device index for Vulkan devices.
    pub const fn vulkan_index(self) -> Option<u32> {
        match self {
            DeviceId::Cpu => None,
            DeviceId::Vulkan(id) => Some(id.index()),
        }
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.diagnostic())
    }
}

/// A device this process can execute on, as reported by the backend that owns
/// it. `name` is a human-readable adapter/model name; for the CPU it is
/// `"cpu"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// The identifier to pass to a plan or a backend.
    pub id: DeviceId,
    /// Human-readable name.
    pub name: String,
}

impl DeviceInfo {
    /// Construct a device description.
    pub fn new(id: DeviceId, name: impl Into<String>) -> Self {
        DeviceInfo {
            id,
            name: name.into(),
        }
    }

    /// Label used in "available devices" lists. Omits the adapter name when it
    /// adds nothing (the CPU entry).
    pub fn label(&self) -> String {
        if self.name == self.id.kind_name() {
            self.id.diagnostic()
        } else {
            format!("{} ({})", self.id.diagnostic(), self.name)
        }
    }
}

/// A caller's device preference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceRequest {
    /// Let the build choose. The only request allowed to resolve to a device
    /// other than the one named.
    Auto,
    /// Require the host CPU.
    Cpu,
    /// Require a Vulkan physical device. `index = None` means "the first
    /// Vulkan device"; it is still an explicit request and never resolves to
    /// the CPU.
    Vulkan {
        /// Physical-device index, or `None` for the lowest available index.
        index: Option<u32>,
    },
    /// Require the Vulkan device whose adapter name matches `name`
    /// (case-insensitive exact match).
    VulkanNamed {
        /// Adapter name to match.
        name: String,
    },
}

impl DeviceRequest {
    /// `true` when the caller named a device, so resolution must not substitute
    /// another one.
    pub const fn is_explicit(&self) -> bool {
        !matches!(self, DeviceRequest::Auto)
    }

    /// Resolve against a caller-supplied device list.
    pub fn resolve(&self, available: &[DeviceInfo]) -> Result<DeviceId, DeviceResolveError> {
        match self {
            DeviceRequest::Auto => {
                if available.iter().any(|info| info.id.is_cpu()) {
                    Ok(DeviceId::Cpu)
                } else if let Some(first) = available.first() {
                    Ok(first.id)
                } else {
                    Err(DeviceResolveError::new(
                        self.to_string(),
                        available,
                        "no devices are available in this build",
                    ))
                }
            }
            DeviceRequest::Cpu => {
                if available.iter().any(|info| info.id.is_cpu()) {
                    Ok(DeviceId::Cpu)
                } else {
                    Err(DeviceResolveError::new(
                        self.to_string(),
                        available,
                        "this build has no CPU backend",
                    ))
                }
            }
            DeviceRequest::Vulkan { index } => {
                let mut vulkan: Vec<&DeviceInfo> = available
                    .iter()
                    .filter(|info| info.id.vulkan_index().is_some())
                    .collect();
                vulkan.sort_by_key(|info| info.id.vulkan_index());
                let reason = if vulkan.is_empty() {
                    "this build has no Vulkan backend (the `vulkan` feature is not enabled)"
                } else {
                    "no Vulkan physical device matches the requested index"
                };
                match index {
                    None => vulkan.first().map(|info| info.id).ok_or_else(|| {
                        DeviceResolveError::new(self.to_string(), available, reason)
                    }),
                    Some(requested) => vulkan
                        .iter()
                        .find(|info| info.id.vulkan_index() == Some(*requested))
                        .map(|info| info.id)
                        .ok_or_else(|| {
                            DeviceResolveError::new(self.to_string(), available, reason)
                        }),
                }
            }
            DeviceRequest::VulkanNamed { name } => {
                let mut matches: Vec<&DeviceInfo> = available
                    .iter()
                    .filter(|info| {
                        info.id.vulkan_index().is_some() && info.name.eq_ignore_ascii_case(name)
                    })
                    .collect();
                matches.sort_by_key(|info| info.id.vulkan_index());
                matches.first().map(|info| info.id).ok_or_else(|| {
                    let reason = if available
                        .iter()
                        .any(|info| info.id.vulkan_index().is_some())
                    {
                        "no Vulkan physical device has that adapter name"
                    } else {
                        "this build has no Vulkan backend (the `vulkan` feature is not enabled)"
                    };
                    DeviceResolveError::new(self.to_string(), available, reason)
                })
            }
        }
    }

    /// Resolve against the devices this process actually reports.
    pub fn resolve_on_host(&self) -> Result<DeviceId, DeviceResolveError> {
        self.resolve(&available_devices())
    }
}

impl fmt::Display for DeviceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceRequest::Auto => f.write_str("auto"),
            DeviceRequest::Cpu => f.write_str("cpu"),
            DeviceRequest::Vulkan { index: None } => f.write_str("vulkan"),
            DeviceRequest::Vulkan { index: Some(index) } => write!(f, "vulkan(index={index})"),
            DeviceRequest::VulkanNamed { name } => write!(f, "vulkan(name=\"{name}\")"),
        }
    }
}

/// Why an explicit device request could not be satisfied.
///
/// The message always contains the requested device and the list of devices
/// this process can execute on, so a user can fix the request without reading
/// the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceResolveError {
    requested: String,
    available: Vec<String>,
    reason: String,
}

impl DeviceResolveError {
    /// Construct from a rendered request, the caller's device list and a reason.
    pub fn new(
        requested: impl Into<String>,
        available: &[DeviceInfo],
        reason: impl Into<String>,
    ) -> Self {
        DeviceResolveError {
            requested: requested.into(),
            available: available.iter().map(DeviceInfo::label).collect(),
            reason: reason.into(),
        }
    }

    /// A request for a specific device that this build cannot allocate on.
    pub fn unavailable_device(device: DeviceId) -> Self {
        DeviceResolveError::new(
            device.diagnostic(),
            &available_devices(),
            "no backend for this device is present in the alpha build; FN3-11 (#57) implements Vulkan allocation",
        )
    }

    /// The rejected request, rendered.
    pub fn requested(&self) -> &str {
        &self.requested
    }

    /// Labels of the devices that were available.
    pub fn available(&self) -> &[String] {
        &self.available
    }

    /// The reason resolution failed.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for DeviceResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.available.is_empty() {
            write!(
                f,
                "requested device {} is unavailable: {}; no devices are available",
                self.requested, self.reason
            )
        } else {
            write!(
                f,
                "requested device {} is unavailable: {}; available devices: [{}]",
                self.requested,
                self.reason,
                self.available.join(", ")
            )
        }
    }
}

impl std::error::Error for DeviceResolveError {}

/// Devices this build can execute on, in preference order for
/// [`DeviceRequest::Auto`].
///
/// FN3-11 (#57) appends Vulkan physical devices here under its feature gate.
pub fn available_devices() -> Vec<DeviceInfo> {
    vec![DeviceInfo::new(DeviceId::Cpu, "cpu")]
}

/// Resolve a request against an explicit device list.
pub fn resolve(
    request: &DeviceRequest,
    available: &[DeviceInfo],
) -> Result<DeviceId, DeviceResolveError> {
    request.resolve(available)
}

/// Resolve a request against the devices this process reports.
pub fn resolve_on_host(request: &DeviceRequest) -> Result<DeviceId, DeviceResolveError> {
    request.resolve_on_host()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vk(index: u32, name: &str) -> DeviceInfo {
        DeviceInfo::new(DeviceId::Vulkan(PhysicalDeviceId::new(index)), name)
    }

    #[test]
    fn explicit_vulkan_never_falls_back_to_cpu() {
        let available = available_devices();
        let error = DeviceRequest::Vulkan { index: Some(0) }
            .resolve(&available)
            .expect_err("no vulkan in this build");
        assert_eq!(error.requested(), "vulkan(index=0)");
        assert_eq!(error.available(), ["cpu"]);
        let text = error.to_string();
        assert!(text.contains("vulkan(index=0)"), "{text}");
        assert!(text.contains("available devices: [cpu]"), "{text}");
    }

    #[test]
    fn auto_may_choose_cpu() {
        assert_eq!(
            DeviceRequest::Auto.resolve(&available_devices()),
            Ok(DeviceId::Cpu)
        );
    }

    #[test]
    fn named_request_matches_case_insensitively_and_keeps_index_order() {
        let available = vec![
            DeviceInfo::new(DeviceId::Cpu, "cpu"),
            vk(1, "RTX 4090"),
            vk(0, "RTX 4090"),
        ];
        assert_eq!(
            DeviceRequest::VulkanNamed {
                name: "rtx 4090".to_string()
            }
            .resolve(&available),
            Ok(DeviceId::Vulkan(PhysicalDeviceId::new(0)))
        );
    }

    #[test]
    fn named_request_failure_lists_available_adapters() {
        let available = vec![DeviceInfo::new(DeviceId::Cpu, "cpu"), vk(0, "RTX 4090")];
        let error = DeviceRequest::VulkanNamed {
            name: "MI300".to_string(),
        }
        .resolve(&available)
        .unwrap_err();
        let text = error.to_string();
        assert!(text.contains("MI300"), "{text}");
        assert!(text.contains("cpu"), "{text}");
        assert!(text.contains("vulkan(0) (RTX 4090)"), "{text}");
    }
}
