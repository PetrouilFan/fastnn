//! Loader, instance and validation plumbing.
//!
//! [`Instance`] is an explicit owner: it owns the `ash::Entry` (the `dlopen`
//! handle), the `vk::Instance` and, when validation is enabled, the debug
//! messenger. Nothing in the platform layer destroys the instance behind a
//! user's back; the backend owns exactly one instance and drops it last.
//!
//! # Lifetime rule
//!
//! Loader → instance → physical devices → logical device → every buffer,
//! pipeline, pool and command buffer. In Rust this is encoded by:
//!
//! - [`Instance`] holds `entry` and destroys the `vk::Instance` before
//!   unloading the loader (field order is not enough: `Drop` is explicit).
//! - [`crate::inference::backend::vulkan::Device`] holds `Arc<Instance>`.
//! - every resource holds `Arc<Device>`.
//!
//! # Validation
//!
//! `VK_LAYER_KHRONOS_validation` is opt-in through [`InstanceConfig`] or
//! `FN3_VK_VALIDATION=1`. Messages go to stderr with the stable
//! `[fn3vk][validation][ERROR|WARN|INFO]` prefix the FN3-05 spike established,
//! and every error/warning is counted so a test can assert the count is zero.
//! Validation is never compiled out: the release path simply passes
//! `validation: false`, which costs one boolean.

use std::ffi::CStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use ash::vk;

use super::device_info::{self, AdapterInfo};
use super::error::{Error, Result};

static VALIDATION_ERRORS: AtomicU64 = AtomicU64::new(0);
static VALIDATION_WARNINGS: AtomicU64 = AtomicU64::new(0);
static DESTROYED_INSTANCES: AtomicU64 = AtomicU64::new(0);

/// How many `vk::Instance`s this process has destroyed.
///
/// A leak-path teardown must leave this unchanged: destroying the instance
/// over a leaked device is the use-after-free this counter guards.
pub fn destroyed_instance_total() -> u64 {
    DESTROYED_INSTANCES.load(Ordering::Relaxed)
}

/// Error and warning counts emitted by the validation layer in this process.
pub fn validation_counts() -> (u64, u64) {
    (
        VALIDATION_ERRORS.load(Ordering::Relaxed),
        VALIDATION_WARNINGS.load(Ordering::Relaxed),
    )
}

/// `true` when `FN3_VK_VALIDATION` asks for the validation layer.
///
/// Accepted values: `1`, `true`, `on`, `yes`. Anything else (including unset)
/// leaves validation off.
pub fn validation_from_env() -> bool {
    match std::env::var("FN3_VK_VALIDATION") {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        Err(_) => false,
    }
}

/// How an instance is created.
#[derive(Debug, Clone)]
pub struct InstanceConfig {
    /// Enable `VK_LAYER_KHRONOS_validation` and a debug messenger on stderr.
    pub validation: bool,
    /// Application name reported to the driver.
    pub application_name: &'static str,
}

impl Default for InstanceConfig {
    fn default() -> Self {
        InstanceConfig {
            validation: validation_from_env(),
            application_name: "fastnn-fn3",
        }
    }
}

impl InstanceConfig {
    /// Explicit validation choice.
    pub fn with_validation(validation: bool) -> Self {
        InstanceConfig {
            validation,
            ..InstanceConfig::default()
        }
    }
}

struct DebugState {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

/// An owning Vulkan instance.
pub struct Instance {
    // Declared first so that a plain field drop would unload the loader before
    // the instance; `Drop` explicitly destroys the instance and then clears
    // this field.
    entry: Option<ash::Entry>,
    instance: ash::Instance,
    debug: Option<DebugState>,
    loader_api: Option<u32>,
    validation: bool,
    adapters: Vec<AdapterInfo>,
    physical_devices: Vec<vk::PhysicalDevice>,
    destroyed: bool,
}

impl Instance {
    /// Load the Vulkan loader and create an instance.
    pub fn new(config: &InstanceConfig) -> Result<Arc<Instance>> {
        // SAFETY: loading the Vulkan loader is a process-global operation; the
        // returned `Entry` owns the `dlopen` handle for its whole lifetime and
        // `Instance` keeps it alive until after `vkDestroyInstance`.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|error| Error::Loader(format!("dlopen(libvulkan.so.1) failed: {error}")))?;

        // SAFETY: querying the loader version has no side effects; the value is
        // returned by copy.
        let loader_api = unsafe { entry.try_enumerate_instance_version() }.map_err(|code| {
            Error::vk(
                "vkEnumerateInstanceVersion",
                code,
                "querying the loader's supported instance version",
            )
        })?;
        if let Some(version) = loader_api {
            if version < vk::API_VERSION_1_1 {
                return Err(Error::internal(format!(
                    "the Vulkan loader only supports {}, but the FastNN 3.0 Vulkan backend \
                     requires 1.1 for vkGetPhysicalDeviceProperties2 and the maintenance3 \
                     properties that report maxMemoryAllocationSize",
                    device_info::format_version(version)
                )));
            }
        }

        let mut enabled_layers: Vec<*const std::ffi::c_char> = Vec::new();
        let mut enabled_extensions: Vec<*const std::ffi::c_char> = Vec::new();
        let mut debug_create = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        if config.validation {
            // SAFETY: plain enumeration of loader-owned layer properties.
            let layers =
                unsafe { entry.enumerate_instance_layer_properties() }.map_err(|code| {
                    Error::vk(
                        "vkEnumerateInstanceLayerProperties",
                        code,
                        "while checking for VK_LAYER_KHRONOS_validation",
                    )
                })?;
            let available: Vec<String> = layers
                .iter()
                .filter_map(|layer| {
                    layer
                        .layer_name_as_c_str()
                        .ok()
                        .map(|name| name.to_string_lossy().into_owned())
                })
                .collect();
            if !available
                .iter()
                .any(|name| name == "VK_LAYER_KHRONOS_validation")
            {
                return Err(Error::internal(format!(
                    "validation was requested but VK_LAYER_KHRONOS_validation is not \
                     installed; available layers: {}",
                    if available.is_empty() {
                        "(none)".to_string()
                    } else {
                        available.join(", ")
                    }
                )));
            }

            // SAFETY: plain enumeration of loader-owned instance extensions.
            let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|code| {
                    Error::vk(
                        "vkEnumerateInstanceExtensionProperties",
                        code,
                        "while checking for VK_EXT_debug_utils",
                    )
                })?;
            if !extensions.iter().any(|extension| {
                extension
                    .extension_name_as_c_str()
                    .map(|name| name == c"VK_EXT_debug_utils")
                    .unwrap_or(false)
            }) {
                return Err(Error::internal(
                    "validation was requested but VK_EXT_debug_utils is not exposed by the loader",
                ));
            }
            enabled_layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            enabled_extensions.push(c"VK_EXT_debug_utils".as_ptr());
        }

        let application_name = std::ffi::CString::new(config.application_name).map_err(|_| {
            Error::internal("instrumentation: application name contains a NUL byte")
        })?;
        let application_info = vk::ApplicationInfo::default()
            .application_name(&application_name)
            .application_version(vk::make_api_version(0, 3, 0, 0))
            .engine_name(&application_name)
            .engine_version(vk::make_api_version(0, 3, 0, 0))
            .api_version(vk::API_VERSION_1_1);

        let mut instance_create = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_extensions);
        if config.validation {
            // Routes messages emitted during vkCreateInstance itself.
            instance_create = instance_create.push_next(&mut debug_create);
        }

        // SAFETY: `application_name`, the layer/extension name pointers, and the
        // debug create-info chain all live until `create_instance` returns; the
        // name pointers are NUL-terminated static C strings.
        let instance =
            unsafe { entry.create_instance(&instance_create, None) }.map_err(|code| {
                Error::vk(
                    "vkCreateInstance",
                    code,
                    format!(
                        "requesting Vulkan 1.1 with layers [{}] and extensions [{}]",
                        if config.validation {
                            "VK_LAYER_KHRONOS_validation"
                        } else {
                            ""
                        },
                        if config.validation {
                            "VK_EXT_debug_utils"
                        } else {
                            ""
                        }
                    ),
                )
            })?;

        let debug = if config.validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            // SAFETY: the instance enabled VK_EXT_debug_utils;
            // `debug_create` contains a valid callback and no user data; the
            // messenger is destroyed before the instance in `Drop`.
            let messenger = unsafe { loader.create_debug_utils_messenger(&debug_create, None) }
                .map_err(|code| {
                    Error::vk(
                        "vkCreateDebugUtilsMessengerEXT",
                        code,
                        "installing the validation-output messenger",
                    )
                })?;
            Some(DebugState { loader, messenger })
        } else {
            None
        };

        // SAFETY: enumeration of devices owned by the instance just created.
        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.map_err(|code| {
                Error::vk(
                    "vkEnumeratePhysicalDevices",
                    code,
                    "the loader returned no device list",
                )
            })?;
        let adapters = physical_devices
            .iter()
            .enumerate()
            .map(|(index, physical)| AdapterInfo::probe(&instance, *physical, index as u32))
            .collect::<Result<Vec<_>>>()?;

        Ok(Arc::new(Instance {
            entry: Some(entry),
            instance,
            debug,
            loader_api,
            validation: config.validation,
            adapters,
            physical_devices,
            destroyed: false,
        }))
    }

    /// The raw instance handle.
    pub fn ash(&self) -> &ash::Instance {
        &self.instance
    }

    /// `true` when the validation layer is enabled on this instance.
    pub fn validation(&self) -> bool {
        self.validation
    }

    /// Loader API version, when the loader reports one.
    pub fn loader_api(&self) -> Option<u32> {
        self.loader_api
    }

    /// Adapters probed at construction, in discovery order.
    ///
    /// The index of an adapter in this slice is exactly the index carried by
    /// [`crate::inference::device::PhysicalDeviceId`].
    pub fn adapters(&self) -> &[AdapterInfo] {
        &self.adapters
    }

    /// The raw physical device at a discovery index.
    pub fn physical(&self, index: u32) -> Option<vk::PhysicalDevice> {
        self.physical_devices.get(index as usize).copied()
    }

    /// Find an adapter by device UUID (the only identity the loader preserves
    /// across instances).
    pub fn find_by_uuid(&self, uuid: &[u8; vk::UUID_SIZE]) -> Option<&AdapterInfo> {
        self.adapters
            .iter()
            .find(|adapter| &adapter.device_uuid == uuid)
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if self.destroyed {
            return;
        }
        if let Some(debug) = self.debug.take() {
            // SAFETY: the messenger was created by this loader for this
            // instance and is destroyed exactly once, before the instance.
            unsafe {
                debug
                    .loader
                    .destroy_debug_utils_messenger(debug.messenger, None)
            };
        }
        // SAFETY: every device and child object is already gone (they hold
        // `Arc<Instance>` and therefore keep this instance alive); the instance
        // is destroyed exactly once.
        unsafe { self.instance.destroy_instance(None) };
        DESTROYED_INSTANCES.fetch_add(1, Ordering::Relaxed);
        // Unload the loader only after the instance is gone.
        self.entry = None;
        self.destroyed = true;
    }
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("validation", &self.validation)
            .field("loader_api", &self.loader_api)
            .field("adapters", &self.adapters.len())
            .finish_non_exhaustive()
    }
}

/// The process-wide probe instance used by
/// [`crate::inference::device::available_devices`].
///
/// It is created once and intentionally never destroyed: device indices in
/// [`crate::inference::device::PhysicalDeviceId`] are defined by this probe's
/// discovery order, so the table must not change under a running process. The
/// backend never executes on this instance; it creates its own instance and
/// resolves the probed UUID against it. A failure is cached so a missing loader
/// is reported once and then cheap to repeat.
pub fn shared() -> Result<&'static Arc<Instance>> {
    static SHARED: OnceLock<std::result::Result<Arc<Instance>, String>> = OnceLock::new();
    match SHARED.get_or_init(|| {
        Instance::new(&InstanceConfig::default()).map_err(|error| error.to_string())
    }) {
        Ok(instance) => Ok(instance),
        Err(message) => Err(Error::Loader(message.clone())),
    }
}

/// Devices reported by the shared probe, or an empty list when the loader or an
/// ICD is missing. Never panics: this feeds a diagnostic API.
pub fn shared_adapters() -> &'static [AdapterInfo] {
    match shared() {
        Ok(instance) => instance.adapters(),
        Err(_) => &[],
    }
}

/// Report a loader failure once, in the prefix the evidence logs use.
pub fn report_loader_failure() {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_err() {
        return;
    }
    if let Err(error) = shared() {
        eprintln!("[fn3vk][warn] Vulkan device discovery unavailable: {error}");
    }
}

// SAFETY: per VK_EXT_debug_utils, `callback_data` is a valid pointer for the
// duration of the callback and its string pointers are NUL-terminated; the
// callback only reads them and never retains the borrow.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // SAFETY: see the function comment; `as_ref` returns `None` for null.
    let Some(data) = (unsafe { callback_data.as_ref() }) else {
        return vk::FALSE;
    };

    let message = if data.p_message.is_null() {
        "<null message>".to_string()
    } else {
        // SAFETY: guaranteed NUL-terminated by the validation layer for the
        // duration of this callback.
        unsafe { CStr::from_ptr(data.p_message) }
            .to_string_lossy()
            .into_owned()
    };
    let id = if data.p_message_id_name.is_null() {
        String::new()
    } else {
        // SAFETY: same guarantee as `p_message`.
        let name = unsafe { CStr::from_ptr(data.p_message_id_name) }
            .to_string_lossy()
            .into_owned();
        format!(" {name}({})", data.message_id_number)
    };

    let label = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        "ERROR"
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        VALIDATION_WARNINGS.fetch_add(1, Ordering::Relaxed);
        "WARN"
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        "INFO"
    } else {
        "VERBOSE"
    };

    let mut kinds = Vec::new();
    if message_types.contains(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL) {
        kinds.push("general");
    }
    if message_types.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        kinds.push("validation");
    }
    if message_types.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        kinds.push("performance");
    }

    eprintln!(
        "[fn3vk][validation][{label}]{id} ({}) {message}",
        kinds.join(",")
    );
    vk::FALSE
}
