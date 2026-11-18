//! Instance/loader lifetime, validation plumbing, device enumeration and
//! explicit device selection.
//!
//! The instance owns the loader and the debug messenger. Physical devices are
//! enumerated from it; a [`crate::device::DeviceContext`] borrows this type so
//! the instance cannot be destroyed before the device and the loader cannot be
//! unloaded before the instance (see `destroy`).

use std::ffi::{c_char, CStr};
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use crate::args::{Selector, LAVAPIPE_ICD};
use crate::error::{Error, Result};
use crate::ledger::Ledger;
use crate::probe::{self, DeviceInfo};

static VALIDATION_ERRORS: AtomicU64 = AtomicU64::new(0);
static VALIDATION_WARNINGS: AtomicU64 = AtomicU64::new(0);

/// Validation-layer messages seen so far (errors, warnings).
pub fn validation_counts() -> (u64, u64) {
    (
        VALIDATION_ERRORS.load(Ordering::Relaxed),
        VALIDATION_WARNINGS.load(Ordering::Relaxed),
    )
}

pub struct InstanceHandle {
    // Declared first so that a plain field-drop would unload the loader before
    // the instance; `destroy` explicitly destroys the instance first and then
    // clears this field.
    entry: Option<ash::Entry>,
    instance: ash::Instance,
    debug: Option<DebugState>,
    pub loader_api: Option<u32>,
    pub validation: bool,
    pub ledger: Ledger,
    destroyed: bool,
}

struct DebugState {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl InstanceHandle {
    pub fn create(validation: bool, loader_path: Option<&str>) -> Result<Self> {
        // SAFETY: loading the Vulkan loader is a process-global operation done
        // before any worker thread exists. `Entry` owns the dlopen handle for
        // its whole lifetime; `load_from` only reads the supplied path.
        let entry = unsafe {
            match loader_path {
                Some(path) => ash::Entry::load_from(path),
                None => ash::Entry::load(),
            }
        }
        .map_err(|err| {
            Error::Loader(match loader_path {
                Some(path) => format!("dlopen({path:?}) failed: {err}"),
                None => format!("dlopen(libvulkan.so.1) failed: {err}"),
            })
        })?;

        // SAFETY: querying the loader version has no side effects and writes
        // through the returned value only.
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
                    "the Vulkan loader only supports {}, but this spike requires 1.1 \
                     for vkGetPhysicalDeviceProperties2 / VkPhysicalDeviceMaintenance3Properties",
                    probe::format_version(version)
                )));
            }
        }

        let mut ledger = Ledger::default();
        let mut enabled_layers: Vec<*const c_char> = Vec::new();
        let mut enabled_extensions: Vec<*const c_char> = Vec::new();
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

        if validation {
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
                .filter_map(|l| {
                    l.layer_name_as_c_str()
                        .ok()
                        .map(|c| c.to_string_lossy().into_owned())
                })
                .collect();
            if !available.iter().any(|n| n == "VK_LAYER_KHRONOS_validation") {
                return Err(Error::internal(format!(
                    "--validation requested but VK_LAYER_KHRONOS_validation is not installed; \
                     available layers: {}",
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
            let has_debug_utils = extensions.iter().any(|e| {
                e.extension_name_as_c_str()
                    .map(|c| c == c"VK_EXT_debug_utils")
                    .unwrap_or(false)
            });
            if !has_debug_utils {
                return Err(Error::internal(
                    "--validation requested but VK_EXT_debug_utils is not exposed by the loader",
                ));
            }
            enabled_layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            enabled_extensions.push(c"VK_EXT_debug_utils".as_ptr());
        }

        let app_name = c"fn3-vk-bringup";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(app_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_1);

        let mut instance_create = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_extensions);
        if validation {
            // Also route messages emitted during vkCreateInstance itself.
            instance_create = instance_create.push_next(&mut debug_create);
        }

        // SAFETY: `app_info`, the layer/extension name pointers, and the debug
        // create-info chain all live until `create_instance` returns. The
        // pointers are NUL-terminated static C strings.
        let instance =
            unsafe { entry.create_instance(&instance_create, None) }.map_err(|code| {
                Error::vk(
                    "vkCreateInstance",
                    code,
                    format!(
                        "requesting Vulkan 1.1 with layers [{}] and extensions [{}]",
                        if validation {
                            "VK_LAYER_KHRONOS_validation"
                        } else {
                            ""
                        },
                        if validation { "VK_EXT_debug_utils" } else { "" }
                    ),
                )
            })?;
        ledger.create("instance");

        let debug = if validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            // SAFETY: the instance enables VK_EXT_debug_utils; `debug_create`
            // contains a valid callback and no user data; the returned
            // messenger is owned by `self` and destroyed before the instance.
            let messenger = unsafe { loader.create_debug_utils_messenger(&debug_create, None) }
                .map_err(|code| {
                    Error::vk(
                        "vkCreateDebugUtilsMessengerEXT",
                        code,
                        "installing the validation-output messenger",
                    )
                })?;
            ledger.create("debug_messenger");
            Some(DebugState { loader, messenger })
        } else {
            None
        };

        Ok(Self {
            entry: Some(entry),
            instance,
            debug,
            loader_api,
            validation,
            ledger,
            destroyed: false,
        })
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn enumerate(&self) -> Result<Vec<DeviceInfo>> {
        // SAFETY: standard enumeration of physical devices owned by this
        // instance; ash allocates the result vector.
        let physical_devices =
            unsafe { self.instance.enumerate_physical_devices() }.map_err(|code| {
                Error::vk(
                    "vkEnumeratePhysicalDevices",
                    code,
                    "no physical devices were returned by the loader",
                )
            })?;
        if physical_devices.is_empty() {
            return Err(Error::internal(
                "the Vulkan loader reported zero physical devices; \
                 check that an ICD is installed and VK_ICD_FILENAMES is valid",
            ));
        }

        physical_devices
            .iter()
            .enumerate()
            .map(|(index, physical)| DeviceInfo::probe(&self.instance, *physical, index as u32))
            .collect()
    }

    pub fn physical(&self, index: u32) -> Result<vk::PhysicalDevice> {
        // SAFETY: re-enumeration returns the same driver-owned handles; the
        // list is only used to resolve the index selected in `choose_device`.
        let devices = unsafe { self.instance.enumerate_physical_devices() }.map_err(|code| {
            Error::vk(
                "vkEnumeratePhysicalDevices",
                code,
                "resolving the selected device",
            )
        })?;
        devices.get(index as usize).copied().ok_or_else(|| {
            Error::internal(format!(
                "physical device index {index} disappeared between enumeration and selection"
            ))
        })
    }

    pub fn teardown(&mut self) {
        if self.destroyed {
            return;
        }
        if let Some(debug) = self.debug.take() {
            // SAFETY: `messenger` was created by this loader for this instance
            // and is destroyed exactly once, before the instance.
            unsafe {
                debug
                    .loader
                    .destroy_debug_utils_messenger(debug.messenger, None)
            };
            self.ledger.destroy("debug_messenger");
        }
        // SAFETY: no device or child object may still exist; the borrow
        // checker guarantees any DeviceContext was dropped first. `teardown`
        // is idempotent via `self.destroyed`.
        unsafe { self.instance.destroy_instance(None) };
        self.ledger.destroy("instance");
        // Unload the loader only after the instance is gone.
        self.entry = None;
        self.destroyed = true;
    }
}

impl Drop for InstanceHandle {
    fn drop(&mut self) {
        self.teardown();
    }
}

/// Resolve a selector against the probed device list.
///
/// With no selector the first device that has a compute queue is chosen,
/// preferring discrete GPUs, then integrated GPUs, then anything else.
pub fn choose_device(devices: &[DeviceInfo], selector: Option<&Selector>) -> Result<usize> {
    let menu = || {
        let mut text = String::from("available devices:");
        for device in devices {
            text.push_str("\n  ");
            text.push_str(&device.summary_line());
        }
        text
    };

    let chosen = match selector {
        None => {
            let mut candidates: Vec<(u8, usize)> = devices
                .iter()
                .enumerate()
                .filter(|(_, d)| d.has_compute_queue())
                .map(|(i, d)| (device_type_rank(d.device_type), i))
                .collect();
            candidates.sort_unstable();
            match candidates.first() {
                Some((_, index)) => *index,
                None => {
                    return Err(Error::internal(format!(
                        "no physical device exposes a COMPUTE queue family; {}",
                        menu()
                    )))
                }
            }
        }
        Some(Selector::Index(index)) => {
            let index = *index as usize;
            if index >= devices.len() {
                return Err(Error::Usage(format!(
                    "--device-index {index} does not exist: the loader exposed {} device(s)\n{}",
                    devices.len(),
                    menu()
                )));
            }
            index
        }
        Some(Selector::Name(needle)) => {
            let needle_lower = needle.to_lowercase();
            let matches: Vec<usize> = devices
                .iter()
                .enumerate()
                .filter(|(_, d)| d.name.to_lowercase().contains(&needle_lower))
                .map(|(i, _)| i)
                .collect();
            match matches.as_slice() {
                [] => {
                    return Err(Error::Usage(format!(
                        "--device-name {needle:?} matched no device (case-insensitive substring match)\n{}",
                        menu()
                    )))
                }
                [only] => *only,
                many => {
                    let mut detail = format!(
                        "--device-name {needle:?} is ambiguous; {} devices match:",
                        many.len()
                    );
                    for index in many {
                        detail.push_str("\n  ");
                        detail.push_str(&devices[*index].summary_line());
                    }
                    detail.push_str("\npass a full UUID or an index to disambiguate");
                    return Err(Error::Usage(detail));
                }
            }
        }
        Some(Selector::Uuid(text)) => {
            let wanted = probe::parse_uuid(text)?;
            match devices
                .iter()
                .enumerate()
                .find(|(_, d)| d.device_uuid == wanted)
            {
                Some((index, _)) => index,
                None => {
                    return Err(Error::Usage(format!(
                        "--device-uuid {text:?} matched no device\n{}",
                        menu()
                    )))
                }
            }
        }
    };

    let device = &devices[chosen];
    if !device.has_compute_queue() {
        return Err(Error::internal(format!(
            "device {} ({}) exposes no COMPUTE queue family; queue families: {}",
            chosen,
            device.name,
            device
                .queue_families
                .iter()
                .map(|f| format!("family[{}]={}", f.index, probe::queue_flags_string(f.flags)))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(chosen)
}

/// Check `--require-*` capability requirements against a selected device.
pub fn check_requirements(
    device: &DeviceInfo,
    required_extensions: &[String],
    required_min_api: Option<u32>,
) -> Result<()> {
    if let Some(required) = required_min_api {
        if device.api_version < required {
            return Err(Error::internal(format!(
                "device \"{}\" supports Vulkan {}, but --require-min-api requires {}",
                device.name,
                probe::format_version(device.api_version),
                probe::format_version(required)
            )));
        }
    }
    for extension in required_extensions {
        if !device.has_extension(extension) {
            return Err(Error::internal(format!(
                "device \"{}\" does not expose required extension {extension} \
                 (it exposes {} device extensions; run --list to inspect them)",
                device.name,
                device.extensions.len()
            )));
        }
    }
    Ok(())
}

/// Path used by `--software`. Kept here so the CLI contract and the error
/// message mention the same file.
pub fn software_icd_path() -> &'static str {
    LAVAPIPE_ICD
}

fn device_type_rank(device_type: vk::PhysicalDeviceType) -> u8 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 3,
        _ => 4,
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // SAFETY: per VK_EXT_debug_utils, `callback_data` is a valid pointer for
    // the duration of the callback and its string pointers are NUL-terminated;
    // the callback only reads them and never retains the borrow.
    let data = unsafe { callback_data.as_ref() };
    let Some(data) = data else {
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
        "[fn3vk][validation][{label}]{} ({}) {message}",
        id,
        kinds.join(",")
    );
    vk::FALSE
}
