//! Command-line and environment contract for the bring-up binary.
//!
//! Precedence is: explicit CLI flag > environment variable > default. Exactly
//! one device selector may be active; asking for two is a usage error rather
//! than silently ignoring one.

use crate::error::{Error, Result};

/// Default software ICD installed on this machine (Mesa lavapipe).
pub const LAVAPIPE_ICD: &str = "/usr/share/vulkan/icd.d/lvp_icd.json";

#[derive(Debug, Clone)]
pub enum Selector {
    Index(u32),
    /// Case-insensitive substring of the device name.
    Name(String),
    /// 8-4-4-4-12 UUID or 32 hex digits, case-insensitive.
    Uuid(String),
}

#[derive(Debug, Clone)]
pub struct Options {
    pub list: bool,
    pub validation: bool,
    pub software: bool,
    pub icd: Option<String>,
    pub loader_path: Option<String>,
    pub selector: Option<Selector>,
    pub require_extensions: Vec<String>,
    pub require_min_api: Option<u32>,
    pub count: u32,
    pub repeats: u32,
    pub help: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            list: false,
            validation: false,
            software: false,
            icd: None,
            loader_path: None,
            selector: None,
            require_extensions: Vec::new(),
            require_min_api: None,
            count: 262_147,
            repeats: 100,
            help: false,
        }
    }
}

pub fn parse() -> Result<Options> {
    let mut opts = Options::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--list" => opts.list = true,
            "--validation" => opts.validation = true,
            "--software" => opts.software = true,
            "--help" | "-h" => opts.help = true,
            "--icd" => opts.icd = Some(next_value(&mut args, "--icd")?),
            "--loader-path" => opts.loader_path = Some(next_value(&mut args, "--loader-path")?),
            "--device-index" => {
                let value = next_value(&mut args, "--device-index")?;
                let index = value.parse::<u32>().map_err(|_| {
                    Error::Usage(format!("--device-index expects an integer, got {value:?}"))
                })?;
                set_selector(&mut opts, Selector::Index(index))?;
            }
            "--device-name" => {
                let value = next_value(&mut args, "--device-name")?;
                set_selector(&mut opts, Selector::Name(value))?;
            }
            "--device-uuid" => {
                let value = next_value(&mut args, "--device-uuid")?;
                set_selector(&mut opts, Selector::Uuid(value))?;
            }
            "--require-extension" => {
                let value = next_value(&mut args, "--require-extension")?;
                opts.require_extensions.push(value);
            }
            "--require-min-api" => {
                let value = next_value(&mut args, "--require-min-api")?;
                opts.require_min_api = Some(parse_api_version(&value)?);
            }
            "--count" => {
                let value = next_value(&mut args, "--count")?;
                opts.count = value.parse::<u32>().map_err(|_| {
                    Error::Usage(format!("--count expects a positive integer, got {value:?}"))
                })?;
                if opts.count == 0 {
                    return Err(Error::Usage("--count must be at least 1".into()));
                }
            }
            "--repeats" => {
                let value = next_value(&mut args, "--repeats")?;
                opts.repeats = value.parse::<u32>().map_err(|_| {
                    Error::Usage(format!(
                        "--repeats expects a positive integer, got {value:?}"
                    ))
                })?;
                if opts.repeats == 0 {
                    return Err(Error::Usage("--repeats must be at least 1".into()));
                }
            }
            other => {
                return Err(Error::Usage(format!(
                    "unknown argument {other:?}; run with --help for the contract"
                )))
            }
        }
    }

    if opts.help {
        return Ok(opts);
    }

    if opts.software && opts.icd.is_some() {
        return Err(Error::Usage(
            "--software and --icd are mutually exclusive; --software selects \
             the installed lavapipe ICD"
                .into(),
        ));
    }

    // Environment fallbacks. CLI wins when already set.
    if !opts.validation {
        opts.validation = env_truthy("FN3_VK_VALIDATION");
    }
    if !opts.software && opts.icd.is_none() {
        opts.software = env_truthy("FN3_VK_SOFTWARE");
    }
    if opts.selector.is_none() {
        if let Ok(value) = std::env::var("FN3_VK_DEVICE_INDEX") {
            let index = value.parse::<u32>().map_err(|_| {
                Error::Usage(format!(
                    "FN3_VK_DEVICE_INDEX expects an integer, got {value:?}"
                ))
            })?;
            set_selector(&mut opts, Selector::Index(index))?;
        } else if let Ok(value) = std::env::var("FN3_VK_DEVICE_NAME") {
            set_selector(&mut opts, Selector::Name(value))?;
        } else if let Ok(value) = std::env::var("FN3_VK_DEVICE_UUID") {
            set_selector(&mut opts, Selector::Uuid(value))?;
        }
    }

    Ok(opts)
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| Error::Usage(format!("{flag} requires a value")))
}

fn set_selector(opts: &mut Options, selector: Selector) -> Result<()> {
    if opts.selector.is_some() {
        return Err(Error::Usage(
            "more than one device selector supplied; use exactly one of \
             --device-index, --device-name, --device-uuid"
                .into(),
        ));
    }
    opts.selector = Some(selector);
    Ok(())
}

fn env_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"),
        Err(_) => false,
    }
}

fn parse_api_version(value: &str) -> Result<u32> {
    let mut parts = value.split('.');
    let major = parts
        .next()
        .and_then(|p| p.parse::<u32>().ok())
        .ok_or_else(|| {
            Error::Usage(format!(
                "invalid API version {value:?}; expected MAJOR.MINOR"
            ))
        })?;
    let minor = parts
        .next()
        .and_then(|p| p.parse::<u32>().ok())
        .ok_or_else(|| {
            Error::Usage(format!(
                "invalid API version {value:?}; expected MAJOR.MINOR"
            ))
        })?;
    if parts.next().is_some() {
        return Err(Error::Usage(format!(
            "invalid API version {value:?}; expected MAJOR.MINOR"
        )));
    }
    Ok(ash::vk::make_api_version(0, major, minor, 0))
}

pub fn help() -> &'static str {
    r#"fn3-vk-bringup — FN3-05 native Vulkan compute bring-up

USAGE:
    vk-bringup [OPTIONS]

DEVICE SELECTION (exactly one, optional):
    --device-index N        Select the physical device with enumeration index N
    --device-name SUBSTR    Case-insensitive substring of the device name
    --device-uuid UUID      Device UUID as 8-4-4-4-12 hex or 32 hex digits
    --list                  Enumerate every physical device and exit

ENVIRONMENT:
    --software              Use the installed lavapipe ICD (CI-only evidence)
    --icd PATH              Override VK_ICD_FILENAMES with an explicit ICD JSON
    --loader-path PATH      Load a specific Vulkan loader (diagnostics)

VALIDATION:
    --validation            Enable VK_LAYER_KHRONOS_validation and route
                            VK_EXT_debug_utils output to stderr

CAPABILITY REQUIREMENTS:
    --require-extension NAME    Fail unless the selected device exposes NAME
                                (repeatable; check-only, not enabled)
    --require-min-api X.Y       Fail unless the device API version >= X.Y

WORKLOAD:
    --count N               Element count for the FMA dispatch (default 262147)
    --repeats N             Repeated submissions reusing all handles (default 100)

GENERAL:
    -h, --help              Print this help

ENVIRONMENT VARIABLES:
    FN3_VK_VALIDATION=1     Same as --validation
    FN3_VK_SOFTWARE=1       Same as --software
    FN3_VK_DEVICE_INDEX     Same as --device-index
    FN3_VK_DEVICE_NAME      Same as --device-name
    FN3_VK_DEVICE_UUID      Same as --device-uuid
    VK_ICD_FILENAMES        Honoured by the Vulkan loader directly

EXIT CODES:
    0 success, 1 dispatch/validation failure, 2 usage or Vulkan error.
"#
}
