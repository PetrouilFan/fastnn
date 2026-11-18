//! Device requests, backend opening and the diagnosis matrix.
//!
//! The core rule is the architecture's no-silent-fallback rule: an explicit
//! device either opens exactly that device or the command fails with the
//! request, the available devices and a next step. `auto` is the only request
//! allowed to choose, and today it chooses the CPU.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use crate::fn3::model::format::PreparedHeader;
use crate::fn3::{Backend, CpuBackend, DeviceId, DeviceRequest};

use super::args::DiagnoseArgs;
use super::CliError;

/// A parsed `--device` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceSpec {
    /// Let the build choose.
    Auto,
    /// The host CPU.
    Cpu,
    /// A Vulkan physical device.
    Vulkan(VulkanSelector),
}

/// How a Vulkan device was selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VulkanSelector {
    /// The first enumerated Vulkan device.
    First,
    /// A stable discovery index.
    Index(u32),
    /// A device UUID, with or without dashes.
    Uuid(String),
}

impl DeviceSpec {
    /// Parse one `--device` value.
    pub fn parse(text: &str) -> Result<DeviceSpec, CliError> {
        let lower = text.to_ascii_lowercase();
        match lower.as_str() {
            "auto" => Ok(DeviceSpec::Auto),
            "cpu" => Ok(DeviceSpec::Cpu),
            "vulkan" => Ok(DeviceSpec::Vulkan(VulkanSelector::First)),
            _ => {
                let Some(rest) = lower.strip_prefix("vulkan:") else {
                    return Err(CliError::usage(format!(
                        "unknown device `{text}`; accepted forms: auto, cpu, vulkan, \
                         vulkan:<index>, vulkan:<uuid>"
                    )));
                };
                if rest.is_empty() {
                    return Ok(DeviceSpec::Vulkan(VulkanSelector::First));
                }
                if rest.chars().all(|character| character.is_ascii_digit()) {
                    let index = rest.parse::<u32>().map_err(|_| {
                        CliError::usage(format!(
                            "`vulkan:{rest}` looks like an index but does not fit in 32 bits"
                        ))
                    })?;
                    return Ok(DeviceSpec::Vulkan(VulkanSelector::Index(index)));
                }
                let uuid = normalize_uuid(rest).ok_or_else(|| {
                    CliError::usage(format!(
                        "`vulkan:{rest}` is neither a device index nor a UUID; pass \
                         `vulkan:<index>` or `vulkan:<uuid>` (32 hex digits, with or without \
                         dashes)"
                    ))
                })?;
                Ok(DeviceSpec::Vulkan(VulkanSelector::Uuid(uuid)))
            }
        }
    }

    /// A test/user-facing constructor for `vulkan:<index>`.
    pub fn vulkan_index(index: u32) -> DeviceSpec {
        DeviceSpec::Vulkan(VulkanSelector::Index(index))
    }

    /// The request as the user spelled it.
    pub fn label(&self) -> String {
        match self {
            DeviceSpec::Auto => "auto".to_string(),
            DeviceSpec::Cpu => "cpu".to_string(),
            DeviceSpec::Vulkan(VulkanSelector::First) => "vulkan".to_string(),
            DeviceSpec::Vulkan(VulkanSelector::Index(index)) => format!("vulkan:{index}"),
            DeviceSpec::Vulkan(VulkanSelector::Uuid(uuid)) => format!("vulkan:{uuid}"),
        }
    }

    /// Resolve against the devices this build reports.
    ///
    /// A Vulkan request in a build without the `vulkan` feature fails here,
    /// before any file is loaded, with the rebuild command.
    pub fn resolve(&self) -> Result<DeviceId, CliError> {
        if matches!(self, DeviceSpec::Vulkan(_)) && !cfg!(feature = "vulkan") {
            return Err(CliError::device(
                "this build was compiled without the Vulkan feature, so no Vulkan device can \
                 be selected; the request is not silently turned into a CPU run",
            )
            .with_next(
                "rebuild with `cargo build --release --no-default-features --features cli,vulkan`, \
                 or pass `--device cpu`; `fn3-infer diagnose` shows the devices in this build",
            ));
        }
        let request = self.request()?;
        request.resolve_on_host().map_err(|error| {
            CliError::device(error.to_string()).with_next(
                "run `fn3-infer diagnose` to see every enumerated device; choose `--device cpu` \
                 to run the reference backend",
            )
        })
    }

    /// Build the core request, resolving a UUID to an index when possible.
    pub fn request(&self) -> Result<DeviceRequest, CliError> {
        match self {
            DeviceSpec::Auto => Ok(DeviceRequest::Auto),
            DeviceSpec::Cpu => Ok(DeviceRequest::Cpu),
            DeviceSpec::Vulkan(VulkanSelector::First) => Ok(DeviceRequest::Vulkan { index: None }),
            DeviceSpec::Vulkan(VulkanSelector::Index(index)) => Ok(DeviceRequest::Vulkan {
                index: Some(*index),
            }),
            DeviceSpec::Vulkan(VulkanSelector::Uuid(uuid)) => {
                #[cfg(feature = "vulkan")]
                {
                    let adapters = crate::fn3::backend::vulkan::adapters();
                    for adapter in adapters {
                        if normalize_uuid(&adapter.uuid_string()).as_deref() == Some(uuid) {
                            return Ok(DeviceRequest::Vulkan {
                                index: Some(adapter.index),
                            });
                        }
                    }
                    let listed = adapters
                        .iter()
                        .map(|adapter| {
                            format!("vulkan:{} ({})", adapter.uuid_string(), adapter.name)
                        })
                        .collect::<Vec<_>>();
                    let listed = if listed.is_empty() {
                        "no Vulkan adapters are enumerated".to_string()
                    } else {
                        format!("enumerated: [{}]", listed.join(", "))
                    };
                    Err(
                        CliError::device(format!("no Vulkan adapter has UUID `{uuid}`; {listed}"))
                            .with_next(
                                "run `fn3-infer diagnose` and copy the UUID it prints, or pass \
                         `vulkan:<index>`",
                            ),
                    )
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    Err(CliError::device(format!(
                        "this build was compiled without the Vulkan feature, so UUID `{uuid}` \
                         cannot be resolved"
                    ))
                    .with_next(
                        "rebuild with `--no-default-features --features cli,vulkan`, or use \
                         `--device cpu`",
                    ))
                }
            }
        }
    }
}

/// One opened backend and the facts needed to report it honestly.
pub struct OpenedBackend {
    /// The executor every instruction of the session goes through.
    pub backend: Arc<dyn Backend>,
    /// What the user requested.
    pub requested: String,
    /// The device the backend reports.
    pub device: DeviceId,
    /// Extra facts (adapter name, software flag, validation).
    pub detail: String,
}

impl OpenedBackend {
    /// Open the backend for `spec`, or fail naming the request and the devices.
    pub fn open(spec: &DeviceSpec) -> Result<OpenedBackend, CliError> {
        #[cfg(feature = "vulkan")]
        if let DeviceSpec::Vulkan(_) = spec {
            // Do not pre-resolve a Vulkan request through the core device list:
            // that list is empty when the loader cannot create an instance, and
            // its generic "no Vulkan backend" reason would hide the driver/ICD
            // diagnosis `VulkanBackend::open` produces.
            let request = spec.request()?;
            return open_vulkan(spec, request);
        }
        let device = spec.resolve()?;
        match device {
            DeviceId::Cpu => Ok(OpenedBackend {
                backend: Arc::new(CpuBackend::new()),
                requested: spec.label(),
                device,
                detail: "reference implementation".to_string(),
            }),
            DeviceId::Vulkan(id) => {
                #[cfg(feature = "vulkan")]
                {
                    let request = DeviceRequest::Vulkan {
                        index: Some(id.index()),
                    };
                    let backend = crate::fn3::backend::vulkan::VulkanBackend::open(&request)
                        .map_err(|error| {
                            CliError::device(format!("cannot open {device}: {error}")).with_next(
                                "install or repair the Vulkan ICD/GPU driver, or run `--device \
                                 cpu`; `fn3-infer diagnose` shows the loader's view",
                            )
                        })?;
                    let detail = format!(
                        "adapter={} software={} validation={}",
                        backend.adapter().name,
                        backend.is_software(),
                        backend.validation_enabled()
                    );
                    let backend: Arc<dyn Backend> = backend;
                    Ok(OpenedBackend {
                        backend,
                        requested: spec.label(),
                        device,
                        detail,
                    })
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    let _ = id;
                    Err(CliError::device(
                        "this build has no Vulkan backend, but the device resolved to Vulkan",
                    ))
                }
            }
        }
    }
}

#[cfg(feature = "vulkan")]
fn open_vulkan(spec: &DeviceSpec, request: DeviceRequest) -> Result<OpenedBackend, CliError> {
    let backend = crate::fn3::backend::vulkan::VulkanBackend::open(&request).map_err(|error| {
        CliError::device(format!("cannot open the requested Vulkan device: {error}")).with_next(
            "install or repair the Vulkan ICD/GPU driver, or run `--device cpu`; \
             `fn3-infer diagnose` shows the loader's view",
        )
    })?;
    let device = backend.device_id();
    let detail = format!(
        "adapter={} software={} validation={}",
        backend.adapter().name,
        backend.is_software(),
        backend.validation_enabled()
    );
    let backend: Arc<dyn Backend> = backend;
    Ok(OpenedBackend {
        backend,
        requested: spec.label(),
        device,
        detail,
    })
}

/// Normalize a UUID to 32 lowercase hex digits, or `None` if it is not one.
///
/// Both the canonical 8-4-4-4-12 form and 32 bare hex digits are accepted.
fn normalize_uuid(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let compact: Vec<u8> = if bytes.len() == 32 {
        bytes.to_vec()
    } else if bytes.len() == 36
        && bytes[8] == b'-'
        && bytes[13] == b'-'
        && bytes[18] == b'-'
        && bytes[23] == b'-'
    {
        bytes
            .iter()
            .enumerate()
            .filter(|(position, _)| !matches!(position, 8 | 13 | 18 | 23))
            .map(|(_, byte)| *byte)
            .collect()
    } else {
        return None;
    };
    if !compact.iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    Some(
        compact
            .iter()
            .map(|byte| char::from(*byte).to_ascii_lowercase())
            .collect(),
    )
}

/// Run `diagnose`.
pub fn execute_diagnose(args: &DiagnoseArgs, out: &mut dyn Write) -> Result<(), CliError> {
    let _ = writeln!(
        out,
        "fn3-infer {} — build/device/backend diagnosis",
        env!("CARGO_PKG_VERSION")
    );
    let _ = writeln!(
        out,
        "  vulkan feature: {}",
        if cfg!(feature = "vulkan") {
            "enabled"
        } else {
            "disabled (rebuild with --features cli,vulkan)"
        }
    );

    if let Some(path) = args.artifact.as_ref() {
        diagnose_artifact(path, out)?;
    }

    let _ = writeln!(out, "\ndevice matrix:");
    let _ = writeln!(out, "  {:<16} {:<12} {}", "request", "state", "evidence");
    let _ = writeln!(
        out,
        "  {:<16} {:<12} {}",
        "cpu", "ready", "backend cpu; reference implementation"
    );

    if cfg!(feature = "vulkan") {
        #[cfg(feature = "vulkan")]
        diagnose_vulkan(args.device.as_ref(), out)?;
    } else {
        let _ = writeln!(
            out,
            "  {:<16} {:<12} this build has no Vulkan backend",
            "vulkan", "unavailable"
        );
    }

    if let Some(spec) = args.device.as_ref() {
        match OpenedBackend::open(spec) {
            Ok(opened) => {
                let capabilities = opened.backend.capabilities();
                let _ = writeln!(
                    out,
                    "\nrequested device `{}`: opened {} with backend {}, {}",
                    opened.requested,
                    opened.device,
                    opened.backend.name(),
                    opened.detail
                );
                let _ = writeln!(
                    out,
                    "  capabilities: dtypes={:?} max_buffer_bytes={} max_rank={} threads={}",
                    capabilities.dtypes,
                    capabilities.max_buffer_bytes,
                    capabilities.max_shape_rank,
                    capabilities.max_threads
                );
            }
            Err(error) => {
                let _ = writeln!(out, "\nrequested device could not be opened");
                return Err(error);
            }
        }
    }

    let _ = writeln!(
        out,
        "\nlegend: enumerated = the loader listed it; opened = the backend built its device \
         and pipelines;\n        ran the model = this command did not execute inference. An \
         advertised GPU that is not opened,\n        and an opened GPU that never ran the \
         model, are not evidence of a working inference path."
    );
    Ok(())
}

fn diagnose_artifact(path: &Path, out: &mut dyn Write) -> Result<(), CliError> {
    let bytes = std::fs::read(path).map_err(|error| {
        CliError::model(format!(
            "cannot read artifact `{}`: {error}",
            path.display()
        ))
        .with_next("check the path, or create it with `fn3-infer prepare`")
    })?;
    let header = PreparedHeader::parse(&bytes).map_err(|error| {
        CliError::model(format!(
            "`{}` is not a usable artifact: {error}",
            path.display()
        ))
        .with_next("re-run `fn3-infer prepare` with the intended ONNX file")
    })?;
    let artifact =
        crate::fn3::model::format::PreparedArtifact::from_bytes(&bytes).map_err(|error| {
            CliError::model(format!(
                "`{}` failed its payload checks: {error}",
                path.display()
            ))
            .with_next("the file is corrupt; re-run `fn3-infer prepare` from the source ONNX")
        })?;
    let geometry = artifact.geometry();
    let envelope = artifact.envelope();
    let _ = writeln!(
        out,
        "\nartifact: {} ({} bytes, crc32 {:#010x})",
        path.display(),
        header.file_bytes,
        header.payload_crc32
    );
    let _ = writeln!(
        out,
        "  geometry: layers={} hidden={} mlp={} heads={}/{} head_dim={} vocab={}",
        geometry.layers,
        geometry.hidden_size,
        geometry.mlp_hidden_size,
        geometry.query_heads,
        geometry.kv_heads,
        geometry.head_dim,
        geometry.vocab_size
    );
    let _ = writeln!(
        out,
        "  envelope: batch {}..={} seq {}..={} past {}..={} total {}..={}",
        envelope.get(crate::fn3::Symbol::Batch).min(),
        envelope.get(crate::fn3::Symbol::Batch).max(),
        envelope.get(crate::fn3::Symbol::Seq).min(),
        envelope.get(crate::fn3::Symbol::Seq).max(),
        envelope.get(crate::fn3::Symbol::Past).min(),
        envelope.get(crate::fn3::Symbol::Past).max(),
        envelope.get(crate::fn3::Symbol::Total).min(),
        envelope.get(crate::fn3::Symbol::Total).max(),
    );
    Ok(())
}

#[cfg(feature = "vulkan")]
fn diagnose_vulkan(request: Option<&DeviceSpec>, out: &mut dyn Write) -> Result<(), CliError> {
    use crate::fn3::backend::vulkan::{adapters, VulkanBackend};

    let adapters = adapters();
    if adapters.is_empty() {
        let _ = writeln!(
            out,
            "  {:<16} {:<12} {}",
            "vulkan", "unavailable", "the loader reported zero physical devices"
        );
        return Ok(());
    }
    for adapter in adapters {
        let uuid = adapter.uuid_string();
        let _ = writeln!(
            out,
            "  {:<16} {:<12} [{}] {} uuid={} compute={} software={}",
            format!("vulkan({})", adapter.index),
            "enumerated",
            adapter.index,
            adapter.name,
            uuid,
            adapter.has_compute_queue(),
            adapter.is_software()
        );
        if !adapter.has_compute_queue() {
            let _ = writeln!(out, "  {:<16} {:<12} no compute queue", "", "not-opened");
            continue;
        }
        if let Some(spec) = request {
            // Only open the requested device; the others stay enumerated.
            let requested = spec.request()?;
            let matches = match requested {
                DeviceRequest::Vulkan { index: None } => true,
                DeviceRequest::Vulkan { index } => index == Some(adapter.index),
                _ => false,
            };
            if !matches {
                continue;
            }
        }
        match VulkanBackend::open_with(
            &DeviceRequest::Vulkan {
                index: Some(adapter.index),
            },
            crate::fn3::backend::vulkan::VulkanConfig::default(),
        ) {
            Ok(backend) => {
                let capabilities = backend.capabilities();
                let _ = writeln!(
                    out,
                    "  {:<16} {:<12} adapter={} software={} validation={} dtypes={:?} \
                     max_buffer_bytes={}",
                    "",
                    "opened",
                    backend.adapter().name,
                    backend.is_software(),
                    backend.validation_enabled(),
                    capabilities.dtypes,
                    capabilities.max_buffer_bytes
                );
            }
            Err(error) => {
                let _ = writeln!(out, "  {:<16} {:<12} {error}", "", "open-failed");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_forms_map_to_requests() {
        assert_eq!(DeviceSpec::parse("auto").unwrap(), DeviceSpec::Auto);
        assert_eq!(DeviceSpec::parse("cpu").unwrap(), DeviceSpec::Cpu);
        assert_eq!(
            DeviceSpec::parse("vulkan").unwrap(),
            DeviceSpec::Vulkan(VulkanSelector::First)
        );
        assert_eq!(
            DeviceSpec::parse("vulkan:2").unwrap(),
            DeviceSpec::Vulkan(VulkanSelector::Index(2))
        );
    }

    #[test]
    fn uuids_accept_both_forms_and_reject_garbage() {
        let dashed = "471b2bda-51e1-6451-e692-ff5d1e146f1e";
        let compact = "471b2bda51e16451e692ff5d1e146f1e";
        let expected = Some(compact.to_string());
        assert_eq!(normalize_uuid(dashed), expected);
        assert_eq!(normalize_uuid(compact), expected);
        assert_eq!(normalize_uuid("471b2bda-51e1-6451-e692"), None);
        assert_eq!(normalize_uuid("not-a-uuid"), None);
        assert_eq!(normalize_uuid("471b2bda51e16451e692ff5d1e146f1z"), None);
    }

    #[test]
    fn unknown_or_malformed_devices_name_the_accepted_forms() {
        let error = DeviceSpec::parse("gpu").expect_err("unknown");
        assert!(error.message().contains("vulkan:<uuid>"), "{error:?}");
        let error = DeviceSpec::parse("vulkan:wat").expect_err("bad uuid");
        assert!(error.message().contains("index"), "{error:?}");
    }

    #[test]
    fn an_explicit_vulkan_request_never_falls_back_to_cpu() {
        if cfg!(feature = "vulkan") {
            // This machine has a Vulkan device; the no-fallback path is pinned
            // by the no-vulkan build and the CLI integration test.
            return;
        }
        let error = DeviceSpec::parse("vulkan").unwrap().resolve().unwrap_err();
        assert_eq!(error.code(), super::super::EXIT_DEVICE);
        assert!(error.message().contains("Vulkan") || error.message().contains("vulkan"));
    }
}
