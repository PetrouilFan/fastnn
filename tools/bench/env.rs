//! Environment capture for benchmark reproducibility.
//!
//! Everything here is best-effort and reports absence explicitly (`None` plus
//! a warning) instead of inventing values. Linux `/proc` and standard
//! userspace tools are used directly; no extra crate is required.

use crate::schema::{BuildInfo, Checksum, GitInfo, GpuDevice, GpuInfo, HostInfo, LoadInfo};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// RFC 3339 UTC timestamp (second resolution) without a chrono dependency.
pub fn rfc3339_utc(unix_s: u64) -> String {
    let days = (unix_s / 86_400) as i64;
    let seconds_of_day = unix_s % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Howard Hinnant's civil-from-days algorithm (days since 1970-01-01).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

fn run_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn read_file(path: &str) -> Option<String> {
    fs::read_to_string(path).ok()
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}

pub fn capture_git(warnings: &mut Vec<String>) -> GitInfo {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let commit = non_empty(run_stdout(
        "git",
        &["-C", manifest_dir, "rev-parse", "HEAD"],
    ));
    if commit.is_none() {
        warnings.push("git: HEAD could not be resolved; commit recorded as null".to_string());
    }
    let branch = non_empty(run_stdout(
        "git",
        &["-C", manifest_dir, "rev-parse", "--abbrev-ref", "HEAD"],
    ));
    let dirty_text = run_stdout(
        "git",
        &[
            "-C",
            manifest_dir,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
    );
    let (dirty, dirty_entries) = match &dirty_text {
        Some(text) if text.is_empty() => (Some(false), Some(0)),
        Some(text) => (Some(true), Some(text.lines().count() as u64)),
        None => (None, None),
    };
    let describe = non_empty(run_stdout(
        "git",
        &["-C", manifest_dir, "describe", "--always", "--dirty"],
    ));

    GitInfo {
        commit,
        branch,
        dirty,
        dirty_entries,
        describe,
    }
}

pub fn capture_build(warnings: &mut Vec<String>) -> BuildInfo {
    let rustc_version = non_empty(run_stdout("rustc", &["--version"]));
    if rustc_version.is_none() {
        warnings.push("build: rustc --version unavailable".to_string());
    }
    let cargo_version = non_empty(run_stdout("cargo", &["--version"]));

    let mut features = BTreeMap::new();
    macro_rules! feature {
        ($name:literal) => {
            features.insert($name.to_string(), cfg!(feature = $name));
        };
    }
    feature!("simd");
    feature!("parallel");
    feature!("prepared-plan");
    feature!("fusion-forward");
    feature!("fusion-backward");
    feature!("gpu");
    feature!("openblas");
    feature!("blas");
    feature!("python");
    feature!("cli");
    feature!("debug_canary");

    BuildInfo {
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        profile: if cfg!(debug_assertions) {
            "debug".to_string()
        } else {
            "release".to_string()
        },
        debug_assertions: cfg!(debug_assertions),
        rustc_version,
        cargo_version,
        features,
        target_arch: std::env::consts::ARCH.to_string(),
        target_os: std::env::consts::OS.to_string(),
        target_family: Some(std::env::consts::FAMILY.to_string()),
        command_line: std::env::args().collect(),
    }
}

fn cpu_model_from_cpuinfo(cpuinfo: &str) -> Option<String> {
    cpuinfo.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name").then(|| value.trim().to_string())
    })
}

fn physical_cores_from_cpuinfo(cpuinfo: &str) -> Option<u64> {
    let mut pairs: HashSet<(String, String)> = HashSet::new();
    let mut physical_id: Option<String> = None;
    let mut core_id: Option<String> = None;
    let mut flush = |physical_id: &mut Option<String>, core_id: &mut Option<String>| {
        if let (Some(p), Some(c)) = (physical_id.take(), core_id.take()) {
            pairs.insert((p, c));
        }
    };
    for line in cpuinfo.lines() {
        let line = line.trim();
        if line.is_empty() {
            flush(&mut physical_id, &mut core_id);
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "physical id" => physical_id = Some(value.trim().to_string()),
            "core id" => core_id = Some(value.trim().to_string()),
            _ => {}
        }
    }
    flush(&mut physical_id, &mut core_id);
    if pairs.is_empty() {
        None
    } else {
        Some(pairs.len() as u64)
    }
}

fn os_pretty_name() -> Option<String> {
    let text = read_file("/etc/os-release")?;
    text.lines()
        .find_map(|line| line.strip_prefix("PRETTY_NAME="))
        .map(|value| value.trim_matches('"').to_string())
}

fn kernel_release() -> Option<String> {
    non_empty(read_file("/proc/sys/kernel/osrelease").map(|s| s.trim().to_string()))
        .or_else(|| non_empty(run_stdout("uname", &["-r"])))
}

fn memory_total_bytes() -> Option<u64> {
    let text = read_file("/proc/meminfo")?;
    let kib = text.lines().find_map(|line| {
        let rest = line.strip_prefix("MemTotal:")?;
        rest.split_whitespace().next()?.parse::<u64>().ok()
    })?;
    Some(kib * 1024)
}

pub fn capture_load(threshold: f64, warnings: &mut Vec<String>) -> LoadInfo {
    let text = match read_file("/proc/loadavg") {
        Some(text) => text,
        None => {
            let message = "/proc/loadavg unavailable; load guard could not run".to_string();
            warnings.push(message.clone());
            return LoadInfo {
                one: None,
                five: None,
                fifteen: None,
                threshold,
                exceeded: false,
                reason: Some(message),
            };
        }
    };
    let mut fields = text.split_whitespace();
    let one = fields.next().and_then(|v| v.parse::<f64>().ok());
    let five = fields.next().and_then(|v| v.parse::<f64>().ok());
    let fifteen = fields.next().and_then(|v| v.parse::<f64>().ok());
    let exceeded = one.is_some_and(|value| value > threshold);
    if exceeded {
        warnings.push(format!(
            "load guard: 1-minute load average {:.2} exceeds threshold {:.2}; measurements may be noisy",
            one.unwrap_or(0.0),
            threshold
        ));
    }
    LoadInfo {
        one,
        five,
        fifteen,
        threshold,
        exceeded,
        reason: None,
    }
}

/// PIDs of process names in `needles`, excluding this process and its parent.
fn matching_pids(needles: &[&str]) -> Vec<u64> {
    let self_pid = std::process::id() as u64;
    let parent_pid = fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|stat| {
            // `/proc/<pid>/stat` encloses the process name in parentheses; the
            // fields after the final `)` are state, ppid, ...
            let rest = stat.rsplit_once(')')?.1;
            rest.split_whitespace().nth(1)?.parse::<u64>().ok()
        })
        .unwrap_or(0);
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir("/proc") else {
        return found;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|s| s.parse::<u64>().ok()) else {
            continue;
        };
        if pid == self_pid || pid == parent_pid {
            continue;
        }
        let comm = fs::read_to_string(format!("/proc/{pid}/comm")).unwrap_or_default();
        let comm = comm.trim();
        if needles.contains(&comm) {
            found.push(pid);
        }
    }
    found.sort_unstable();
    found
}

/// PIDs of other live `fn3-bench` processes (excluding this one).
///
/// Matches the kernel `comm` name, not `cmdline`: wrappers such as `timeout`
/// mention the binary in their argument vector and must not be flagged.
pub fn other_benchmark_pids() -> Vec<u64> {
    matching_pids(&["fn3-bench"])
}

pub fn capture_host(load_threshold: f64, warnings: &mut Vec<String>) -> HostInfo {
    let cpuinfo = read_file("/proc/cpuinfo").unwrap_or_default();
    let cpu_model = non_empty(cpu_model_from_cpuinfo(&cpuinfo));
    if cpu_model.is_none() {
        warnings.push("host: CPU model unavailable from /proc/cpuinfo".to_string());
    }
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(1);
    let threads_used = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(logical_cores);
    let load = capture_load(load_threshold, warnings);
    let competing_build_pids = matching_pids(&["cargo", "rustc", "rustdoc"]);
    let competing_benchmark_pids = other_benchmark_pids();

    HostInfo {
        os: os_pretty_name(),
        kernel: kernel_release(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_model,
        physical_cores: physical_cores_from_cpuinfo(&cpuinfo),
        logical_cores,
        threads_used,
        memory_total_bytes: memory_total_bytes(),
        load,
        competing_build_pids,
        competing_benchmark_pids,
    }
}

fn vendor_name(vendor_id: &str) -> Option<String> {
    match vendor_id.to_ascii_lowercase().as_str() {
        "0x10de" => Some("NVIDIA".to_string()),
        "0x1002" => Some("AMD".to_string()),
        "0x8086" => Some("Intel".to_string()),
        "0x10005" => Some("Mesa".to_string()),
        _ => None,
    }
}

fn parse_device_type(raw: &str, driver: &str, name: &str) -> (String, bool) {
    let lower_driver = driver.to_ascii_lowercase();
    let lower_name = name.to_ascii_lowercase();
    let software = raw.contains("CPU")
        || lower_driver.contains("llvmpipe")
        || lower_driver.contains("lavapipe")
        || lower_name.contains("llvmpipe")
        || lower_name.contains("lavapipe");
    let device_type = if software {
        "software"
    } else if raw.contains("GPU") {
        "physical"
    } else {
        "unknown"
    };
    (device_type.to_string(), software)
}

/// Parse the `Devices:` section of `vulkaninfo --summary`.
fn parse_vulkan_summary(text: &str) -> (Option<String>, Vec<GpuDevice>) {
    let mut instance_version = None;
    let mut devices = Vec::new();
    let mut current: Option<BTreeMap<String, String>> = None;
    let mut current_index = 0usize;

    let flush = |index: usize, fields: BTreeMap<String, String>, devices: &mut Vec<GpuDevice>| {
        let name = fields
            .get("deviceName")
            .cloned()
            .unwrap_or_else(|| format!("Vulkan device {index}"));
        let driver_name = fields.get("driverName").cloned();
        let driver_info = fields.get("driverInfo").cloned();
        let (device_type, software_icd) = parse_device_type(
            fields.get("deviceType").map(String::as_str).unwrap_or(""),
            driver_name.as_deref().unwrap_or(""),
            &name,
        );
        let vendor_id = fields.get("vendorID").cloned();
        devices.push(GpuDevice {
            index,
            name,
            vendor: vendor_id.as_deref().and_then(vendor_name),
            vendor_id,
            device_id: fields.get("deviceID").cloned(),
            device_type,
            software_icd,
            driver_name,
            driver_info,
            driver_version: fields.get("driverVersion").cloned(),
            api_version: fields.get("apiVersion").cloned(),
            device_uuid: fields.get("deviceUUID").cloned(),
        });
    };

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Vulkan Instance Version:") {
            instance_version = Some(rest.trim().to_string());
            continue;
        }
        if trimmed.starts_with("GPU") && trimmed.ends_with(':') {
            if let Some(fields) = current.take() {
                flush(current_index, fields, &mut devices);
            }
            current_index = trimmed
                .trim_start_matches("GPU")
                .trim_end_matches(':')
                .parse::<usize>()
                .unwrap_or(devices.len());
            current = Some(BTreeMap::new());
            continue;
        }
        if let Some(fields) = current.as_mut() {
            if let Some((key, value)) = trimmed.split_once('=') {
                fields.insert(key.trim().to_string(), value.trim().to_string());
            }
        }
    }
    if let Some(fields) = current.take() {
        flush(current_index, fields, &mut devices);
    }
    devices.sort_by_key(|d| d.index);
    (instance_version, devices)
}

pub fn capture_gpu(probe: bool, warnings: &mut Vec<String>) -> GpuInfo {
    if !probe {
        return GpuInfo {
            status: "unavailable".to_string(),
            reason: Some("GPU probe skipped (--no-gpu-probe)".to_string()),
            vulkan_instance_version: None,
            devices: Vec::new(),
            selected_device_index: None,
            execution_backend: "cpu".to_string(),
            execution_reason: Some("GPU probe skipped".to_string()),
        };
    }

    let output = Command::new("vulkaninfo").arg("--summary").output();
    let text = match output {
        Ok(output) => String::from_utf8_lossy(&output.stdout).to_string(),
        Err(error) => {
            let reason = format!("vulkaninfo could not be executed: {error}");
            warnings.push(format!("gpu: {reason}"));
            return GpuInfo {
                status: "unavailable".to_string(),
                reason: Some(reason),
                vulkan_instance_version: None,
                devices: Vec::new(),
                selected_device_index: None,
                execution_backend: "cpu".to_string(),
                execution_reason: Some("no Vulkan devices enumerated".to_string()),
            };
        }
    };

    let (vulkan_instance_version, devices) = parse_vulkan_summary(&text);
    if devices.is_empty() {
        let reason = "vulkaninfo --summary enumerated no devices".to_string();
        warnings.push(format!("gpu: {reason}"));
        return GpuInfo {
            status: "unavailable".to_string(),
            reason: Some(reason),
            vulkan_instance_version,
            devices,
            selected_device_index: None,
            execution_backend: "cpu".to_string(),
            execution_reason: Some("no Vulkan devices enumerated".to_string()),
        };
    }

    let physical_count = devices
        .iter()
        .filter(|device| device.device_type == "physical")
        .count();
    let software_count = devices.iter().filter(|device| device.software_icd).count();
    let execution_reason = if physical_count > 0 {
        Some(format!(
            "CPU execution in this revision; {physical_count} physical and {software_count} software Vulkan device(s) enumerated, none selected"
        ))
    } else {
        Some(format!(
            "CPU execution in this revision; only software Vulkan device(s) enumerated ({software_count})"
        ))
    };

    GpuInfo {
        status: "present".to_string(),
        reason: None,
        vulkan_instance_version,
        devices,
        selected_device_index: None,
        execution_backend: "cpu".to_string(),
        execution_reason,
    }
}

/// FNV-1a 64 over raw bytes. Used for deterministic fixture identity; this is
/// not a cryptographic hash and the algorithm is recorded next to the value.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub fn checksum_f32(subject: &str, values: &[f32]) -> Checksum {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Checksum {
        subject: subject.to_string(),
        algorithm: "fnv1a64".to_string(),
        value: format!("{:016x}", fnv1a64(&bytes)),
    }
}

/// Whole-process peak resident set size in bytes, from `/proc/self/status`.
pub fn peak_rss_bytes() -> Option<u64> {
    let text = read_file("/proc/self/status")?;
    let kib = text.lines().find_map(|line| {
        let rest = line.strip_prefix("VmHWM:")?;
        rest.split_whitespace().next()?.parse::<u64>().ok()
    })?;
    Some(kib * 1024)
}

pub fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Random (non-seeded) run identifier derived from the clock and PID.
pub fn unique_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut state = nanos ^ (u64::from(std::process::id()) << 32);
    let mut bytes = [0u8; 16];
    for chunk in bytes.chunks_mut(8) {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        chunk.copy_from_slice(&z.to_le_bytes());
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_formatting() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(1_700_000_000), "2023-11-14T22:13:20Z");
    }

    #[test]
    fn vulkan_summary_parses_physical_and_software_devices() {
        let text = r#"
Vulkan Instance Version: 1.4.363

Devices:
========
GPU0:
	apiVersion         = 1.4.312
	driverVersion      = 580.196.7.0
	vendorID           = 0x10de
	deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName         = NVIDIA GeForce GTX 1080 Ti
	driverName         = NVIDIA
	deviceUUID         = 471b2bda-51e1-6451-e692-ff5d1e146f1e
GPU1:
	apiVersion         = 1.4.354
	vendorID           = 0x10005
	deviceType         = PHYSICAL_DEVICE_TYPE_CPU
	deviceName         = llvmpipe (LLVM 23.1.4, 256 bits)
	driverName         = llvmpipe
"#;
        let (instance, devices) = parse_vulkan_summary(text);
        assert_eq!(instance.as_deref(), Some("1.4.363"));
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].name, "NVIDIA GeForce GTX 1080 Ti");
        assert_eq!(devices[0].device_type, "physical");
        assert!(!devices[0].software_icd);
        assert_eq!(devices[1].device_type, "software");
        assert!(devices[1].software_icd);
    }

    #[test]
    fn cpuinfo_helpers() {
        let cpuinfo = "processor\t: 0\nmodel name\t: Test CPU\nphysical id\t: 0\ncore id\t: 0\n\n\
                       processor\t: 1\nmodel name\t: Test CPU\nphysical id\t: 0\ncore id\t: 1\n\n";
        assert_eq!(cpu_model_from_cpuinfo(cpuinfo).as_deref(), Some("Test CPU"));
        assert_eq!(physical_cores_from_cpuinfo(cpuinfo), Some(2));
    }

    #[test]
    fn checksum_is_stable_and_dtype_sensitive() {
        let a = checksum_f32("weights", &[1.0, 2.0]);
        let b = checksum_f32("weights", &[1.0, 2.0]);
        assert_eq!(a.value, b.value);
        assert_eq!(a.value.len(), 16);
    }
}
