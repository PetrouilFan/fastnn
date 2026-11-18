//! FN3-05 native Vulkan compute bring-up.
//!
//! Runs one end-to-end storage-buffer FMA dispatch plus a repeated-submission
//! reuse test on a selected physical device, with optional validation layers.
//! See README.md for the CLI contract and the evidence commands.

mod args;
mod compute;
mod device;
mod error;
mod ledger;
mod platform;
mod probe;

use ash::vk;

use compute::{DispatchReport, RunConfig};
use error::{Error, Result};

fn main() {
    match run() {
        Ok(()) => {}
        Err(err) => {
            eprintln!("[fn3vk][error] {err}");
            if let Some(hint) = err.hint() {
                eprintln!("[fn3vk][hint] {hint}");
            }
            let code = match err {
                Error::Usage(_) | Error::Loader(_) | Error::Vk { .. } => 2,
                Error::Internal(_) => 1,
            };
            std::process::exit(code);
        }
    }
}

fn run() -> Result<()> {
    let opts = args::parse()?;
    if opts.help {
        print!("{}", args::help());
        return Ok(());
    }

    println!(
        "[fn3vk] fn3-vk-bringup {} (ash 0.38.0, raw Vulkan loader)",
        env!("CARGO_PKG_VERSION")
    );

    // The loader reads VK_ICD_FILENAMES when the instance is created, so the
    // override must be installed before that call. This happens before any
    // worker thread exists, hence no concurrent environment access.
    let icd_override = match (&opts.icd, opts.software) {
        (Some(path), _) => Some(path.clone()),
        (None, true) => Some(platform::software_icd_path().to_string()),
        (None, false) => None,
    };
    if let Some(path) = &icd_override {
        println!("[fn3vk] ICD override: VK_ICD_FILENAMES={path}");
        if opts.software {
            println!(
                "[fn3vk] software lane: lavapipe evidence is CI-only and does NOT \
                 satisfy the physical-GPU gate"
            );
        }
        std::env::set_var("VK_ICD_FILENAMES", path);
    }

    let mut instance =
        platform::InstanceHandle::create(opts.validation, opts.loader_path.as_deref())?;

    match instance.loader_api {
        Some(version) => println!(
            "[fn3vk] loader instance API: {}",
            probe::format_version(version)
        ),
        None => println!("[fn3vk] loader instance API: 1.0 (vkEnumerateInstanceVersion absent)"),
    }
    println!(
        "[fn3vk] validation: {}",
        if instance.validation {
            "ENABLED (VK_LAYER_KHRONOS_validation -> stderr)"
        } else {
            "disabled"
        }
    );

    let devices = instance.enumerate()?;
    println!(
        "[fn3vk] selector: {}",
        match &opts.selector {
            Some(selector) => format!("{selector:?}"),
            None => "none (default: compute-capable, discrete GPU preferred)".to_string(),
        }
    );
    println!("[fn3vk] physical devices: {}", devices.len());
    for device in &devices {
        println!("[fn3vk]   {}", device.summary_line());
    }

    if opts.list {
        println!();
        for device in &devices {
            probe::print_device(device, true);
            println!();
        }
        instance.teardown();
        assert_balanced("instance", &instance.ledger)?;
        println!("[fn3vk] result: LIST OK");
        return Ok(());
    }

    let chosen = platform::choose_device(&devices, opts.selector.as_ref())?;
    platform::check_requirements(
        &devices[chosen],
        &opts.require_extensions,
        opts.require_min_api,
    )?;
    let info = devices[chosen].clone();

    println!();
    println!("[fn3vk] selected device {} of {}", chosen, devices.len());
    probe::print_device(&info, false);

    {
        let mut ctx = device::DeviceContext::create(&instance, info)?;
        println!();
        println!(
            "[fn3vk] logical device: queue family {} ({})",
            ctx.family_index,
            probe::queue_flags_string(
                ctx.info
                    .queue_families
                    .iter()
                    .find(|f| f.index == ctx.family_index)
                    .map(|f| f.flags)
                    .unwrap_or(vk::QueueFlags::empty())
            )
        );

        let cfg = RunConfig {
            count: opts.count,
            repeats: opts.repeats,
        };
        println!(
            "[fn3vk] workload: count={} workgroups={} repeats={} local_size={}",
            cfg.count,
            cfg.count.div_ceil(64),
            cfg.repeats,
            64
        );
        println!(
            "[fn3vk] buffers: 4 x {} B device-local storage, {} B upload staging, {} B readback staging",
            u64::from(cfg.count) * 4,
            3 * u64::from(cfg.count) * 4,
            u64::from(cfg.count) * 4
        );

        let engine = compute::Engine::new(&ctx, cfg)?;
        let (report, compute_ledger) = engine.run()?;
        print_report(&report);
        assert_balanced("compute", &compute_ledger)?;

        ctx.teardown();
        assert_balanced("device", &ctx.ledger)?;
    }

    instance.teardown();
    assert_balanced("instance", &instance.ledger)?;

    let (errors, warnings) = platform::validation_counts();
    if instance.validation {
        println!("[fn3vk] validation summary: errors={errors} warnings={warnings}");
        if errors > 0 {
            return Err(Error::internal(format!(
                "validation reported {errors} error(s); see the stderr lines above"
            )));
        }
    } else {
        println!("[fn3vk] validation summary: disabled (no layer installed by this run)");
    }

    println!("[fn3vk] result: PASS");
    Ok(())
}

fn print_report(report: &DispatchReport) {
    println!(
        "[fn3vk] dispatch: count={} workgroups={} upload={} B readback={} B wall={:.3} ms",
        report.count,
        report.workgroups,
        report.upload_bytes,
        report.readback_bytes,
        report.primary_ms
    );
    println!(
        "[fn3vk] verify: max_abs_error={} exact_matches={}/{}",
        report.max_abs_error, report.exact_matches, report.count
    );
    println!(
        "[fn3vk] repeat: submissions={} samples={} sample_mismatches={} \
         min={:.3} ms mean={:.3} ms max={:.3} ms",
        report.repeats,
        report.sample_checks,
        report.sample_mismatches,
        report.repeat_min_ms,
        report.repeat_mean_ms,
        report.repeat_max_ms
    );
    println!(
        "[fn3vk] repeat verify: max_abs_error={} exact_matches={}/{}",
        report.final_max_abs_error, report.final_exact, report.count
    );
    println!(
        "[fn3vk] submissions total: {} (1 upload+dispatch, {} repeated dispatches, 1 final readback)",
        report.submits, report.repeats
    );
}

fn assert_balanced(label: &str, ledger: &ledger::Ledger) -> Result<()> {
    print!("[fn3vk] handles[{label}]:\n{}", ledger.report());
    let outstanding = ledger.outstanding();
    if ledger.is_balanced() {
        Ok(())
    } else {
        Err(Error::internal(format!(
            "handle ledger is not balanced for {label}: {}",
            outstanding.join("; ")
        )))
    }
}
