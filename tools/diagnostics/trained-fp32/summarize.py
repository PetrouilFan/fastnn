#!/usr/bin/env python3
"""Assemble the #76 evidence tables from the raw probe outputs.

Reads the fn3 layer probe, the ORT intermediate dump, the float64 reference and
the parity-test report, then writes:

* ``summary.json`` — one machine-readable record of every decision metric;
* ``layer-table.md`` — the per-layer error table used in the report.

Nothing here recomputes a model; it only reduces the captured tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROMPT_MEAN_ABS_BOUND = 5e-5
TINY_MEAN_ABS_BOUND = 2.5e-6
ATOL = 5e-4
RTOL = 1e-4


def load_index(root: Path, phase: str) -> dict[str, np.ndarray]:
    index = json.loads((root / phase / "index.json").read_text())
    entries = index.get("tensors") or index.get("captured")
    values = {}
    for entry in entries:
        path = root / phase / entry["file"]
        data = np.load(path) if path.suffix == ".npy" else np.fromfile(path, dtype="<f4")
        values[entry["name"]] = data.reshape(entry["shape"])
    return values


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    a = actual.astype(np.float64).reshape(-1)
    r = reference.astype(np.float64).reshape(-1)
    delta = a - r
    rms_reference = float(np.sqrt(np.mean(r * r)))
    rms_delta = float(np.sqrt(np.mean(delta * delta)))
    centered = delta - delta.mean()
    return {
        "mean_abs": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_signed": float(delta.mean()),
        "std": float(delta.std()),
        "centered_rms": float(np.sqrt(np.mean(centered * centered))),
        "relative_rms": rms_delta / rms_reference if rms_reference else None,
        "rms_reference": rms_reference,
        "reference_mean_abs": float(np.mean(np.abs(r))),
        "reference_mean": float(r.mean()),
    }


def logit_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    entry = metrics(actual, reference)
    a = actual.astype(np.float64).reshape(-1)
    r = reference.astype(np.float64).reshape(-1)
    delta = a - r
    bound = ATOL + RTOL * np.abs(r)
    order = np.argsort(-r)
    actual_order = np.argsort(-a)
    p = np.exp(r - r.max())
    p /= p.sum()
    q = np.exp(a - a.max())
    q /= q.sum()
    entry.update(
        {
            "elements": int(a.size),
            "violations": int(np.count_nonzero(np.abs(delta) > bound)),
            "mean_abs_within_5e-5": bool(np.mean(np.abs(delta)) <= PROMPT_MEAN_ABS_BOUND),
            "top1": int(np.argmax(a)),
            "reference_top1": int(np.argmax(r)),
            "top5_overlap": len(
                set(int(i) for i in actual_order[:5]) & set(int(i) for i in order[:5])
            ),
            "reference_margin": float(r[order[0]] - r[order[1]]),
            "kl": float(np.sum(p * np.log((p + 1e-300) / (q + 1e-300)))),
            "scale_aware_bound": PROMPT_MEAN_ABS_BOUND
            + 5e-6 * float(np.mean(np.abs(r))),
        }
    )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn3", required=True, type=Path)
    parser.add_argument("--ort", required=True, type=Path)
    parser.add_argument("--f64", required=True, type=Path)
    parser.add_argument("--parity", required=True, type=Path)
    parser.add_argument("--vulkan-parity", type=Path)
    parser.add_argument("--tiny-log", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "existing_mean_abs_bound": PROMPT_MEAN_ABS_BOUND,
        "tiny_mean_abs_bound": TINY_MEAN_ABS_BOUND,
        "phases": {},
    }
    for phase in ("prefill", "decode1", "decode2"):
        fn3 = load_index(args.fn3, phase)
        ort = load_index(args.ort, phase)
        f64 = load_index(args.f64, phase)
        def last(logits: np.ndarray) -> np.ndarray:
            # The committed trained gate compares the final prompt position;
            # decode phases are already [1, 1, vocab].
            return logits[:, -1, :] if phase == "prefill" else logits

        phase_entry = {
            "fn3_vs_ort_logits": logit_metrics(last(fn3["logits"]), last(ort["logits"])),
            "fn3_vs_f64_logits": logit_metrics(last(fn3["logits"]), last(f64["logits"])),
            "ort_vs_f64_logits": logit_metrics(last(ort["logits"]), last(f64["logits"])),
            "full_prefill_fn3_vs_ort": (
                logit_metrics(fn3["logits"], ort["logits"])
                if phase == "prefill"
                else None
            ),
        }
        stages = {}
        for layer in range(12):
            name = f"/transformer/h.{layer}/Add_1_output_0"
            if name not in fn3 or name not in ort or name not in f64:
                continue
            stages[f"{layer:02d}"] = {
                "name": name,
                "fn3_vs_ort": metrics(fn3[name], ort[name]),
                "fn3_vs_f64": metrics(fn3[name], f64[name]),
                "ort_vs_f64": metrics(ort[name], f64[name]),
            }
        phase_entry["block_outputs"] = stages
        report["phases"][phase] = phase_entry

    # The acceptance test's own report is authoritative for the committed
    # payload and the Vulkan arm.
    report["parity_test_cpu"] = json.loads(args.parity.read_text())
    if args.vulkan_parity and args.vulkan_parity.is_file():
        report["parity_test_vulkan_staged"] = json.loads(args.vulkan_parity.read_text())

    # Decode step 2 has no committed payload, so neither parity report carries
    # a full metric for it. The logits files do, and the ORT probe is the
    # reference.
    def read_f32(path: Path) -> np.ndarray:
        return np.fromfile(path, dtype="<f4").astype(np.float64)

    def ort_logits(phase: str) -> np.ndarray:
        index = json.loads((args.ort / phase / "index.json").read_text())
        entry = next(item for item in index["tensors"] if item["name"] == "logits")
        return np.load(args.ort / phase / entry["file"]).reshape(-1)

    parity_dir = args.parity.parent
    vulkan_dir = args.vulkan_parity.parent if args.vulkan_parity else None
    extra = {}
    for phase, suffix in (("decode2", "decode.step2"),):
        reference = ort_logits(phase)
        cpu = read_f32(parity_dir / f"cpu.{suffix}.logits.f32")
        extra[f"cpu_{phase}_vs_ort"] = logit_metrics(cpu, reference)
        if vulkan_dir and (vulkan_dir / f"vulkan.{suffix}.logits.f32").is_file():
            vulkan = read_f32(vulkan_dir / f"vulkan.{suffix}.logits.f32")
            extra[f"vulkan_{phase}_vs_ort"] = logit_metrics(vulkan, reference)
            extra[f"vulkan_{phase}_vs_cpu"] = logit_metrics(vulkan, cpu)
    report["uncommitted_decode2"] = extra

    # Relative-to-scale comparison of the two suite bounds, computed from the
    # committed summary, not from fn3's output.
    tiny_reference_mean_abs = 0.17982319  # fixtures/parity/tiny-gpt2 payload
    trained_reference_mean_abs = 49.45502253115944  # committed last-logits summary
    report["bound_scale"] = {
        "tiny": {
            "reference_mean_abs": tiny_reference_mean_abs,
            "bound": TINY_MEAN_ABS_BOUND,
            "bound_relative_to_reference": TINY_MEAN_ABS_BOUND / tiny_reference_mean_abs,
        },
        "trained": {
            "reference_mean_abs": trained_reference_mean_abs,
            "bound": PROMPT_MEAN_ABS_BOUND,
            "bound_relative_to_reference": PROMPT_MEAN_ABS_BOUND
            / trained_reference_mean_abs,
        },
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Per-layer error, trained GPT-2, fn3 FP32 vs ORT and float64",
        "",
        "Generated by `tools/diagnostics/trained-fp32/summarize.py`. "
        "`mean_abs` is the suite metric; `centered_rms` removes the common-mode "
        "shift; `rel_rms` is `rms(delta)/rms(reference)`.",
        "",
        "## Prefill p13, every block output",
        "",
        "| Layer | fn3-ORT mean_abs | fn3-ORT max_abs | fn3-ORT centered_rms | fn3-ORT rel_rms | fn3-f64 rel_rms | ORT-f64 rel_rms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer, entry in report["phases"]["prefill"]["block_outputs"].items():
        a, b, c = entry["fn3_vs_ort"], entry["fn3_vs_f64"], entry["ort_vs_f64"]
        lines.append(
            f"| {layer} | {a['mean_abs']:.3e} | {a['max_abs']:.3e} | "
            f"{a['centered_rms']:.3e} | {a['relative_rms']:.3e} | "
            f"{b['relative_rms']:.3e} | {c['relative_rms']:.3e} |"
        )
    lines += [
        "",
        "## Logits",
        "",
        "| Phase | Reference | mean_abs | max_abs | signed | centered_rms | rel_rms | violations | top1 | top5 | KL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in ("prefill", "decode1", "decode2"):
        for label, key in (
            ("fn3-ORT", "fn3_vs_ort_logits"),
            ("fn3-f64", "fn3_vs_f64_logits"),
            ("ORT-f64", "ort_vs_f64_logits"),
        ):
            m = report["phases"][phase][key]
            lines.append(
                f"| {phase} | {label} | {m['mean_abs']:.3e} | {m['max_abs']:.3e} | "
                f"{m['mean_signed']:+.3e} | {m['centered_rms']:.3e} | "
                f"{m['relative_rms']:.3e} | {m['violations']} | {m['top1']} | "
                f"{m['top5_overlap']}/5 | {m['kl']:.3e} |"
            )
    (args.out / "layer-table.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out/'summary.json'}")
    print(f"wrote {args.out/'layer-table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
