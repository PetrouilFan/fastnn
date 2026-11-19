#!/usr/bin/env python3
"""Artifact-only full-vocab correctness check (CLI logits rounded to 4 decimals)."""

import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import sys


UUID = "471b2bda-51e1-6451-e692-ff5d1e146f1e"
PROMPT = "464,2068,7586,21831,18045,625,262,16931,3290,13"
SHA = [
    "33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122",
    "2626606ee4f079b43524f27f739be9414fa32fb52e9eff4b8b97e50cb095fa19",
]


def run(binary, artifact, device, quant):
    env = os.environ.copy()
    for key in ("VK_INSTANCE_LAYERS", "FN3_VK_STAGED"):
        env.pop(key, None)
    env.update(FN3_VK_VALIDATION="1", FN3_VK_REQUIRE_PHYSICAL="1", FN3_VK_DEVICE_UUID=UUID)
    command = [binary, "run", "--model", artifact, "--device", device, "--tokens", PROMPT,
               "--decode", "4874,2402", "--steps", "--top-k", "50257"]
    if quant:
        command.append("--weight-int8")
    if device != "cpu":
        command[:0] = ["flock", "/tmp/opencode/fn3-measure.lock"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, (command, result.stdout[-3000:], result.stderr)
    assert "source: none (artifact-only" in result.stdout
    if device != "cpu":
        assert f"uuid={UUID}" in result.stderr, result.stderr
        assert "software=false" in result.stderr and "validation=true" in result.stderr
        assert "[validation][ERROR]" not in result.stderr, result.stderr
    routes = re.findall(r"packed-matmuls=(\d+)/(\d+) fp32-matmuls=(\d+)", result.stdout)
    if quant:
        assert len(routes) == 3 and all(int(p) + int(f) == int(t) and int(f) for p, t, f in routes)
    else:
        assert not routes
    rows = [dict((int(i), float(v)) for i, v in re.findall(r"(\d+) \((-?[\d.]+)\)", line))
            for line in result.stdout.splitlines() if line.strip().startswith("top-50257:")]
    assert len(rows) == 3 and all(len(row) == 50257 for row in rows)
    return rows, routes


def compare(label, reference, actual, tolerance):
    for step, (ref, got) in enumerate(zip(reference, actual)):
        delta = [abs(ref[i] - got[i]) for i in ref]
        rms = math.sqrt(sum(d*d for d in delta) / len(delta))
        ref_top, got_top = list(ref)[:5], list(got)[:5]
        print(f"{label} step={step} max={max(delta):.5f} rms={rms:.5f} "
              f"argmax={ref_top[0]}/{got_top[0]} top5={len(set(ref_top) & set(got_top))}", flush=True)
        assert max(delta) <= tolerance
        assert ref_top[0] == got_top[0] and set(ref_top) == set(got_top)


if __name__ == "__main__":
    binary, tiny, trained = sys.argv[1:]
    for artifact, sha, packed, total, tolerance in zip(
        (tiny, trained), SHA, (8, 48), (13, 73), (0.01, 0.85)
    ):
        with Path(artifact).open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == sha
        cpu, _ = run(binary, artifact, "cpu", False)
        cpuq, cpu_routes = run(binary, artifact, "cpu", True)
        gpu, _ = run(binary, artifact, f"vulkan:{UUID}", False)
        gpuq, gpu_routes = run(binary, artifact, f"vulkan:{UUID}", True)
        model = Path(artifact).name
        print(f"{model} packed/total/fallback cpu={cpu_routes} gpu={gpu_routes}")
        assert cpu_routes == gpu_routes == [(str(packed), str(total), str(total - packed))] * 3
        compare(f"{model} CPU FP32 vs INT8", cpu, cpuq, tolerance)
        compare(f"{model} Vulkan FP32 vs INT8", gpu, gpuq, tolerance)
        compare(f"{model} CPU vs Vulkan INT8", cpuq, gpuq, 0.001)
