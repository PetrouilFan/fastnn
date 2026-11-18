# CI lanes — operating contract

FastNN 3.0 has three kinds of GPU-adjacent evidence and they are not
interchangeable:

| Evidence | Produced by | What it can support |
|---|---|---|
| compile check | `vulkan-build` lane | "the feature and shaders compile" |
| software smoke | `vulkan-software-smoke` lane (lavapipe) | "the code runs under a CPU Vulkan implementation" — CI-ONLY |
| physical run | `tools/ci/run-gpu-lane.sh` on a trusted machine | "the named GPU and driver executed this and validation was clean" |

A lane that did not run is **NOT VERIFIED**. It is never "passed by default".
`tools/ci/lane_status.py render` reads `tools/ci/lanes.json`, lists every lane,
and fails the run when a lane that applies to the current tree has no PASSED
record.

## Files

| Path | Purpose |
|---|---|
| `lanes.json` | Lane inventory: jobs, what each proves / does not prove, when it becomes required |
| `lane_status.py` | `record` a lane result, `render` the run summary, `describe`/`resolve` for humans |
| `model-pin.json` | The only model-fixture pin CI reads; owned by FN3-03 (#49) |
| `model_pin.py` | `check`, `seal`, `selftest` for the pin; exit 69 means NOT VERIFIED |
| `verify_evidence.py` | Rejects anything that is not a physical-GPU evidence record |
| `run-gpu-lane.sh` | Local/protected physical-GPU evidence path; refused on hosted runners |
| `gpu_evidence.py` | Implementation behind `run-gpu-lane.sh` |
| `lane-lib.sh` | Shared bash helpers: lane identity, reproduction output, status recording |
| `cpu-lane.sh`, `python-lane.sh` | Lane entry points |
| `vulkan-build-lane.sh`, `vulkan-software-lane.sh` | Vulkan lanes |
| `model-pin-lane.sh` | Offline pinned-model lane |
| `gpu-evidence-status.sh` | Hosted job: validates committed evidence, never runs a GPU |
| `packaging_lane.py`, `wheel_smoke.py` | Wheel build/install/smoke lane |

## Local commands

```bash
tools/ci/cpu-lane.sh                    # fmt, clippy budget, build, Rust tests
tools/ci/python-lane.sh                 # maturin develop + pytest -m "not slow"
tools/ci/vulkan-build-lane.sh           # spirv-val + feature build (no device)
tools/ci/vulkan-software-lane.sh        # CI-ONLY lavapipe smoke (no GPU)
tools/ci/model-pin-lane.sh              # offline; NOT VERIFIED until #49
python3 tools/ci/packaging_lane.py      # wheel build/install/smoke
tools/ci/run-gpu-lane.sh --out docs/reports/fn3-20/gpu-evidence/<name>.json
```

Status records go to `$LANE_STATUS_DIR` (default `target/ci-lane-status`).
`GITHUB_STEP_SUMMARY`, when set, receives the same lines.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | the lane produced the evidence it claims |
| 1 | the lane failed, or a record/pin is invalid |
| 69 | NOT VERIFIED: nothing failed, but the check could not run (missing pin, missing backend, missing device) |

Exit 69 is deliberate: a lane script may exit 0 so the job can finish and record
its status, but the status is `not_verified` and the summary says so.

## Adding a lane

1. Add an entry to `lanes.json` with `proves`, `does_not_prove`, `jobs`,
   `reproduce`, and either `required` or `required_when`.
2. Add a job with a name that states the lane and its evidence boundary.
3. Have the job call a script that prints the lane identity (`ci::describe`),
   prints `REPRODUCE:` on failure, and records an explicit status.
4. Add an `upload-artifact` step for `lane-status-*`.
5. Add the job to `lane-summary.needs`.

Never add a hosted job that claims physical-GPU proof. `lane_status.py` rejects
that mechanically.

## Physical-GPU evidence rule

Only a record written by `tools/ci/run-gpu-lane.sh` and accepted by
`tools/ci/verify_evidence.py` counts. A record must contain:

* `device.kind == "physical"`, device name, canonical UUID, `deviceType`
  (not `VK_PHYSICAL_DEVICE_TYPE_CPU`), driver name/version, Vulkan API version;
* `validation.enabled == true` with zero validation errors;
* `checks` including a passed `fastnn_vulkan_smoke` when `result.status` is
  `passed`;
* timings, git commit, and the commands that were run.

Lavapipe/llvmpipe/SwiftShader names are rejected. A `not_verified` record is
valid to commit (it is truthful) but never satisfies the lane.

Hosted CI must never run the physical lane: the script refuses when
`GITHUB_ACTIONS=true` unless a human deliberately overrides it on a trusted
machine.
