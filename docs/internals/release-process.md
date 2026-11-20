# Release Process

Release workflow, checklist, and version management for fastnn.

## Current Release Status

The current release is **v3.0.0**, tagged at `4db656e`; the crate and the wheel
both carry version `3.0.0`. `main` is the release branch; release tags are
created only from a reviewed, green `main`, and `validate-tag` rejects a tag
whose commit is not the tip of `origin/main`.

### Version stores

Rust/SemVer and Python/PEP 440 spell a prerelease differently, so do not compare
the main files with string equality. `scripts/ci/check-versions.py` is the only
version rule: it normalises both spellings, checks every store (`Cargo.toml`,
both Cargo lockfiles, `pyproject.toml`, `uv.lock`, `fastnn/__init__.py` and the
changelog) and accepts `--tag` so the release workflow can validate the pushed
tag against the same rule.

### Recent Release History

| Tag | Status | Notes |
|-----|--------|-------|
| v3.0.0 | Good | Current GitHub release with Linux, Windows, and macOS wheels |
| v2.6.0 | Superseded | Previous GitHub release with Linux, Windows, and macOS wheels |
| v2.5.0 | Superseded | Superseded by v2.6.0 |
| v2.4.0 | Superseded | Full compiler pipeline release |
| v2.2.4 | Good | Stable patch; macOS lib tests skipped |
| v2.2.3 | Failed | Artifact publishing broken in workflow |
| v2.2.2 | Failed | macOS tests failed due to runner runtime paths |
| v2.2.1 | Good | First post-v2.2 with portable CI fixes |
| v2.2.0 | Good | Major feature release (compiled training, WGPU, FlashAttention) |

The v2.2.2 and v2.2.3 failures were CI/workflow issues, not code regressions.
Both were resolved by v2.2.4 which skips macOS lib tests and fixes artifact paths.

## Release Workflow

Releases are triggered by pushing a `v*` tag. The workflow in
`.github/workflows/release.yml`:

1. **Validate tag** -- require a `vMAJOR.MINOR.PATCH` tag (a SemVer prerelease
   suffix is allowed) whose commit is exactly the tip of `origin/main`, then
   run `scripts/ci/check-versions.py --tag` against the tag.
2. **Quality job** -- `tools/ci/cpu-lane.sh`: formatting, version agreement,
   Clippy with warnings denied, a `--no-default-features` build, and the Rust
   unit, fn3 contract and fn3 CI smoke suites.
3. **Portable build job** -- `cargo build --release --no-default-features` on
   Ubuntu, Windows, and macOS.
4. **Build wheels job** -- Python wheels via maturin on Ubuntu, Windows, and
   macOS, followed by the pinned-model fetch, a strict clean-directory
   install/import/inference smoke test, and the wheel and evidence artifacts.
5. **Create release job** -- attach `dist/*.whl` to a GitHub Release with
   generated notes; a tag containing `-` is marked as a prerelease.

The workflow publishes wheels to GitHub Releases only. It does not publish to
PyPI or crates.io, and `fastnn-infer` is a source build
(`required-features = ["cli"]`), so no CLI binary is attached.

### Pinned-model inference in the release wheel

The release wheel is never accepted on an import alone. In every wheel job the
workflow first fetches `onnx/model.onnx` from the revision recorded in
`fixtures/parity/tiny-gpt2/model.lock.json`:

- the URL is derived from that file's `model_id` and `revision`, never typed by
  hand;
- the downloaded size and SHA-256 are compared with the same lock entry;
- a fetch, size or digest failure ends the job with a named
  `::error::pinned-model ...`; it is never recorded as `not_run`.

The lane then runs
`python tools/ci/packaging_lane.py --skip-build --require-inference`. With that
flag a missing model fails the lane before wheel installation. The lane
installs the wheel in a fresh virtualenv outside the checkout and checks that
both the Python facade and native extension import from that environment.
The installed wheel verifies the staged ONNX size and SHA-256 and prepares a
new `.fnn3`. The lane then removes the staging directory and starts a separate
isolated Python process with only the artifact in its working directory and
no ONNX argument or model environment variable. It loads without `source=` on
the explicit CPU device and asserts `loaded.source is None`, finite FP32
logits, prefill top-5 `[46737, 28597, 16571, 7868, 13]`, and KV cache growth.
The three decode inputs are the separate tokens `640, 11, 612` (not the
10-token prefill prompt); their top-1 results `[32493, 11724, 612]` correspond
to cache lengths 11, 12 and 13. The workflow uploads the JSON evidence,
including the observed source state and both digests, but no model bytes.

Inference is opt-in through `FASTNN_FN3_ONNX_MODEL`; there is no default model
path. A local run without it prints the unset reason, writes `status: not_run`
with `strict: false`, records the `wheel_inference` check as `not_run`, and
exits 0. Hosted packaging CI names an absent path explicitly so `not_run` is
deterministic. That is a packaging check, not a release gate, and the
distinction is recorded in `tools/ci/lanes.json`.

This is the `tiny-gpt2-cpu-fp32` Python-wheel case, extended with an INT8-weight
replay of the same artifact. Both replays are separate source-free processes;
the INT8 one opts in at load time, so the artifact bytes are identical. A
quantized CLI release gate would still need its own pinned inputs, a packaged
CLI, a source-free replay and a quantized acceptance oracle; these CPU cases are
not evidence for trained models or Vulkan.

## How to Run a Release

### Preflight Checks

```bash
bash scripts/ci/preflight-release.sh
```

Checks: clean worktree, branch sanity (`main`/`dev`), version consistency
through `scripts/ci/check-versions.py`, release-workflow existence and tag
trigger, stale build artifacts, and lockfile presence. `--run-tests`
additionally runs formatting, Clippy with warnings denied, and the Rust test
suites.

### Update Version Numbers

The three primary sources must be updated atomically (the lockfiles and
changelog follow; `scripts/ci/check-versions.py` enforces all of them):

| File | Field |
|------|-------|
| `Cargo.toml` | `version = "X.Y.Z"` |
| `pyproject.toml` | `version = "X.Y.Z"` |
| `fastnn/__init__.py` | `__version__ = "X.Y.Z"` |

Do not move a published tag. If a release fails after publication, fix it under
the next patch version. The workflow accepts `vX.Y.Z` tags, including a SemVer
prerelease suffix, provided the tagged commit is the tip of `origin/main`.

### Update CHANGELOG.md

Add a section for the new version at the top of `CHANGELOG.md`.

### Commit and Tag

```bash
git add -A && git commit -m "chore: prepare vX.Y.Z"
# Open and merge a release PR from dev to main, then verify main CI.
git switch main
git pull --ff-only origin main
git tag -a vX.Y.Z -m "fastnn vX.Y.Z"
git push origin vX.Y.Z
```

The tag push triggers the release workflow automatically.

### Verify the Release

Check GitHub Releases for the new tag and wheel artifacts for all three platforms
(Ubuntu, Windows, macOS). Download a published wheel and smoke-test its install,
import, reported version, eager execution, and AOT execution in a clean environment.

## Dry Run

```bash
bash scripts/ci/preflight-release.sh --dry-run
```

## Checklist Template

```
Release: vX.Y.Z
Date: YYYY-MM-DD

[ ] Preflight script passes
[ ] `python3 scripts/ci/check-versions.py` passes (manifests + all lockfiles + facade + changelog)
[ ] CHANGELOG.md updated
[ ] No stale artifacts in git status
[ ] Worktree is clean
[ ] Release PR merged and main CI green
[ ] Tag pushed, workflow started
[ ] All 3 platform wheels built successfully
[ ] Release wheel job reports strict pinned-model inference PASSED (never not_run)
[ ] GitHub Release created with artifacts
[ ] Downloaded release wheel passes clean-environment smoke tests
```

## See also

- [Development](development.md) -- Codebase walkthrough and how-to guides
- [Architecture](architecture.md) -- AOT compiler pipeline documentation
- [docs/index.md](../index.md) -- Documentation home
- [CONTRIBUTING.md](../../CONTRIBUTING.md) -- Repository setup, PR process, coding standards
