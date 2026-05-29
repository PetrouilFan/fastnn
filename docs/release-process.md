# Release Process

## Current Release Status

The current good release is **v2.2.4** (tagged, wheels published to GitHub Releases).

### Recent Release History

| Tag   | Status | Notes |
|-------|--------|-------|
| v2.2.4 | Good | macOS release lib tests skipped; final stable release |
| v2.2.3 | Failed | Release workflow artifact publishing was broken |
| v2.2.2 | Failed | Release tests failed on macOS due to runner-specific runtime paths |
| v2.2.1 | Good | First post-v2.2 release with portable CI fixes |
| v2.2.0 | Good | Major feature release (compiled training, WGPU packed shaders, FlashAttention SIMD) |

v2.2.2 and v2.2.3 were both CI/workflow issues, not code regressions. The problems were:

- **v2.2.2**: macOS Rust lib tests failed on GitHub Actions runners due to runtime path issues. Fixed by making `cargo test` use portable `RUSTFLAGS` and running coverage without the Python extension feature.
- **v2.2.3**: The `softprops/action-gh-release` action failed to find/upload wheel artifacts. Fixed by aligning the `dist/` path in the workflow.

Both were resolved by v2.2.4 which skips macOS lib tests and properly configures artifact paths.

## Release Workflow

Releases are triggered by pushing a `v*` tag. The workflow in `.github/workflows/release.yml`:

1. **Test job**: Runs `cargo test` on Ubuntu and Windows (macOS skipped for lib tests).
2. **Build wheels job**: Builds Python wheels via maturin on Ubuntu, Windows, and macOS.
3. **Create release job**: Downloads all wheel artifacts and publishes them to a GitHub Release.

## How to Run a Release

### 1. Preflight Checks

Before tagging, run the preflight script to catch common issues:

```bash
bash scripts/ci/preflight-release.sh
```

This checks:
- Clean worktree (no uncommitted changes)
- Branch sanity (main/dev/agent/*)
- Version consistency across Cargo.toml, pyproject.toml, fastnn/__init__.py
- Release workflow exists and references wheel artifacts
- No stale build artifacts staged (e.g. test_macro_positions)
- Lockfile presence

Add `--run-tests` to also execute `cargo test` as part of preflight.

### 2. Update Version Numbers

All three version sources must be updated atomically:

| File | Field |
|------|-------|
| `Cargo.toml` | `version = "X.Y.Z"` |
| `pyproject.toml` | `version = "X.Y.Z"` |
| `fastnn/__init__.py` | `__version__ = "X.Y.Z"` |

**Do not burn version numbers prematurely.** If a release candidate fails, increment the patch version for the next attempt (e.g. v2.2.2 failed → v2.2.3 tried → v2.2.4 succeeded).

### 3. Update CHANGELOG.md

Add a section for the new version at the top of CHANGELOG.md following the existing format.

### 4. Commit and Tag

```bash
git add -A
git commit -m "chore: release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

The tag push triggers the release workflow automatically.

### 5. Verify the Release

After the workflow completes:
- Check GitHub Releases for the new tag and wheel artifacts
- Verify wheel artifacts for all three platforms (ubuntu, windows, macos)
- Download and test install a wheel in a clean environment

## How Not to Burn Version Numbers

1. **Always run preflight before tagging.** This catches most issues.
2. **Use release candidates for risky changes.** Tag `v2.2.4-rc1` first, test, then promote to `v2.2.4`.
3. **Do not skip the preflight step.** The v2.2.2 and v2.2.3 failures could have been caught by checking workflow artifact paths and macOS test status beforehand.
4. **Pin dependency versions in lockfiles.** Both Cargo.lock and uv.lock should be committed.
5. **Keep the worktree clean.** Stale build artifacts (like the test_macro_positions binary) can accidentally get staged.

## Dry Run

To test the preflight checks without making changes:

```bash
bash scripts/ci/preflight-release.sh --dry-run
```

## Checklist Template

Copy this for each release:

```
Release: vX.Y.Z
Date: YYYY-MM-DD

[ ] Preflight script passes
[ ] Version updated in Cargo.toml, pyproject.toml, fastnn/__init__.py
[ ] CHANGELOG.md updated
[ ] No stale artifacts in git status
[ ] Worktree is clean
[ ] Tag pushed, workflow started
[ ] All 3 platform wheels built successfully
[ ] GitHub Release created with artifacts
[ ] Smoke test install from wheel
```
