# Release Process

Release workflow, checklist, and version management for fastnn.

## Current Release Status

The current good release is **v2.4.0** (tagged, wheels published to GitHub Releases).

### Recent Release History

| Tag | Status | Notes |
|-----|--------|-------|
| v2.4.0 | Good | Current stable release with full compiler pipeline |
| v2.2.4 | Good | Previous stable; macOS lib tests skipped |
| v2.2.3 | Failed | Artifact publishing broken in workflow |
| v2.2.2 | Failed | macOS tests failed due to runner runtime paths |
| v2.2.1 | Good | First post-v2.2 with portable CI fixes |
| v2.2.0 | Good | Major feature release (compiled training, WGPU, FlashAttention) |

The v2.2.2 and v2.2.3 failures were CI/workflow issues, not code regressions.
Both were resolved by v2.2.4 which skips macOS lib tests and fixes artifact paths.

## Release Workflow

Releases are triggered by pushing a `v*` tag. Workflow in `.github/workflows/release.yml`:

1. **Test job** -- `cargo test` on Ubuntu and Windows (macOS skipped for lib tests).
2. **Build wheels job** -- Python wheels via maturin on Ubuntu, Windows, macOS.
3. **Create release job** -- Publish all wheel artifacts to a GitHub Release.

## How to Run a Release

### Preflight Checks

```bash
bash scripts/ci/preflight-release.sh
```

Checks: clean worktree, branch sanity (`main`/`dev`/`agent/*`), version consistency
across `Cargo.toml`, `pyproject.toml`, `fastnn/__init__.py`, workflow existence,
stale build artifacts, lockfile presence. Add `--run-tests` to also run `cargo test`.

### Update Version Numbers

All three sources must be updated atomically:

| File | Field |
|------|-------|
| `Cargo.toml` | `version = "X.Y.Z"` |
| `pyproject.toml` | `version = "X.Y.Z"` |
| `fastnn/__init__.py` | `__version__ = "X.Y.Z"` |

Do not burn version numbers. If an RC fails, increment the patch version. Use
release candidates (`vX.Y.Z-rc1`) for risky changes and promote after testing.

### Update CHANGELOG.md

Add a section for the new version at the top of `CHANGELOG.md`.

### Commit and Tag

```bash
git add -A && git commit -m "chore: release vX.Y.Z"
git tag vX.Y.Z && git push origin main --tags
```

The tag push triggers the release workflow automatically.

### Verify the Release

Check GitHub Releases for the new tag and wheel artifacts for all three platforms
(Ubuntu, Windows, macOS). Smoke test install a wheel in a clean environment.

## Dry Run

```bash
bash scripts/ci/preflight-release.sh --dry-run
```

## Checklist Template

```
Release: vX.Y.Z
Date: YYYY-MM-DD

[ ] Preflight script passes
[ ] Version updated in all three files
[ ] CHANGELOG.md updated
[ ] No stale artifacts in git status
[ ] Worktree is clean
[ ] Tag pushed, workflow started
[ ] All 3 platform wheels built successfully
[ ] GitHub Release created with artifacts
[ ] Smoke test install from wheel
```

## See also

- [Development](development.md) -- Codebase walkthrough and how-to guides
- [Architecture](architecture.md) -- AOT compiler pipeline documentation
- [docs/index.md](../index.md) -- Documentation home
- [CONTRIBUTING.md](../../CONTRIBUTING.md) -- Repository setup, PR process, coding standards
