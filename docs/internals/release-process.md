# Release Process

Release workflow, checklist, and version management for fastnn.

## Current Release Status

The current published release is **v2.5.0**. The **v2.6.0** release is being
prepared on `dev`; release tags are created only from a reviewed, green `main`.

### Recent Release History

| Tag | Status | Notes |
|-----|--------|-------|
| v2.5.0 | Good | Current GitHub release with Linux, Windows, and macOS wheels |
| v2.4.0 | Superseded | Full compiler pipeline release |
| v2.2.4 | Good | Previous stable; macOS lib tests skipped |
| v2.2.3 | Failed | Artifact publishing broken in workflow |
| v2.2.2 | Failed | macOS tests failed due to runner runtime paths |
| v2.2.1 | Good | First post-v2.2 with portable CI fixes |
| v2.2.0 | Good | Major feature release (compiled training, WGPU, FlashAttention) |

The v2.2.2 and v2.2.3 failures were CI/workflow issues, not code regressions.
Both were resolved by v2.2.4 which skips macOS lib tests and fixes artifact paths.

## Release Workflow

Releases are triggered by pushing a `v*` tag. The workflow in
`.github/workflows/release.yml`:

1. **Validate tag and package versions** -- require an exact `vX.Y.Z` tag from
   `main` and matching Cargo/Python versions.
2. **Quality job** -- formatting, AGENTS freshness, Clippy, serial Rust tests,
   integration tests, and release-mode malformed-input coverage on Ubuntu.
3. **Portable build job** -- release builds on Ubuntu, Windows, and macOS.
4. **Build wheels job** -- Python wheels via maturin on Ubuntu, Windows, and
   macOS, followed by a clean-directory install/import/version smoke test.
5. **Create release job** -- publish all wheel artifacts to a GitHub Release.

The workflow publishes wheels to GitHub Releases only. It does not publish to
PyPI or crates.io.

## How to Run a Release

### Preflight Checks

```bash
bash scripts/ci/preflight-release.sh
```

Checks: clean worktree, branch sanity (`main`/`dev`), version consistency
across `Cargo.toml`, `pyproject.toml`, `fastnn/__init__.py`, workflow existence,
matching changelog entry, stale build artifacts, and lockfile presence. Add
`--run-tests` to run the release preflight test suite.

### Update Version Numbers

All three sources must be updated atomically:

| File | Field |
|------|-------|
| `Cargo.toml` | `version = "X.Y.Z"` |
| `pyproject.toml` | `version = "X.Y.Z"` |
| `fastnn/__init__.py` | `__version__ = "X.Y.Z"` |

Do not move a published tag. If a release fails after publication, fix it under
the next patch version. The current workflow accepts stable `vX.Y.Z` tags only.

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
[ ] Version updated in all three version sources and all three lockfiles
[ ] CHANGELOG.md updated
[ ] No stale artifacts in git status
[ ] Worktree is clean
[ ] Release PR merged and main CI green
[ ] Tag pushed, workflow started
[ ] All 3 platform wheels built successfully
[ ] GitHub Release created with artifacts
[ ] Downloaded release wheel passes clean-environment smoke tests
```

## See also

- [Development](development.md) -- Codebase walkthrough and how-to guides
- [Architecture](architecture.md) -- AOT compiler pipeline documentation
- [docs/index.md](../index.md) -- Documentation home
- [CONTRIBUTING.md](../../CONTRIBUTING.md) -- Repository setup, PR process, coding standards
