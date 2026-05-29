#!/usr/bin/env bash
set -euo pipefail

# Release preflight checks for fastnn.
# Usage: bash scripts/ci/preflight-release.sh [--dry-run] [--run-tests]
#
# --dry-run     Print what would be checked / run, but skip destructive or slow steps.
# --run-tests   Actually run cargo test (default: print the command only).

DRY_RUN=false
RUN_TESTS=false

for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY_RUN=true ;;
    --run-tests) RUN_TESTS=true ;;
    *)           echo "Unknown flag: $arg"; exit 1 ;;
  esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

FAILURES=0

pass() { echo -e "${GREEN}PASS${NC}  $1"; }
warn() { echo -e "${YELLOW}WARN${NC}  $1"; }
fail() { echo -e "${RED}FAIL${NC}  $1"; FAILURES=$((FAILURES + 1)); }

# ── 1. Clean worktree ────────────────────────────────────────────────────────
echo "=== Release Preflight ==="
echo ""

if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
  pass "Worktree is clean (no unstaged, uncommitted, or untracked changes)"
else
  fail "Worktree is dirty — commit or stash changes before tagging"
  git status --short
fi

# ── 2. Branch / tag sanity ───────────────────────────────────────────────────
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "detached")
echo ""
echo "Current branch: $BRANCH"

if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "dev" ] || [[ "$BRANCH" == agent/* ]]; then
  pass "Branch '$BRANCH' is acceptable for a release"
else
  warn "Branch '$BRANCH' is not a typical release branch (main/dev/agent/*). Intentional?"
fi

if [ "$(git log --oneline -1 --format='%H' HEAD)" = "$(git log --oneline -1 --format='%H' v2.2.4 2>/dev/null)" ]; then
  pass "HEAD matches v2.2.4 tag"
else
  warn "HEAD does not match v2.2.4 — a new tag will be created on a different commit"
fi

# ── 3. Version consistency ───────────────────────────────────────────────────
echo ""
echo "--- Version consistency ---"

CARGO_VER=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
PYPROJECT_VER=$(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
INIT_VER=$(grep -m1 '__version__' fastnn/__init__.py | sed 's/.*"\(.*\)".*/\1/')

echo "  Cargo.toml:     $CARGO_VER"
echo "  pyproject.toml: $PYPROJECT_VER"
echo "  __init__.py:    $INIT_VER"

if [ "$CARGO_VER" = "$PYPROJECT_VER" ] && [ "$PYPROJECT_VER" = "$INIT_VER" ]; then
  pass "All version strings match ($CARGO_VER)"
else
  fail "Version mismatch: Cargo=$CARGO_VER pyproject=$PYPROJECT_VER init=$INIT_VER"
fi

# ── 4. Release workflow exists and references wheel artifacts ─────────────────
echo ""
echo "--- Release workflow ---"

WORKFLOW=".github/workflows/release.yml"
if [ -f "$WORKFLOW" ]; then
  pass "Release workflow exists at $WORKFLOW"
else
  fail "Release workflow missing at $WORKFLOW"
fi

if grep -q 'dist/\*\.whl' "$WORKFLOW" 2>/dev/null; then
  pass "Release workflow uploads wheel artifacts (dist/*.whl)"
else
  warn "Release workflow does not explicitly reference dist/*.whl in release files"
fi

if grep -q 'tags:' "$WORKFLOW" 2>/dev/null; then
  pass "Release workflow is tag-triggered"
else
  warn "Release workflow has no tag trigger — check on: config"
fi

# ── 5. Stale build artifacts not staged ──────────────────────────────────────
echo ""
echo "--- Stale build artifacts ---"

KNOWN_ARTIFACTS=(
  test_macro_positions
  target
  build
  dist
  '*.pyc'
  '__pycache__'
)

STAGED_ARTIFACTS=()
for pat in "${KNOWN_ARTIFACTS[@]}"; do
  matched=$(git ls-files -- "$pat" 2>/dev/null || true)
  if [ -n "$matched" ]; then
    for f in $matched; do
      STAGED_ARTIFACTS+=("$f")
    done
  fi
done

# Check for untracked artifact executables
UNTRACKED_EXES=$(git ls-files --others --exclude-standard -z 2>/dev/null | xargs -0 file 2>/dev/null | grep 'ELF\|executable\|Mach-O' | cut -d: -f1 || true)

if [ ${#STAGED_ARTIFACTS[@]} -eq 0 ] && [ -z "$UNTRACKED_EXES" ]; then
  pass "No stale build artifacts detected in working tree"
else
  if [ ${#STAGED_ARTIFACTS[@]} -gt 0 ]; then
    fail "Staged artifacts found:"
    for a in "${STAGED_ARTIFACTS[@]}"; do
      echo "    $a"
    done
  fi
  if [ -n "$UNTRACKED_EXES" ]; then
    warn "Untracked executables in tree (not staged, but may clutter):"
    echo "$UNTRACKED_EXES" | while read -r line; do echo "    $line"; done
  fi
fi

# Also check git ls-files for the known binary
if git ls-files -- test_macro_positions | grep -q .; then
  fail "test_macro_positions is tracked in git — it is a stale build artifact"
fi

# ── 6. Lockfiles exist and are consistent ────────────────────────────────────
echo ""
echo "--- Lockfiles ---"

if [ -f "Cargo.lock" ]; then
  pass "Cargo.lock exists"
else
  warn "Cargo.lock missing (may be expected for library crates, but release should pin deps)"
fi

if [ -f "uv.lock" ]; then
  pass "uv.lock exists"
else
  warn "uv.lock missing"
fi

# ── 7. cargo test command ────────────────────────────────────────────────────
echo ""
echo "--- cargo test ---"

TEST_CMD="cargo test --lib --no-default-features --features \"simd,parallel,fusion-forward\""
echo "Command: $TEST_CMD"

if [ "$RUN_TESTS" = true ] && [ "$DRY_RUN" = false ]; then
  echo "Running cargo test..."
  if eval "$TEST_CMD"; then
    pass "cargo test passed"
  else
    fail "cargo test failed"
  fi
else
  if [ "$DRY_RUN" = true ]; then
    echo "(dry-run — skipping test execution)"
  else
    echo "(use --run-tests to execute)"
  fi
fi

# ── 8. Checklist for manual review ───────────────────────────────────────────
echo ""
echo "--- Manual checklist ---"
echo "  [ ] Confirm CHANGELOG.md has an entry for the target version"
echo "  [ ] Confirm no source changes since last release commit"
echo "  [ ] Confirm GitHub Actions secrets (PYPI_TOKEN etc.) are valid"
echo "  [ ] Confirm no open blockers in issues/PRs"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
if [ "$FAILURES" -gt 0 ]; then
  echo -e "${RED}Preflight failed with $FAILURES failure(s)${NC}"
  exit 1
else
  echo -e "${GREEN}Preflight passed — ready to tag${NC}"
  exit 0
fi
