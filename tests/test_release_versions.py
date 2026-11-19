"""The release version rule is executable, not a convention.

`scripts/ci/check-versions.py` is the single owner of the Cargo/Python spelling
difference. These tests prove it passes on this tree and that it fails when a
copy drifts, so "the versions agree" cannot silently become false again.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "ci" / "check-versions.py"

def cargo_version() -> str:
    """Read the one authoritative version, so this test follows the release."""
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match is not None, "Cargo.toml has no [package] version"
    return match.group(1)


def pep440(cargo: str) -> str:
    """The Cargo spelling of a prerelease, normalised the way the checker does."""
    return re.sub(r"-(alpha|beta|rc)\.", lambda m: {"alpha": "a", "beta": "b", "rc": "rc"}[m.group(1)], cargo)


CARGO = cargo_version()
PEP440 = pep440(CARGO)


def run_checker(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def write_tree(root: pathlib.Path, *, pyproject: str = PEP440) -> None:
    (root / "fuzz").mkdir(parents=True)
    (root / "fastnn").mkdir()
    (root / "Cargo.toml").write_text(f'[package]\nname = "fastnn"\nversion = "{CARGO}"\n')
    (root / "Cargo.lock").write_text(f'[[package]]\nname = "fastnn"\nversion = "{CARGO}"\n')
    (root / "fuzz" / "Cargo.lock").write_text(
        f'[[package]]\nname = "fastnn"\nversion = "{CARGO}"\n'
    )
    (root / "pyproject.toml").write_text(f'[project]\nname = "fastnn"\nversion = "{pyproject}"\n')
    (root / "uv.lock").write_text(f'[[package]]\nname = "fastnn"\nversion = "{PEP440}"\n')
    (root / "fastnn" / "__init__.py").write_text(f'__version__ = "{PEP440}"\n')
    (root / "CHANGELOG.md").write_text(f"# Changelog\n\n## [{CARGO}] - 2026-11-19\n")


def test_this_checkout_has_one_normalised_version_everywhere():
    result = run_checker("--tag", f"v{CARGO}")
    assert result.returncode == 0, result.stderr
    assert "VERSIONS AGREE" in result.stdout
    assert PEP440 in result.stdout


def test_the_tag_must_use_the_cargo_spelling():
    accepted = run_checker("--tag", f"v{CARGO}")
    assert accepted.returncode == 0, accepted.stderr

    rejected = run_checker("--tag", "v3.0.0a1")
    assert rejected.returncode != 0
    assert "tag" in rejected.stderr


def test_a_drifted_python_version_fails_the_checker(tmp_path):
    write_tree(tmp_path)
    assert run_checker("--root", str(tmp_path)).returncode == 0

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "fastnn"\nversion = "2.5.0"\n'
    )
    result = run_checker("--root", str(tmp_path))
    assert result.returncode != 0
    assert "pyproject.toml" in result.stderr
    assert "3.0.0b1" in result.stderr


def test_a_missing_changelog_entry_fails_the_checker(tmp_path):
    write_tree(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n")
    result = run_checker("--root", str(tmp_path))
    assert result.returncode != 0
    assert "CHANGELOG.md" in result.stderr


def test_the_checker_is_the_script_the_ci_lane_runs():
    lane = (ROOT / "tools" / "ci" / "cpu-lane.sh").read_text()
    assert "scripts/ci/check-versions.py" in lane
    assert "CLIPPY_MAX_WARNINGS=\"${CLIPPY_MAX_WARNINGS:-0}\"" in lane
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "scripts/ci/check-versions.py --tag" in release
