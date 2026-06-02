import subprocess
import sys
from pathlib import Path


def test_doc_truthfulness_script_rejects_missing_local_markdown_link():
    repo_root = Path(__file__).resolve().parents[1]
    doc_path = repo_root / "docs" / "training.md"
    original = doc_path.read_text(encoding="utf-8")

    try:
        doc_path.write_text(
            original + "\n[Broken local doc link](missing-doc-for-guardrail.md)\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, "scripts/ci/check-doc-truthfulness.py"],
            cwd=repo_root,
            text=True,
            capture_output=True,
        )
    finally:
        doc_path.write_text(original, encoding="utf-8")

    assert result.returncode != 0
    assert "docs/training.md" in result.stderr
    assert "missing-doc-for-guardrail.md" in result.stderr
