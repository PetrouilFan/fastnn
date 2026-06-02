import json
import subprocess
import sys
from pathlib import Path


def test_criterion_to_json_rejects_empty_benchmark_tree(tmp_path: Path):
    criterion_dir = tmp_path / "criterion"
    criterion_dir.mkdir()
    output_path = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/criterion_to_json.py",
            "--criterion-dir",
            str(criterion_dir),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "no Criterion benchmark estimates found" in result.stderr
    assert not output_path.exists()


def test_criterion_to_json_writes_sorted_benchmark_summary(tmp_path: Path):
    criterion_dir = tmp_path / "criterion"
    estimates_path = criterion_dir / "group" / "bench" / "new" / "estimates.json"
    estimates_path.parent.mkdir(parents=True)
    estimates_path.write_text(
        json.dumps(
            {
                "mean": {"point_estimate": 10.0},
                "median": {"point_estimate": 9.0},
                "std_dev": {"point_estimate": 1.5},
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/criterion_to_json.py",
            "--criterion-dir",
            str(criterion_dir),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["criterion_dir"] == str(criterion_dir)
    assert summary["benchmarks"] == [
        {
            "group": "group",
            "benchmark": "bench",
            "mean_ns": 10.0,
            "median_ns": 9.0,
            "std_dev_ns": 1.5,
        }
    ]
