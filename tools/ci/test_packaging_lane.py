"""Packaging gate controls, runnable without a wheel, NumPy, or model weights."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import packaging_lane as lane
import wheel_smoke as smoke


class PackagingLaneTests(unittest.TestCase):
    def _installed_lane(self, out: Path):
        """Mock the build/install steps so lane.main reaches the inference stage."""
        dist = out / "dist"
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "fastnn-3.0.0-test.whl").write_bytes(b"wheel")
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(lane, "_version_from_cargo", return_value="3.0.0"))
        stack.enter_context(patch.object(lane, "_python_version", return_value="3.0.0"))
        stack.enter_context(patch.object(lane, "_run", return_value=subprocess.CompletedProcess([], 0)))
        stack.enter_context(patch.object(lane, "_venv_python", return_value=Path("python")))
        stack.enter_context(patch.object(lane, "_installed_distributions", return_value={"numpy", "fastnn"}))
        stack.enter_context(patch.object(lane, "_declared_dependencies", return_value={"numpy"}))
        stack.enter_context(patch.object(lane, "_record"))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return stack

    def test_non_strict_missing_fixture_records_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            missing = out / "missing.onnx"
            with self._installed_lane(out), patch.dict(os.environ, {smoke.MODEL_ENV: str(missing)}):
                self.assertEqual(lane.main(["--skip-build", "--out", str(out)]), 0)
            report = json.loads((out / "wheel-inference.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "not_run")
            self.assertIs(report["strict"], False)
            self.assertIn(str(missing), report["reason"])

    def test_unset_model_variable_attempts_no_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self._installed_lane(out), patch.dict(os.environ, {smoke.MODEL_ENV: ""}):
                self.assertEqual(lane.main(["--skip-build", "--out", str(out)]), 0)
            report = json.loads((out / "wheel-inference.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "not_run")
            self.assertIn(f"{smoke.MODEL_ENV} is not set", report["reason"])

    def test_failed_replay_marks_the_lane_and_report_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            model = out / "model.onnx"
            model.write_bytes(b"present so the lane reaches replay")
            with (
                self._installed_lane(out),
                patch.dict(os.environ, {smoke.MODEL_ENV: str(model)}),
                patch.object(lane, "_artifact_inference", side_effect=ValueError("replay exploded")),
            ):
                self.assertEqual(lane.main(["--skip-build", "--require-inference", "--out", str(out)]), 1)
            report = json.loads((out / "wheel-inference.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertIn("replay exploded", report["reason"])

    def test_artifact_only_false_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory).resolve()
            model = scratch / "model.onnx"
            model.write_bytes(b"pinned ONNX bytes")
            pin = lane.ModelPin(model.stat().st_size, smoke.sha256(model), "https://example.invalid")
            artifact = b"prepared artifact"
            digest = hashlib.sha256(artifact).hexdigest()

            def worker(command, *, cwd, env):
                if command[1] == "prepare":
                    (cwd / smoke.ARTIFACT).write_bytes(artifact)
                    fields = {"artifact_sha256": digest}
                else:
                    fields = {"artifact_only": False, "artifact_sha256": digest}
                Path(command[-1]).write_text(json.dumps({"status": "passed", **fields}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.object(lane, "_run", side_effect=worker),
                patch.dict(os.environ, {smoke.MODEL_ENV: str(model)}),
                self.assertRaisesRegex(ValueError, "did not attest to artifact-only inference"),
            ):
                lane._artifact_inference(["worker"], scratch, model, pin)

    def test_load_requires_the_model_to_report_no_source(self):
        fn3 = Mock()
        loaded = Mock(source=None)
        fn3.load.return_value = loaded
        self.assertIs(smoke._load_artifact_only(fn3, Path("artifact.fnn3")), loaded)
        fn3.load.assert_called_once_with(Path("artifact.fnn3"), device="cpu")
        loaded.source = "model.onnx"
        with self.assertRaisesRegex(AssertionError, "source attached"):
            smoke._load_artifact_only(fn3, Path("artifact.fnn3"))

    def test_phase_ignores_a_stale_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory) / "replay"
            cwd.mkdir()
            (cwd.parent / "replay.json").write_text('{"status":"passed"}', encoding="utf-8")
            with (
                patch.object(lane, "_run", return_value=subprocess.CompletedProcess([], 0)),
                self.assertRaisesRegex(ValueError, "replay produced no valid evidence"),
            ):
                lane._phase([], phase="replay", cwd=cwd, env={})

    def test_missing_required_fixture_fails_and_replaces_stale_success(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            report = out / "wheel-inference.json"
            report.write_text('{"status":"passed"}', encoding="utf-8")
            missing = out / "missing.onnx"
            with (
                patch.dict(os.environ, {smoke.MODEL_ENV: str(missing)}),
                patch.object(lane, "_record") as record,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(lane.main(["--skip-build", "--require-inference", "--out", str(out)]), 1)
            evidence = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed")
            self.assertIn(str(missing), evidence["reason"])
            self.assertIn(lane._pinned_model().sha256, evidence["reason"])
            self.assertEqual(record.call_args.args[1], "failed")

    def test_prepare_replay_use_disjoint_directories_and_no_source_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory).resolve()
            model = scratch / "user-model.onnx"
            model.write_bytes(b"pinned ONNX bytes")
            pin = lane.ModelPin(model.stat().st_size, smoke.sha256(model), "https://example.invalid")
            artifact = b"prepared artifact"
            digest = hashlib.sha256(artifact).hexdigest()
            phases = []
            prepare_dir = None

            def worker(command, *, cwd, env):
                nonlocal prepare_dir
                self.assertNotIn(smoke.MODEL_ENV, env)
                phases.append(command[1])
                if command[1] == "prepare":
                    prepare_dir = cwd
                    staged_model = cwd / command[3]
                    self.assertNotEqual(staged_model, model)
                    self.assertEqual(staged_model.read_bytes(), model.read_bytes())
                    self.assertIn(pin.sha256, command)
                    (cwd / smoke.ARTIFACT).write_bytes(artifact)
                    fields = {"model_sha256": pin.sha256, "artifact_sha256": digest}
                else:
                    self.assertIsNotNone(prepare_dir)
                    self.assertFalse(prepare_dir.exists())
                    self.assertEqual(command[1:-2], ["replay", "--expect-artifact-sha256", digest])
                    self.assertEqual(list(cwd.iterdir()), [cwd / smoke.ARTIFACT])
                    self.assertEqual((cwd / smoke.ARTIFACT).read_bytes(), artifact)
                    fields = {"artifact_only": True, "artifact_sha256": digest}
                Path(command[-1]).write_text(json.dumps({"status": "passed", **fields}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(lane, "_run", side_effect=worker), patch.dict(os.environ, {smoke.MODEL_ENV: str(model)}):
                evidence = lane._artifact_inference(["worker"], scratch, model, pin)
            self.assertEqual(phases, ["prepare", "replay"])
            self.assertEqual(evidence["model_sha256"], pin.sha256)
            self.assertTrue(evidence["artifact_only"])
            self.assertTrue(model.is_file())

    def test_phase_fails_if_success_exit_has_no_report(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory) / "replay"
            cwd.mkdir()
            with (
                patch.object(lane, "_run", return_value=subprocess.CompletedProcess([], 0)),
                self.assertRaisesRegex(ValueError, "replay produced no valid evidence"),
            ):
                lane._phase([], phase="replay", cwd=cwd, env={})

    def test_prepare_rejects_wrong_size_or_digest_before_importing_onnx(self):
        fn3 = Mock()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"bad")
            with self.assertRaisesRegex(AssertionError, "size mismatch"):
                smoke.prepare(fn3, model, 4, "0" * 64)
            with self.assertRaisesRegex(AssertionError, "sha256 mismatch"):
                smoke.prepare(fn3, model, 3, "0" * 64)
        fn3.prepare.assert_not_called()

    def test_replay_rejects_adjacent_source_and_corrupted_artifact_before_load(self):
        fn3 = Mock()
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            artifact = Path(smoke.ARTIFACT)
            artifact.write_bytes(b"corrupt artifact")
            model = Path("model.onnx")
            model.touch()
            with self.assertRaisesRegex(AssertionError, "containing only"):
                smoke.replay(fn3, "0" * 64)
            model.unlink()
            with patch.dict(os.environ, {smoke.MODEL_ENV: "model.onnx"}):
                with self.assertRaisesRegex(AssertionError, "must not inherit"):
                    smoke.replay(fn3, "0" * 64)
                del os.environ[smoke.MODEL_ENV]
                with self.assertRaisesRegex(AssertionError, "artifact sha256 differs"):
                    smoke.replay(fn3, "0" * 64)
        fn3.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
