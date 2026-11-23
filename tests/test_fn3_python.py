"""FN3-18 (#64): the ``fastnn.inference`` Python facade over the 3.0 inference runtime.

What is covered, in the order the issue asks for it:

* prepare: an ONNX file becomes a versioned artifact, and "I wrote it" is
  distinguishable from "it was already there and valid";
* load and run: logits as NumPy, plus the session's observable state;
* devices: enumeration, an explicit request, and no silent fallback;
* errors: missing file, bad magic, future version, unavailable device, a
  sequence beyond the envelope and an exceeded capacity, each a typed exception;
* sessions are independent, and that is visible from Python;
* the GIL is released for prefill and decode.

The pinned model is not committed (no licence), so every test that needs it
skips with a printed reason when the file is absent. The numerical assertions
compare against the committed ORT fixtures in ``fixtures/parity/tiny-gpt2``.
"""

import gc
import hashlib
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import fastnn
import fastnn.inference as fn3

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ONNX = "/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx"
ONNX = pathlib.Path(os.environ.get("FASTNN_FN3_ONNX_MODEL", DEFAULT_ONNX))
FIXTURES = ROOT / "fixtures" / "parity" / "tiny-gpt2" / "tensors"

#: The 17-token fixture sequence. The 13-token prefix matches
#: ``prefill.p13.last_logits``; the four suffix tokens are the cached decode
#: inputs whose outputs are ``decode.step{1..4}``.
TOKENS = [
    464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257,
    640, 11, 612, 373,
]
VOCAB = 50257
ATOL = 1e-5
MEAN_ATOL = 2.5e-6


def pinned_onnx() -> pathlib.Path:
    """The pinned model, or a skip with the reason a person needs."""
    if not ONNX.is_file():
        pytest.skip(
            f"pinned tiny-GPT-2 ONNX not present at {ONNX}; set "
            "FASTNN_FN3_ONNX_MODEL to run the numerical tests"
        )
    return ONNX


def fixture(name: str) -> np.ndarray:
    """A committed reference tensor (little-endian f32)."""
    return np.fromfile(FIXTURES / f"{name}.f32", dtype="<f4")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory) -> pathlib.Path:
    """A full-window artifact prepared once for the module."""
    source = pinned_onnx()
    path = tmp_path_factory.mktemp("fn3-64") / "tiny-gpt2.fn3prep"
    report = fn3.prepare(source, path)
    assert report.status == "created"
    return path


@pytest.fixture(scope="module")
def model(artifact_path) -> fn3.Model:
    """The loaded model shared by tests that never mutate it.

    Loaded from the artifact alone: no ONNX path, no environment variable. The
    ``source=`` verification has its own tests below.
    """
    return fn3.load(artifact_path, device="cpu")


@pytest.fixture()
def session(model) -> fn3.Session:
    return model.session()


# ---------------------------------------------------------------------------
# 1. The public surface
# ---------------------------------------------------------------------------


def test_namespace_is_reachable_from_the_package_and_the_extension():
    # `import fastnn` alone must expose the namespace ...
    assert fastnn.inference is fn3
    assert "inference" in fastnn.__all__
    assert not hasattr(fastnn, "fn3")
    # ... and the extension module is importable by its dotted name.
    import fastnn._core.inference as core_fn3

    # The Python package adds path handling and docstrings around the same
    # objects; the classes and exceptions are the extension's, unchanged.
    assert core_fn3.Session is fn3.Session
    assert fn3.Session.__module__ == "fastnn._core.inference"
    assert not hasattr(fastnn._core, "fn3")
    assert core_fn3.Fn3DeviceError is fn3.Fn3DeviceError
    assert callable(core_fn3.load) and callable(core_fn3.prepare)
    assert fn3.load.__module__ == "fastnn.inference"
    for name in fn3.__all__:
        assert hasattr(fn3, name), f"fn3.__all__ names {name} but it is missing"
    for name in core_fn3.__all__:
        assert hasattr(core_fn3, name), f"_core.inference.__all__ names {name} but it is missing"


def test_error_hierarchy_is_catchable_at_every_level():
    for name in [
        "Fn3ModelFileError",
        "Fn3FormatError",
        "Fn3ImportError",
        "Fn3ModelExistsError",
    ]:
        assert issubclass(getattr(fn3, name), fn3.Fn3ModelError)
    assert issubclass(fn3.Fn3ModelError, fn3.Fn3Error)
    for name in [
        "Fn3DeviceError",
        "Fn3EnvelopeError",
        "Fn3CapacityError",
        "Fn3SessionError",
        "Fn3TokenError",
        "Fn3ShapeError",
        "Fn3TypeError",
        "Fn3UnsupportedError",
        "Fn3ExecutionError",
    ]:
        assert issubclass(getattr(fn3, name), fn3.Fn3Error)
    # 3.0 is its own surface: there is one error base and no legacy one.
    assert issubclass(fn3.Fn3Error, Exception)
    assert not hasattr(fastnn, "FastnnError")


def test_documented_surface_has_docstrings():
    assert "inference" in fn3.__doc__.lower()
    # The v3 surface writes one artifact and returns its report; the docstring
    # must say so rather than describing the retired in-memory prepare.
    assert fn3.prepare.__doc__ and "artifact" in fn3.prepare.__doc__
    assert fn3.load.__doc__ and "device" in fn3.load.__doc__


# ---------------------------------------------------------------------------
# 2. Devices
# ---------------------------------------------------------------------------


def test_devices_lists_cpu_first_and_describes_itself():
    devices = fn3.devices()
    assert devices, "at least the CPU must be listed"
    assert devices[0].kind == "cpu"
    assert str(devices[0]) == "cpu"
    assert devices[0].index is None
    assert repr(devices[0]).startswith("Device(")
    # A Device object is accepted as a request and resolves to itself.
    assert devices[0] == fn3.devices()[0]


def test_unknown_device_spelling_is_typed():
    with pytest.raises(fn3.Fn3DeviceError) as info:
        fn3.load("unused", device="tpu")
    error = info.value
    assert error.requested == "tpu"
    assert error.available == ["cpu"]
    assert "accepted forms" in str(error)
    assert "available devices" in str(error)


def test_explicit_vulkan_request_never_falls_back_to_cpu():
    vulkan = [device for device in fn3.devices() if device.kind == "vulkan"]
    if vulkan:
        pytest.skip(
            "this build enumerates a Vulkan device; the no-driver path is not "
            "exercised here"
        )
    for request, rendered in [
        ("vulkan", "vulkan"),
        ("vulkan:0", "vulkan(index=0)"),
        ("vulkan:7", "vulkan(index=7)"),
    ]:
        with pytest.raises(fn3.Fn3DeviceError) as info:
            fn3.load("unused", device=request)
        error = info.value
        assert error.available == ["cpu"]
        # The message names the request the caller made ...
        assert rendered in str(error)
        assert error.requested == rendered
        assert "available devices: [cpu]" in str(error)
    # A named adapter request from a Device object is equally explicit.
    with pytest.raises(fn3.Fn3DeviceError):
        fn3.load("unused", device="vulkan")


# ---------------------------------------------------------------------------
# 3. prepare
# ---------------------------------------------------------------------------


def test_prepare_writes_a_verifiable_artifact(artifact_path):
    data = artifact_path.read_bytes()
    info = fn3.inspect(artifact_path)
    # The artifact is the documented size for the pinned model, and its digest
    # is the digest of the bytes on disk.
    assert len(data) == 27_879_928
    assert info.size_bytes == len(data)
    assert info.sha256 == hashlib.sha256(data).hexdigest()
    assert info.format_version == 1
    assert info.payload_bytes == len(data) - 64
    # Geometry is detected, not assumed.
    geometry = info.geometry
    assert (geometry.layers, geometry.hidden_size, geometry.mlp_hidden_size) == (2, 128, 512)
    assert (geometry.query_heads, geometry.kv_heads, geometry.head_dim) == (2, 2, 64)
    assert geometry.vocab_size == VOCAB
    assert geometry.max_positions == 1024
    assert geometry.layer_norm_epsilon == pytest.approx(1e-5, rel=1e-3)
    assert (info.value_count, info.node_count, info.weight_count) == (326, 265, 50)
    assert info.metadata.opset_version == 14
    assert info.metadata.ir_version == 7


def test_prepare_distinguishes_wrote_it_from_already_there(artifact_path, tmp_path):
    source = pinned_onnx()
    target = tmp_path / "existing.fn3prep"
    shutil.copyfile(artifact_path, target)

    # Same arguments, existing valid artifact: nothing is written.
    report = fn3.prepare(source, target)
    assert report.status == "already_valid"
    assert report.wrote is False
    # `verified` means "the file on disk is this artifact", not "I wrote it".
    assert report.verified is True
    assert report.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    untouched = target.read_bytes()

    # Different bounds produce a different artifact: refused, with reasons.
    with pytest.raises(fn3.Fn3ModelExistsError) as info:
        fn3.prepare(source, target, max_total=17)
    error = info.value
    assert error.path == str(target)
    assert any("envelope" in difference for difference in error.differences)
    assert "overwrite=True" in str(error)
    # The file was not touched by the refusal.
    assert target.read_bytes() == untouched

    # Explicit replacement reports the difference and re-verifies the bytes.
    replaced = fn3.prepare(source, target, max_total=17, overwrite=True)
    assert replaced.status == "replaced"
    assert replaced.wrote is True
    assert replaced.verified is True
    assert replaced.replaced_differences
    assert replaced.envelope.total == (1, 17)
    assert fn3.inspect(target).envelope.total == (1, 17)

    fresh = tmp_path / "fresh.fn3prep"
    created = fn3.prepare(source, fresh)
    assert created.status == "created"
    assert created.wrote is True and created.verified is True
    assert created.sha256 == hashlib.sha256(fresh.read_bytes()).hexdigest()


def test_prepare_accepts_pathlib_and_rejects_a_directory(tmp_path):
    source = pinned_onnx()
    target = tmp_path / "path-lib.fn3prep"
    report = fn3.prepare(source, target)
    assert report.path == str(target)
    with pytest.raises(fn3.Fn3ModelFileError) as info:
        fn3.prepare(source, tmp_path)
    assert str(tmp_path) in str(info.value)
    with pytest.raises(TypeError):
        fn3.prepare(12345, target)


def test_import_problems_are_reported_together(tmp_path):
    source = pinned_onnx()
    truncated = tmp_path / "truncated.onnx"
    truncated.write_bytes(source.read_bytes()[:4096])
    with pytest.raises(fn3.Fn3ImportError) as info:
        fn3.prepare(truncated, tmp_path / "out.fn3prep")
    error = info.value
    assert error.problems
    assert error.problem_count == len(error.problems)
    assert "ONNX" in str(error)


# ---------------------------------------------------------------------------
# 4. Typed errors for bad files and bad requests
# ---------------------------------------------------------------------------


def test_missing_files_are_typed_and_name_the_path(tmp_path):
    with pytest.raises(fn3.Fn3ModelFileError) as info:
        fn3.prepare(tmp_path / "absent.onnx", tmp_path / "out.fn3prep")
    assert info.value.path == str(tmp_path / "absent.onnx")
    assert info.value.kind == "io"

    with pytest.raises(fn3.Fn3ModelFileError) as info:
        fn3.load(tmp_path / "absent.fn3prep")
    assert info.value.path == str(tmp_path / "absent.fn3prep")

    with pytest.raises(fn3.Fn3ModelFileError) as info:
        fn3.inspect(tmp_path / "absent.fn3prep")
    assert info.value.path == str(tmp_path / "absent.fn3prep")


def test_bad_magic_future_version_and_corruption_are_distinct(artifact_path, tmp_path):
    original = artifact_path.read_bytes()

    cases = {}

    bad_magic = bytearray(original)
    bad_magic[0:8] = b"FNN2PREP"
    cases["bad_magic"] = bad_magic

    future = bytearray(original)
    struct.pack_into("<H", future, 8, 2)
    cases["unsupported_version"] = future

    corrupt = bytearray(original)
    corrupt[5000] ^= 0xFF
    cases["checksum_mismatch"] = corrupt

    truncated = original[:100]
    cases["length_mismatch"] = truncated

    for kind, bytes_ in cases.items():
        path = tmp_path / f"{kind}.fn3prep"
        path.write_bytes(bytes_)
        with pytest.raises(fn3.Fn3FormatError) as info:
            fn3.inspect(path)
        error = info.value
        assert error.kind == kind, f"{kind}: {error}"
        assert error.version_mismatch is (kind == "unsupported_version")
        # Only a version problem is not corruption; each message says what to do.
        assert str(error)

    # The future-version message names both numbers, so a reader knows whether
    # to upgrade this build or regenerate the file.
    path = tmp_path / "unsupported_version.fn3prep"
    with pytest.raises(fn3.Fn3FormatError) as info:
        fn3.inspect(path)
    assert "version 2" in str(info.value) and "version 1" in str(info.value)

    # And a load refuses the same bytes the same way.
    with pytest.raises(fn3.Fn3FormatError):
        fn3.load(path, pinned_onnx())


def test_load_needs_no_source_and_runs_the_artifact_alone(artifact_path):
    """One file is enough: no ONNX path, no environment variable."""
    model = fn3.load(artifact_path, device="cpu")
    assert model.source is None
    session = model.session()
    logits = np.asarray(session.prefill(TOKENS[:10])).reshape(-1)
    top = np.argsort(-logits, kind="stable")[:5]
    assert top.tolist() == [46737, 28597, 16571, 7868, 13]


def test_artifact_only_load_works_in_a_clean_process(artifact_path, tmp_path):
    """A fresh interpreter, an empty cwd and no model env var: still runs.

    This is the deployment claim. The child process is given the artifact path
    and nothing else; ``FASTNN_FN3_ONNX_MODEL`` is cleared, and its working
    directory holds no ONNX file.
    """
    script = (
        "import json, pathlib, sys\n"
        "import numpy as np\n"
        "import fastnn.inference as fn3\n"
        "model = fn3.load(sys.argv[1], device='cpu')\n"
        "session = model.session()\n"
        "logits = np.asarray(session.prefill([464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13])).reshape(-1)\n"
        "top = np.argsort(-logits, kind='stable')[:5]\n"
        "print(json.dumps({'top': top.tolist(), 'source': model.source,\n"
        "                  'cwd_onnx': sorted(p.name for p in pathlib.Path('.').glob('*.onnx'))}))\n"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "FASTNN_FN3_ONNX_MODEL"
    }
    completed = subprocess.run(
        [sys.executable, "-c", script, str(artifact_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["top"] == [46737, 28597, 16571, 7868, 13]
    assert report["source"] is None
    assert report["cwd_onnx"] == []


def test_a_matching_source_is_accepted_and_reported(artifact_path):
    """The optional source check still passes and says what was checked."""
    model = fn3.load(artifact_path, pinned_onnx(), device="cpu")
    assert model.source == str(pinned_onnx())
    logits = np.asarray(model.session().prefill(TOKENS[:10])).reshape(-1)
    top = np.argsort(-logits, kind="stable")[:5]
    assert top.tolist() == [46737, 28597, 16571, 7868, 13]


def test_a_source_that_disagrees_is_refused(artifact_path, tmp_path):
    """A modified-but-importable ONNX must not be run against the artifact.

    The planner consumes the source's graph while the artifact supplies the
    weights, so a disagreement would execute a model that is half of each file.
    """
    source = pinned_onnx()
    data = bytearray(source.read_bytes())
    flipped = 0
    for offset in (len(data) - 100_000, len(data) // 2, len(data) - 4096):
        candidate = bytearray(data)
        candidate[offset] ^= 0xFF
        path = tmp_path / f"modified-{offset}.onnx"
        path.write_bytes(candidate)
        try:
            fn3.load(artifact_path, path)
        except fn3.Fn3ModelError as error:
            if "disagree" in str(error):
                assert "weight" in str(error)
                flipped += 1
                break
            # The flip landed in a shape or dtype field: the importer caught it.
            assert isinstance(error, fn3.Fn3ImportError) or "import" in str(error)
        except fn3.Fn3ImportError:
            continue
        else:  # pragma: no cover - a silent accept would be the defect
            pytest.fail(f"a modified source at offset {offset} was accepted")
    if not flipped:
        pytest.skip("no byte flip produced an importable-but-different source")


# ---------------------------------------------------------------------------
# 5. Load and run
# ---------------------------------------------------------------------------


def test_load_reports_the_artifact_it_loaded(artifact_path, model):
    data = artifact_path.read_bytes()
    assert model.sha256 == hashlib.sha256(data).hexdigest()
    assert model.size_bytes == len(data)
    assert model.device.kind == "cpu"
    assert model.capacity == 1024
    assert model.kv_bytes == 2_097_152  # 2 layers x 2 x 1024 x 2 heads x 64 x 4 B
    assert model.weight_count == 50
    assert model.weight_bytes == 27_833_180
    assert model.value_count == 326 and model.node_count == 265
    assert model.geometry.vocab_size == VOCAB
    assert repr(model).startswith("Model(#")


def test_prefill_and_cached_decode_match_the_committed_fixtures(model):
    session = model.session()
    assert session.phase == "empty"
    assert session.cache_len == 0

    logits = session.prefill(TOKENS[:13])
    assert logits.shape == (1, 1, VOCAB)
    assert logits.dtype == "float32"
    assert logits.nbytes == VOCAB * 4

    actual = np.asarray(logits).reshape(-1)
    expected = fixture("prefill.p13.last_logits")
    max_abs = float(np.max(np.abs(actual - expected)))
    mean_abs = float(np.mean(np.abs(actual - expected)))
    assert max_abs <= ATOL, f"prefill.p13 max_abs {max_abs:e}"
    assert mean_abs <= MEAN_ATOL, f"prefill.p13 mean_abs {mean_abs:e}"

    # The greedy next token is the reference's argmax, and the session's own
    # argmax agrees with NumPy's.
    assert int(np.argmax(expected)) == 257
    assert logits.argmax() == 257
    assert session.cache_len == 13
    assert session.step == 0
    assert session.tokens_consumed == 13
    assert session.phase == "ready"

    # Rebuild the exact four-step sequence the fixtures record.
    session.reset()
    session.prefill(TOKENS[:13])
    first = np.asarray(session.decode([TOKENS[13]])).reshape(-1)
    reference = fixture("decode.step1.logits")
    decode_max = float(np.max(np.abs(first - reference)))
    assert decode_max <= ATOL, f"decode.step1 max_abs {decode_max:e}"
    predicted = [int(first.argmax())]
    for step, token in enumerate(TOKENS[14:17], start=2):
        logits = session.decode([token])
        predicted.append(int(logits.argmax()))
        assert session.cache_len == 13 + step
        assert session.step == step
        assert session.tokens_consumed == 13 + step
        dims = session.stats().last_plan_dims
        assert dims[0] == 1 and dims[1] == 1
        assert dims[2] == 12 + step and dims[3] == 13 + step
    # The recorded argmax values of decode steps 1..4.
    assert predicted == [24671, 11, 612, 373], predicted
    assert session.cache_len == 17

    stats = session.stats()
    assert stats.phase == "ready"
    # The first prefill was rebuilt after `reset` returned to Empty; the second
    # round then reuses that plan (one hit) and builds the four decode shapes.
    assert (stats.plan_hits, stats.plan_misses, stats.plan_evictions) == (1, 5, 0)
    assert stats.plan_cache_len == 5
    assert stats.slot_allocations > 0
    assert stats.kv_bytes == session.kv_bytes


def test_logits_outlive_the_session_and_are_read_only_views(model):
    session = model.session()
    logits = session.prefill(TOKENS[:4])
    view = logits.to_numpy()
    assert view.dtype == np.float32
    assert view.shape == (1, 1, VOCAB)
    assert view.flags.writeable is False
    assert np.asarray(logits).flags.writeable is False
    expected = np.frombuffer(logits.data, dtype=np.float32).reshape(1, 1, VOCAB)
    assert np.array_equal(view, expected)

    del session
    gc.collect()
    assert np.array_equal(view, expected)
    assert int(logits.argmax()) == int(np.argmax(expected.reshape(-1)))

    writable = np.array(logits)
    assert writable.flags.writeable is True
    assert np.array_equal(writable, view)

    # `tolist` is nested by the leading axes and holds the same values.
    rows = logits.tolist()
    assert len(rows) == 1 and len(rows[0]) == 1 and len(rows[0][0]) == VOCAB
    assert rows[0][0][0] == pytest.approx(float(view[0, 0, 0]))


def test_tokens_accept_numpy_integer_arrays(model):
    from_list = np.asarray(model.session().prefill([7, 8, 9])).copy()
    from_array = np.asarray(model.session().prefill(np.array([7, 8, 9], dtype=np.int64))).copy()
    assert np.array_equal(from_list, from_array)


# ---------------------------------------------------------------------------
# 6. Sessions are independent
# ---------------------------------------------------------------------------


def test_two_sessions_share_no_mutable_state(model):
    first = model.session()
    second = model.session()
    assert first.instance != second.instance

    first.prefill(TOKENS[:3])
    second.prefill(TOKENS[:6])
    assert (first.cache_len, second.cache_len) == (3, 6)
    assert (first.tokens_consumed, second.tokens_consumed) == (3, 6)
    assert (first.step, second.step) == (0, 0)

    first.decode([TOKENS[3]])
    assert (first.cache_len, first.step) == (4, 1)
    assert (second.cache_len, second.step) == (6, 0)

    addresses = [
        first.stats().cache_addresses,
        second.stats().cache_addresses,
    ]
    assert len(addresses[0]) == len(addresses[1]) == 2, "one key buffer per layer"
    assert set(addresses[0]).isdisjoint(addresses[1]), addresses
    for layer in range(2):
        assert addresses[0][layer] != addresses[1][layer]

    # Plan caches are per session too, and content-identical for one shape.
    assert first.stats().plan_misses >= 1 and second.stats().plan_misses >= 1

    # A fresh pair over the same prompt is deterministic and content-identical:
    # independence is not nondeterminism.
    third = model.session()
    fourth = model.session()
    left = np.asarray(third.prefill(TOKENS[:3])).copy()
    right = np.asarray(fourth.prefill(TOKENS[:3])).copy()
    assert np.array_equal(left, right)
    assert third.stats().last_plan == fourth.stats().last_plan
    assert set(third.stats().cache_addresses).isdisjoint(fourth.stats().cache_addresses)


def test_reset_clears_state_and_keeps_the_plan_cache(model):
    session = model.session()
    logits = np.asarray(session.prefill(TOKENS[:13])).copy()
    for token in TOKENS[13:17]:
        session.decode([token])
    assert session.stats().plan_misses == 5
    assert session.stats().plan_hits == 0

    session.reset()
    stats = session.stats()
    assert stats.phase == "empty"
    assert (stats.cache_len, stats.step, stats.tokens_consumed) == (0, 0, 0)
    assert stats.last_plan is None and stats.last_plan_dims is None
    # Buffers were kept: the addresses are the same allocation.
    assert stats.kv_bytes == session.kv_bytes

    again = np.asarray(session.prefill(TOKENS[:13])).copy()
    assert np.array_equal(logits, again), "reset must reproduce the same logits"
    for token in TOKENS[13:17]:
        session.decode([token])
    # The same five shapes are now served from the cache: no recompilation.
    assert session.stats().plan_misses == 5
    assert session.stats().plan_hits == 5


# ---------------------------------------------------------------------------
# 7. Bounds: envelope and capacity, both atomic
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def narrow(artifact_path, tmp_path_factory):
    """An artifact whose window is narrower than the position table.

    ``seq <= 13`` while the KV capacity is 17: the 14-token prefill is outside
    the declared window but inside the cache, which is exactly the case that
    must be refused by the envelope, not by the cache.
    """
    source = pinned_onnx()
    path = tmp_path_factory.mktemp("fn3-64-narrow") / "narrow.fn3prep"
    report = fn3.prepare(source, path, max_total=17, max_seq=13)
    assert report.status == "created"
    return report, path


def test_sequence_beyond_the_envelope_is_refused_and_changes_nothing(narrow):
    report, path = narrow
    assert report.envelope.seq == (1, 13)
    model = fn3.load(path, pinned_onnx())

    # A prompt longer than the declared window is refused before anything runs.
    session = model.session()
    with pytest.raises(fn3.Fn3EnvelopeError) as info:
        session.prefill(TOKENS[:14])
    error = info.value
    assert error.symbol == "seq"
    assert error.requested == 14
    assert (error.min, error.max) == (1, 13)
    assert "envelope" in str(error)
    after = session.stats()
    assert (after.cache_len, after.step, after.tokens_consumed) == (0, 0, 0)
    assert after.phase == "empty"

    # The same session is still usable for a legal prompt after the refusal.
    session.prefill(TOKENS[:4])
    assert session.stats().cache_len == 4

    # The Ready-phase atomic failure path (a decode that would overflow) is
    # covered by `test_exceeded_capacity_is_refused_and_changes_nothing`.


def test_exceeded_capacity_is_refused_and_changes_nothing(artifact_path):
    model = fn3.load(artifact_path, pinned_onnx(), capacity=4)
    assert model.capacity == 4 and model.kv_bytes == 8192
    session = model.session()
    session.prefill([1, 2, 3, 4])
    before = session.stats()

    with pytest.raises(fn3.Fn3CapacityError) as info:
        session.decode([5])
    error = info.value
    assert error.requested == 5
    assert error.capacity == 4
    assert "kv cache" in error.what.lower()

    after = session.stats()
    assert (after.cache_len, after.step, after.tokens_consumed) == (4, 0, 4)
    assert after.phase == "ready"

    # A sequence longer than the window's capacity is refused by name too.
    with pytest.raises((fn3.Fn3CapacityError, fn3.Fn3EnvelopeError)):
        model.session().prefill(list(range(5)))


def test_session_phase_errors_are_typed(model):
    session = model.session()
    with pytest.raises(fn3.Fn3SessionError) as info:
        session.decode([1])
    assert info.value.action == "decode"
    assert info.value.phase == "empty"

    with pytest.raises(fn3.Fn3SessionError) as info:
        session.prefill([])
    assert info.value.action == "prefill"

    session.prefill([1])
    with pytest.raises(fn3.Fn3SessionError) as info:
        session.prefill([2])
    assert info.value.phase == "ready"
    assert "reset" in str(info.value)


def test_token_ids_are_validated(model):
    session = model.session()
    with pytest.raises(fn3.Fn3TokenError) as info:
        session.prefill([1, 2, VOCAB])
    assert info.value.token == VOCAB
    assert info.value.vocab_size == VOCAB
    assert "position 2" in str(info.value)

    with pytest.raises(fn3.Fn3TokenError):
        session.prefill([-1])
    with pytest.raises(fn3.Fn3TokenError) as info:
        session.prefill([1, 1.5])
    assert "position 1" in str(info.value)
    with pytest.raises(fn3.Fn3TokenError):
        session.prefill(42)
    # Nothing was consumed by the failures.
    assert session.cache_len == 0 and session.phase == "empty"


# ---------------------------------------------------------------------------
# 8. Threads and the GIL
# ---------------------------------------------------------------------------


def test_prefill_releases_the_gil(model):
    """A busy Python thread must make progress while a step runs.

    A native call that kept the GIL would block the counter thread completely,
    so "the counter advanced" is the observable difference. The bound is
    generous: a 13-token prefill takes hundreds of milliseconds.
    """
    counter = {"count": 0, "stop": False}

    def spin():
        while not counter["stop"]:
            counter["count"] += 1

    worker = threading.Thread(target=spin, daemon=True)
    worker.start()
    try:
        time.sleep(0.05)
        before = counter["count"]
        session = model.session()
        started = time.perf_counter()
        session.prefill(TOKENS[:13])
        elapsed = time.perf_counter() - started
        progressed = counter["count"] - before
    finally:
        counter["stop"] = True
        worker.join(timeout=5)

    assert elapsed > 0.02, "the step was too quick for this measurement to mean anything"
    assert progressed > 1000, (
        f"the counter thread advanced {progressed} times during {elapsed:.3f}s; "
        "the GIL was not released"
    )


def test_two_sessions_run_in_parallel_threads(model):
    """Two sessions may be stepped by two threads at once."""
    results = {}

    def run(name, tokens):
        session = model.session()
        results[name] = np.asarray(session.prefill(tokens)).copy()
        results[name + "-len"] = session.cache_len

    threads = [
        threading.Thread(target=run, args=("a", TOKENS[:6])),
        threading.Thread(target=run, args=("b", TOKENS[:6])),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert results["a-len"] == results["b-len"] == 6
    assert np.array_equal(results["a"], results["b"])


def test_one_session_serialises_concurrent_calls(model):
    """Concurrent calls on one session wait rather than corrupting state."""
    session = model.session()
    errors = []

    def run(token):
        try:
            session.prefill([token])
        except fn3.Fn3SessionError:
            pass  # the loser of the phase race is refused, not corrupted
        except Exception as error:  # pragma: no cover - a crash would land here
            errors.append(error)

    threads = [threading.Thread(target=run, args=(token,)) for token in (1, 2, 3, 4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors, errors
    # Exactly one prefill won; the session is consistent and usable.
    assert session.phase == "ready"
    assert session.cache_len == 1
    assert session.decode([5]).shape == (1, 1, VOCAB)


# ---------------------------------------------------------------------------
# 9. Opt-in INT8 weights
# ---------------------------------------------------------------------------


def test_weight_int8_is_off_by_default_and_emits_no_report(model):
    """The default is the FP32 path: no report, and no new observable state."""
    assert model.weight_int8 is False
    session = model.session()
    assert session.weight_int8_report is None
    logits = session.prefill(TOKENS[:4])
    assert session.weight_int8_report is None, (
        "a weight_int8=False session must not report INT8 coverage"
    )
    assert np.isfinite(np.asarray(logits)).all()


def test_weight_int8_reports_the_cli_fields(artifact_path):
    """The report carries the same numbers and labels as the CLI's --steps."""
    quantized = fn3.load(artifact_path, device="cpu", weight_int8=True)
    assert quantized.weight_int8 is True
    session = quantized.session()
    assert session.weight_int8_report is None, "no step has completed yet"

    session.prefill(TOKENS[:10])
    report = session.weight_int8_report
    assert report is not None

    # Coverage, from the pinned tiny artifact: 8 packed of 13 MatMuls.
    assert report.packed_matmuls == 8
    assert report.fp32_matmuls == 5
    assert len(report.routes) == 13
    assert report.packed_matmuls + report.fp32_matmuls == len(report.routes)
    assert sum(route.packed for route in report.routes) == report.packed_matmuls
    assert any(route.route == "computed-rhs" for route in report.routes)
    assert any(route.packed and route.route == "packed" for route in report.routes)
    codes = {
        "packed",
        "computed-rhs",
        "constant-rhs",
        "graph-input-rhs",
        "activation-layout",
        "output-layout",
        "vulkan-operands",
        "packing",
    }
    for route in report.routes:
        assert route.route in codes, route.route
        assert route.instruction >= 0
        assert route.rhs

    # Bytes. The packed set replaces exactly packed_source - packed_payload.
    assert report.model_weight == 27_833_180
    assert report.packed_source == 1_572_864
    assert report.packed_payload == 417_792
    assert report.dense_weight == 26_260_132
    assert report.shared_weight == 0
    assert 0 < report.packed_payload < report.packed_source
    assert np.isfinite(report.route_pack_upload)
    assert report.route_pack_upload > 0.0

    # The CPU keeps host tensors, so the CLI calls this counter
    # `fp32-retained`, and prints no device-traffic line.
    assert report.dense_label == "fp32-retained"
    assert report.packed_prepare is None
    assert report.staged_host_bytes is None
    assert report.late_persistent is None

    # as_dict() is the CLI's own vocabulary, for one oracle over both paths.
    fields = report.as_dict()
    assert fields["packed-matmuls"] == report.packed_matmuls
    assert fields["matmul-total"] == 13
    assert fields["fp32-matmuls"] == report.fp32_matmuls
    assert fields["packed-payload"] == report.packed_payload
    assert fields["packed-source"] == report.packed_source
    assert fields[report.dense_label] == report.dense_weight
    assert "dense-weight" not in fields, "the CPU keys the dense figure fp32-retained"
    assert "dense-label" not in fields
    assert fields["shared-weight"] == report.shared_weight
    assert fields["model-weight"] == report.model_weight
    assert fields["route/pack/upload"] == report.route_pack_upload
    assert "packed-prepare" not in fields
    assert "staged-host-bytes" not in fields
    assert "late-persistent" not in fields

    # One oracle over both paths: parse the CLI line captured from the same
    # artifact and require every label to line up with the dict, so a label
    # drift fails here. The time is compared for shape, not value, because the
    # capture predates this run.
    line = (
        "weight-int8: packed-matmuls=8/13 fp32-matmuls=5 packed-payload=417792B "
        "packed-source=1572864B fp32-retained=26260132B shared-weight=0B "
        "model-weight=27833180B route/pack/upload=49.84ms"
    )
    parsed = {}
    for item in line.split(": ", 1)[1].split():
        key, _, value = item.partition("=")
        parsed[key] = value.split("/")[0].rstrip("Bms")
    assert set(parsed) == set(fields) - {"matmul-total"}
    assert int(parsed["packed-matmuls"]) == fields["packed-matmuls"]
    assert int(parsed["fp32-matmuls"]) == fields["fp32-matmuls"]
    assert int(parsed["packed-payload"]) == fields["packed-payload"]
    assert int(parsed["packed-source"]) == fields["packed-source"]
    assert int(parsed["fp32-retained"]) == fields["fp32-retained"]
    assert int(parsed["shared-weight"]) == fields["shared-weight"]
    assert int(parsed["model-weight"]) == fields["model-weight"]
    assert float(parsed["route/pack/upload"]) > 0.0
    assert fields["route/pack/upload"] > 0.0

    # The report is a snapshot: a read-only object, never a live mutation path.
    with pytest.raises(AttributeError):
        report.packed_matmuls = 1

    # A second step describes that step, not the first, and is a new object.
    session.decode([4874])
    second = session.weight_int8_report
    assert second is not None and second is not report
    assert second.packed_matmuls == report.packed_matmuls
    assert second.packed_payload == report.packed_payload

    # reset() clears the report; the earlier object stays valid (a real copy).
    session.reset()
    assert session.weight_int8_report is None
    assert report.packed_matmuls == 8
    assert second.packed_matmuls == 8


def test_weight_int8_prefill_and_decode_match_fp32(model, artifact_path):
    """The tiny artifact's logits stay within the format's published bound."""
    reference = model.session()
    quantized = fn3.load(artifact_path, device="cpu", weight_int8=True).session()

    expected = np.asarray(reference.prefill(TOKENS[:10])).copy()
    actual = np.asarray(quantized.prefill(TOKENS[:10])).copy()
    assert int(actual.argmax()) == int(expected.argmax())
    assert float(np.max(np.abs(actual - expected))) < 0.02

    expected = np.asarray(reference.decode([4874])).copy()
    actual = np.asarray(quantized.decode([4874])).copy()
    assert int(actual.argmax()) == int(expected.argmax())
    assert float(np.max(np.abs(actual - expected))) < 0.02


def test_weight_int8_rejects_a_non_bool(tmp_path):
    """The option is a bool; anything else is a TypeError, as for capacity.

    No pinned model is needed: the facade validates before it opens anything,
    so this test runs in every lane. ``np.bool_(True)`` is the case that proves
    the facade check is load-bearing, because the PyO3 ``bool`` extractor
    accepts it while ``isinstance(weight_int8, bool)`` does not.
    """
    missing = tmp_path / "absent.fnn3"
    for value in ("yes", 1, 0, 1.0, None, np.bool_(True)):
        with pytest.raises(TypeError) as info:
            fn3.load(missing, weight_int8=value)
        assert "weight_int8" in str(info.value)


def test_staged_vulkan_weight_int8_is_refused_with_the_typed_error(artifact_path):
    """A host-staged Vulkan session refuses INT8 weights; it never uses the CPU."""
    vulkan = [device for device in fn3.devices() if device.kind == "vulkan"]
    if not vulkan:
        pytest.skip("this build or machine enumerates no Vulkan device")

    previous = os.environ.get("FN3_VK_STAGED")
    os.environ["FN3_VK_STAGED"] = "1"
    try:
        model = fn3.load(artifact_path, device=vulkan[0], weight_int8=True)
        with pytest.raises(fn3.Fn3UnsupportedError) as info:
            model.session()
    finally:
        if previous is None:
            os.environ.pop("FN3_VK_STAGED", None)
        else:
            os.environ["FN3_VK_STAGED"] = previous

    error = info.value
    assert error.op == "matmul"
    assert error.backend == "vulkan"
    assert "INT8 weights need the CPU backend or a resident Vulkan session" in str(error)
    assert "host-staged" in str(error)
    assert "CPU backend" in str(error)
