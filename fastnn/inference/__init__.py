"""fastnn.inference — the FastNN 3.0 inference API.

This is the surface that survives the 2.x removal (#65). It is a thin Python
layer over the same Rust engine the CLI and the Rust tests use: one versioned
artifact, an explicit device, sessions that own their cache, and logits as
NumPy arrays.

Quick start
-----------

Prepare once, then load and run::

    import fastnn.inference as fn3

    report = fn3.prepare("model.onnx", "model.fn3prep")
    print(report.status, report.size_bytes, report.sha256)
    # created 27879928 dc03b3c2...     (the artifact's digest, not the ONNX's)

    model = fn3.load("model.fn3prep", source="model.onnx", device="cpu")
    session = model.session()
    logits = session.prefill(prompt_token_ids)
    print(int(logits.argmax()), session.cache_len)   # 257 13

    for token in [640, 11, 612, 373]:
        logits = session.decode([token])
        print(int(logits.argmax()), session.cache_len)
        # 24671 14 / 11 15 / 612 16 / 373 17

`logits.to_numpy()` (or ``numpy.asarray(logits)``) gives a read-only ``float32``
view over one copy of the session output; the session can be reset or dropped
afterwards. ``numpy.array(logits)`` copies if you need a writable array.

Devices
-------

``fn3.devices()`` lists what this build can actually execute on. Passing
``device="vulkan"`` on a build without the Vulkan feature, or on a machine
without a driver, raises :class:`Fn3DeviceError` naming the request, the reason
and the available devices. It never returns CPU results.
"""

from __future__ import annotations

import os

from fastnn._core.inference import (
    ArtifactInfo,
    Device,
    Envelope,
    Geometry,
    Logits,
    Metadata,
    Model,
    PrepareReport,
    Session,
    SessionStats,
    devices as _devices,
    inspect as _inspect,
    load as _load,
    prepare as _prepare,
)
from fastnn._core.inference import (
    Fn3CapacityError,
    Fn3DeviceError,
    Fn3EnvelopeError,
    Fn3Error,
    Fn3ExecutionError,
    Fn3FormatError,
    Fn3ImportError,
    Fn3ModelError,
    Fn3ModelExistsError,
    Fn3ModelFileError,
    Fn3SessionError,
    Fn3ShapeError,
    Fn3TokenError,
    Fn3TypeError,
    Fn3UnsupportedError,
)


def _path(value) -> str:
    """Accept ``str`` and ``os.PathLike``; reject anything else clearly."""
    if isinstance(value, str):
        return value
    try:
        return os.fspath(value)
    except TypeError:
        raise TypeError(
            f"expected a path (str or os.PathLike), found {type(value).__name__}"
        ) from None


def devices() -> list[Device]:
    """Devices this build can execute on, in preference order.

    The CPU is always first. A ``vulkan``-feature build appends every adapter
    the loader enumerates, including software adapters, which
    :func:`load` refuses unless the backend was configured to allow them.
    """
    return _devices()


def prepare(
    onnx,
    artifact,
    *,
    max_total: int | None = None,
    max_seq: int | None = None,
    max_past: int | None = None,
    overwrite: bool = False,
) -> PrepareReport:
    """Import ``onnx`` and write (or confirm) a versioned artifact at ``artifact``.

    Parameters
    ----------
    onnx:
        Path of the ONNX file (opset 13..=17, decoder architecture).
    artifact:
        Destination path.
    max_total:
        Largest ``past + seq`` the artifact authorises, i.e. the context window.
        Defaults to the model's position table, capped at 1024.
    max_seq:
        Longest single call (a prefill). Defaults to ``max_total``.
    max_past:
        Largest cached prefix. Defaults to ``max_total - 1``.
    overwrite:
        Replace a destination that holds a *different* valid artifact.

    Returns
    -------
    PrepareReport
        ``status`` is ``"created"``, ``"replaced"`` or ``"already_valid"``;
        ``wrote`` is false only for ``"already_valid"``. The report carries the
        file size, its SHA-256 (comparable with ``sha256sum``), the artifact's
        own payload CRC-32, the geometry and the declared window.

    Notes
    -----
    A destination that already holds exactly these bytes is left alone. A
    destination that holds a different valid artifact raises
    :class:`Fn3ModelExistsError` (with ``.differences``) unless ``overwrite`` is
    true, and a destination that is not a valid artifact raises
    :class:`Fn3FormatError`. Every successful write is re-read, re-parsed,
    re-validated and compared with the digest of the bytes that were written.
    """
    return _prepare(
        _path(onnx),
        _path(artifact),
        max_total=max_total,
        max_seq=max_seq,
        max_past=max_past,
        overwrite=overwrite,
    )


def inspect(artifact) -> ArtifactInfo:
    """Describe an artifact file without opening a device or a session.

    Reads the whole file, checks its framing, payload CRC and internal
    consistency, and reports the geometry, window, provenance, table sizes and
    SHA-256. A file that fails any check raises :class:`Fn3FormatError` (with
    ``.kind``) or :class:`Fn3ModelFileError`.
    """
    return _inspect(_path(artifact))


def load(
    artifact,
    source=None,
    *,
    device=None,
    capacity: int | None = None,
) -> Model:
    """Load a prepared artifact and bind it to an explicit device.

    The artifact is the whole input: it carries the graph, the weights, the
    geometry and the declared window, so one file is enough and no ONNX is read.

    Parameters
    ----------
    artifact:
        Path of a ``.fn3prep`` artifact produced by :func:`prepare`.
    source:
        Optional path of the ONNX file the artifact was prepared from. When
        given, the two files are proved to describe the same model before
        anything executes — provenance, geometry, every value, every operation
        and every weight byte — and a disagreement raises
        :class:`Fn3ModelError`. When omitted, the artifact alone is planned and
        run; nothing about the session comes from an ONNX file in either case.
    device:
        ``None``/``"auto"``, ``"cpu"``, ``"vulkan"``, ``"vulkan:<index>"`` or a
        :class:`Device` from :func:`devices`. An explicit request either resolves
        to exactly that device or raises :class:`Fn3DeviceError`; there is no
        fallback.
    capacity:
        KV cache capacity in tokens for each session. Defaults to the artifact's
        declared window (``max_total``).

    Returns
    -------
    Model
        Immutable and shareable between threads. ``model.session()`` creates one
        independent session (its own KV cache, plan cache and counters).
    """
    return _load(
        _path(artifact),
        source=None if source is None else _path(source),
        device=device,
        capacity=capacity,
    )


__all__ = [
    "ArtifactInfo",
    "Device",
    "Envelope",
    "Fn3CapacityError",
    "Fn3DeviceError",
    "Fn3EnvelopeError",
    "Fn3Error",
    "Fn3ExecutionError",
    "Fn3FormatError",
    "Fn3ImportError",
    "Fn3ModelError",
    "Fn3ModelExistsError",
    "Fn3ModelFileError",
    "Fn3SessionError",
    "Fn3ShapeError",
    "Fn3TokenError",
    "Fn3TypeError",
    "Fn3UnsupportedError",
    "Geometry",
    "Logits",
    "Metadata",
    "Model",
    "PrepareReport",
    "Session",
    "SessionStats",
    "devices",
    "inspect",
    "load",
    "prepare",
]
