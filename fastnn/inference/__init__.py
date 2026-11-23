"""fastnn.inference — the FastNN 3.0 inference API.

One prepared artifact is the whole input. ``prepare`` imports ONNX, plans and
writes one versioned file; ``load`` maps it (no ONNX, no re-planning) and
``Model.session().run({...})`` executes one forward.

Quick start
-----------

::

    import fastnn.inference as fn3

    report = fn3.prepare("model.onnx", "model.fnn3", target="all")
    print(report.status, report.size_bytes, report.sha256)

    model = fn3.load("model.fnn3")
    session = model.session()
    outputs = session.run({"images": [0.0] * (3 * 320 * 320)})
    print(outputs["output0"].shape, outputs["output0"].to_numpy())

``inspect`` describes the file without mapping its weight payload; ``doctor``
diagnoses an ONNX model before preparing it. A failure is a typed exception from
the hierarchy below, with the core's own message.
"""

from __future__ import annotations

import os

from fastnn._core.inference import (
    ArtifactInfo,
    Device,
    DoctorIssue,
    DoctorReport,
    Metadata,
    Model,
    PlanInfo,
    PrepareReport,
    Section,
    Session,
    Tensor,
    devices as _devices,
    doctor as _doctor,
    inspect as _inspect,
    load as _load,
    prepare as _prepare,
    targets as _targets,
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
    """Devices this build can execute on, in preference order."""
    return _devices()


def targets() -> tuple[str, ...]:
    """Target profile names ``prepare`` accepts."""
    return tuple(_targets())


def prepare(
    onnx,
    artifact,
    *,
    target: str = "cpu",
    precision: str = "f32",
    overwrite: bool = False,
) -> PrepareReport:
    """Import ``onnx`` and write one versioned artifact at ``artifact``.

    ``target`` is one of :func:`targets` (``"cpu"``, ``"cpu-reference"`` or
    ``"all"``); ``precision`` is ``"f32"``. A destination that already holds the
    same bytes is left alone and reported as ``status="already_valid"``. A
    different destination needs ``overwrite=True``.
    """
    return _prepare(
        _path(onnx),
        _path(artifact),
        target=target,
        precision=precision,
        overwrite=overwrite,
    )


def inspect(artifact) -> ArtifactInfo:
    """Describe an artifact without mapping its weight payload.

    ``mapped_bytes`` is 0 and ``bytes_read`` is the evidence: the reader reads
    the header, directory, ``GRAPH``, ``PLANS``, ``QUANT``, ``HASHES`` and the
    ``WEIGHTS`` table, never the weight payload.
    """
    return _inspect(_path(artifact))


def load(artifact, *, target: str | None = None, device=None) -> Model:
    """Map an artifact and return an immutable :class:`Model`.

    No ONNX file is read and the plan stored in the artifact is decoded, not
    rebuilt. ``target`` selects one of the stored plans; ``None`` uses the
    artifact's default. ``device`` accepts ``None``/``"cpu"``; the typed-graph
    runtime is CPU-only and refuses another device instead of falling back.
    """
    return _load(_path(artifact), target=target, device=device)


def doctor(onnx) -> DoctorReport:
    """Diagnose an ONNX model before preparing it.

    Reports unsupported operators, dynamic-shape symbols, dtype problems and
    the expected resident memory, with the source node on every finding.
    """
    return _doctor(_path(onnx))


__all__ = [
    "ArtifactInfo",
    "Device",
    "DoctorIssue",
    "DoctorReport",
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
    "Metadata",
    "Model",
    "PlanInfo",
    "PrepareReport",
    "Section",
    "Session",
    "Tensor",
    "devices",
    "doctor",
    "inspect",
    "load",
    "prepare",
    "targets",
]
