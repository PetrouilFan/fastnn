#!/usr/bin/env bash
# Lane: cpu (job: cpu-python)
#
# Builds the extension with maturin and runs the Python suite marked
# "not slow". This is the second half of the cpu lane: it still touches no GPU.
#
# Today the suite exercises the frozen 2.x Python facade; FN3-18 (#64) adds the
# 3.0 Python surface and FN3-19 (#65) deletes the 2.x tests.
set -uo pipefail

CI_LANE=cpu
CI_JOB=cpu-python
source "$(dirname "${BASH_SOURCE[0]}")/lane-lib.sh"

VENV="${FASTNN_VENV:-$CI_ROOT/.venv}"
PYTHON="${FASTNN_PYTHON:-python3}"
ONNX_PINS="${FASTNN_ONNX_PINS:-onnx==1.17.0 onnxruntime==1.20.1}"

ci::describe cpu

if ! command -v uv >/dev/null 2>&1; then
    ci::failed "uv is required to reproduce this lane" "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

ci::run "Create virtualenv" "uv venv '$VENV' --python 3.12 --allow-existing"
ci::run "Install build/test dependencies" \
    "uv pip install --python '$VENV/bin/python' 'maturin>=1.7,<2.0' pytest numpy $ONNX_PINS"
ci::run "Build and install the extension (maturin develop)" \
    "VIRTUAL_ENV='$VENV' '$VENV/bin/maturin' develop"

# The test suite imports numpy/onnx directly; prove the environment is honest
# before attributing any failure to fastnn.
ci::run "Verify test dependencies import" \
    "'$VENV/bin/python' -c 'import numpy, onnx, onnxruntime; print(numpy.__version__, onnx.__version__, onnxruntime.__version__)'"

ci::run "Python tests (not slow)" \
    "'$VENV/bin/python' -m pytest tests/ -m 'not slow' -v"

ci::passed "extension built with maturin and the Python suite marked 'not slow' passed (currently the frozen 2.x facade)"
