#!/usr/bin/env bash
# Regenerate the FastNN parity fixtures from pinned model revisions.
#
# Default: CI suites (ops + tiny-gpt2).  Model files are fetched and checked
# against fixtures/parity/*/model.lock.json before generation.
#
#   tools/parity/regenerate.sh
#   tools/parity/regenerate.sh --include-trained
#   tools/parity/regenerate.sh --out /tmp/parity-check
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/fixtures/parity"
CACHE="$ROOT/tools/parity/.cache/models"
VENV="$ROOT/tools/parity/.cache/venv"
SUITES="ci"
PYTHON_BIN="${PARITY_PYTHON:-3.13}"

usage() {
    cat <<'EOF'
usage: regenerate.sh [--out DIR] [--cache DIR] [--venv DIR] [--include-trained]

  --out DIR            output fixture root (default: fixtures/parity)
  --cache DIR          model cache (default: tools/parity/.cache/models)
  --venv DIR           virtual environment (default: tools/parity/.cache/venv)
  --include-trained    also fetch/generate trained-gpt2 (~498 MB)
  -h, --help           show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --cache) CACHE="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        --include-trained) SUITES="all"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required (https://docs.astral.sh/uv/)" >&2
    exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "+ uv venv --python $PYTHON_BIN $VENV"
    uv venv --python "$PYTHON_BIN" "$VENV"
fi

echo "+ uv pip install torch from the CPU wheel index"
uv pip install \
    --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu \
    -r "$ROOT/tools/parity/requirements-torch.lock.txt"

echo "+ uv pip install tools/parity/requirements.lock.txt"
uv pip install \
    --python "$VENV/bin/python" \
    --index-url https://pypi.org/simple \
    -r "$ROOT/tools/parity/requirements.lock.txt"

export PYTHONPATH="$ROOT/tools/parity${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$SUITES" == "all" ]]; then
    echo "+ fetch tiny-gpt2 and trained-gpt2"
    "$VENV/bin/python" "$ROOT/tools/parity/fetch_models.py" --cache "$CACHE" --suite all
else
    echo "+ fetch tiny-gpt2"
    "$VENV/bin/python" "$ROOT/tools/parity/fetch_models.py" --cache "$CACHE" --suite tiny-gpt2
fi

echo "+ generate fixtures into $OUT"
"$VENV/bin/python" "$ROOT/tools/parity/generate_fixtures.py" \
    --out "$OUT" --cache "$CACHE" --suites "$SUITES"

echo "+ verify fixtures offline"
python3 "$ROOT/tools/parity/verify_fixtures.py" --root "$OUT"

echo "done: $OUT"
