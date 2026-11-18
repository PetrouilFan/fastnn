#!/usr/bin/env bash
# Physical-GPU evidence path for the gpu-physical lane.
#
# THIS IS THE ONLY ACCEPTED FORM OF PHYSICAL-GPU EVIDENCE.
# It is never run by hosted CI. Run it on a trusted local/protected machine:
#
#   tools/ci/run-gpu-lane.sh --out docs/reports/fn3-20/gpu-evidence/<name>.json
#
# The record it writes contains the device name, UUID, driver version, Vulkan
# API version, validation results, pass/fail and timings. A software/lavapipe
# record is rejected by tools/ci/verify_evidence.py and cannot satisfy the lane.
#
# Rules for running it:
#   * never on a shared hosted runner, and never for untrusted code;
#   * the machine must have a physical Vulkan device (lavapipe alone is not
#     enough);
#   * commit the record (plus any log the script leaves next to it) when using
#     it as evidence; CI validates committed records and never creates one.
#
# Exit codes: 0 passed, 1 failed/refused, 69 not verified (truthful record was
# still written, e.g. the Vulkan backend does not exist yet).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=""
EXTRA=()

usage() {
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while (( $# > 0 )); do
    case "$1" in
        --out)
            OUT="${2:?--out needs a path}"
            shift 2
            ;;
        --device-uuid)
            EXTRA+=(--device-uuid "${2:?--device-uuid needs a UUID}")
            shift 2
            ;;
        --timeout-seconds)
            EXTRA+=(--timeout-seconds "${2:?--timeout-seconds needs a number}")
            shift 2
            ;;
        --no-spike)
            EXTRA+=(--no-spike)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "${GITHUB_ACTIONS:-}" == "true" && "${FN3_ALLOW_HOSTED_GPU:-}" != "1" ]]; then
    echo "REFUSED: physical-GPU evidence must be produced on a trusted local or protected machine," >&2
    echo "never on a hosted runner and never for untrusted code." >&2
    exit 1
fi

if [[ -z "$OUT" ]]; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    OUT="$ROOT/target/ci-gpu-evidence/gpu-evidence-${stamp}.json"
fi

echo "LANE gpu-physical — physical device run"
echo "  PROVES: the named physical Vulkan device, driver and API version, validation results,"
echo "          and the fastnn Vulkan smoke outcome on this machine."
echo "  DOES NOT PROVE: any other device, driver or OS; hosted CI never runs this lane."
echo "  RECORD: $OUT"
echo

python3 "$ROOT/tools/ci/gpu_evidence.py" --out "$OUT" "${EXTRA[@]}"
status=$?

if [[ -f "$OUT" ]]; then
    echo
    echo "== Validate the record against the evidence contract =="
    echo "\$ python3 tools/ci/verify_evidence.py '$OUT'"
    python3 "$ROOT/tools/ci/verify_evidence.py" "$OUT" || true
fi

case "$status" in
    0) echo "PHYSICAL-GPU RESULT: PASSED — commit '$OUT' as evidence." ;;
    69) echo "PHYSICAL-GPU RESULT: NOT VERIFIED — the record is truthful but is not evidence of a working Vulkan backend." ;;
    *) echo "PHYSICAL-GPU RESULT: FAILED — do not commit this record as evidence." ;;
esac
exit "$status"
