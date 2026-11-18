#!/usr/bin/env bash
# Lane: gpu-physical  (evidence ingestion only — this job has no GPU)
#
# Hosted CI has no GPU and must never run one for untrusted code. This script
# therefore never creates a Vulkan device. It only validates committed records
# produced by tools/ci/run-gpu-lane.sh on a trusted local/protected machine.
#
# No record  -> NOT VERIFIED, naming the script that can produce one.
# Valid record with result not_verified -> NOT VERIFIED (truthful, not evidence).
# Valid record with result passed -> PASSED.
# Malformed record -> FAILED (a hand-written record must not slip through).
set -uo pipefail

CI_LANE=gpu-physical
CI_JOB=gpu-evidence
source "$(dirname "${BASH_SOURCE[0]}")/lane-lib.sh"

EVIDENCE_DIR="${FN3_GPU_EVIDENCE_DIR:-$CI_ROOT/docs/reports/fn3-20/gpu-evidence}"

ci::describe gpu-physical
echo
echo "This job does not have a GPU and does not run one."
echo "Only tools/ci/run-gpu-lane.sh output is accepted as physical-GPU evidence."

shopt -s nullglob
RECORDS=("$EVIDENCE_DIR"/*.json)
shopt -u nullglob

if (( ${#RECORDS[@]} == 0 )); then
    ci::not_verified "no physical-GPU evidence record exists in ${EVIDENCE_DIR#$CI_ROOT/}. Hosted CI has no GPU and never runs this lane; run tools/ci/run-gpu-lane.sh on a machine with a physical Vulkan device, then commit the record. GPU is NOT VERIFIED."
fi

echo
echo "== Validate committed evidence records =="
echo "\$ python3 tools/ci/verify_evidence.py '${EVIDENCE_DIR#$CI_ROOT/}'/*.json"
if ! python3 "$CI_ROOT/tools/ci/verify_evidence.py" "${RECORDS[@]}"; then
    ci::failed "one or more committed physical-GPU evidence records are invalid" \
        "python3 tools/ci/verify_evidence.py docs/reports/fn3-20/gpu-evidence/*.json"
fi

PASSED_RECORDS=()
NOT_VERIFIED_RECORDS=()
for record in "${RECORDS[@]}"; do
    status="$(python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("result") or {}).get("status",""))' "$record")"
    case "$status" in
        passed) PASSED_RECORDS+=("$record") ;;
        not_verified) NOT_VERIFIED_RECORDS+=("$record") ;;
        *)
            ci::failed "record $record has unsupported result.status '$status'" \
                "python3 tools/ci/verify_evidence.py '$record'"
            ;;
    esac
done

if (( ${#PASSED_RECORDS[@]} == 0 )); then
    ci::not_verified "${#NOT_VERIFIED_RECORDS[@]} committed record(s) are valid but report result=not_verified; GPU is NOT VERIFIED (Vulkan backend not yet implemented, #57)"
fi

ci::record passed "validated ${#PASSED_RECORDS[@]} physical-GPU evidence record(s) from tools/ci/run-gpu-lane.sh" \
    --evidence "${PASSED_RECORDS[@]}"
echo "LANE $CI_LANE / $CI_JOB: PASSED — validated physical-GPU evidence"
exit 0
