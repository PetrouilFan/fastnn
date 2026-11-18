#!/usr/bin/env bash
# Lane: pinned-model
#
# Model-dependent CI must use exactly one revision, named by
# tools/ci/model-pin.json. FN3-03 (#49) owns fixtures/ and the revision; this
# script reads the consumer-side pin and never guesses.
#
# An absent or unpinned file is NOT VERIFIED, with a message naming the file
# and the owning issue. It is never a silent pass. The run itself is offline:
# HF_HUB_OFFLINE and CARGO_NET_OFFLINE are set so CI cannot download a model.
set -uo pipefail

CI_LANE=pinned-model
CI_JOB=pinned-model
source "$(dirname "${BASH_SOURCE[0]}")/lane-lib.sh"

PIN="$CI_ROOT/tools/ci/model-pin.json"

ci::describe pinned-model

echo
echo "== Read the single consumer-side model pin =="
echo "\$ python3 tools/ci/model_pin.py check"
PIN_MESSAGE="$(python3 "$CI_ROOT/tools/ci/model_pin.py" check 2>&1)"
PIN_STATUS=$?
echo "$PIN_MESSAGE"

case "$PIN_STATUS" in
    0)
        ;;
    69)
        ci::not_verified "model-dependent lane skipped: $PIN_MESSAGE"
        ;;
    *)
        ci::failed "the model pin is invalid: $PIN_MESSAGE" "python3 tools/ci/model_pin.py check"
        ;;
esac

E2E_TEST=""
for candidate in tests/fn3_e2e.rs tests/fn3_e2e/main.rs; do
    if [[ -f "$CI_ROOT/$candidate" ]]; then
        E2E_TEST="$candidate"
        break
    fi
done
if [[ -z "$E2E_TEST" ]]; then
    ci::not_verified "the pin is $PIN_MESSAGE but the end-to-end fixture test does not exist yet (FN3-16 / #62): tests/fn3_e2e.rs or tests/fn3_e2e/main.rs"
fi

ci::run "Pinned end-to-end test (offline)" \
    "cd '$CI_ROOT' && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CARGO_NET_OFFLINE=true cargo test --test fn3_e2e -- --test-threads=1"

ci::passed "end-to-end test ran offline against the revision pinned in tools/ci/model-pin.json"
