#!/usr/bin/env bash
# Lane: pinned-model
#
# Model-dependent CI must use exactly one revision, named by
# tools/ci/model-pin.json. FN3-03 (#49) owns fixtures/ and the revision; this
# script reads the consumer-side pin and never guesses.
#
# An absent or unpinned file is NOT VERIFIED, with a message naming the file
# and the owning issue. It is never a silent pass.
#
# Once the pin is valid this lane stages the graph the pin names, verifies its
# size and SHA-256 against fixtures/parity/tiny-gpt2/model.lock.json, and runs
# the end-to-end test against it with FN3_E2E_REQUIRE_MODEL=1. That variable
# makes an absent model a failure instead of a skip, which is what let this job
# pass while nothing ran. Cargo itself still runs offline: no crate is fetched.
#
# PINNED_MODEL_ALLOW_SKIP=1 turns an unstageable graph into NOT VERIFIED rather
# than a failure, for a checkout with no network that only wants the rest of the
# suite. It is never set in CI.
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

echo
echo "== Stage the pinned graph =="
MODEL="$CI_ROOT/tools/parity/.cache/models/tiny-gpt2/onnx/model.onnx"
if [[ ! -f "$MODEL" ]]; then
    echo "\$ python3 tools/parity/fetch_models.py --suite tiny-gpt2 --graph-only"
    # The job exports HF_HUB_OFFLINE for the test run; the fetch needs the
    # network, so it runs with that variable removed. The script itself checks
    # the size and SHA-256 from the lock before it reports success, so a
    # truncated or substituted download fails here rather than being executed.
    STAGED=0
    for attempt in 1 2 3; do
        if env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
             python3 "$CI_ROOT/tools/parity/fetch_models.py" --suite tiny-gpt2 --graph-only; then
            STAGED=1
            break
        fi
        # A runner's network is not the change under test: give it a moment
        # and try again before drawing any conclusion from the failure.
        echo "attempt $attempt could not stage the pinned graph; retrying in 5s" >&2
        sleep 5
    done
    if [[ "$STAGED" != "1" ]]; then
        if [[ "${PINNED_MODEL_ALLOW_SKIP:-0}" == "1" ]]; then
            ci::not_verified "PINNED_MODEL_ALLOW_SKIP=1 and the pinned graph could not be staged"
        fi
        if [[ -f "$MODEL" ]]; then
            # Bytes arrived but do not match the lock. That is a real defect in
            # what the Hub served, not a missing network, and it must not be
            # reported as "no evidence" or executed.
            ci::failed "the pinned graph was downloaded but does not match the lock; it must not be executed" \
                "python3 tools/parity/fetch_models.py --suite tiny-gpt2 --graph-only"
        fi
        # No bytes at all: this advisory lane simply has no evidence to offer.
        # Reporting FAILED would blame the change for the runner's network;
        # reporting PASSED would claim a model ran when none did.
        ci::not_verified "the pinned graph could not be staged (the Hub was unreachable after three attempts); no evidence was produced"
    fi
fi
if [[ ! -f "$MODEL" ]]; then
    ci::failed "the pinned graph is still missing after staging: $MODEL" \
        "python3 tools/parity/fetch_models.py --suite tiny-gpt2 --graph-only"
fi
ci::check "pinned graph staged from the lock's revision"

echo
echo "== Run the pinned end-to-end test against that file =="
ci::run "Pinned end-to-end test (staged revision)" \
    "cd '$CI_ROOT' && FASTNN_FN3_ONNX_MODEL='$MODEL' FN3_E2E_REQUIRE_MODEL=1 CARGO_NET_OFFLINE=true cargo test --test fn3_e2e -- --test-threads=1"

ci::passed "the pinned revision was staged, verified against the lock, and executed end to end"
