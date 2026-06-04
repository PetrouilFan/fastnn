# Prepared fallback overhead benchmark

`scripts/prepared_fallback_overhead.py` compares default AOT `forward()` against prepared-plan opt-in fallback paths on the same YOLO input.

Run:

```bash
.venv/bin/python scripts/prepared_fallback_overhead.py --onnx yolov8n.onnx --warmup 3 --iters 20 --json /tmp/prepared_fallback_overhead.json
```

Local result after `d55d5e3`:

```text
forward:                         mean=54.687 ms median=54.388 ms overhead=+0.000 ms (+0.00%) max_abs=0
forward_prepared_fallback:       mean=60.221 ms median=60.185 ms overhead=+5.534 ms (+10.12%) max_abs=0
forward_prepared_arena_fallback: mean=54.917 ms median=54.650 ms overhead=+0.229 ms (+0.42%) max_abs=0
```

Interpretation:

- The plain prepared fallback adds meaningful overhead because it validates then dispatches the original plan with all `WriteConst` work intact.
- The arena fallback that preloads constants and skips redundant Conv weight writes is roughly neutral in this sample.
- The rejected transposed Conv runtime spike was therefore not blocked solely by prepared fallback scaffolding overhead; its transposed-weight kernel/rewrite path was not enough to beat the default Conv path on full YOLO.

Next useful measurement:

- Add lower-level phase timing only if another runtime specialization gets close to beating default `forward()`.
- For now, use this script as the standard before/after check for prepared fallback overhead.
