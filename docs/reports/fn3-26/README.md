# FN3-26 measurement artifacts

Every file here was produced by `tests/fn3_vulkan_perf.rs` on the physical
GTX 1080 Ti (UUID `471b2bda-51e1-6451-e692-ff5d1e146f1e`), 3 warm-up iterations
followed by 20 measured iterations of: new session → prefill 10 tokens
(`464,2068,7586,21831,18045,625,262,16931,3290,13`) → read logits → decode
`640,11,612` (each with its logits read).

| File | Contents |
|---|---|
| `base.json` | `672cec9` matched result (the alpha) |
| `step1.json` | `ea8cd70` persistent weights |
| `step2.json` | `0a8077e` device-resident intermediates and KV |
| `step3.json` | `7ec666f` one submission per resident step |
| `*.csv` | raw per-sample wall times (`sample,kind,ms`) |
| `*-per-op.txt` | per-op instructions, upload bytes, readback bytes, host µs, device µs per iteration |
| `*-measured.log` | the whole measurement window: lock acquisition, load average, `pgrep -a 'cargo|rustc'` snapshots before and after, the release build, and the run |

Reproduce one arm (the arm's tree must contain the harness from this directory):

```bash
flock -w 1800 /tmp/opencode/fn3-measure.lock -c '
  while pgrep -f "cargo|rustc" >/dev/null; do sleep 5; done
  cd <arm-tree> && FN3_PERF=1 \
    FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
    FN3_VK_REQUIRE_PHYSICAL=1 \
    FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx \
    CARGO_BUILD_JOBS=5 cargo test --release --features vulkan \
      --test fn3_vulkan_perf -- --nocapture --test-threads=1
'
```

`tests/fn3_vulkan_perf.rs` uses only APIs that exist at `672cec9`, so the base
and every step run identical measurement code. The source tree of each arm is
`git archive <sha> | tar -x -C <dir>` with this harness copied into `tests/`.
