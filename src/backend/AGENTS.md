# Backend guide

The backend translates compiled IR plus a memory plan into executable
instructions and dispatches them. The authoritative AOT pass order is
`GraphExecutor::compile_with_weight_dtype` in `src/backend/executor.rs`.

## Ownership

- `mod.rs`: backend trait, executable instruction/plan contracts, and shared
  backend types.
- `executor.rs`: compilation orchestration, dynamic-shape handling, and plan
  caching.
- `prepared.rs`: prepared-plan construction/execution and persistent weights.
- `runtime.rs`: loading and running persisted executable plans.
- `cpu/` and `wgpu/`: backend-specific execution.

## Invariants

- `ExecutablePlan`, `MemoryPlan`, instruction semantics, kernel names, and
  `.fnnc` serialization may be redesigned when doing so creates a cleaner
  backend boundary. Preserve only deliberate current behavior with tests.
- Lowering and dispatch are separate responsibilities. Keep backend-specific
  instruction routing out of compiler passes.
- Dynamic shape resolution must use `ShapeEnv` consistently with memory-plan
  tightening.
- Prepared execution must preserve persistent-weight and arena-lifetime
  assumptions.

For CPU safety and kernel rules, read `cpu/AGENTS.md`. For pass ordering, read
`../compiler/passes/AGENTS.md`. Validate backend moves with the roadmap gates,
including prepared-plan and allocator coverage where applicable.