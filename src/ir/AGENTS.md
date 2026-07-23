# IR guide

The IR is the graph contract consumed by compiler passes and backends. Its
current public surface includes `ComputeGraph`, `IRNode`, `Opcode`, graph type
metadata, `GraphBuilder`, and `GraphTensor`.

## Invariants

- Structural graph mutation must invalidate/rebuild the graph caches through
  the graph's mutation protocol.
- `GraphBuilder` is single-threaded construction state; `ComputeGraph` cache
  behavior must remain safe for its existing concurrent access model.
- Preserve node IDs, input/output ordering, multi-output semantics, symbolic
  shape behavior, and attribute encoding unless an explicit IR redesign changes
  them together with their consumers.
- Graph serialization may be redesigned or removed if the resulting graph
  ownership is clearer; cover the replacement behavior with tests.

## Boundaries

- Keep graph types, opcode definitions, graph storage/cache logic, builder
  operations, compilation/cache helpers, and builder tests as distinct
  ownership domains.
- Remove obsolete `ir::node` paths when the new layout is complete.
- `GraphBuilder` may construct backward graphs through autograd, but backend
  compilation belongs in explicit compile/cache ownership rather than general
  operation builders.

Any new operation requires a traced implementation path through IR, inference,
lowering, execution, and relevant import/training support. Use targeted IR,
shape, quantized-pipeline, and graph-executor tests.