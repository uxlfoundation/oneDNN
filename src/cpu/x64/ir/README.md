# IR-Based JIT Kernel Generation for x64

This directory holds an intermediate representation (IR) for x64 JIT CPU kernels
and the shared passes that lower it to machine code. A kernel is written as
target-neutral operations over virtual registers. The shared pipeline allocates
registers and an ISA-specific emitter lowers the IR to instructions. Kernel code
carries no manual register management and no data-type or ISA branching.

IR-based kernels coexist with the existing Xbyak generators and are enabled
incrementally. It is not an optimizing compiler, has no IR-to-IR passes, and
does not target non-x64 backends.

## Architecture

The kernel is a class derived from `jit_generator_t`. It knows the concrete ISA
and data types and can use them directly, for example to set up post-op
callbacks. Its `generate()` calls the stages below in order. Only the *builder* stage is
kernel-specific. Everything else is shared.

```
generate()
|
|   kernel-specific
+-> +-------------+
|   | Builder     |  emits target-neutral IR: param loads, loops, math
|   +-------------+
|
|   shared
+-> +-------------+
|   | Reg config  |  builds ISA-agnostic register pools
|   +-------------+
|
+-> +-------------+
|   | Reg alloc   |  liveness and linear scan over gpr/vec/mask
|   +-------------+
|
+-> +-------------+
|   | Emitter     |  lowers IR to instructions, one backend per ISA
|   +-------------+
|
+-> +-------------+
    | Static data |  mask and post-op tables, after the postamble
    +-------------+
```

The ABI preamble, the spill-frame reservation, and the postamble wrap the emit
step. They are omitted above for clarity.

### Components

* **Builder.** The only kernel-specific part. It loads parameters, builds the
  loop nest, and expresses the math in target-neutral operations. It is data-type
  agnostic (the type is only a tag on vector values) and ISA agnostic (masks,
  vector, and general-purpose registers are abstract).
* **IR.** A linear list of operations over *virtual registers*. One fixed
  struct represents every operation, and an `op` kind selects which fields it
  uses. Every virtual register has a kind (`gpr`, `vec`, or `mask`), and a `vec`
  register also carries the element data type of the values it holds.
* **Register configuration.** An ISA-aware step that produces ISA-agnostic
  register pools (integer indices per register kind) plus the reserved registers:
  the stack pointer, the kernel-argument pointer, and a few scratch registers the
  emitter uses for spilled values. It encodes per-ISA facts such as register
  counts and whether the target has dedicated mask (k) registers.
* **Register allocator.** Maps unlimited virtual registers onto physical ones,
  spilling to the stack under pressure. It knows only register kinds and control
  flow. Liveness analysis (which values are still needed at each operation) is
  backward data-flow over the IR's trivial control-flow graph, iterated to a fixed
  point so loop back-edges propagate. Linear scan then
  reduces each value to a single live interval and spills to the stack when the
  active set outgrows the register file.
* **Emitter.** The only part aware of the ISA and data types, because it produces
  the code. It walks the allocated IR once and lowers each operation to
  instructions using the physical registers the allocator chose. Spilled values
  are loaded into scratch registers around each use. There is one backend per ISA
  family (for example, AVX2\* and AVX-512\*), and a dispatch step selects the
  matching backend.
* **Static data.** Some lowerings need constants, such as the AVX2 mask tables,
  that must live after the ABI postamble, though the references to them are
  emitted during lowering. The emitter fills a data-section structure with the
  bytes and an unresolved label. After the postamble, the bytes are written and
  the labels are bound.
* **Post-ops injector.** The existing Xbyak post-ops injector is reused unchanged,
  plugged in as a single `inject_postops` operation. Its variable-length arguments
  live in a side table indexed from the operation's immediate field, so the
  operation itself carries only that index. For binary post-ops the table also
  holds each accumulator's output offset and active-element count, which the
  injector needs to address its right-hand-side argument. The injector saves and
  restores its own registers and does not participate in IR allocation.

### Control and Data Flow

The IR is produced in full, then consumed read-only by the allocator and the
emitter. `generate()` runs a fixed sequence: build the IR, build the register
configuration, allocate registers, emit the ABI preamble, reserve the spill
frame, emit the lowered code, tear down the frame, emit the postamble, and write
the static data. Each kernel assembles this sequence in its own `generate()`
today. A shared runner for the fixed part is a follow-up.

## Design Principles

* **The whole kernel exists before lowering.** This is what distinguishes the
  approach from single-pass generation and what makes global register allocation
  possible.
* **Flat, fixed-struct IR.** Instructions do not nest, and one struct fits every
  operation. This keeps the builder, allocator, and emitter simple and avoids any
  rewrite-pass machinery.
* **No IR-to-IR optimization passes.** The developer writes the structural
  optimizations (blocking, unrolling, loop order), which are emitted as written.
  The IR only takes over register allocation.
* **Mutable virtual registers.** A virtual register is a named value that may be
  written more than once (for example, a pointer advanced each
  iteration), and it occupies a single physical location for its entire live
  range. The cost is that live-range splitting is ruled out.
* **Single-register addressing.** A memory operand is a base register plus a
  build-time constant displacement, with no index and no scale. Any distance known
  only at run time is folded into the base pointer with an explicit `add`. Every
  memory operation then reads at most one pointer, which lowers register pressure
  and simplifies spilling.
* **Structured loops.** Loops are explicit nodes with a runtime counter.
* **Separation of concerns is the invariant to protect.** The builder is
  target-neutral, ISA and data-type knowledge lives only in the emitter and the
  register configuration, and the allocator knows only kinds and control flow.
* **`def_use()` must match the emitter.** Liveness is computed from the reads and
  writes reported by `def_use()`. If a lowering reads or writes a virtual register
  that `def_use()` does not report, allocation is wrong, so the two must stay in
  sync.

## Directory Structure

* `ir.hpp`, `ir.cpp`: the IR itself, that is, the operation kinds, virtual
  registers, the builder helpers, `def_use()`, and the loop-emission helpers.
* `reg_config.hpp`, `reg_config.cpp`: builds the per-ISA register configuration,
  that is, the allocatable pools plus the reserved and scratch registers.
* `reg_alloc.hpp`, `reg_alloc.cpp`: liveness analysis and the linear-scan
  allocator, producing the assignment for each virtual register and the size of
  the spill frame.
* `emitter/emitter.hpp`, `emitter/emitter.cpp`: the lowering pass, which walks the
  allocated IR, dispatches by ISA family, and manages the static-data section.
* `emitter/backend_avx2.hpp`: the AVX2-family backend, the reference example of
  per-operation instruction selection.
* `postops_injector.hpp`, `postops_injector.cpp`: the driver for the JIT post-ops
  injector, which lowers the `inject_postops` operation.

The kernel-specific builders live outside this directory. For example,
`src/cpu/x64/brgemm/brgemv_ir.{hpp,cpp}` holds the GEMV builder and shows how
`generate()` runs the full pipeline.

## Developer Guidelines

New functionality maps to a specific layer. The IR infrastructure grows by adding
data types, ISAs, and the fused instructions each ISA supports, and each kind of
change belongs in one place.

1. **A new data type** is a tag on a `vec` register. The builder stays the same,
   and the emitter maps the operation and data type to the right instruction.
2. **A new ISA** adds a new emitter backend or extends an existing one, along with
   the matching register-configuration facts.
3. **A new variant of an existing instruction** is a lowering rule in the emitter,
   not a new IR operation, and is invisible to the builder. For example,
   `vload_masked` already lowers to `vmovss`, `vmovups`, or `vmaskmovps` depending
   on how many elements are active.
4. **A new behavior** that no existing operation can express becomes a new IR
   operation, with its own definition, `def_use()` entry, and lowering.

To tell the third from the fourth, look at `def_use()`. If the reads and writes
stay the same and only the emitted instructions differ (by data type, element
count, or ISA), it is a lowering rule. If the behavior needs different reads or
writes, it is a new operation. This keeps the set of operations target-neutral.

Additional rules to follow.

* Keep the builder target-neutral. No ISA or data-type conditionals in builder
  code, since that is exactly the branching this design removes.
* Update `def_use()` with every change to an operation so liveness stays correct.
* Confine ISA and data-type knowledge to the emitter and the register
  configuration.
* Prefer a running pointer over a computed index, to respect single-register
  addressing.
* The public entry points in these headers are exported for unit testing.

## Tests

The IR, allocator, and emitter have dedicated unit tests
(`test_internals_cpu_ir`). IR-based kernels are also tested through benchdnn.

## References

* Design document (RFC): *IR-Based JIT Kernel Generation for x64 CPUs*, at
  https://github.com/uxlfoundation/oneDNN/pull/5460
  (`rfcs/20260630-ir-x64/README.md`). It covers the motivation, goals, and
  design rationale.
