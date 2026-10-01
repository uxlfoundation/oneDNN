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
  register also carries the element data type it holds. A vector load or store
  carries a second data type, `mem_dt`, for what memory holds. The two differ
  when the access converts.
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
  operation itself carries only that index. The table also holds each
  accumulator's output offset, which a binary post-op needs to address its
  right-hand-side argument and a sum post-op needs to read the previous
  destination value. The injector saves and restores its own registers and does
  not participate in IR allocation.

### Control and Data Flow

The IR is produced in full, then consumed read-only by the allocator and the
emitter. `generate()` runs a fixed sequence: build the IR, build the register
configuration, allocate registers, emit the ABI preamble, reserve the spill
frame, emit the lowered code, tear down the frame, emit the postamble, write
the static data, and print the debug output with `ir::print_kernel_dump()`
(see Debug Output). Each kernel assembles this sequence in its own
`generate()` today. A shared runner for the fixed part is a follow-up.

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
* `emitter/backend_avx2.hpp`, `emitter/backend_avx512.hpp`: the per-ISA-family
  backends, which do the per-operation instruction selection. They are peers,
  each self-contained.
* `postops_injector.hpp`, `postops_injector.cpp`: the driver for the JIT post-ops
  injector, which lowers the `inject_postops` operation.
* `dump.hpp`, `dump.cpp`: the debug output that `ONEDNN_VERBOSE=x64ir=<level>`
  enables, that is, the kernel summary and the IR dump (see Debug Output).

The kernel-specific builders live outside this directory. For example,
`src/cpu/x64/brgemm/brgemv_ir.{hpp,cpp}` holds the GEMV builder and shows how
`generate()` runs the full pipeline.

## Developer Guidelines

New functionality maps to a specific layer. The IR infrastructure grows by adding
data types, ISAs, and the fused instructions each ISA supports, and each kind of
change belongs in one place.

1. **A new data type** is a tag on a `vec` register, plus the `mem_dt` on the
   loads and stores that reach it. The builder stays the same, and the emitter
   maps the operation and the data types to the right instruction.
2. **A new ISA** adds a new emitter backend or extends an existing one, along with
   the matching register-configuration facts.
3. **A new variant of an existing instruction** is a lowering rule in the emitter,
   not a new IR operation, and is invisible to the builder. For example,
   `vload_masked` already lowers to `vmaskmovps` on AVX2 and to `vmovups` under
   an EVEX write mask on AVX-512.
4. **A new behavior** that no existing operation can express becomes a new IR
   operation, with its own definition, `def_use()` entry, and lowering.

When adding new operations conform to the following rules:

* **Keep parameters out of the name.** An operation takes its data types as
  parameters, so the name must not mention a data type. There is one `vdot`,
  not `vdot_f32` and `vdot_bf16`, and the emitter selects `vfmadd231ps` or
  `vdpbf16ps` from the operand's `dt`. The same applies to the ISA. If the
  operation cannot be named without naming a data type, it is a lowering rule
  for an existing operation, not a new operation.
* **Decide how many operations the family needs.** Related forms can be one
  operation whose parameters select the form, or one operation per form. A
  single load taking an element count and an optional mask would cover the
  full-vector, single-element, and masked cases, and it would work. The cost
  is that the meaning of a call depends on the combination of arguments, and
  some arguments are unused in some combinations. This IR uses separate
  operations: `vload`, `vload_scalar`, `vload_bcast`, and `vload_masked`. Each
  signature takes only the arguments its form needs. Both options are valid.
  Prefer separate operations, because they are harder to call incorrectly.
* **Check that the operation works for other data types and ISAs.** A new
  operation usually has one lowering at first, and it is tempting to name it
  after that instruction. `fma` would have been a reasonable name while the
  only lowering was `vfmadd231ps`. That name does not work for bf16, where
  `vdpbf16ps` adds two products per f32 lane, or for int8, where VNNI adds
  four. The operation is named `vdot` because all of these compute a dot
  product, and the length depends on the data type. Before adding an
  operation, look up the instruction you would emit for every ISA and data
  type you expect to support, and define the operation at the level where they
  agree. If they do not agree at any level, these are two operations.

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

## Debug Output

The IR pipeline can print a text description of each kernel that it
generates. The description includes the IR of the kernel. The IR shows the
kernel in the same form as the builder code: the operations, the virtual
registers that they use, and the loops around them. The description also
includes a short summary. The summary gives numbers from the IR, such as the
number of operations, and numbers from the generated code, such as its size.

The description helps a developer to check a kernel without a debugger. The
builder creates the IR in C++ code, so the IR is not visible anywhere else.
`ONEDNN_JIT_DUMP` shows only the final machine code. In machine code, a loop is
a label, a decrement, and a conditional jump. In the IR, it is a pair of
`loop_begin` and `loop_end` operations. The description is useful in these
cases:

* **Checking a builder.** The IR dump shows the loops with their counters and
  iteration counts, the pointer increments (`add_imm`), the memory offsets, and
  the data types that the builder produced. A developer can compare them with
  what the builder code is supposed to produce.
* **Comparing two versions.** The output does not change from run to run. A
  `diff` of the output before and after a change shows how the change affected
  each kernel.

### Enabling the Output

The output can be enabled only in dev-mode builds (`ONEDNN_DEV_MODE=ON`). Set
`ONEDNN_VERBOSE` to `x64ir=<level>` to enable it:

```
ONEDNN_VERBOSE=x64ir=1 ./benchdnn --matmul --dt=f32 64x256:256x1
```

The `x64ir` token works together with other `ONEDNN_VERBOSE` tokens, for
example, `ONEDNN_VERBOSE=dispatch,x64ir=1`. `ONEDNN_VERBOSE=all` and
`debuginfo=` do not enable it. `all` enables the standard verbose output, and
backend-specific dumps should not mix with it.

The output is printed only when a kernel is generated. A primitive cache hit
generates no kernel, so it prints nothing.

The level selects how much the output shows. A higher level prints everything
that a lower level prints, and it adds more. The output for each kernel starts
with a `begin` line and ends with an `end` line. The sections below describe
each level.

### Level 1: Summary

Level 1 prints a short summary for each generated kernel. The command from the
previous section generates one IR kernel and prints the following:

```
begin x64ir #1 jit_brgemv_ir_kernel_t isa=avx2
ir: 66 ops, 19 vregs (gpr 8, vec 11, mask 0), 2 loops, nesting depth 2, 0 branches
code: 492 bytes (instructions 492, static data 0)
end x64ir #1
```

The lines mean the following:

* `begin x64ir #1` starts the output for one kernel. `#1` is the number of the
  kernel. Kernels are numbered in the order in which they are printed. The
  kernel name and the ISA follow.
* `ir:` gives numbers from the IR. It gives the number of operations and
  virtual registers. The numbers in parentheses split the virtual registers by
  kind. It also gives the number of loops, the nesting depth, and the number of
  branches (`jz` and `jmp`).
* The nesting depth is the largest number of loops around one operation. Code
  without loops has depth 0. One loop gives depth 1. A loop inside another loop
  gives depth 2.
* `code:` gives numbers from the generated code. The first number is the size
  of the whole kernel in bytes. It is the same size that `ONEDNN_JIT_DUMP`
  writes. The instructions are the ABI preamble and postamble, the spill-frame
  setup, and the lowered operations. The static data is the constants after
  the postamble, such as mask tables and post-op tables. It includes the
  padding that aligns the constants.
* `end x64ir #1` ends the output for the kernel.

### Level 2: Details

Level 2 prints the same output as level 1 for now.

### Level 3: IR Dump

Level 3 adds the IR dump after the summary. The IR dump has one line per
operation. The example below shows a part of the output for the same command.
The lines marked `...` are left out.

```
begin x64ir #1 jit_brgemv_ir_kernel_t isa=avx2
ir: 66 ops, 19 vregs (gpr 8, vec 11, mask 0), 2 loops, nesting depth 2, 0 branches
code: 492 bytes (instructions 492, static data 0)
    0 | load r0, [param+24]
    1 | load r1, [param+16]
    2 | mov_imm r2, 0
    3 | loop r3 = 2 {
    4 |   vzero f32:v4
...
   17 |   loop r15 = 32 {
   18 |     prefetch [r14+512]
   19 |     vload f32:v16, f32:[r14+0]
   20 |     prefetch [r13+512]
   21 |     vload f32:v17, f32:[r13+0]
   22 |     vdot f32:v4, f32:v17, f32:v16
...
   46 |   } // r15 -= 1, repeat while > 0
   47 |   vhreduce f32:v4, f32:v18
...
   55 |   vstore_scalar f32:[r0+0], f32:v4
...
   65 | } // r3 -= 1, repeat while > 0
end x64ir #1
```

Each line starts with the operation index. An operation inside a loop is
indented by two spaces for each loop around it.

An operation prints as its name, then its operands. The name is the
`op_kind_t` name, for example, `vload` or `vdot`. The destination operand
comes first, as in Intel-syntax x64 assembly. Operands print as follows:

* `r<id>` and `m<id>` are virtual registers of kind gpr and mask. `<id>` is
  the virtual register id.
* `<dt>:v<id>` is a virtual register of kind vec that holds the data type
  `<dt>`, for example, `f32:v4`.
* The ids are unique across all three kinds, so `v5` is the virtual register
  with id 5. A debugger shows the same number for the `vreg_t` value.
* `[r<id>+<disp>]` is the memory at the address in `r<id>` plus the byte
  offset `<disp>`. The offset is a decimal number.
* `[param+<disp>]` is a field of the kernel argument struct at the byte offset
  `<disp>`.
* A vector load or store puts the data type in memory before the memory
  operand, for example, `f32:[r13+0]`. A load or store that converts between
  data types shows two different types. For example, `vload f32:v3,
  bf16:[r0+0]` reads bf16 values and converts them to f32.
* `L<id>` is an IR label, the target of `jmp` and `jz`. Loops do not use IR
  labels. The emitter creates the labels for loops during lowering.

Two kinds of operations print in a special form:

* A loop prints as two lines. `loop r3 = 2 {` is the `loop_begin` operation.
  `r3` is the loop counter, and `2` is the number of iterations. The number of
  iterations can also come from a register, for example, `loop r4 = r1 {`. The
  line `} // r3 -= 1, repeat while > 0` is the `loop_end` operation.
* `L0:` is the `label` operation for the label `L0`. `jz r1, L0` jumps to
  `L0` when `r1` is zero.

## References

* Design document (RFC): *IR-Based JIT Kernel Generation for x64 CPUs*, at
  https://github.com/uxlfoundation/oneDNN/pull/5460
  (`rfcs/20260630-ir-x64/README.md`). It covers the motivation, goals, and
  design rationale.
