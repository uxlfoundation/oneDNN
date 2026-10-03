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
  the spill frame. On request, the allocator also returns statistics for the
  debug output.
* `emitter/emitter.hpp`, `emitter/emitter.cpp`: the lowering pass, which walks the
  allocated IR, dispatches by ISA family, and manages the static-data section.
* `emitter/backend_avx2.hpp`, `emitter/backend_avx512.hpp`: the per-ISA-family
  backends, which do the per-operation instruction selection. They are peers,
  each self-contained.
* `postops_injector.hpp`, `postops_injector.cpp`: the driver for the JIT post-ops
  injector, which lowers the `inject_postops` operation.
* `dump.hpp`, `dump.cpp`: the debug output that `ONEDNN_VERBOSE=x64ir=<level>`
  enables, that is, the kernel summary, the spill details, and the IR dump (see
  Debug Output).

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
number of operations, numbers from the register allocation, such as the number
of spilled virtual registers, and numbers from the generated code, such as its
size.

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
* **Finding spills.** The summary shows how many virtual registers of each
  register file the allocator keeps on the stack instead of in a register.
  Level 2 shows which virtual registers these are and why the register file
  spills. Level 3 shows the physical register or the stack slot of each virtual
  register in each operation.
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
alloc gpr: pool 12, vregs 8, peak 7 at op 15, spilled 0, scratch 2 unused
alloc vec+mask: pool 13, vregs 11, peak 11 at op 21, spilled 0, scratch 3 unused
alloc: frame 0 bytes, liveness passes 3
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
* `alloc gpr:` and `alloc vec+mask:` describe the register files, one line per
  file. A register file is the set of physical registers that the allocator
  assigns to one or more kinds of virtual registers. The name of the file lists
  these kinds. On AVX2, a mask is a vector register, so masks and vecs share
  the file `vec+mask`. On AVX-512, the files are `gpr`, `vec`, and `mask`.
* `pool` is the number of physical registers that the allocator can assign in
  the file. For example, the gpr pool does not include the stack pointer, the
  register that holds the kernel arguments, and the scratch registers.
* `vregs` is the number of virtual registers that use the file.
* `peak` is the largest number of virtual registers that need a register of the
  file at the same operation. The index of the first such operation follows. A
  virtual register needs a register at an operation when the operation reads
  or writes it, or when a later operation reads its current value.
* `spilled` is the number of virtual registers that the allocator keeps on the
  stack. The emitter loads such a value into a scratch register before each
  read, and it stores the value back after each write.
* `scratch` is the number of scratch registers that the emitter reserves in the
  file. Only spill code uses them. `unused` means that the file has no spilled
  virtual register, so the generated code does not use them.
* `alloc: frame` gives the size of the stack area for spilled values in bytes.
  It also gives the number of passes that the liveness analysis needed. The
  analysis repeats until a pass changes nothing, so loops need more passes.
* `end x64ir #1` ends the output for the kernel.

### Level 2: Details

Level 2 adds the details of the spills. For each register file that has a
spilled virtual register, it prints a `hint` line and one `spill` line for each
spilled virtual register. For a kernel without spills, level 2 prints the same
output as level 1.

On an AVX-512 machine, the following command generates a BRGEMM kernel that
spills one gpr:

```
ONEDNN_VERBOSE=x64ir=2 ./benchdnn --brgemm --bia_dt=f32 --bs=16 16x64:64x32
```

The command prints the following:

```
begin x64ir #1 jit_brgemm_ir_kernel_t isa=avx512_core
ir: 193 ops, 34 vregs (gpr 13, vec 21, mask 0), 3 loops, nesting depth 3, 3 branches
code: 1330 bytes (instructions 1330, static data 0)
alloc gpr: pool 12, vregs 13, peak 13 at op 36, spilled 1, scratch 2
alloc vec: pool 29, vregs 21, peak 19 at op 39, spilled 0, scratch 3 unused
alloc mask: pool 7, vregs 0, peak 0, spilled 0
alloc: frame 16 bytes, liveness passes 3
hint gpr: peak 13 at op 36 > pool 12: the pool is too small for the live vregs
spill r1@[rsp+0]: weight 11, interval 1..192, refs 1 at depth 0, 1 at depth 1
end x64ir #1
```

The new lines mean the following:

* `hint gpr:` tells why the register file spills. The allocator gives each
  virtual register one interval, from the first operation that needs it to the
  last. The allocator spills only when more intervals overlap at one operation
  than the pool has registers. The hint compares this with the peak.
* When the peak is larger than the pool, as in this example, more virtual
  registers need a register at one operation than the pool has. Any allocation
  with this pool spills. The hint says that the pool is too small for the live
  virtual registers.
* When the peak fits in the pool, the hint shows the largest overlap instead,
  for example, `overlap 14 at op 30 > pool 13, peak 12 <= pool`. Then some
  intervals contain a dead gap. In a dead gap, a virtual register holds no value
  that a later operation reads, but it keeps its register. The builder can
  remove a dead gap with a new virtual register for the value after the gap.
* `spill r1@[rsp+0]` names a spilled virtual register and its stack slot. The
  slot is a byte offset from `rsp`. When intervals overlap, the allocator
  spills the virtual register with the lowest weight.
* `weight` is the spill weight. Each read and each write of the virtual
  register adds 1 outside loops, 10 inside one loop, 100 inside two loops, and
  so on. `refs` gives the number of reads and writes at each loop depth. In
  this example, 1 at depth 0 and 1 at depth 1 give the weight 1 + 10 = 11.
* `interval` gives the first and the last operation of the interval.

### Level 3: IR Dump

Level 3 adds the IR dump after the lines of level 2. The IR dump has one line
per operation. The example below shows a part of the output for the command in
Enabling the Output. The lines marked `...` are left out.

```
begin x64ir #1 jit_brgemv_ir_kernel_t isa=avx2
ir: 66 ops, 19 vregs (gpr 8, vec 11, mask 0), 2 loops, nesting depth 2, 0 branches
code: 492 bytes (instructions 492, static data 0)
alloc gpr: pool 12, vregs 8, peak 7 at op 15, spilled 0, scratch 2 unused
alloc vec+mask: pool 13, vregs 11, peak 11 at op 21, spilled 0, scratch 3 unused
alloc: frame 0 bytes, liveness passes 3
index | gpr vec+mask | operation
    0 |   1        1 | load r0@rax, [param+24]
    1 |   2        1 | load r1@rcx, [param+16]
    2 |   3        1 | mov_imm r2@rdx, 0
    3 |   4        1 | loop r3@rbx = 2 {
    4 |   4        2 |   vzero f32:v4@ymm1
...
   17 |   7        9 |   loop r15@rbp = 32 {
   18 |   7        9 |     prefetch [r14@r8+512]
   19 |   7       10 |     vload f32:v16@ymm9, f32:[r14@r8+0]
   20 |   7       10 |     prefetch [r13@rsi+512]
   21 |   7       11 |     vload f32:v17@ymm10, f32:[r13@rsi+0]
   22 |   7       11 |     vdot f32:v4@ymm1, f32:v17@ymm10, f32:v16@ymm9
...
   46 |   7        9 |   } // r15@rbp -= 1, repeat while > 0
   47 |   4        9 |   vhreduce f32:v4@ymm1, f32:v18@ymm0
...
   55 |   4        9 |   vstore_scalar f32:[r0@rax+0], f32:v4@ymm1
...
   65 |   4        1 | } // r3@rbx -= 1, repeat while > 0
end x64ir #1
```

The first line of the IR dump names the columns. Each of the other lines has
three columns:

* The first column is the operation index.
* The second column is the register pressure. For each register file, it gives
  the number of virtual registers that need a register of the file at the
  operation. The largest number in a column is the `peak` of the summary.
* The third column is the operation. An operation inside a loop is indented by
  two spaces for each loop around it.

An operation prints as its name, then its operands. The name is the
`op_kind_t` name, for example, `vload` or `vdot`. The destination operand
comes first, as in Intel-syntax x64 assembly. Operands print as follows:

* `r<id>` and `m<id>` are virtual registers of kind gpr and mask. `<id>` is
  the virtual register id.
* `<dt>:v<id>` is a virtual register of kind vec that holds the data type
  `<dt>`, for example, `f32:v4`.
* The ids are unique across all three kinds, so `v5` is the virtual register
  with id 5. A debugger shows the same number for the `vreg_t` value.
* `@` follows each virtual register and gives its location. The location is
  the physical register, for example, `r0@rax` or `f32:v4@ymm1`. The register
  names are the names that the emitter uses. On AVX2, vecs and masks are in
  `ymm` registers. On AVX-512, vecs are in `zmm` registers and masks are in `k`
  registers. For a spilled virtual register, the location is its stack slot,
  for example, `r1@[rsp+0]`.
* `[r<id>+<disp>]` is the memory at the address in `r<id>` plus the byte
  offset `<disp>`. The offset is a decimal number.
* `[param+<disp>]` is a field of the kernel argument struct at the byte offset
  `<disp>`.
* A vector load or store puts the data type in memory before the memory
  operand, for example, `f32:[r13@rsi+0]`. A load or store that converts
  between data types shows two different types. For example, `vload
  f32:v3@ymm2, bf16:[r0@rax+0]` reads bf16 values and converts them to f32.
* `L<id>` is an IR label, the target of `jmp` and `jz`. Loops do not use IR
  labels. The emitter creates the labels for loops during lowering.

Two kinds of operations print in a special form:

* A loop prints as two lines. `loop r3@rbx = 2 {` is the `loop_begin`
  operation. `r3` is the loop counter, and `2` is the number of iterations. The
  number of iterations can also come from a register, for example, `loop
  r4@rdx = r1@rcx {`. The line `} // r3@rbx -= 1, repeat while > 0` is the
  `loop_end` operation.
* `L0:` is the `label` operation for the label `L0`. `jz r1@rcx, L0` jumps to
  `L0` when `r1` is zero.

## References

* Design document (RFC): *IR-Based JIT Kernel Generation for x64 CPUs*, at
  https://github.com/uxlfoundation/oneDNN/pull/5460
  (`rfcs/20260630-ir-x64/README.md`). It covers the motivation, goals, and
  design rationale.
