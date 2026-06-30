# IR-Based JIT Kernel Generation for x64 CPUs

## Background

Every optimized x64 CPU kernel in oneDNN is written as a JIT generator, a C++ class that
emits machine instructions one by one through Xbyak. This is assembly
programming wrapped in C++. The model has worked, but as the number of data
types, ISAs, and fused post-ops grew, the cost of writing and maintaining
kernels this way grew with it. The recurring problems are the following.

* **Manual register management.** The author has to track, by reading the code,
  which physical registers are live at each point, assign them by hand, and
  decide where to spill and restore. This is time consuming and error prone, and
  it has to be redone whenever the kernel changes. The number of available
  vector and general-purpose registers differs across platforms, so the same
  logic has to be re-balanced per target.

* **Data-type branching.** Supporting a new data type means adding another set of
  conditionals throughout the kernel body.

* **ISA branching.** A single kernel often targets several ISAs, so it carries
  branches on the ISA and handles details such as AVX2 versus AVX-512 masking in
  the same code path. There is no ISA-isolated space. This is how bugs such as
  illegal-instruction escapes slip through. The code that should not run on a
  given ISA still sits on the same path.

* **Duplication.** The same building blocks (load and store, type conversion,
  broadcast, and others) are re-implemented across kernels. Some helpers exist,
  but they tend to be narrow. They fit the case they were written for and break
  down for a slightly different one. At some point, combining everything into one
  general solution becomes more confusing than repeating the code.

* **Hard to extend.** Adding GEMV to brgemm required fitting new microkernels
  into the existing loop-nest infrastructure. Isolating the GEMV path took
  significant effort and still raised the overall complexity of the brgemm
  kernel. Implementing GEMV as a separate kernel would instead have duplicated
  the loop nest and supporting code, and generalizing the loop-nest
  infrastructure to serve both cleanly is significant work in itself.

* **Hard to debug.** Failures such as an undefined `Label`, produced by a buggy
  generation path, are difficult to trace back to their cause.

This list is not exhaustive, but it captures the kinds of problems that recur
across the codebase.

## What Was Tried First

Register management is one of the largest of these problems, so it was the first
target. The idea was to replace the manual spiller (`reg64_savable_*` mechanism)
with one that allocates registers of every kind (vector, GPR, mask, tile) and
spills automatically. A prototype was built and enabled in brgemm.

The prototype ran into a fundamental limitation. JIT generators are single-pass.
As the generator walks the kernel and emits instructions, the allocator sees
register uses one at a time and has no view of how a register will be used
later. With only local information its decisions were poor, producing many
redundant spills and an overall worse allocation than the manual one.

Adding hints helped. Telling the allocator where loops begin and end, and giving
it lifetime hints, brought spill counts much closer to the manual result, within
5-10% for most configurations. But some still regressed by up to 40%, for
example pure-f32 brgemm on AVX-512. Those regressions occurred because spills
were chosen in a single pass without considering future register use. The result
of a single-pass allocator was still the same manual management, just hidden
behind another layer of abstraction. The developer was still deciding the
allocation by hand, only now using a different mechanism to express it. The main
problem did not go away. Allocation needs global information that a single pass
does not have, so manual control was still doing the real work.

That is the motivation for the change proposed here. Instead of trying to add
automatic allocation into single-pass generation, we introduce an intermediate
representation (IR) that is built in full before any code is emitted, so every
later pass has a complete view of the kernel.

## Goals

* Remove manual register management from kernel development. Allocation and
  spilling become automatic, driven by a global view of the kernel.
* Let a kernel be written once, free of data-type and ISA branches, and lowered
  to the target instructions by a separate pass.
* Reuse the allocation and lowering infrastructure across all kernels, so common
  building blocks are written once.
* Match the performance of the hand-written kernels (see the validation
  section).

## Non-Goals

* This is not an optimizing compiler. The IR has no transformation or
  optimization passes, and adding them is out of scope (see "On Optimization
  Passes").
* This does not replace existing kernels wholesale. It is an additional way of
  writing kernels that coexists with the current generators and is adopted
  incrementally. The infrastructure is expected to mature as more kernels are
  ported. For instance, GEMV could later be reimplemented on the IR across
  AVX-512 and more data types and dropped from brgemm.
* This does not target non-x64 CPU backends or commit to sharing the IR with
  them (see "Other CPU Vendors").

## Proposal

Introduce an IR-based kernel generation pipeline for x64. A kernel is described
by a *builder* that emits target-neutral IR operations instead of instructions.
The IR then runs through register allocation and is lowered to machine code by
an *emitter*. The IR layer knows nothing about any specific kernel, so the same
pipeline serves every builder.

The benefits come from having the whole kernel available before lowering.

1. Register allocation is automatic and uses a global view of the kernel rather
   than a local one.
2. The allocator and emitter are shared infrastructure, so kernels do not
   duplicate them, and improvements to allocation or lowering benefit every
   kernel at once.
3. The builder is data-type and ISA agnostic, which removes the branches that
   make today's kernels hard to write, read, and extend.

### Pipeline

`generate()` runs the following stages. The builder stage is kernel-specific.
Everything below it is shared.

```
 jit_generator_t            Xbyak-based generator base class
        |  inheritance (no Vmm template, no ISA template)
        v
 <kernel>_ir_t              IR-based kernel. generate() runs the pipeline.
        |                   Knows the concrete ISA and data types.
        v
 build IR                   builder stage: parameter loads, loop nests, math
        |
        v
 register config            ISA-agnostic pools (built by an ISA-aware step)
        |
        v
 allocate registers         liveness and linear scan over gpr/vec/mask (tile)
        |
        v
 preamble                   standard ABI preamble
        |
        v
 stack allocation           reserve the spill frame the allocator sized
        |
        v
 emit                       lower IR to machine code for the target ISA
        |
        v
 stack cleanup
        |
        v
 postamble                  standard ABI postamble
        |
        v
 static data               mask tables, post-op tables, and other constants
```

The current pipeline is manually assembled inside each kernel's `generate()`.
A follow-up will factor the fixed sequence (build, allocate, preamble, emit,
postamble, data) into a shared runner that builders plug into.

### The IR

The IR is a flat, linear list of instructions. Each instruction is one fixed
struct shared by all operations. An `op` kind defines the operation, and each
operation uses only the fields it needs (`dst`, two sources `s0`/`s1`, an
immediate `imm`, a memory address, and a few control-flow fields) and ignores
the rest. A fixed-size instruction keeps the builder, allocator, and emitter
simple.

Each instruction reads and writes *virtual registers*, the integer-named
placeholders that are used instead of physical registers until allocation. A virtual
register has one of a few kinds (`gpr`, `vec`, `mask`, and `tile` for AMX), which
fixes the kind of physical register it can later occupy. A `vec` register also
carries the element data type of the values it holds.

A `gpr` register carries no data type. Once a value is loaded into it, the
register holds raw bytes, and each operation interprets those bytes by its own
definition. The one place where interpretation matters is the load itself,
because that is where a narrower value is widened to the register width. The load
operation therefore takes a byte count and a signedness so the emitter can pick
sign or zero extension (for example `movsxd` versus `movzx`). The store operation
takes only a byte count, since truncation to a narrower width needs no such
choice.

Three design decisions keep the representation small and cheap to build and
analyze.

* **Mutable virtual registers.** A virtual register is a named value that may be
  written more than once. A pointer register, for example, is loaded once and
  then advanced with `add` every loop iteration under the same name. Each
  virtual register occupies one physical location for its whole live range, so a
  write overwrites that location in place. Reusing one name instead of creating a
  new name per write keeps the number of virtual registers small and lets the
  allocator rely only on liveness, the live interval of each name. The cost is
  that each name keeps a single home (one register, or one stack slot) for its
  whole life. A value cannot, for example, stay in a register while heavily used
  and then be evicted to free that register during a long gap before its next
  use. That technique, live-range splitting, is therefore ruled out today (see
  the allocator section).

* **Single-register addressing.** A memory operand is "base register +
  displacement", where the displacement is a single build-time constant encoded
  in the instruction. There is no index register and no scale. Any distance
  known only at run time is folded into the base pointer with an explicit `add`,
  that is, a running pointer advanced each iteration. Restricting addresses to
  one register means every memory op reads at most one pointer, which lowers
  register pressure and simplifies spilling.

* **Structured loops.** Loops are explicit nodes with a runtime counter
  (`loop_begin`/`loop_end`). Loops whose counter is known at build time are
  unrolled by the builder.

### The Builder

The builder is the only kernel-specific part. It is a high-level version of
today's JIT kernel. It loads parameters, builds the loop nest, and expresses the
math, but does so in target-neutral operations.

It is data-type agnostic. The data type is used only as a tag on `vec`
registers, and the operations carry no data-type branches. An operation such as
`vfma` is lowered by the emitter to the instruction for the tagged type. This
alone removes a large class of branches.

It is ISA agnostic. There is no ISA-specific logic in the builder. A `mask` is
an abstract element predicate that the emitter realizes per ISA (a k-register on
AVX-512, or a vector register of all-ones elements on AVX2). Vector and
general-purpose registers are likewise abstract and are mapped to concrete
registers (YMM or ZMM, and the GPRs) during lowering.

The builder currently threads an `ir_t` object through its functions and calls
methods on it. A later option is to expose the same operations as free functions
over an implicit context, so the object does not have to be passed around, which
is the style the GPU JIT builder uses. The trade-off is a context that has to be
set up and torn down and only one IR being built at a time. It is a
straightforward refactoring and is not required now, so it is left for later.

### A Worked Example: The GEMV Reduction Step

The difference is easiest to see on the innermost reduction step, which loads
one chunk of `x` and multiply-adds it into each accumulator. Below is the IR
builder for it (the full f32 implementation) next to the corresponding slice of
the hand-written GEMV inside brgemm.

IR builder (`brgemv_ir.cpp`), full and tail:

```cpp
void emit_microkernel(ir_t &ir, const conf_t &cfg,
        const std::vector<int> &acc, int a_ptr, int x_ptr) {
    const int x = ir.new_vec(cfg.dt_x);
    const int a = ir.new_vec(cfg.dt_a);
    ir.vload(x, x_ptr, 0);
    for (int i = 0; i < (int)acc.size(); i++) {
        ir.vload(a, a_ptr, cfg.dt_sz_a * (dim_t)i * cfg.lda);
        ir.vfma(acc[i], a, x);
    }
}

// The K-tail is a separate function with the same shape, using masked loads
// instead of propagating an `is_tail` flag through the body:
//   ir.vload_masked(x, x_ptr, 0, mask, cfg.k_tail);
//   ir.vload_masked(a, a_ptr, ..., mask, cfg.k_tail);
//   ir.vfma(acc[i], a, x);
```

Hand-written GEMV (`jit_brgemm_kernel.cpp`, non-transposed slice):

```cpp
// Registers are assigned by hand from the top of the vector file:
//   gemv_load_a() -> index max_effective_vregs - 1 - gemv_bd_block() - 0
//   gemv_load_b() -> index max_effective_vregs - 1 - gemv_bd_block() - 1
const auto a_vmm = gemv_load_a();
const auto b_vmm = gemv_load_b();

maybe_set_avx_mask(is_rd_tail);                 // ISA-specific mask setup
load_and_convert_to_f32(b_vmm, ptr[reg_aux_B], brg.dt_b, is_rd_tail);
for (dim_t bd = 0; bd < bd_block; bd++) {
    const auto acc = accm(1, bd, 0);
    load_and_convert_to_f32(a_vmm, ptr[reg_aux_A + A_offset(bd, 0)],
            brg.dt_a, is_rd_tail);
    uni_vfmadd231ps(acc, a_vmm, b_vmm);
}

// ...where the load itself is the branch the IR removes:
void load_and_convert_to_f32(Vmm vmm, Address addr, data_type_t dt, bool tail) {
    if (is_superset(brg.isa_impl, avx512_core)) {
        const auto kmask = tail ? rd_tail_mask : gemv_full_mask;
        if (dt == bf16) {
            uni_vpmovzxwd(vmm | kmask | T_z, addr);
            uni_vpslld(vmm, vmm, 16);
        } else if (dt == f16) {
            vmovdqu16(Vmm_lower_t(vmm.getIdx()) | kmask | T_z, addr);
            vcvtph2ps(vmm, Vmm_lower_t(vmm.getIdx()));
        }
    } else { // avx2
        if (tail)
            vmaskmovps(vmm, vmm_tail_mask(), addr);
        else
            uni_vmovups(vmm, addr);
    }
}
```

The IR builder is shorter partly because it covers less today (f32, AVX2), so
this is not a like-for-like line count. The point is what the extra hand-written
lines are, and which shared stage (the emitter or the allocator) takes over each
one so the builder never has to.

* **Data-type branches.** The data type is a tag on the vec register, and the
  emitter picks the instruction.
* **ISA branches.** The builder has no ISA logic, and the emitter lowers per
  ISA.
* **Tail masking.** The mask is abstract, and the emitter realizes it per ISA
  (a k-register or a vector mask).
* **Manual register indexing.** The allocator assigns the registers.

None of these live in the builder. Adding bf16, f16, or AVX-512 to the IR GEMV
adds code to the shared emitter once, not a new branch to every builder that
loads a value.

### The Register Allocator

The allocator turns the IR's unlimited virtual registers into the CPU's real
ones, spilling to the stack when more values are live at once than there are
registers. It knows only register kinds and control flow, nothing about the
computation or the ISA. It works in two steps.

1. **Liveness analysis.** A value is live at a point if it will be read again
   before being overwritten. The IR forms a trivial control-flow graph (each
   instruction is a node, and the only non-linear edges are loop back-edges and
   the two branch ops), and liveness is computed by backward data-flow iterated
   to a fixed point. The back-edge is what forces more than one pass. A pointer
   read at the top of a loop body and advanced at the bottom is live across the
   whole body, and that fact only propagates on a later pass. The number of
   passes scales with loop-nesting depth.

2. **Linear scan.** Each value's liveness is collapsed to a single
   `[start, end]` interval, intervals are sorted by start, and registers are
   assigned per register file. When no register is free, the interval whose end
   is furthest away is spilled.

Linear scan was chosen over graph coloring on the basis of compile time and the
performance we actually need. Graph coloring can produce a slightly better
allocation in some cases, but linear scan is much faster, which matters under
JIT generation-time constraints. The target is to be on par with hand-written
allocation, and linear scan is expected to reach that.

The algorithm leaves room for three refinements, listed in the order we expect to
need them.

* **Spill weights by loop depth.** The current "furthest end wins" rule can
  spill a hot, long-lived value (such as a loop-invariant base pointer) simply
  because it lives long. Weighting spill choice by loop-nesting depth biases
  spills toward cold values and keeps hot ones resident. This is the refinement
  we expect to implement.

* **Free-register spill reloads.** The emitter currently reloads spilled values
  into a fixed set of scratch registers reserved for the whole kernel (see
  below). The allocator can instead find a register that is already free at each
  spill site and reload into it, so no register is reserved up front. This
  removes the standing cost of the reserved scratch registers, which matters for
  high-pressure kernels such as brgemm.

* **Hole-aware interval splitting.** Collapsing liveness to one interval ignores
  holes (a value live in `3..7` and `20..25` is treated as live `3..25`). Under
  register pressure those holes could be reused. This is a larger change and may
  not be needed. The weighting above is expected to cover our cases.

The register pool the allocator draws from is built per ISA. The pool itself is
ISA-agnostic, holding only integer indices of the registers available for each
kind. The function that builds it is ISA-aware because it needs to know how many
vector registers exist, how many GPRs are available (which depends on APX
availability), and whether the target has dedicated mask (k) registers. On AVX2
a mask is a vector register, so `vec` and `mask` allocate from the same file and
compete for the same pool. On AVX-512 a mask is a k-register with its own file.

AMX tile registers are treated as another register file that the allocator
manages, alongside `gpr`, `vec`, and `mask`. Two constraints come from how
kernels use tiles. Tile registers are not spilled, and they are not
oversubscribed. If a kernel asks for more tiles than the target provides, or a
spill would be needed, the allocator reports an error rather than emitting slow
or incorrect code.

A small number of physical registers are reserved before allocation and excluded
from the pool: the stack pointer, the kernel-argument pointer, and a fixed set of
scratch registers that the emitter uses to load and store spilled values. Three
vector registers are reserved, enough to cover an operation whose destination and
both sources are spilled at once, and two general-purpose registers, since an IR
operation uses at most a destination and one source in that file.

Reserving these registers for the whole kernel is acceptable under low register
pressure, but it does not scale. Under high register pressure a kernel can run
close to full vector pressure, up to 16 of 16 on AVX2 and 32 of 32 on AVX-512,
and holding vector registers out of the pool there is too costly. The plan is to
drop the fixed scratch registers in favor of the free-register spill reloads
described above, so no register is reserved for the whole kernel.

### The Emitter

The emitter is the only part that is aware of everything, because it is the part
that produces code. It walks the allocated IR once and lowers each abstract
operation to target-specific instructions using the physical registers the
allocator chose. Spilled values are loaded into the reserved scratch registers
before use and stored back after.

There is a separate emitter per ISA family.

* **AVX2\*** (avx2, avx2_vnni2, and so on)
* **AVX-512\*** (avx512_core, avx512_core_bf16, and so on)

Each family emitter handles every ISA in its family. For example, lowering
`vfma` for bf16 (`vdpbf16ps`) requires avx512_core_bf16, but it is still handled
by the AVX-512 emitter. The AMX and ACE emitters are expected to share
infrastructure with the AVX-512 emitter.


### IR Infrastructure Extension Rules

The IR infrastructure grows by adding data types, ISAs, and the fused
instructions each ISA supports. Each kind of change belongs in one place.

1. **A new data type** is a tag on a `vec` register. The builder stays the same.
   The emitter maps the target-neutral operation and data type to the right
   instruction.
2. **A new ISA** adds a new emitter or extends an existing one.
3. **A new variant of an existing instruction**, such as an instruction with a
   memory operand that fuses a load and an operation, is handled by a lowering
   rule in the emitter. It is *not* a new IR operation and is invisible to the
   builder.
4. **A new behavior** that no existing operations can express becomes a new IR
   operation, with its own definition, uses, and lowering.

To tell the third from the fourth: if existing operations can already express
the behavior and the only difference is the ISA encoding, use a lowering rule.
Otherwise, add a new IR operation. This keeps the set of operations
target-neutral.

### Static Data

Some lowerings need static data, such as the mask tables AVX2 uses for
masking. That data must be written after the ABI postamble, but the jumps to it
are emitted during lowering, before its address is known. To bridge this, the
emitter is given a data-section structure that it fills with the bytes plus an
unresolved Xbyak `Label`. After lowering, the structure holds everything needed,
and once the postamble is emitted the data is written and the labels are bound
to their final addresses.

### Post-Ops Injector

The post-ops injector is an existing Xbyak-based component, and we do not want
to rewrite it as an IR builder, at least not now. Instead we plug it into
IR-based code by treating "apply post-ops" as a single IR instruction,
`inject_postops`.

The complication is that the injector takes a variable number of accumulator
registers, while an IR instruction has a fixed-size operand list. We resolve
this by storing the injector arguments (the accumulator list and the output base
pointer) in a side table on the IR, and having the `inject_postops` instruction
carry, in its immediate field, an index into that table. The IR core therefore
carries only virtual-register ids and has no dependency on the injector.

The injector is created in `generate()`, lives across the whole codegen flow,
and is destroyed on exit. A callback registered with the emitter forwards the
physical registers chosen by the allocator to the injector's apply entry point.
The emitter invokes it when it lowers `inject_postops`.

Keeping the injector outside IR register allocation is a deliberate
simplification, not a fundamental design choice. The injector has the capability
to save and restore the registers it uses, and that capability is what makes the
simplification possible. The injector also knows at construction time how many
registers it needs, so a later change can pass that to the allocator and let it
account for those registers directly, removing the spills the current approach
can cause. This is not needed yet because the kernels we migrate first already
rely on the injector preserving the registers it uses.

## On Optimization Passes

The IR has no IR-to-IR optimization passes, and that is a deliberate choice.

CPU JIT kernels are single-threaded microkernels invoked from their drivers.
Their optimization is *structural*. The developer writes the kernel already in
an optimized shape (blocking, unrolling, loop structure).
An IR-level optimizer would have little left to do, and that small gain does not
justify building and maintaining a pass framework or the generation time it
would add. We rely on the kernel structure for optimization instead.

One clarification matters, because it relates to whether IR-generated code can
match the performance of the hand-written kernels. The IR takes over one thing
the developer used to control: register allocation and spilling. The developer
still writes the optimized structure (blocking, unrolling, loop order,
scheduling) and that structure is emitted as written. Register allocation, by
contrast, is now automatic. For the performance claim to hold, the allocator's
choices must be at least as good as the ones a developer made by hand (see the
validation section).

This holds as long as the kernel does not oversubscribe the physical registers.
Because the IR uses unlimited virtual registers, a builder can ask for more live
values than there are physical registers, which forces the allocator to spill
inside the hot loops. A spill inside a hot loop adds extra loads and stores that
the developer did not plan for and slows the loop down. So the developer is still
responsible for keeping the number of live values in a hot loop within the number
of physical registers, which keeps the loop free of spills. This is not new work.
Hand-written kernels already do register blocking, deciding how many registers go
to accumulators and how many to loads, and that responsibility stays with the
developer. The allocator assigns physical registers but does not create more of
them. The allocator's loop-depth spill weighting (see the allocator section)
keeps any unavoidable spills on cold paths.

## Relationship to the GPU JIT

oneDNN already has a JIT IR on the GPU side, and it is reasonable to ask why the
CPU does not reuse it. The two solve different problems.

* The GPU JIT generates the whole computation for a problem, the entire loop
  nest and data movement that on the CPU lives in a C++ driver around a small
  microkernel. With that much more code to produce, it relies on optimization
  passes to transform and refine its IR, closer to a domain-specific compiler.
  The CPU JIT, as mentioned above, emits already-optimized microkernels directly,
  so a pass framework is unnecessary.
* This leads to different IR designs. The GPU favors a representation tuned for
  transformation. The CPU uses a simpler representation that is cheap to
  construct and analyze.
* The backends target fundamentally different architectures, so there is little
  code-generation infrastructure to share.

Reusing the GPU IR would mean adopting a heavier representation and pass
framework built for a different target and a different way of generating code.
The simpler CPU-specific IR is a better fit.

## Other CPU Vendors

The IR is ISA-neutral by design, so another CPU backend could in principle
supply its own emitter and reuse the rest. But "neutral by design" and "shared,
co-owned code" are different things, and we propose the first without committing to
the second yet.

The reason is development speed. oneDNN has several CPU backends, each with its
own ISA and maintainers. The IR is new and its interfaces will change
substantially as we learn what works. If the IR were shared from the start,
every such change would become a cross-team negotiation, slowing us down exactly
when we need to speed it up. Within x86 this is not a concern. Intel and AMD
are largely compatible, so vendor-specific tweaks are easy to add and stay
low-risk because everything still runs on the same ISA. The cost appears only
when crossing into a genuinely different ISA.

The proposal is therefore to let the interfaces settle on x86 first. Another
backend then has two options. It can copy the IR now and maintain its own
version, with no later merge expected. The cost is a duplicated IR
infrastructure, but some CPU backends already duplicate large components such as
the brgemm kernel, so one more may be acceptable to them. Alternatively, it can
wait until the IR matures, then extract the shared part and migrate to it, with
no duplication. Both paths avoid locking down an immature interface across
backends too early.

## Additional Benefits

Beyond addressing the problems above, the IR enables the following.

* **Simplifying existing kernels.** Xbyak-based kernels can be ported
  incrementally. The brgemm copy-kernel zoo, for instance, is a good candidate
  for consolidation once the IR covers the needed data types and ISAs.
* **Faster development of new kernels.** New kernels such as SDPA become quicker
  to write because the builder works at a higher level.
* **Readable IR dumps.** Because the whole kernel exists in an abstract form
  before lowering, it can be printed in a readable form for debugging. The dump
  shows virtual registers, the loop structure with indentation for nested loops,
  and the loop counters with how they step each iteration. It can be printed
  through the existing verbose feature, so tracing a generation problem does not
  require dumping the JIT output and reading raw assembly, and the IR format is
  more readable than the emitted instructions.
* **Room for AMD-specific tweaks.** Because AMD shares the x86 ISA, small
  AMD-specific tweaks can be added without disturbing the Intel path, and they
  stay low-risk because everything still runs on the same ISA. The pipeline also
  gives a clean place to isolate them, a small lowering hook in the emitter
  rather than branches scattered through the kernel. This in-tree flexibility is
  specific to AMD. A vendor on a genuinely different ISA does not gain it.

## Proof of Concept

The PoC implements the full pipeline for one kernel: a non-transposed GEMV
builder, with the linear-scan allocator and an AVX2 emitter behind it. It is f32
only and supports bias and eltwise/binary post-ops. Sum, scales, and zero points
are not yet supported.

## Validation, Limitations, and Open Questions

The IR-based GEMV PoC demonstrates instruction-level parity with the
hand-written kernel. It emits the same sequence as the brgemm GEMV
implementation, differing only in register assignment, and achieves zero spills
under low register pressure. This confirms functional equivalence for the GEMV
case.

Generation overhead is higher than the single-pass Xbyak baseline due to added
build, liveness, and allocation passes. Cost ranges from about 20% higher
without tail handling to up to 2x with tails, but remains sub-millisecond
(0.23-0.70 ms). Since kernels are cached, this overhead is currently amortized
and not on the hot path. These passes are also unoptimized (liveness, for
example, recomputes per-instruction state to a fixed point), so the
generation time is a baseline that can likely be reduced.

However, the PoC does not stress the allocator under high register pressure.
GEMV's low-pressure nature avoids spill behavior entirely, so spill parity with
hand-written kernels remains unproven for workloads such as brgemm. While
global-liveness allocation is expected to improve behavior over the prior
single-pass allocator prototype (which saw a 40% spill regression in the worst
case), this has not yet been validated. Loop-depth spill weighting is
likely required before brgemm can be supported robustly.

Similarly, generation time must be re-evaluated on larger, heavily unrolled
kernels, where both IR size and liveness computation grow significantly beyond
GEMV's profile.

The IR infrastructure is still simple and largely unoptimized (the allocator,
for example, has no spill weighting or interval splitting yet). There is room to
improve it, so these results are a baseline rather than a ceiling.

The next step is to migrate brgemm for f32 on AVX-512, the exact configuration
where the single-pass prototype regressed 40%, and therefore the sharpest test
of the allocator.

### Open Questions

1. Does the allocator maintain spill parity with hand-written code under high
   register pressure (e.g., f32 brgemm on AVX-512), and is loop-depth spill
   weighting sufficient to achieve it?
2. Does kernel generation time remain acceptable at brgemm scale, given larger
   IR and more expensive fixed-point liveness computation?
