# In-place binary post-ops: DST as RHS

## Overview

The Sum post-op in oneDNN is not like the other post-ops: it performs a binary
operation but requires no right-hand-side buffer, instead taking its input from
DST. This behavior can in principle be replicated in any of the Binary post-ops,
extending the buffer semantics of Sum beyond addition. This RFC discusses how,
and proposes doing so for multiplication, which is the operation the use case
outlined below calls for.

Many modern LLMs like LLaMa, especially in OpenVINO, contain 3-GEMM subgraphs
that can benefit from a variant of Sum that performs multiplication instead of
addition, saving ~4–24% of E2E VRAM, depending on the model and the prompt
length, compared to a regular `binary_mul` which inevitably requires an RHS
buffer that should exist in VRAM beside DST.

The shape of that need is measurable. Across multiple E2E runs over a ~75GB pool
of such networks on BMG, the Binary post-ops OpenVINO emits — with the data type
of the right-hand side each of them is given — are:

| Binary post-op | Occurrences |
|----------------|-------------|
| `binary_add:f16` | 3744 |
| `binary_mul:f16` | 1248 |
| `binary_mul:f32` | 672 |

No other Binary algorithm appears — no `binary_sub`, no `binary_div`, none of
the comparisons — and every one of these cases is a single unquantized
floating-point type throughout. Since in-place `binary_add` is what Sum already
provides, in-place `binary_mul` on `f16` and `f32` is the entirety of what the
feature is known to be needed for. A quantized destination is not out of reach —
Caveats > Quantization covers what it would and would not permit — but it is not
what the use case asks for, and the proposal is not shaped around it.

## Options

There are several approaches to adding an in-place multiplication post-op:

1. Add a new one-off Sum-like post-op.
    * Pros: The most conceptually straightforward option.
    * Cons: Will essentially duplicate (and then triplicate, for potential
    other binary operations) the already very different codepath for Sum in
    many of the primitives' implementations.
2. Extend Sum to support other algorithms.
    * Pros: Existing codepath for Sum in most of the primitives.
    * Cons: Will still duplicate at least some Binary post-op code, furthermore
    Sum was never even intended to support other algorithms beside addition, so
    extending the Sum PO will require a significant amount of unnecessary work.
    This option was prototyped: in GEMMstone, where Sum is not a post-op proper
    but an extension of β in the BLAS formula `C = αAB + βC`, β was extended to
    also carry an algorithm kind. The prototype worked, but it left β with a
    meaning unrelated to BLAS for every algorithm other than `binary_add`, and
    reimplemented what the Binary post-op pipeline already does — so it was
    scrapped in favor of extending Binary instead.
3. Add a flag to `post_ops_t::binary_t` that would signal that the `src1` buffer
of this Binary aliases DST.
    * Pros: Virtually no changes to most of primitives' implementations.
    * Cons: No forward compatibility with other buffered post-ops like PReLU or
    e.g. potential `post_ops_t::ternary_t` — or, for that matter, the existing
    ternary called `binary_select` that uses `src2`.
4. Add a flag to `attr_t` (rather than to `post_ops_t`) to tell the parent
primitive that some of the post-ops might be aliasing DST.
    * Pros: Same as `post_ops_t::binary_t` flag, without its cons. Covers PReLU
    and any future buffered post-op without further API changes. Reuses the
    established attribute machinery: `skip_mask_t`, the `attr-*` conventions in
    verbose and benchdnn, and the existing rules for where attributes live.
    * Cons: The flag lives outside the post-ops whose behavior it governs, so
    it is not carried along when they are. Where a parent primitive nests
    several *different* sub-primitives that need *different* attributes, those
    attributes are not cloned from the parent: each is constructed empty and
    filled with whatever that sub-primitive needs. The flag then defaults to
    absent, and a sub-primitive writing to the parent's DST is free to use it
    as intermediate storage — which is exactly what the flag exists to forbid.
    The omission is silent: it yields wrong results rather than a rejected
    primitive descriptor. No implementation in the library today is in that
    position (see Implementation), so this is a prospective cost rather than a
    present defect — but a prospective cost with no existing site to copy the
    correct handling from. It is fair to say that attributes already hold
    entries that are lost by copying `post_ops_t` alone, but unlike those, this
    flag is load-bearing for correctness rather than performance. A per-post-op
    flag, on the other hand, carries information no implementation is expected
    to use, since aliasing is a property of the whole post-op chain in practice.
5. Add a new algorithm, e.g. `binary_mul_inplace`, only accepted as a post-op.
    * Pros: All of the pros of [3] plus no new public API surface whatsoever —
    no attributes and no new entry points; the post-op is appended through the
    existing `append_binary()` and reuses the existing `binary_mul` machinery.
    The Binary primitive proper enumerates the algorithms it accepts, so the new
    algorithm cannot leak into it. Being part of the post-op entry, the in-place
    property travels with the post-ops wherever they are copied: there's nothing
    to propagate separately into a nested primitive and nothing to lose by
    omission; it is covered by `operator==()`, by the primitive cache key and by
    serialization at no cost, since the algorithm kind already is. Being
    per-entry also makes the requirements on the aliased operands checkable at
    all: the library knows which of them alias DST, so it can validate them
    against DST, whereas [4] states only that *some* post-op may alias DST and
    can express no such rule. Smallest possible change for the use case that
    motivates this RFC.
    * Cons: Does not generalize: every further in-place algorithm needs its own
    `alg_kind_t` entry, which reproduces the combinatorial growth of [1] at the
    algorithm level, and does nothing for PReLU or other buffered post-ops, both
    present and planned. It also encodes buffer binding into `alg_kind_t`, which
    otherwise only describes the operation being performed. Gating it still
    takes a `skip_mask_t` flag, since implementations validate post-ops by kind
    rather than by algorithm — but that flag is internal, so unlike [4] it costs
    no public API.

Both [3] and [4] were discussed at length during the review of this RFC, and the
discussion first converged on [4]. In its turn, [5] was proposed later as a
deliberately narrower alternative, and it is the one the review ultimately
settled on; the rest of this document specifies it.

The use case data in the Overview supports that choice: `binary_mul` is the only
algorithm the feature is known to be indispensable for. The case for [3] and [4]
never rested on demand for other algorithms, which at the time is absent, but on
the cost of potentially adding them later one at a time and on covering other
buffered post-ops without further changes to the internals of oneDNN. Against a
use case that is proven to be narrow, such generality does not look worth the
extra testing surface and maintenance cost.

## Proposal

The new oneDNN interface for in-place Binary post-ops is a Binary algorithm kind
only accepted as a post-op. It is the first such entry in the Binary section of
`dnnl_alg_kind_t`.

The restriction is not arbitrary: at primitive level, in-place is already a
documented execution-time contract — source 0 may alias the destination — and
for a commutative operation that contract covers the in-place case completely:
`DST = DST × M` is `binary_mul` with the multiplicand M in the source 1 slot and
DST passed for both source 0 and the destination. A post-op's right-hand side
carries no such contract, and the aliasing has to be known at primitive creation
because it constrains what the implementation may do with DST. Hence an
algorithm kind, and hence one the Binary primitive's own allowlist rejects.

```diff
 typedef enum {
 
 …
 
     /// Binary not equal
     dnnl_binary_ne = 0x1fffb,
     /// Binary select
     dnnl_binary_select = 0x1fffc,
+    /// In-place multiplication: RHS = DST
+    dnnl_binary_mul_inplace = 0x1fffd,
     /// Nearest Neighbor Resampling Method
     dnnl_resampling_nearest = 0x2fff0,
     /// Linear Resampling Method
     dnnl_resampling_linear = 0x2fff1,
 
 …
 
 } dnnl_alg_kind_t;
```

— with the usual counterpart in `dnnl::algorithm`:

```diff
 enum class algorithm {
 
 …
 
     /// Binary not equal
     binary_ne = dnnl_binary_ne,
     /// Binary select
     binary_select = dnnl_binary_select,
+    /// In-place multiplication: RHS = DST
+    binary_mul_inplace = dnnl_binary_mul_inplace,
     /// Nearest Neighbor resampling method
     resampling_nearest = dnnl_resampling_nearest,
     /// Linear (Bilinear, Trilinear) resampling method
     resampling_linear = dnnl_resampling_linear,
 
 …
 
 };
```

No new entry point is required: the post-op is appended via the existing
`dnnl_post_ops_append_binary()` and its C++ equivalent. An in-place Binary is a
regular Binary post-op in every respect but one — its right-hand side is the
destination buffer, and the algorithm kind is what says so.

From a semantical standpoint, this is how an in-place Binary differs from the
regular ones:

```
ACC = ACC + RHS — binary_add
ACC = ACC × RHS — binary_mul
ACC = ACC / RHS — binary_div
…
ACC = ACC + DST — Sum
ACC = ACC × DST — binary_mul_inplace
```

— where DST is the destination buffer, ACC is the accumulator register block
within the primitive, and RHS is the separate right-hand-side buffer used in
regular Binary post-ops.

Keep in mind that DST is not intended to get rewritten until after all post-ops
are applied, so for primitives' implementations where that cannot be guaranteed,
there should be either no support for in-place post-ops or some memory movement
to guard the DST data against the intermediate writes.

The post-op buffer that aliases DST — the one described by `user_src1_desc` —
must be declared with a memory descriptor that equals that of `DNNL_ARG_DST` in
shape and in data type. Its layout has to match DST as well, but it need not be
spelled out: declaring it as `format_kind::any` is both allowed and preferable
when in doubt, since it then resolves to whatever layout the primitive picks for
DST — see Applicability. Where DST is itself declared as `format_kind::any`, it
is in fact the only valid spelling, there being no layout yet to match.

Quite naturally, this means that an in-place Binary cannot be broadcast, as
that would break the MD equality requirement.

Requiring the exact type, rather than merely a type of the same bit size, is a
deliberate narrowing. The looser rule would be defensible on structural grounds:
the aliased buffer is DST's own storage read back under a different
interpretation, so the operand occupies exactly the bytes DST occupies and only
the element type used to decode them differs, while everything that governs
addressing — shape and layout — has to match either way. Sum admits such a
reinterpretation, and it would in principle allow e.g. an `s8` multiplicand
carried in a `u8` destination. It is not free, though: the library cannot tell
an intended reinterpretation from a mistake, so it gives up an error the strict
rule catches, and it needs a second rule spanning the whole chain, because among
several aliased operands — all of them the bit size of DST — at most one
distinct declared type can be DST's own, so two in-place Binaries declaring
different types are provably wrong in part whatever DST's type turns out to be.
Set against that, no case in the data above calls for the relaxation at all. The
strict rule is therefore what this RFC proposes, and the relaxation is left as a
possible later extension — nothing else here depends on the choice, and
loosening a rule later breaks no existing user. Under the strict rule every
aliased operand is DST's type by construction, so the cross-chain question does
not arise at all.

Unlike under [4], the requirement is checked rather than merely stated. The
algorithm names the aliased operand, so the library knows at primitive creation
exactly which descriptor has to agree with DST, and
`post_ops_t::entry_t::validate_binary()` — which already receives `dst_md` and
already compares `ndims` against it — is the natural place to reject a mismatch,
right next to the block that validates the extra operand of `binary_select`:

```diff
     status_t post_ops_t::entry_t::validate_binary(
             engine_kind_t engine_kind, const memory_desc_t *dst_md) const {
 
 …
 
+        if (is_inplace_binary()) {
+            const memory_desc_wrapper src1_d(binary.user_src1_desc);
+            const memory_desc_wrapper dst_d(dst_md);
+            VCHECK_ATTR(src1_d.data_type() == dst_d.data_type(),
+                    VERBOSE_INCONSISTENT_DT, "bin_po src1", "dst");
+            VCHECK_ATTR(utils::array_cmp(
+                                src1_d.dims(), dst_d.dims(), dst_d.ndims()),
+                    VERBOSE_INCONSISTENT_MDS, "bin_po src1", "dst");
+            VCHECK_ATTR(IMPLICATION(!src1_d.format_any(),
+                                !dst_d.format_any()
+                                        && src1_d.similar_to(dst_d)),
+                    VERBOSE_INCONSISTENT_MDS, "bin_po src1", "dst");
+        }
+
         return status::success;
     }
```

The layout is compared only when the operand declares one. `format_kind::any`
there — the encouraged spelling — defers it to whatever the primitive picks for
DST, and there is nothing to compare yet. A spelled-out operand layout, though,
requires a spelled-out DST layout that matches it: every caller passes
`desc.dst_desc`, straight from the user, so DST itself may well be `any` at this
point, and admitting a concrete operand layout against an unresolved DST would
leave it unverified for good, since the implementation picks DST's layout
afterwards and nothing revisits the post-op.
`memory_desc_wrapper::similar_to()`, which compares dims, strides and blocking,
holds for no descriptor of format kind `any` by documented design, which is why
it sits under that implication rather than being called outright. The data type
and the dimensions are compared unconditionally.

No such check would be available under [4]: an attribute saying that some
post-op may alias DST leaves no entry to validate against DST to begin with.

Since the algorithm determines the operand, the Binary MD is redundant:
everything about it follows from DST. It is kept because doing so costs no new
API, because the existing `append_binary()` requires it anyway, and because it
is where a later relaxation of the type rule would live.

What remains a contract on the user is the buffer itself. Being a regular Binary
post-op, an in-place Binary still takes its right-hand side at execution as
`DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1`, and the library has no
reliable way to tell whether the handle it is given is in fact DST. Passing
anything else there is undefined behavior.

## Implementation

Most of the work is the bookkeeping that every new algorithm kind requires:

* `dnnl_alg_kind_t` entry in `dnnl_types.h`, its mirror in `alg_kind` from
  `c_types_map.hpp`, and a corresponding `dnnl::algorithm` entry in `dnnl.hpp`;
* the Binary algorithm allowlist in `post_ops_t::validate_binary()`, which is
  what admits the new kind as a post-op in the first place;
* the algorithm name tables — `dnnl_debug_autogenerated.cpp` and its generator,
  and `tests/benchdnn/dnn_types.cpp` for the benchdnn side.

Three additions are specific to the feature rather than to the enumeration:

* `entry_t::validate_binary()` gains the descriptor check described in the
  proposal;
* an `entry_t::is_inplace_binary()` predicate, modelled on the adjacent
  `is_binary_with_ternary_op()` which discriminates `binary_select` the same way,
  and a `post_ops_t::has_inplace_binary()` helper on top of it — the `entry_t`
  predicate being needed because `is_binary()` deliberately says nothing about
  the algorithm;
* a `skip_mask_t` bit for the opt-in, enforced in `has_default_values()` through
  that helper — see Applicability.

What is worth noting is the list of places that do *not* have to be touched.
`post_ops_t::operator==()`, `primitive_hashing::get_attr_hash()`,
`primitive_serialization.cpp` and `verbose.cpp` all handle the algorithm kind of
a post-op entry already, so an in-place Binary is distinguished from a regular
one, cached separately, serialized correctly and printed in a form that can be
converted back into a reproducer without a single line being added to any of
them. For the same reason, nothing has to be propagated manually into a nested
primitive: the property travels inside the post-op entry, and a nested primitive
that is handed the post-ops is handed the property with them.

The Binary primitive proper keeps its own algorithm allowlist, separate from the
post-op one, so a `binary_mul_inplace` passed to it is rejected at primitive
descriptor creation without any special handling — there is no destination to
alias in that context, and nothing about it needs to be special-cased.

With the algorithm in place, on execution the primitive expects a reference to
the `DNNL_ARG_DST` buffer passed as the right-hand side of that post-op, in the
usual `DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1` slot, provided that
buffer satisfies the requirements outlined in the proposal: same shape, same
layout, same type.

That way, no aspect of the existing Binary post-op pipeline is to be altered
in any form, since DST is, after all, just another regular buffer. This is not
a hypothetical:

* Intel GPU JIT v2 Conv builder already dispatches Sum as `alg_kind::binary_add`
  with the destination buffer supplied as the right-hand side, through the same
  codepath as a regular Binary post-op;
* The v1 IR builder, used in Conv, Deconv, Pool, and Reorder, likewise imports
  DST as an ordinary post-op input tensor;
* GPU JIT GEMM is the exception where Sum is a separate codepath rather than an
  injected post-op, but GPU JIT GEMM has a regular Binary codepath as well.

## Caveats

### Quantization

The Sum post-op includes two immediate constants for scales and zero points
that can be applied to the DST buffer prior to performing the addition. This
can come in handy when DST is quantized — e.g. if its data type is `u8` or `s8`
or `f8_*`. Binary post-ops have no such capability (and they don't need it, for
reasons discussed further below), but Sum-like behavior can be replicated for
every algorithm below, with high accuracy when the post-op accumulator is `f32`
(`A` = ACC, `D` = DST, `s` = scale, `z` = zero point):

| oneDNN op | Sum-like quantized form | Representable `s` and `z` | Equivalent decomposition |
|-----------|-------------------------|---------------------------|--------------------------|
| `binary_add` | `A + (sD + z)`   | `s ∈ ℝ \ {0}, z ∈ ℝ` | `linear(binary_add(linear(A, 1/s, 0), D), s, z)` |
| `binary_sub` | `A – (sD + z)`   | `s ∈ ℝ \ {0}, z ∈ ℝ` | `linear(binary_sub(linear(A, 1/s, 0), D), s, –z)` |
| `binary_min` | `min(A, sD + z)` | `s ∈ ℝ \ {0}, z ∈ ℝ` | `linear(binary_min(linear(A, 1/s, –z/s), D), s, z), s > 0`<br>`linear(binary_max(linear(A, 1/s, –z/s), D), s, z), s < 0` |
| `binary_max` | `max(A, sD + z)` | `s ∈ ℝ \ {0}, z ∈ ℝ` | `linear(binary_max(linear(A, 1/s, –z/s), D), s, z), s > 0`<br>`linear(binary_min(linear(A, 1/s, –z/s), D), s, z), s < 0` |
| `binary_lt`  | `A < (sD + z)`   | `s ∈ ℝ \ {0}, z ∈ ℝ` | `binary_lt(linear(A, 1/s, –z/s), D), s > 0`<br>`binary_gt(linear(A, 1/s, –z/s), D), s < 0` |
| `binary_le`  | `A ≤ (sD + z)`   | `s ∈ ℝ \ {0}, z ∈ ℝ` | `binary_le(linear(A, 1/s, –z/s), D), s > 0`<br>`binary_ge(linear(A, 1/s, –z/s), D), s < 0` |
| `binary_gt`  | `A > (sD + z)`   | `s ∈ ℝ \ {0}, z ∈ ℝ` | `binary_gt(linear(A, 1/s, –z/s), D), s > 0`<br>`binary_lt(linear(A, 1/s, –z/s), D), s < 0` |
| `binary_ge`  | `A ≥ (sD + z)`   | `s ∈ ℝ \ {0}, z ∈ ℝ` | `binary_ge(linear(A, 1/s, –z/s), D), s > 0`<br>`binary_le(linear(A, 1/s, –z/s), D), s < 0` |
| `binary_eq`  | `A = (sD + z)`   | `s ∈ ℝ \ {0}, z ∈ ℝ` | `binary_eq(linear(A, 1/s, –z/s), D)` |
| `binary_ne`  | `A ≠ (sD + z)`   | `s ∈ ℝ \ {0}, z ∈ ℝ` | `binary_ne(linear(A, 1/s, –z/s), D)` |
| `binary_mul` | `A × (sD + z)`   | `s ∈ ℝ, z = 0` | `binary_mul(linear(A, s, 0), D)` |
| `binary_div` | `A / (sD + z)`   | `s ∈ ℝ \ {0}, z = 0` | `binary_div(linear(A, 1/s, 0), D)` |

Note: with multiplicative post-ops like `binary_mul`, `A × (sD + z) = sAD + zA`
unless `z = 0`; `z ≠ 0` isn't representable with oneDNN post-ops as of now.

The decompositions assume an `f32` post-op accumulator. That is not universal —
e.g. the CPU primitives
[do support](https://github.com/uxlfoundation/oneDNN/pull/5341) `f16` and `bf16`
accumulators — and the intermediate `1/s` scaling a decomposition introduces can
lose precision or leave the range there, so with accumulation shorter than `f32`
the Sum-like forms above are to be treated as unavailable rather than merely
less accurate. This limits the workaround, not the feature:
`binary_mul_inplace` carries no scale or zero point and reads DST exactly as a
regular Binary reads its RHS, so nothing about the algorithm itself depends on a
decomposition.

Only the `binary_mul` row of the table is in scope for this proposal, since
`binary_mul_inplace` is the only in-place algorithm it introduces. The remaining
rows are retained because they describe the general shape of the problem, and
because they are what a later in-place algorithm would have to be weighed
against.

An in-place Binary reads DST as an ordinary right-hand-side buffer, under DST's
own data type as the proposal requires. Every type DST can have is permitted
there, the low-precision ones (`u8`, `s8`, `s4`, the `f8_*` family, etc.)
included — and it being low precision does not imply it needs dequantization,
just like with regular RHS buffers for Binary post-ops.
DST scales and DST zero points describe how ACC is quantized on the way out:
they are applied to ACC immediately before the store, not to DST itself.
Consequently they have no effect on the values an in-place Binary reads from
DST.

Post-ops in oneDNN do not generally carry quantization parameters — except Sum,
which has support for rudimentary (immediate and scalar) ZP/scale values for
historical reasons. Generalizing the buffer semantics of Sum to the Binary
post-ops doesn't — and doesn't have to — generalize its quantization semantics:
giving Binary scales and ZPs would introduce a per-post-op dequantization
concept that doesn't exist in the library yet and is out of the scope of this
RFC. Sum's immediate scalar ZP/scale pair is already insufficient for modern
quantized networks (as modern quantization involves per-channel ZPs/scales that
need to be fetched from a memory buffer, not hard-coded at graph transformation
stage), and adding buffered ZPs/scales would make post-ops something else
entirely. In any case, wherever Sum-like dequantization is expressible in
existing terms, the operation compositions above cover it. Where it isn't
(`binary_mul`/`binary_div` with a non-0 ZP), the case cannot be represented.

### Applicability

Not all of the oneDNN primitive implementations can support in-place Binary
post-ops — namely those that write intermediate data to DST, e.g. when
accumulating with atomics. To guard against improper usage, a new `skip_mask_t`
flag is to be introduced:

```diff
 enum class skip_mask_t : unsigned {
     none = 0,
     scales = 1u << 1,
     scales_groups = (unsigned)scales | (1u << 2),
     scales_data_type = (unsigned)scales | (1u << 3),
     zero_points = 1u << 4,
     zero_points_groups = (unsigned)zero_points | (1u << 5),
     zero_points_data_type = (unsigned)zero_points | (1u << 6),
     post_ops = 1u << 7,
     sum_dt = 1u << 8,
     rnn_data_qparams = 1u << 9,
     rnn_weights_qparams = 1u << 10,
     rnn_tparams = 1u << 11,
     rnn_weights_projection_qparams = 1u << 12,
     gpu_attr = 1u << 13,
     accumulation_mode = 1u << 14,
     fpmath_mode = 1u << 15,
     dropout = 1u << 16,
     rounding_mode = 1u << 17,
     precomputed_reductions = 1u << 18,
+    post_ops_inplace = (unsigned)post_ops | (1u << 19),
 };
```

The flag is enforced centrally, in `primitive_attr_t::has_default_values()`,
which is where every implementation's `skip_mask_t` is applied. Sum's data type
sets the precedent: it is likewise a property of a post-op entry rather than an
attribute of its own, and is likewise gated by a `skip_mask_t` bit that the
central check tests against the post-op chain:

```diff
     CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::sum_dt),
             post_ops_.sum_with_default_dt(dst_dt)));
+    CHECK_ARG(IMPLICATION((bool)(~mask & smask_t::post_ops_inplace),
+            !post_ops_.has_inplace_binary()));
```

That way in-place post-ops become an opt-in feature, so by default no primitive
would accept them unless the implementation allows it explicitly — and, unlike
an opt-in that each implementation has to remember to check for itself, this one
takes effect for every implementation the moment the bit is defined.

This matters more than it might appear, because implementations validate
post-ops by kind and very rarely by algorithm. `post_ops_t::entry_t::is_binary()`
tests `kind == primitive_kind::binary` and says nothing about the algorithm, and
the consumers follow suit: the Intel GPU JIT v1 IR builder, for instance,
branches on `is_binary()` and then maps whatever algorithm it finds, so an
unrecognized one reaches `alg_kind_to_op_kind()` and its
`gpu_error_not_expected()` — a failure at kernel build time rather than a
rejected primitive descriptor. Explicit algorithm allowlists exist in only a
handful of places, JIT GEMM's `supported_binary_op()`, the zen64 Matmul and the
generic SYCL post-op evaluator among them, and the reference CPU path checks the
algorithm in an `assert` only. An algorithm kind that most implementations would
silently accept and then mishandle is precisely what the central check prevents.

## Interface for benchdnn

benchdnn has to be told not only that aliasing is in play but also which of the
post-op buffers is aliased, and with the algorithm kind it is told both by the
same token: the run line names the algorithm, and the algorithm names the
operand. No new knob and no extension of the post-op grammar is needed — only
the new algorithm name in the parser's table, alongside `mul`, `add` and the
rest:

```
--attr-post-ops=mul_inplace:f16+add:f16 …
```

Every combination is expressible this way without dedicated syntax: with two
Binary post-ops, aliasing in the first, in the second, in both or in neither are
four run lines written the same way as any other post-op chain. The verbose
token that reproduces them needs no work either, being the algorithm name that
`dnnl_debug` already prints.

What does require attention on the benchdnn side is the reference computation.
The in-place algorithm has to read DST as it stood before the post-op chain was
applied, which is the same thing the Sum reference does; otherwise, a reference
that reads the running accumulator by mistake would agree with an implementation
that made the same mistake, and the comparison would pass.

Filling is the other half of it, and Sum has already paved that road too. An
in-place post-op's right-hand side must not be filled as a post-op buffer — it
is DST — so the branch that fills every Binary `src1` has to skip these entries,
and DST itself has to be filled as the declared type instead, which is what
`deduce_cfg_data_type()` already does for Sum.

Also of note is that `verbose_converter.py` would automatically pick up the new
Binary algorithm with no extra edits.

## Conclusion

In-place Binary post-ops are available at a modest cost to the library: one new
algorithm kind and one internal opt-in flag, with no addition to the public API
whatsoever. Nothing has to be propagated, cached, serialized or compared by hand
since all of that already happens for the algorithm kind of a post-op entry. In
return they enable memory savings and new approaches to DNN graph optimization.

The narrowness is deliberate, and it is the point on which the review settled:
the feature is delivered for the one algorithm it is known to be needed for,
and the broader question — how a buffered post-op should declare that it reads
DST in general — is left open rather than answered pre-emptively.
