# In-place binary post-ops: DST as RHS

## Overview

The Sum post-op in oneDNN is not like the other post-ops: it performs a binary
operation but requires no right-hand-side buffer, instead taking its input from
DST. This behavior can be replicated in all of the Binary post-ops, extending
the buffer semantics of Sum to all other Binaries.

The need for this, beside purely architectural, is also practical: many modern
LLMs like LLaMa, especially in OpenVINO, contain 3-GEMM subgraphs that can
benefit from a variant of Sum that performs multiplication instead of addition,
potentially saving dozens of % of E2E VRAM compared to regular binary_mul which
inevitably requires an RHS buffer that should exist in VRAM beside DST.

## Options

There are several approaches to adding an implicit multiplication post-op:

1. Add a new one-off Sum-like post-op.
    * Pros: The most conceptually straightforward option.
    * Cons: Will essentially duplicate (and then triplicate, for potential
    other binary operations) the already very different codepath for Sum in
    most of the primitives' implementations.
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
    * Cons: Such 'global' flag could be easily lost when copying the post-ops
    to nested primitives, since the post-ops themselves do not store the flag
    that governs their behavior, and the primitives being nested are required
    to also support DST aliasing. Note that this is not solved by implementing
    attribute copying correctly: a nested primitive is typically given a fresh,
    empty `attr_t` that is then filled in with whatever the sub-primitive needs,
    so the flag defaults to being absent and has to be propagated explicitly at
    every such site — see Implementation. It is fair to say that attributes
    already hold entries that are lost by copying `post_ops_t` alone, but unlike
    those, this flag is load-bearing for correctness rather than performance.
    A per-post-op flag, on the other hand, carries information no implementation
    is expected to use, since aliasing is a property of the whole post-op chain
    in practice.
5. Add a new in-place-only Binary algorithm, e.g. `binary_mul_inplace`, accepted
as a post-op only.
    * Pros: No new API surface whatsoever — no attribute, no flag, no new entry
    point. Reuses the existing `binary_mul` machinery and the existing algorithm
    validation, which already restricts which algorithms are accepted where, so
    the new algorithm cannot leak into contexts that do not expect it. Smallest
    possible change for the use case that motivates this RFC.
    * Cons: Does not generalize: every further in-place algorithm needs its own
    `alg_kind_t` entry, which reproduces the combinatorial growth of Option 1 at
    the algorithm level, and does nothing for PReLU or a future ternary. It also
    encodes buffer binding into `alg_kind_t`, which otherwise describes only the
    operation being performed.

Options 3 and 4 were discussed at length during the review of this RFC, and the
discussion converged on Option 4, which the rest of this document specifies.
Option 5 was proposed later as a deliberately narrower alternative and is still
open; choosing between it and Option 4 is largely a question of whether the
feature should be scoped to the currently requested case or made general, which
in turn depends on the use case data requested in the Overview.

## Proposal

The new oneDNN interface for in-place Binary post-ops is a boolean primitive
attribute, modelled on the existing `deterministic` attribute:

```c
/// Returns the post-ops DST aliasing attribute value.
///
/// @param attr Primitive attributes.
/// @param value Output post-ops DST aliasing attribute value.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_post_ops_may_alias_dst(
        const_dnnl_primitive_attr_t attr, int *value);

/// Sets the post-ops DST aliasing attribute value.
///
/// @param attr Primitive attributes.
/// @param value Boolean value to set the post-ops DST aliasing attribute.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_post_ops_may_alias_dst(
        dnnl_primitive_attr_t attr, int value);
```

Here, `int value` is a boolean flag (for lack of `bool` in ANSI C). The C++ API
mirrors the same convention as `get_deterministic()`/`set_deterministic()`:

```cpp
/// Returns the post-ops DST aliasing attribute value.
bool primitive_attr::get_post_ops_may_alias_dst() const;

/// Sets the post-ops DST aliasing attribute value.
///
/// @param value Specified post-ops DST aliasing mode.
void primitive_attr::set_post_ops_may_alias_dst(bool value);
```

The attribute is a *permission*, not a request: setting it tells the primitive
that one or more post-op input buffers may alias `DNNL_ARG_DST`, and therefore
that DST may not be used as intermediate storage. It does not oblige the user
to actually alias anything, and an implementation that accepts the attribute
must produce correct results whether or not aliasing occurs at execution.

`..._post_ops_dst_alias` is an equally acceptable spelling; `may_alias_dst` is
used here to emphasize that the attribute grants permission rather than
describing a layout.

In general, no primitive should care which of the post-ops reference DST; all
that matters is whether any of them may, since the whole post-op block gets
executed after the main primitive body had already finished execution but
before DST is written to. This is also what makes the attribute applicable to
PReLU and to a potential `post_ops_t::ternary_t` without further API changes.

From a semantical standpoint, this is how the in-place Binaries differ from the
regular ones:

```
ACC = ACC + RHS — binary_add
ACC = ACC × RHS — binary_mul
ACC = ACC / RHS — binary_div
ACC = ACC + DST — Sum
ACC = ACC + DST — binary_add & inplace
ACC = ACC × DST — binary_mul & inplace
ACC = ACC / DST — binary_div & inplace
…
```

— where DST is the destination buffer, ACC is the accumulator register block
within the primitive, and RHS is the separate right-hand-side buffer used in
regular Binary post-ops.

Keep in mind that DST is not intended to get rewritten until after all post-ops
are applied, so for primitives' implementations where that cannot be guaranteed
there should be either no support for in-place post-ops or some memory movement
to guard the DST data against the intermediate writes.

A post-op buffer that is to alias DST — most commonly the one described by
`user_src1_desc` — must be declared with a memory descriptor that equals the
descriptor of `DNNL_ARG_DST` in shape, and has a type of the same bit size as
that of DST, though not necessarily the same type (e.g. `u8` vs. `f8_e5m2`).
Its layout has to match that of DST as well, but it need not be spelled out:
declaring it as `format_kind::any` is both allowed and preferable, since it then
resolves to whatever layout the primitive picks for DST — see Applicability.

Quite naturally, this means that an in-place Binary cannot be broadcast, as
that would break the MD equality requirement.

Since the attribute does not name a specific post-op, the library has no way to
tell at primitive creation which buffer the user intends to alias, and thus no
way to reject a mismatch. These requirements are therefore a contract on the
user rather than a validated constraint: passing DST as a post-op buffer whose
descriptor does not satisfy them is undefined behavior.

## Implementation

Internally, a boolean member of `dnnl_primitive_attr` is proposed, following
`deterministic_` in both placement and handling:

```
 struct dnnl_primitive_attr {
 
 …
 
     bool deterministic_;
+    bool post_ops_may_alias_dst_;
     dnnl::impl::post_ops_t post_ops_;
 
 …
 
 }
```

Being an attribute rather than a post-op field, it has to be threaded through
the same places `deterministic_` is, none of which are optional:

* the default constructor, where it is initialized to `false`;
* `copy_from()`, or the flag is dropped by `clone()` and by copy construction;
* `operator==()`, or attributes differing only in this flag compare equal;
* `primitive_hashing::get_attr_hash()`, or aliasing and non-aliasing primitive
  descriptors collide in the primitive cache;
* `primitive_serialization.cpp`, for the same reason on the serialized path;
* `verbose.cpp`, as `attr-post-ops-may-alias-dst`, so that verbose output can
  be converted back into a reproducer.

Handling `copy_from()` does not, however, resolve the nested-primitive concern
raised for Option 4. Implementations that build a sub-primitive do not generally
clone the parent attributes; they create an empty `primitive_attr_t` and fill in
only what the sub-primitive needs. Such a sub-primitive gets the flag cleared by
default, and is then free to use DST as intermediate storage — which is exactly
what the flag exists to forbid. Every site that constructs attributes for a
nested primitive whose DST is the parent's DST therefore has to propagate the
flag explicitly, and omitting one is a silent correctness bug rather than a
missed optimization. This is a real cost of the attribute-level approach, and it
falls on the library rather than on the user.

The primitive may then consult the attribute directly. No scan of the post-op
entries is required: the attribute alone determines whether DST may be read,
and an implementation that wants to skip the associated restrictions when no
buffered post-op is present can check `post_ops_` on its own.

With the attribute set, on execution the primitive should expect a reference to
the `DNNL_ARG_DST` buffer passed as any buffer of any post-op that features
additional buffers, provided that buffer satisfies the requirements outlined in
the proposal: same shape and layout, same-size type.

That way, no aspect of the existing Binary post-op pipeline is to be altered
in any form, since DST is, after all, just another regular buffer. This is not
a hypothetical: the Intel GPU JIT implementations already treat Sum this way
internally. In the v2 Conv builder, Sum is dispatched as `alg_kind::binary_add`
with the destination buffer supplied as the right-hand side, through the same
code path as a regular Binary post-op; the v1 IR builder, used in Conv, Deconv,
Pool, and Reorder, likewise imports DST as an ordinary post-op input tensor.
JIT GEMM is the exception, where Sum is a separate codepath rather than an
injected post-op.

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
|-----------|------------------------|----------------------|---------------------|
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

An in-place Binary reads DST as an ordinary right-hand-side buffer. Any data
type valid for a regular Binary RHS, subject to the same-bit-size requirement
above, is valid here, including all of the low-precision types (`u8`, `s8`,
`s4`, the `f8_*` family, etc.) — and it being low precision does not imply it
needs dequantization, just like with regular RHS buffers for Binary post-ops.
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

```
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
+    post_ops_may_alias_dst = (unsigned)post_ops | (1u << 19),
 };
```

That way in-place post-ops become an opt-in feature, so by default no primitive
would accept them unless the implementation allows it explicitly.

Note that whether the feature is applicable may also depend on decisions an
implementation only makes during primitive descriptor initialization, such as
the chosen blocking, which happens after the attributes have been checked. This
does not have to turn into a failure to create the primitive, though. Post-op
memory descriptors may be declared with `format_kind::any`, and
`post_ops_t::set_default_formats()` already resolves such a descriptor against
the DST layout the primitive eventually settles on, initializing it from the DST
blocking descriptor. An aliased buffer declared this way matches DST by
construction, no matter what a primitive that picks its own DST layout — GPU
Deconvolution, say — ends up choosing, so no Sum-like special casing is needed
and the behavior is not asymmetric with Sum in this respect.

Creation fails only if the user pins a concrete layout that the primitive does
not choose, which is the pre-existing behavior of any regular Binary post-op and
is under the user's control.

## Interface for benchdnn

Unlike the API, benchdnn has to decide not only whether aliasing is permitted
but also which buffers are actually aliased at execution. That cannot be derived
from the memory descriptors: a buffer intended for aliasing need not have the
same type as DST, only one of the same size, while a buffer that happens to have
DST's exact type need not be intended for aliasing at all. Selecting buffers by
type is therefore a policy, not an inference, which leaves a few options:

1. Add an `inplace` keyword after the Binary type, per post-op:
    ```
    --attr-post-ops=mul:bf16:inplace+add:bf16 …
    ```
    * Pros: Expresses exactly which buffers are aliased, and is the only form
    that covers the per-post-op combinations — with two Binaries there are four
    distinct cases (first, second, both, neither) and this syntax names each of
    them. No policy needs to be invented, since the run line states the intent.
    * Cons: Not 1-to-1 with the API, which carries a single global attribute and
    no per-post-op information — so the keyword describes the harness rather
    than the attribute under test, and may be read as implying a per-post-op API
    that does not exist. Extends the post-op grammar itself, which is shared by
    every post-op type, rather than adding a self-contained knob.
2. Add a global boolean knob, following `--attr-deterministic=BOOL`:
    ```
    --attr-post-ops-may-alias-dst=true …
    ```
    * Pros: 1-to-1 with the API attribute, and reuses the established `--attr-*`
    conventions. Post-op parsing is left untouched. The cheapest of the three.
    * Cons: Cannot express which buffers alias DST, so the policy has to be
    fixed in benchdnn itself — in practice "alias every eligible buffer" — and
    the per-post-op combinations stay untestable.
3. Add the same global knob, with an optional list of post-op indices:
    ```
    --attr-post-ops-may-alias-dst=true:0+2 …
    ```
    * Pros: Keeps the 1-to-1 mapping with the API for the attribute itself while
    still covering the per-post-op cases; the index list is plainly a
    harness-side selection rather than part of the attribute.
    * Cons: More syntax than option 2 for a knob whose default form is expected
    to cover most testing, and the indices are positional, which makes run lines
    more fragile to edit than named keywords.

All three need a new verbose token and documentation updates; they differ in
whether the post-op grammar or the attribute list is the thing being extended.
Options 1 and 3 are the only ones that cover the per-post-op combinations, while
option 2 is the cheapest and matches the existing conventions most closely. As
this affects validation only and carries no API implications, the choice can be
settled during implementation.

## Conclusion

In-place Binary post-ops are a powerful tool available at a modest cost to the
library: a single primitive attribute, an opt-in skip mask per implementation,
and the discipline of propagating the attribute into nested primitives. In
return they enable memory savings and new approaches to DNN graph optimization.
