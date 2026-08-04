# In-place binary post-ops: sum -> binary_add(dst), and more

## Overview

The Sum post-op in oneDNN is not like the other post-ops: it performs a binary
operation but requires no right-hand-side buffer, instead taking its input from
DST. This behavior can be replicated in all of the Binary post-ops, extending
the buffer semantics of Sum to all other Binaries and eventually deprecating
Sum.

The need for this, beside purely architectural, is also practical: many modern
LLMs like LLaMa, especially in OpenVINO, contain 3-GEMM subgraphs that can
benefit from a variant of Sum that performs multiplication instead of addition,
potentially saving dozens of % of E2E VRAM compared to regular binary_mul which
inevitably requires an RHS buffer that should exist in VRAM beside DST.

The alternative would be to add a new one-off Sum-like PO, let's call it Prod.
That would essentially duplicate the already very different codepath for Sum
in all of the primitives, whereas the changes to Binary would be benign at
worst, simultaneously bringing a lot more value (in the form of way more ops
than just Sum or Prod) to the table; the actual Binary post-op code isn't going
to change at all, the most labor-intensive part would be to determine the
primitives implementations and configurations that alter DST before the final
write, and exclude them from the support matrix.

## Proposal

The new oneDNN interface for in-place Binary post-ops should look like this:

```
dnnl_status_t DNNL_API dnnl_post_ops_set_allow_inplace_bin(
        dnnl_post_ops_t post_ops, int value);

dnnl_status_t DNNL_API dnnl_post_ops_get_allow_inplace_bin(
        const_dnnl_post_ops_t post_ops, int *value);
```

Here, `int value` is a boolean flag (for lack of `bool` in ANSI C) that signals
that the user wants to add 1 or more Binary post-ops that read `DNNL_ARG_DST`.

In general, no primitive should care which of the Binary post-ops reference
DST; all that matters is the presence or absence of such Binaries, since the
whole post-op block gets executed after the main primitive body had already
finished execution but before DST is written to.

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

When adding an in-place Binary, `user_src1_desc` should equal the memory
descriptor of `DNNL_ARG_DST` both in shape and in type, and `user_src2_desc`
should equal `nullptr`, otherwise the post-op is to be considered malformed.

## Implementation

Internally, a boolean flag within `dnnl_post_ops` is proposed; its name is up
for debate, but here it is proposed to be `allow_inplace_bin_`. Its helper
method, `bool allow_inplace_bin()`, would tell the caller whether to expect
in-place Binaries:

```
 struct dnnl_post_ops {
 
 …
 
     std::vector<entry_t> entry_;
+    bool allow_inplace_bin_ = false;
 
 …
 
+    bool allow_inplace_bin() const {
+        if (!allow_inplace_bin_) return false;
+        for (auto &e : entry_)
+            if (e.is_like_binary()) return true;
+        return false;
+    }
 
 …
 
 }
```

With this flag set, on execution the primitive that contains this as `i`-th
post-op should expect a reference to the `DNNL_ARG_DST` buffer passed to it in
`DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1`.

That way, no aspect of the existing Binary post-op pipeline is to be altered
in any way, since DST is, after all, just another regular buffer.

## Caveats

Not all of the oneDNN primitives can support in-place Binary post-ops — namely
those that write intermediate data to DST, e.g. when accumulating with atomics.
To guard against improper usage, a new `skip_mask_t` flag is to be introduced:

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
+    inplace_binary_post_ops = (unsigned)post_ops | (1u << 19),
 };
```

That way in-place post-ops become an opt-in feature, so by default no primitive
would accept them unless the implementation allows it explicitly.

## Interface for benchdnn

The benchdnn interface for in-place Binary post-ops is fairly strightforward.
Just adding an `inplace` keyword after the Binary alg name should be enough:

```
… --attr-post-ops=mul:inplace …
```

## Conclusion

In-place Binary post-ops are a powerful tool available at virtually no cost to
the library; they enable significant memory savings and new approaches to DNN
graph optimization.
