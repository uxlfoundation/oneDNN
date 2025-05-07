# User-Supplied Precomputed Zero Points for GEMM

## Introduction

For modern LLM networks, which usually are a bunch of relatively small GEMMs, performance is priority #1.
To optimize LLMs for performance (i.e. use DPAS and DP4A, the hardware matmul instructions of Intel GPUs)
a novel approach has been proposed which is to dynamically downcast both A and B matrices from a floating
point type to INT8 — but to avoid losing too much precision the downcasting algorithm should also produce
zero points.

Of particular interest is the variation where the B matrix, or "weights", which is usually constant, gets
downcast ahead of time. The problem is that zero points on B mean that A (the "sources") is to be reduced
by the K dimension to multiply the resulting reductions to the ZPs extracted from B; see formulas below.

Calculating partial K reductions of A within the GEMM kernel would never be efficient with existing Intel
GPUs; there is, however, a way to perform them painlessly while downcasting A. The only thing that oneDNN
is missing is an argument identifier to take that reduction buffer from the user.

The general case, trivial but impractical:

```math
C_{m,n}=\sum_{k=0}^{K-1}{A_{m,k}(B_{k,n}-Z_B(k,n))}=\sum_{k=0}^{K-1}{A_{m,k}B_{k,n}-\sum_{k=0}^{K-1}A_{m,k}Z_B(k,n)}
```

The grouped B zero point case:

```math
C_{m,n}=\sum_{g=0}^{G-1}\sum_{k={K\over{G}}g}^{{K\over{G}}(g+1)-1}{A_{m,k}(B_{k,n}-Z_B(g,n))}=\sum_{k=0}^{K-1}{A_{m,k}B_{k,n}}-\overbrace{\sum_{g=0}^{G-1}Z_B(g,n)\underbrace{\sum_{k={K\over{G}}g}^{{K\over{G}}(g+1)-1}A_{m,k}}_{R_{m,g}}}^{\zeta},\;K\mathrm{mod}{G}=0
```

Where R is the matrix of partial K reductions of A, i.e. the reduction buffer introduced above.

*N.B.:* In the formula above, the ζ expression in the simplest case where G = 1, is a single product of 2
values: Z<sub>B</sub>(0,n) and R<sub>m,0</sub>. That way for each C<sub>m,n</sub> the pairs of values are
different, so precomputing all of them would mean converting the 2 vectors into an M×N matrix.

This M×N matrix can of course be added to the biases, avoiding the need for everything written below. But
the downside of that approach is a larger performance penalty, estimated 10–15% overhead vs the same GEMM
with no ZPs, compared to the estimated ≤5% overhead for the proposed approach vs the non-ZP GEMM.

## Proposal

A new `DNNL_ARG_ATTR_*` is to be added instead of the long-deprecated `DNNL_ARG_ATTR_OUTPUT_SCALES` along
with a new convenience arg:

```diff
 /// Dropout RNG seed value passed via a buffer.
 #define DNNL_ARG_ATTR_DROPOUT_SEED 511
 
-/// Output scaling factors provided at execution time.
-/// Deprecated value.
-#define DNNL_ARG_ATTR_OUTPUT_SCALES 513
+/// The argument marked with this attribute is user-precomputed.
+/// Used exclusively for zero points at the time of writing.
+#define DNNL_ARG_ATTR_USER_PRECOMP 512
 
 /// Starting index for source arguments for primitives that take a variable
 /// number of source arguments.
```

```diff
 
 /// Zero points provided at execution time.
 #define DNNL_ARG_ATTR_ZERO_POINTS 8192
+/// Convenience constant for execution-time user-precomputed zero points.
+#define DNNL_ARG_ATTR_USER_PRECOMP_ZERO_POINTS \
+    (DNNL_ARG_ATTR_USER_PRECOMP | DNNL_ARG_ATTR_ZERO_POINTS)
 
 /// Arguments for fused depthwise convolution.
 /// See @ref dev_guide_attributes_post_ops_depthwise_fusion
```

API-wise there is nothing more that should be added, as these 2 arguments are capable of doing everything
related to accepting any precomputed buffers from the user. All that's needed from the user is to bitwise
OR either of these constants with the buffer identifier, and pass the result to zero point related oneDNN
primitive attribute functions (e.g. `dnnl_primitive_attr_set_zero_points`) on PD initialization, and then
add the corresponding buffer to the list of kernel arguments on execution.

## Validation

To validate this new functionality the `benchdnn` tool has to be extended as well:

```diff
     --attr-deterministic=BOOL
     --attr-dropout=PROBABILITY[:SEED[:TAG]]
     --attr-scales=ARG:POLICY[:SCALE[:DATA_TYPE[:GROUPS]]][+...]
-    --attr-zero-points=ARG:POLICY[:ZEROPOINT[:DATA_TYPE[:GROUPS]]][+...]
+    --attr-zero-points=ARG:POLICY[:ZEROPOINT[:DATA_TYPE[:GROUPS[:PRECOMP]]]][+...]
     --attr-post-ops=SUM[:SCALE[:ZERO_POINT[:DATA_TYPE]]]
                     ELTWISE[:ALPHA[:BETA[:SCALE]]]
                     DW:KkSsPp[:DST_DT]
```

Where `PRECOMP` is a boolean value (`false` by default) meaning that this buffer's partial reductions are
to be precomputed by the user if `true`.
