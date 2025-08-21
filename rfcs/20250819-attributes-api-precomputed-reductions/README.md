# User's Precomputed Reductions for GeMM

## Introduction

To optimize LLMs' performance (i.e., use DPAS and DP4A, the hardware GeMM
instructions of Intel GPUs) a novel approach has been proposed which is to
dynamically downcast both A and B matrices from a floating point type to INT8 —
but to avoid losing too much precision the downcasting algorithm should also
produce zero points in certain scenarios.

One of the flavors is the B matrix, or "weights", which is usually constant,
gets downcasted ahead of time. The issue is zero points on B mean the A matrix,
or "activations", must be reduced by the K dimension to multiply the resulting
reductions to the ZPs extracted from the B matrix; see formulas below.

Calculating partial K reductions of A inside the GeMM kernel can never be
efficient with existing Intel GPUs. However, they can be obtained when
downcasting A happens. Given that downcasting A should happen on the user side,
the library can provide a way to accept those reductions to maintain the correct
output and better performance.

The general case, trivial but impractical:

```math
C_{m,n}=\sum_{k=0}^{K-1}{A_{m,k}(B_{k,n}-Z_B(k,n))}=\sum_{k=0}^{K-1}{A_{m,k}B_{k,n}-\sum_{k=0}^{K-1}A_{m,k}Z_B(k,n)}
```

The grouped B zero point case:

```math
C_{m,n}=\sum_{g=0}^{G-1}\sum_{k={K\over{G}}g}^{{K\over{G}}(g+1)-1}{A_{m,k}(B_{k,n}-Z_B(g,n))}=\sum_{k=0}^{K-1}{A_{m,k}B_{k,n}}-\overbrace{\sum_{g=0}^{G-1}Z_B(g,n)\underbrace{\sum_{k={K\over{G}}g}^{{K\over{G}}(g+1)-1}A_{m,k}}_{R_{m,g}}}^{\zeta},\;K\mathrm{mod}{G}=0
```

, where R is the matrix of partial K reductions of A, i.e. the reduction buffer
introduced above.

*N.B.:* In the formula above, the `ζ` expression in the simplest case where
`G = 1`, is a single product of 2 values: $Z_B(0,n)$ and
$R_{m,0}$. That way for each $C_{m,n}$ the pairs of values are
different, so precomputing all of them would mean converting the 2 vectors into
an M by N matrix, which can be appended as bias, avoiding the extra
manipulations written below. But this approach results in the same performance
as doing zero-points math conventionally which is far below the target.

Besides, the same A matrix with partial K reductions must be reusable with
multiple B matrices that might have different zero-points application, e.g. with
different group sizes over K.

## Proposal

The suggestion is to provide a new attribute which resembles zero-points/scales
but provides its own signature and configuration for the library internals.

```c
// dnnl.h

// Same signature as for zero-points/scales.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_precomputed_reductions(
        dnnl_primitive_attr_t attr, int arg, int mask, int group_ndims,
        const dnnl_dims_t group_dims, dnnl_data_type_t data_type);

// dnnl.hpp
void set_precomputed_reductions(int arg, int mask,
        const memory::dims &groups,
        memory::data_type data_type = memory::data_type::s32);

// dnnl_types.h
#define DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS 512
```

The following limitations to be applied:
* Can't be used without weights zero-points specified.
* Must have full A matrix mask, e.g., standard M by K times K by N MatMul should
  have the mask of 3 for this feature, meaning broadcast is not supported.

This design keeps in mind the potential extension for zero-points compensation
functionality on CPU which is limited to weights format kind "any" and can't be
applied for any other format, while this API can help to get that case
supported.

### Alternative design

The alternative design approach was to extend the primitive descriptor signature
with an additional flag. It would deduce the grouping argument, the most
critical one, based on other attributes, which doesn't provide flexibility of
choice to the user. In case the deduced value doesn't coincide with what the
user had computed, it would make the feature unusable. It would also be
non-compliant with the reusability requirement. On top of that, it costs more to
extend the signature and doesn't provide scalability to apply it to, for example
convolutions.

## Performance

As performance is the main motivation for this new API, the broad coverage
numbers obtained with a proposed changed may be found in
[this PR](https://github.com/uxlfoundation/oneDNN/pull/3750). The API speeds up
cases with int8 weights and zero-points. int4 cases apply zero-points on-the-fly
and don't benefit from the feature.

Here are just couple lines for better visualization:

| Case | Pure F16, ms | W/o reductions, ms | W/ reductions, ms |
|:-----|:-------------|:-------------------|:------------------|
| (1)  | 2.45         | 7.32               | 1.44              |
| (2)  | 0.015        | 0.013              | 0.007             |

(1): `--dt=s8:u8:f16 --stag=ab --wtag=ba --dtag=ab --attr-scales=src0:per_tensor:f16:1x128+wei:per_oc:f16 --attr-zero-points=wei:per_oc:u8 --attr-precomputed-reductions=src:per_tensor:s32:1x128 --attr-scratchpad=user 2172x4096:4096x14336`
(2): `--bia_mask=2 --bia-dt=f16 --dt=s8:u8:f16 --stag=ab --wtag=ba --dtag=ab --attr-scales=src0:per_tensor:f16:1x128+wei:per_oc:f16 --attr-zero-points=wei:per_oc:u8 --attr-precomputed-reductions=src:per_tensor:s32:1x128 --attr-scratchpad=user 31x2560:2560x2560`

