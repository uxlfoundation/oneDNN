# Proposal to Extend Layer Normalization to Support Root Mean Square Normalization

## Introduction

Root Mean Square Normalization (RMSNorm) is a normalization technique similar
to Layer Normalization (LayerNorm) but computationally more efficient.

In oneDNN, LayerNorm computes the mean and variance using the following formulas:
```math
\mu(t, n) = \frac{1}{C} \sum_{c} src(t, n, c)
```
```math
\sigma^2(t, n) = \frac{1}{C} \sum_{c} (src(t, n, c) - \mu(t, n))^2
```

It then normalizes the input using these statistics:
```math
dst(t, n, c) = \frac{src(t, n, c) - \mu(t, n)}{\sqrt{\sigma^2(t, n) + \epsilon}}
  * \gamma(c) + \beta(c)
```
Here `gamma` and `beta` are optional scale and shift parameters, and `epsilon`
is a small constant to prevent division by zero.

The purpose of computing the mean and variance is to re-center and re-scale the
input data, which helps improve stability and convergence.

RMSNorm, on the other hand, skips the re-centering step and normalizes using only
the root mean square statistic (`RMS`):
```math
RMS(t, n) = \sqrt{\frac{1}{C} \sum_{c} src(t, n, c)^2 }
```

This results in a simpler normalization formula:
```math
dst(t, n, c) = \frac{src(t, n, c)}{\sqrt{\frac{1}{C} \sum_{c} src(t, n, c)^2 + \epsilon}} * \gamma(c) + \beta(c)
```

The `epsilon` is added to avoid further dividing by zero.
Note that it is not always the part of the definition for `RMS` statistics.

TLDR; RMSNorm could be considered a simplified version of
LayerNorm where the mean is assumed to be zero. As a result, the variance is
equal to `RMS^2`, and moreover if the input data is such that it has a mean of zero,
RMSNorm and LayerNorm produce the same results.

For more details, including comparisons with LayerNorm and examples of benefits,
refer to the article by [Biao Zhang and Rico Sennrich](https://arxiv.org/abs/1910.07467).
This article provides more details regarding the method and showcases some experiments
using frameworks like TensorFlow and PyTorch.

Additionally, consider [Keras RMS Normalization Layer](https://github.com/keras-team/keras/blob/v3.9.2/keras/src/layers/normalization/rms_normalization.py)
and [PyTorch RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) APIs.

Note that this proposal currently does not cover the [Keras Layer Normalization](https://keras.io/api/layers/normalization_layers/layer_normalization/)
with the `rms_scaling=True` option, see [keras#21234 Issue](https://github.com/keras-team/keras/issues/21234).

## Proposal

The proposal is to extend the LayerNorm primitive to support RMSNorm
as this is a subset of existing functionality computations-wise.
As for other Normalization primitives, the proposal is to not extend them at this point.

For the work-in-progress POC, refer to [PR 3068](https://github.com/uxlfoundation/oneDNN/pull/3068).
For more details about the proposal, see the following sections.

Note, that previously, if users wanted to implement RMSNorm, they had to rely on the `dnnl_use_global_stats` flag,
manually set the mean to zero, and compute the `RMS^2` statistics themselves.
This approach is not convenient and adds extra complexity (computations, memory overhead) for the users.

### Extension to the Normalization Flags

This support of RMSNorm be achieved by adding a new flag to the existing `dnnl_normalization_flags_t`
enum and `normalization_flags` enum in the C++ API:
```cpp
/// Flags for normalization primitives.
enum class normalization_flags : unsigned {
  <old flags...>

  /// Use Root Mean Square (RMS) Layer Normalization.
  /// In forward propagation, this means that the mean is set to zero, and RMS
  /// is used instead of variance.
  /// The RMS norm is provided as an output during forward propagation for
  /// training.
  /// In backward propagation, the library computes the derivative with respect
  /// to the RMS norm, assuming that the mean is zero.
  rms_norm = dnnl_rms_norm,
};
```

```c
/// Flags for normalization primitives.
typedef enum {
  <old flags...>

  /// Use RMS normalization
  ///
  /// If specified:
  ///  - on forward propagation use RMS normalization instead of true mean and
  ///    variance
  ///  - on backward propagation assume mean as zero
  dnnl_rms_norm = 0x20U,
} dnnl_normalization_flags_t;
```

The new flag `dnnl_rms_norm` will be used to indicate that the RMS
normalization is used instead of the standard Layer Normalization.

**Naming Option 2**:
Alternative namings for the flag could signify that the mean is **assumed** to
be zero, for instance `dnnl_{use_zero, no}_mean`. But this seems misleading
as the mean is not explicitly set to zero, but simply ignored. Also it doesn't
explciitly indicate what is happening with the variance.

#### Compatibility with Existing Flags

The combination of `dnnl_use_global_stats` and `dnnl_rms_norm` will be supported.
However, users must provide pre-computed RMS norm statistics
(instead of both mean and variance as required for Layer Normalization)
using the `DNNL_ARG_VARIANCE` index.

Since RMSNorm is a simplified version of LayerNorm, it works with `dnnl_use_scale`
and `dnnl_use_shift` flags in the same manner.

Additionally, like LayerNorm, it will not support `dnnl_fuse_norm_relu` or `dnnl_fuse_norm_add_relu`.

### Changes to Layer Normalization Primitive Inputs and Outputs

Since the main purpose of RMSNorm is to avoid the overhead of mean computations,
it seems reasonable to omit the mean as input/output and use `RMS^2` in place of the variance, which means:

- For `dnnl_forward_training`, the mean will not be included as an output when the `dnnl_rms_norm`
flag is enabled. Additionally, the output will provide `RMS^2` in place of the variance.
- For `dnnl_backward`, the mean will not be required as an input. For the variance input
it is expected that `RMS^2` will be supplied instead of the true variance.

### Changes to benchdnn

An additional flag for LayerNorm to enable RMSNorm will be added to the `--flags` option.

```
`--flags=[|G|C|H|M]` -- layer normalization flags, default `none`; where
            multiple simultaneous flags are supported.
            `G` is dnnl_use_global_stats;
            `C` is dnnl_use_scale;
            `H` is dnnl_use_shift;
            `M` is dnnl_rms_norm;
            ...
```

Note: If the flag is named `dnnl_{use_zero, no}_mean`, the option will be updated to `--flags=[|G|C|H|Z]`.

Other internal changes would be captured in [PR 3068](https://github.com/uxlfoundation/oneDNN/pull/3068).

## Open Questions

N/A, everything was discussed and agreed upon.
