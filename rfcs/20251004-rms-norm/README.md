# Proposal to Extend Layer Normalization to Support Root Mean Square Normalization

## Introduction

Root Mean Square Normalization (RMSNorm) is a normalization technique similar
to Layer Normalization (LayerNorm) but computationally more efficient.

In OneDNN, LayerNorm computes the mean and variance using the following formulas:
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
RMS(t, n) = \sqrt{\frac{1}{C} \sum_{c} src(t, n, c)^2 + \epsilon}
```
The `epsilon` is added to avoid further dividing by zero.
Note that it is not always part of the definition for `RMS` statistics.

This results in a simpler normalization formula:
```math
dst(t, n, c) = \frac{src(t, n, c)}{RMS(t, n)} * \gamma(c) + \beta(c)
```

In short, RMSNorm could be considered a simplified version of
LayerNorm where the mean is assumed to be zero. As a result, the variance is
equal to `RMS^2`, and moreover if the input data is such that it has a mean of zero,
RMSNorm and LayerNorm produce the same results.

For more details, including comparisons with LayerNorm and examples of benefits,
please refer to the article by [Biao Zhang and Rico Sennrich](https://arxiv.org/abs/1910.07467).
This article provides more details regarding the method and showcases experiments
using frameworks like TensorFlow and PyTorch.

Additionally, consider TensorFlow
[LayerNorm with RMS parameter](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)
and PyTorch [RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html).

## Proposal

The proposal is to extend the existing LayerNorm primitive to support RMSNorm
as this is a subset of existing functionality computations-wise.

As for other Normalization primitives, the proposal is to not extend them.

For the work-in-progress POC, refer to [PR 3068](https://github.com/uxlfoundation/oneDNN/pull/3068).
For more details about the proposal, see the following sections.

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
  /// This flag is incompatible with #dnnl_use_global_stats.
  use_rms_norm = dnnl_use_rms_norm,
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
  /// Not compatible with #dnnl_use_global_stats.
  dnnl_use_rms_norm = 0x20U,
} dnnl_normalization_flags_t;
```

The new flag `dnnl_use_rms_norm` will be used to indicate that the RMS
normalization is used instead of the standard Layer Normalization.

Alternative namings for the flag could signify that the mean is **assumed** to
be zero, for instance `dnnl_{use, force}_zero_mean`. But this seems misleading
as the mean is not explicitly set to zero, but simply ignored. Also it doesn't
explciitly indicate what is happening with the variance.

#### Compatibility with Existing Flags

The `dnnl_use_global_stats` flag technically could be applied to RMSNorm, but
it is not particularly meaningful since RMSNorm never requires the mean computations.
Additionally, if previously users needed RMSNorm, they could have used the
`dnnl_use_global_stats` flag, providing a zeroed mean, and computing the `RMS^2` statistics
themselves, meaning that from the user's point of view, the difference with the
`dnnl_use_rms_norm` flag is that it eliminates the need for mean allocation.
Therefore the proposal is to make it incompatible with the `dnnl_use_global_stats` flag.

Since RMSNorm is a simplified version of LayerNorm, it works with `dnnl_use_scale`
and `dnnl_use_shift` flags in the same manner. Additionally, like LayerNorm,
it does not support `dnnl_fuse_norm_relu` or `dnnl_fuse_norm_add_relu`.

### Changes to Layer Normalization Primitive Inputs and Outputs

Since the primary purpose of RMSNorm is to avoid the overhead of mean computations,
it seems reasonable to omit the mean as input/output and use `RMS^2` in place of the variance, which means:

- For `dnnl_forward_training`, the mean will not be included as an output when the `dnnl_use_rms_norm`
flag is enabled. Instead, the output will provide `RMS^2` in place of the variance.
- For `dnnl_backward`, the mean will not be required as an input. For the variance input
it is expected that `RMS^2` will be supplied instead of the true variance.

### Changes to Primitive Descriptor

TBD

### Changes to benchdnn

TBD

## Open Questions

TBD