Quantization {#dev_guide_attributes_quantization}
=================================================

@anchor dgaq_intro
## Introduction

Some primitives support input and output tensors with INT8 data types,
both signed and unsigned, enabling reduced-precision inference on
supported hardware.

Similarly, some primitives support OFP8-compliant f8 types (8-bit
floating-point formats) designed to accelerate AI workloads, including
training and inference of large neural networks. Lowering precision to
8 bits with f8 enables faster computation and reduced memory usage.

Related materials:
- [Lower Numerical Precision Deep Learning Inference and Training](https://www.intel.com/content/dam/develop/external/us/en/documents/lower-numerical-precision-deep-learning-jan2018-754765.pdf)
- INT8 example with annotations: @ref dev_guide_inference_int8
- f8 example with annotations: @ref matmul_f8_quantization_cpp
- [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)

## Quantization Model

The primary quantization model that the library assumes is the following:
\f[
    x_{f32}[:] = scale_{x} \cdot (x_{int8}[:] - zp_{x})
\f]

where \f$scale_{x}\f$ is a *scaling factor* in float format,
\f$zp_{x}\f$ is the *zero point* in int32 format, and
\f$[:]\f$ is used to denote elementwise application of the formula
to the arrays. In order to provide best performance, oneDNN does not
compute those scaling factors and zero-points as part of primitive
computation. Those should be computed and provided by the user.

These quantization parameters can either be computed ahead of time
using calibration tools (*static* quantization) or at runtime based on
the actual minimum and maximum values of a tensor (*dynamic*
quantization). Either method can be used in conjunction with oneDNN, as
the quantization parameters are passed to the oneDNN primitives at
execution time.

To support int8 quantization, primitives should be created and
executed as follow:

- during primitive descriptor creation, if one or multiple inputs are
  int8 (signed or not), then the primitive will behave as a quantized
  integer operation.
- still during primitive descriptor creation, the dimensionality of
  the scaling factors and zero-point should be provided using masks
  (e.g. one scale per tensor, one scale per channel, ...).
- finally, during primitive execution, the user must provide the
  actual quantization parameters as arguments to the execute function.
  Scales are `f32` values, and zero-points are `s32` values.

For performance reasons, each primitive implementation typically
supports only a subset of quantization parameter masks. For example,
convolution typically supports per-tensor or per-channel scales (no
zero-point) for weights, and per-tensor scaling factor and zero-points
for activation.

This guide does not cover how the appropriate scaling factor can be found.
Refer to the materials in the [Introduction](@ref dgaq_intro).

### Numerical behavior

Primitive implementations are allowed to convert int8 inputs to wider
datatypes (e.g. int16 or int32), as those conversions do not impact
accuracy.

During execution, primitives implementations avoid integer overflows
and maintain integer accuracy by using wider datatypes (e.g. int32)
for intermediate values and accumulators. Those are then converted as
necessary before the result is written to the output memory objects.

When converting to integral datatypes, implementations typically
saturate, whereas for floating-point datatypes, underflow/overflow can
occur. To force saturation in floating-point datatypes use
@ref dev_guide_attributes_post_ops_eltwise with clip algorithm.

@warning
Depending on the architecture, the behavior of int8 computations might slightly
vary. For more details, refer to @ref dev_guide_int8_computations.

When multiple operations are fused in a single primitive using the
[post ops attribute](@ref dev_guide_attributes_post_ops), those are assumed to be
computed in f32 precision. As a result the destination quantization
parameters are applied after the post-ops as follow:

\f[
   \dst[:] = post\_ops(OP(src[:], weights[:], ...)) / scale_{\dst} + zp_{\dst}

\f]

Quantizing/dequantizing values between post-operations can still be
achieved using one of [eltwise](@ref dev_guide_attributes_post_ops_eltwise),
[binary](@ref dev_guide_attributes_post_ops_binary), or the scale parameter of
the appropriate post-operation.

## Quantization APIs: Scaling, Zero-Points, and Precomputed Reductions

The library API to support for int8 was designed for the model described above.
However, it does not require users to follow exactly this model. As long as
users can fit their model into the given functionality everything should work
fine. Having this in mind we tried to design a minimal and simple yet powerful
enough quantization API.

The most common data type for data tensors during int8 inference is
#dnnl::memory::data_type::s8 and #dnnl::memory::data_type::u8. All the
scaling factors related to tensors are not attached in any way to the
oneDNN memory objects and should be maintained by users.

The library essentially extends the ability of the primitives to scale the
output before storing the result to the memory with the destination data type.
That's exactly the minimum that we need to support int8 inference (check the
equations above--only \f$output\_scale\f$ is non-standard).

The scaling happens in the single precision floating point data type
(#dnnl::memory::data_type::f32). Before storing, the result is downconverted
to the destination data type with saturation if required. The rounding happens
according to the current HW setting (for instance, on CPU according to the
MXCSR register).

### Argument Scaling

The library uses @ref dev_guide_attributes API for setting the scaling factors
for most of the primitives. The supporting attributes can be found in the
documentation for each primitive. The unsupported cases are handled according
to the
[attributes error handling section](@ref dev_guide_attributes_error_handling).

#### Available Scaling API Methods

oneDNN provides the following methods for setting scaling factors:

~~~cpp
// Legacy method with simple mask-based scaling
void dnnl::primitive_attr::set_scales_mask(int arg, int mask);

// Generic method with groups support
void dnnl::primitive_attr::set_scales(int arg, int mask,
                                      const dnnl::memory::dims &groups,
                                      dnnl::memory::data_type data_type = dnnl::memory::data_type::f32,
                                      bool is_on_host = false);

// Convenience method for single host-side scalar
void dnnl::primitive_attr::set_host_scale(int arg,
                                          dnnl::memory::data_type data_type = dnnl::memory::data_type::f32);
~~~

##### Concepts

Argument identifiers (`arg`) specify which tensor (primitive input/output) to scale:
- `DNNL_ARG_SRC`: Source tensor
- `DNNL_ARG_WEIGHTS`: Weight tensor
- `DNNL_ARG_DST`: Destination tensor
- `DNNL_ARG_BIAS`: Bias tensor (limited support)

Mask (`mask`) controls which dimensions get individual scaling factors:
- `0`: Single scaling factor for entire tensor (global scaling)
- `1 << dim`: Scaling factors vary along dimension `dim`
- `(1 << dim1) + (1 << dim2)`: Scaling factors vary along multiple dimensions

Groups (`groups`) divide dimensions into blocks for block-wise quantization:
- `{}`: No grouping (default)
- `{G}`: Single group
- `{G1, G2, ...}`: Multi-dimensional grouping

The scaling parameters support multiple data types to accommodate
different quantization workflows and precision requirements:
- `f32`
- `bf16`, `f16`
- `f8_e5m2`, `f8_e4m3`
- `e8m0`

Additionally, scaling factors can be specified as residing on host or device memory
(refer to [the section below](@ref host-side-scalars-and-zero-points) for
more details):
- `is_on_host = false`: Scaling factor values are in device memory
- `is_on_host = true`: Scaling factor values are in host memory


#### Supported Scaling Patterns

oneDNN supports several scaling patterns to support different quantization
schemes.

* **Global scaling** (`mask=0`) uses a single scaling factor for the entire
  tensor, making it the simplest approach.
* **Per-channel scaling** (`mask=1<<dim`) applies different scaling factors
  along a specific dimension, commonly used for CNN weights.
* **Multi-dimensional scaling** (`mask=(1<<dim1)+(1<<dim2)`) provides
  independent scaling factors along multiple tensor dimensions, useful for complex
  activations where both batch and channel dimensions need separate scaling.
* **Group-based quantization** subdivides tensor dimensions into smaller
  blocks with individual scaling factors, important for large transformer
  models and advanced quantization techniques.

##### Global Scaling

In the simplest case, when there is only one common scaling factor the attribute changes
the op behavior from
\f[
    \dst[:] = Op(...)
\f]

to

\f[
    \dst[:] = scale \cdot Op(...).
\f]

~~~cpp
// Using full set_scales API (recommended)
attr.set_scales(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::f32,
                false /*on device*/);

// Using convenience set_host_scale API for single host-side scalar
attr.set_host_scale(DNNL_ARG_SRC, dnnl::memory::data_type::f32);

// Using legacy set_scales_mask API
attr.set_scales_mask(DNNL_ARG_SRC, 0);

// Tensor: [N, C, H, W] = [2, 3, 4, 4]
// Scaling factors: 1 value
// Usage: All elements use same scaling factor
~~~

@note For more details on global scaling with a single scaling factor residing on
host, use @ref host-side-scalars-and-zero-points "host-side scalar scaling"
(`set_host_scale`) to avoid device memory transfer overhead.

Global scaling is demonstrated in
[Convolution with Per-output-channel Quantization](#convolution-with-per-output-channel-quantization) below.

##### Per-Channel Scaling

Per-channel scaling applies different scaling factors along specific tensor
dimensions. For instance, it is commonly used for CNN weights where each
output channel has its own scaling factor.

~~~cpp
// Scaling factor per output channel (dimension 0 of weights)
attr.set_scales(DNNL_ARG_WEIGHTS, 1 << 0, {}, dnnl::memory::data_type::f32,
                false /*on device*/);

// Tensor: [OC, IC, H, W] = [64, 128, 3, 3]
// Scaling factors: 64 values (one per output channel)
// Usage: Each output channel gets its own scaling factor
~~~

Per-channel scaling is demonstrated in the
[Weights Preparation with Per-output-channel Scaling](#weights-preparation-with-per-output-channel-scaling) and
[Convolution with Per-output-channel Quantization](#convolution-with-per-output-channel-quantization).
It's also used in @ref inference_int8_matmul_cpp for weights quantization.

##### Group-Based Quantization

Groups enable block-wise quantization by subdividing tensor dimensions into
smaller blocks, each with its own scaling factor. This might help balance accuracy
and efficiency by providing more granular quantization than global scaling.

~~~cpp
// Weight shape: [K, N] = [1024, 512] with groups [32, 1]
// Creates 32 blocks of size [32, 512] each with its own scaling factor
std::vector<dnnl::memory::dim_t> groups = {32, 1};
attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), groups,
                dnnl::memory::data_type::f32, false);

// Tensor: [K, N] = [1024, 512]
// Scaling factors: 32 values (one per group)
// Usage: Each group gets its own scaling factor
~~~

Group-based quantization is demonstrated in
[Example 1](#example-1-matmul-with-advanced-quantization)
and [Example 2](#example-2-matmul-with-precomputed-reductions-and-advanced-quantization)
below.
See also @ref weights_decompression_matmul_cpp for a complete implementation.

##### Multi-Dimensional Scaling

Multi-dimensional scaling applies scaling factors across multiple tensor dimensions
simultaneously.

For scaling factors per dimensions \f$d_i\f$, set `mask = `\f$\sum_{d_i} 2^{d_i}\f$.

Resulting scaling factor count without groups: \f$\prod_{d_i} D_{d_i}\f$, with groups:
\f$\prod_{d_i} G_{d_i}\f$.

~~~cpp
// Scaling factors vary along batch and channel dimensions
attr.set_scales(DNNL_ARG_SRC, (1 << 0) + (1 << 1), {},
                dnnl::memory::data_type::f32, false);

// Tensor: [N, C, H, W] = [8, 64, 32, 32]
// Scaling factors needed: 8 * 64 = 512 values
// Usage: Each (batch, channel) combination gets its own scaling factor
~~~

Multi-dimensional scaling is demonstrated in
[Example 1](#example-1-matmul-with-advanced-quantization)
and [Example 2](#example-2-matmul-with-precomputed-reductions-and-advanced-quantization)
below.
See also @ref weights_decompression_matmul_cpp for a complete implementation.

### Argument Zero-Points

Zero-points handle the quantization case where the quantized integer range
does not center around zero.

The library uses @ref dev_guide_attributes API for setting zero-points for
most primitives. The supporting attributes can be found in the documentation
for each primitive. The unsupported cases are handled according to the
[attributes error handling section](@ref dev_guide_attributes_error_handling).

#### Available Zero-Point API Methods

oneDNN provides the following methods for setting zero-points:

~~~cpp
// Legacy method with simple mask-based zero-points
void dnnl::primitive_attr::set_zero_points_mask(int arg, int mask);

// Generic method with groups support
void dnnl::primitive_attr::set_zero_points(int arg, int mask,
                                          const dnnl::memory::dims &groups,
                                          dnnl::memory::data_type data_type = dnnl::memory::data_type::s32,
                                          bool is_on_host = false);

// Convenience method for single host-side scalar
void dnnl::primitive_attr::set_host_zero_point(int arg,
                                              dnnl::memory::data_type data_type = dnnl::memory::data_type::s32);
~~~

##### Zero-Point Concepts

Argument identifiers (`arg`) specify which tensor (primitive input/output) to apply zero-points:
- `DNNL_ARG_SRC`: Source tensor zero-points
- `DNNL_ARG_WEIGHTS`: Weight tensor zero-points
- `DNNL_ARG_DST`: Destination tensor zero-points

Mask (`mask`) and Groups (`groups`) follow the same semantics as scaling
factors.

Data Types (`data_type`) supported for zero-points:
- `s32`
- `s8`, `u8`
- `s4`, `u4`

Additionally, zero-point can be specified as residing on host or device memory
(refer to [the section below](@ref host-side-scalars-and-zero-points) for
more details):
- `is_on_host = false`: Zero-point value is in device memory
- `is_on_host = true`: Zero-point value is in host memory

#### Supported Zero-Point Patterns

Zero-point patterns mirror the scaling factor patterns described above. The same mask
and groups concepts apply:

- **Global zero-point** (`mask=0`): Single zero-point for entire tensor
- **Per-channel zero-points** (`mask=1<<dim`): Different zero-points per
  channel
- **Group-based zero-points** (`mask` with `groups`): Block-wise zero-points
- **Multi-dimensional zero-points** (`mask=(1<<dim1)+(1<<dim2)`):
  Independent zero-points across multiple dimensions

~~~cpp
// Global zero-point
attr.set_zero_points(DNNL_ARG_SRC, 0, {}, dnnl::memory::data_type::s32, false);

// Per-channel zero-points
attr.set_zero_points(DNNL_ARG_WEIGHTS, 1 << 0, {}, dnnl::memory::data_type::s8,
                     false);

// Group-based zero-points
std::vector<dnnl::memory::dim_t> groups = {64, 1};
attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), groups,
                     dnnl::memory::data_type::s32, false);
~~~

Zero-point usage is demonstrated in the
[Convolution with Per-output-channel Quantization](#convolution-with-per-output-channel-quantization) and
[Example 2](#example-2-matmul-with-precomputed-reductions-and-advanced-quantization)
below.
See also @ref inference_int8_matmul_cpp and @ref weights_decompression_matmul_cpp
for complete implementations.

@anchor host-side-scalars-and-zero-points
### Special Case: Host-side Scalar Scale and Zero-point

When using the GPU engine, host-side scalar scales and zero-points are
supported to reduce copying of data from host to device. A memory object
for scale or zero-point host value should be created as a host-side scalar
(see @ref dev_guide_host_side_scalars for details) and passed to the primitive
execution function. The host scales or zero-points attributes should also
be set using the following API:

~~~cpp
dnnl::primitive_attr attr;
attr.set_host_scale(DNNL_ARG_DST,
           dnnl::memory::data_type::f32);

attr.set_host_zero_point(DNNL_ARG_DST,
           dnnl::memory::data_type::s32);
~~~

See also @ref matmul_with_host_scalar_scale_cpp for a complete example.

### Precomputed Reductions

Precomputed reductions optimize performance for Large Language Model (LLM) inference
when using grouped weight zero-points with dynamic quantization. When weights have
grouped zero-points, additional reduction operations on the source tensor are required
during computation. These reductions can be pre-computed externally and provided to
the primitive, eliminating expensive operations from the critical execution path and
achieving a speedup for LLM workloads.

The library uses @ref dev_guide_attributes API for setting precomputed reductions.
The supporting attributes can be found in the documentation for each primitive.
The unsupported cases are handled according to the
[attributes error handling section](@ref dev_guide_attributes_error_handling).

#### Available Precomputed Reductions API Method

oneDNN provides the following method for setting precomputed reductions:

~~~cpp
void dnnl::primitive_attr::set_precomputed_reductions(int arg, int mask,
        const memory::dims &groups,
        memory::data_type data_type = memory::data_type::s32);
~~~

##### Precomputed Reductions Concepts

Argument identifier (`arg`):
- `DNNL_ARG_SRC`: Source tensor reductions

Mask (`mask`) and Groups (`groups`) follow the same semantics as scaling
factors and zero-points.

Data Types (`data_type`) supported for precomputed reductions:
- `s32`

#### Limitations

The following limitations apply when using precomputed reductions:

- **Requires weight zero-points**: Cannot be used without weights zero-points specified
- **Full matrix mask required**: Must have full A matrix mask (e.g., for standard
  M×K times K×N MatMul, the mask should be 3), meaning broadcast is not supported

See [Example 2](#example-2-matmul-with-precomputed-reductions-and-advanced-quantization) for complete code.

## int8 Convolution Quantization Breakdown

Consider a convolution with bias. The tensors are represented as:

- \f$\src_{f32}[:] = scale_{\src} \cdot (\src_{int8}[:] - zp_{\src})\f$
- \f$\weights_{f32}[:] = scale_{\weights} \cdot \weights_{int8}[:]\f$
- \f$\dst_{f32}[:] = scale_{\dst} \cdot (\dst_{int8}[:] - zp_{\dst})\f$

Here the \f$\src_{f32}, \weights_{f32}, \dst_{f32}\f$ are not
computed at all, the whole work happens with int8 tensors.So the task
is to compute the \f$\dst_{int8}\f$ tensor, using the \f$\src_{int8}\f$,
\f$\weights_{int8}\f$ tensors passed at execution time, as well as the
corresponding quantization parameters \f$scale_{\src}\f$, \f$scale_{\weights}\f$,
\f$scale_{\dst}\f$, and \f$zp_{\src}\f$, \f$zp_{\dst}\f$.
Mathematically, the computations are:

\f[
   \dst_{int8}[:] =
      \operatorname{f32\_to\_int8}(
         (scale_{\src} \cdot scale_{\weights} \cdot
         \operatorname{s32\_to\_f32}(conv_{s32}(\src_{int8}, \weights_{int8})
	   - zp_{\src} \cdot comp_{s32}) + bias_{f32}) / scale_{\dst}
           + zp_{\dst} )
\f]

where

- \f$\operatorname{conv}_{s32}\f$ is just a regular convolution which takes source and
  weights with int8 data type and compute the result in int32 data type (int32
  is chosen to avoid overflows during the computations);

- \f$comp_{s32}\f$ is a compensation term to account for
  \f$\src\f$ non-zero zero-point. This term is computed by the oneDNN
  library and can typically be pre-computed ahead of time, for example
  during weights reorder.

- \f$\operatorname{f32\_to\_s8}()\f$ converts an `f32` value to `s8` with
  potential saturation if the values are out of the range of the int8 data
  type.

- \f$\operatorname{s32\_to\_f32}()\f$ converts an `int8` value to
  `f32` with potential rounding. This conversion is typically
  necessary to apply `f32` scaling factors.

### Per-Channel Scaling Specifics

Some of the primitives have limited support of multiple scales for a quantized
tensor. The most popular use case is the @ref dev_guide_convolution primitive
that supports per-output-channel scaling factors for the weights, meaning that
the actual convolution computations would need to scale different output
channels differently. This is possible without significant performance loss
because the per-output-channel re-quantization is only required at the very end
of the computations. It seems impossible to implement the same trick for the
input channels, since that would require re-quantization for every input
data point.

- \f$\src_{f32}(n, ic, ih, iw) = scale_{\src} \cdot \src_{int8}(n, ic, ih, iw)\f$

- \f$\weights_{f32}(oc, ic, kh, kw) = scale_{\weights}(oc) \cdot \weights_{int8}(oc, ic, kh, kw)\f$

- \f$\dst_{f32}(n, oc, oh, ow) = scale_{\dst} \cdot \dst_{int8}(n, oc, oh, ow)\f$

Note that now the weights' scaling factor depends on \f$oc\f$.

To compute the \f$\dst_{int8}\f$ we need to perform the following:

\f[

    \dst_{int8}(n, oc, oh, ow) =
        \operatorname{f32\_to\_int8}(
            \frac{scale_{\src} \cdot scale_{\weights}(oc) \cdot
            conv_{s32}(\src_{int8}, \weights_{int8})|_{(n, oc, oh, ow)} + \bias_{f32}}{scale_{\dst}}
        ).
\f]

The user is responsible for preparing quantized weights accordingly. To do that,
oneDNN provides reorders that can perform per-channel scaling:

\f[

    \weights_{int8}(oc, ic, kh, kw) =
        \operatorname{f32\_to\_int8}(
            \weights_{f32}(oc, ic, kh, kw) / scale_{weights}(oc)
        ).
\f]

### Weights Preparation with Per-output-channel Scaling

~~~cpp
   // weights dimensions
   const int OC, IC, KH, KW;

   // original f32 weights in plain format
   dnnl::memory::desc wei_plain_f32_md(
           {OC, IC, KH, KW},                 // dims
           dnnl::memory::data_type::f32,     // the data originally in f32
           dnnl::memory::format_tag::hwigo   // the plain memory format
           );

   // the scaling factors for quantized weights
   // An unique scale for each output-channel.
   std::vector<float> wei_scales(OC) = { /* values */ };
   dnnl::memory();

   // int8 convolution primitive descriptor
   dnnl::convolution_forward::primitive_desc conv_pd(/* see the convolution workflow section */);

   // query the convolution weights memory descriptor
   dnnl::memory::desc wei_conv_s8_md = conv_pd.weights_desc();

   // prepare the attributes for the reorder
   dnnl::primitive_attr attr;
   const int quantization_mask = 0
       | (1 << 0);  // scale per  OC dimension, which is the dim #0
   attr.set_scales_mask(DNNL_ARG_DST, quantization_mask);

   // create reorder that would perform:
   //   wei_s8(oc, ic, kh, kw) <- wei_f32(oc, ic, kh, kw) / scale(oc)
   // including the data format conversion.
   auto wei_reorder_pd = dnnl::reorder::primitive_desc(
           wei_plain_f32_md, engine, // source
           wei_conv_s8_md, engine, // destination,
           attr);
   auto wei_reorder = dnnl::reorder(wei_reorder_pd);

// ...
~~~

### Convolution with Per-output-channel Quantization

Building upon the weights preparation shown above, this section shows
the complete workflow for an int8 convolution that combines per-output-channel
weight scaling with global source and destination scaling.

~~~cpp
   const float src_scale; // src_f32[:] = src_scale * src_s8[:]
   const float dst_scale; // dst_f32[:] = dst_scale * dst_s8[:]

   // the scaling factors for quantized weights (as declared above)
   // An unique scale for each output-channel.
   std::vector<float> wei_scales(OC) = {...};


   // Src, weights, and dst memory descriptors for convolution,
   // with memory format tag == any to allow a convolution implementation
   // to chose the appropriate memory format

   dnnl::memory::desc src_conv_s8_any_md(
           {BATCH, IC, IH, IW},          // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let convolution to choose
           );

   dnnl::memory::desc wei_conv_s8_any_md(
           {OC, IC, KH, KW},             // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let convolution to choose
           );

   dnnl::memory::desc dst_conv_s8_any_md(...);  // ditto

   // prepare the attributes for the convolution
   dnnl::primitive_attr attr;
   const int data_mask = 0; // scale and zero-point per tensor for source and destination
   const int wei_mask = 0
       | (1 << 0); // scale per OC dimension, which is the dim #0 on weights tensor:
                   // (   OC, IC, KH, KW)
                   //      0   1   2   3

   attr.set_scales_mask(DNNL_ARG_SRC, data_mask);
   attr.set_zero_points_mask(DNNL_ARG_SRC, data_mask);

   attr.set_scales_mask(DNNL_ARG_WEIGHTS, wei_mask);

   attr.set_scales_mask(DNNL_ARG_DST, data_mask);
   attr.set_zero_points_mask(DNNL_ARG_DST, data_mask);

   // create a convolution primitive descriptor
   auto conv_pd = dnnl::convolution_forward::primitive_desc(
           dnnl::prop_kind::forward_inference,
           dnnl::algorithm::convolution_direct,
           src_conv_s8_any_md,                     // what's important is that
           wei_conv_s8_any_md,                     // we specified that we want
           dst_conv_s8_any_md,                     // computations in s8
           strides, padding_l, padding_r,
           dnnl::padding_kind::zero
           attr);   // the attributes describe the quantization flow
// ...
~~~

## Additional Examples

### Example 1: matmul with advanced quantization

This example describes a process of weights decompression, or
weights-only-quantization (WoQ), in matmul primitive which may be found when
running Large Language Models (LLM). The advanced quantization here refers to
additional grouping introduced over reduction dimension besides traditional
per-N quantization.

~~~cpp
   // Src, weights, and dst memory descriptors for matmul.
   // Consider simple 2D matmul case.
   dnnl::memory::desc src_f16_any_md(...);
   dnnl::memory::desc wei_s8_any_md(
           {K (256), N (512)},           // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let matmul to choose
           );
   dnnl::memory::desc dst_f16_any_md(...);

   // prepare the attributes
   dnnl::primitive_attr attr;
   // scale per K and N dimensions:
   const int wei_mask = (1 << 0) | (1 << 1);
   // K dimension specifies the group size of `128`. It means that each 128
   // elements over K dimension will share a single value. For a given example,
   // there will be two groups, thus, two values referring to a single N value.
   std::vector<dim_t> wei_groups = {128, 1}

   // the scaling factors for quantized weights (as declared above)
   // A unique scale for each gK (256 / 128 = 2) times N, total 1024 elements.
   std::vector<half> wei_scales(gK, N) = {...};

   attr.set_scales(DNNL_ARG_WEIGHTS, wei_mask, wei_groups, data_type::f16);

   // Additionally, to instruct the library to perform weights decompression,
   // fpmath mode must be set with a flag set to `true`:
   attr.set_fpmath_mode(fpmath_mode::f16, /* apply_to_int = */ true);

   // create a matmul primitive descriptor
   auto matmul_pd = dnnl::matmul::primitive_desc(
           engine,
           src_f16_any_md,
           wei_s8_any_md,
           dst_f16_any_md,
           attr);   // the attributes describe the quantization flow
// ...
~~~

### Example 2: matmul with precomputed reductions and advanced quantization

This example is a complementary addition to the one above. It describes a
process of dynamic quantization with weights's tensor asymmetric quantization
and external precomputed reductions of the source tensor.

The case arises from the technique of quantizing source tensor on-the-fly (on
the application side) and passing both quantized source and weights tensors to
the library.

It's important that precomputed reductions appear from weights zero-points to
provide accurate result when zero-points datatype is s8, in which case it's
impossible to apply them on-the-fly without potential accuracy loss.

~~~cpp
   // Src, weights, and dst memory descriptors for matmul.
   // Consider simple 2D matmul case.
   dnnl::memory::desc src_u8_any_md(
           {M (64), K (256)},            // dims
           dnnl::memory::data_type::u8,  // the data originally in u8
           dnnl::memory::format_tag::any // let matmul to choose
           );
   dnnl::memory::desc wei_s8_any_md(
           {K (256), N (512)},           // dims
           dnnl::memory::data_type::s8,  // the data originally in s8
           dnnl::memory::format_tag::any // let matmul to choose
           );
   dnnl::memory::desc dst_f16_any_md(...);

   // prepare the attributes
   dnnl::primitive_attr attr;
   // scale per K and N dimensions:
   const int wei_mask = (1 << 0) | (1 << 1);
   // K dimension specifies the group size of `128`. It means that each 128
   // elements over K dimension will share a single value. For a given example,
   // there will be two groups, thus, two values referring to a single N value.
   std::vector<dim_t> wei_scales_groups = {128, 1}

   // The scaling factors for quantized weights (as declared above)
   // A unique scale for each scale_gK (256 / 128 = 2) times N, total 1024
   // elements.
   std::vector<half> wei_scales(scale_gK, N) = {...};

   attr.set_scales(DNNL_ARG_WEIGHTS, wei_mask, wei_scales_groups,
           data_type::f16);

   // Zero-points would have the same mask as grouping applies for them as well.
   // For example, let it use the different size of the group.
   std::vector<dim_t> wei_zp_groups = {64, 1};

   // The zero-point factors for quantized weights (as declared above)
   // A unique zero-point for each zp_gK (256 / 64 = 4) times N, total 2048
   // elements.
   std::vector<half> wei_zps(zp_gK, N) = {...};

   attr.set_zero_points(DNNL_ARG_WEIGHTS, wei_mask, wei_zp_groups,
           data_type::s8);

   // Now, specify the precomputed reductions.
   // Note that it's specified for source tensor.
   // It means it should have full-size source tensor mask (which in this
   // example coincides with `wei_mask`), and groups would be over another
   // dimension, same as zero-points group size.
   std::vector<dim_t> src_pr_groups = {1, 64};

   // The precomputed reduction factors for quantized sources.
   // A unique reduction for each M times pr_gK (256 / 64 = 4), total 256
   // elements.
   std::vector<half> src_prs(M, pr_gK) = {...};

   attr.set_precomputed_reductions(DNNL_ARG_SRC, src_tensor_mask,
           src_pr_groups);

   // fpmath mode is not required in case of dynamic quantization as it's
   // treated as classical quantization case.

   // create a matmul primitive descriptor
   auto matmul_pd = dnnl::matmul::primitive_desc(
           engine,
           src_s8_any_md,
           wei_s8_any_md,
           dst_f16_any_md,
           attr);   // the attributes describe the quantization flow
// ...
~~~
