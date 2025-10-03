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

oneDNN support two main categories of quantization:
- static quantization with scales only (symmetric) or scales and
  zero-points (asymmetric), where scales are applied after zero-point.
- dynamic quantization compliant with the Open Compute Project (OCP)
  Microscaling (MX) [formats specification][1].

To support quantization, primitives should be created and executed as
follows:

- during primitive descriptor creation source, weights or destination
  memory descriptors use low precision datatype (e.g. `s8` or
  `fp8_e4m3`).
- during primitive descriptor creation group size, data types, and
  broadcasting masks of the scaling factors and zero-point are
  provided using primitive attributes.
- during primitive execution the actual quantization parameters are
  provided as arguments to the execute function.

For performance reasons, each primitive implementation typically
supports only a subset of quantization parameter masks, group sizes
and data type combinations. Which combination is supported and
optimized is listed in each primitive documentation page.

This guide does not cover how the appropriate scaling factor can be found.
Refer to the materials in the [Introduction](@ref dgaq_intro).

### Static quantization

The only formula for static quantization currently supported
by oneDNN is with scales applied after zero-point as follows:

\f[
x_{f32}[:] = scale_{x} \cdot (x_{quant}[:] - zp_{x})
\f]

where \f$x_{f32}\f$ and \f$x_{quant}\f$ are the non-quantized and
quantized representation of \f$x\f$ respectively, \f$scale_{x}\f$ is a
*scaling factor* in a floating-point format, \f$zp_{x}\f$ is a *zero
point* (typically in integral format), and \f$[:]\f$ is used to denote
elementwise application of the formula to the arrays. 

In this model, oneDNN assumes that quantization parameters are inputs
provided by the user and the library does not compute those scaling
factors and zero-points as part of primitive computation.

These quantization parameters can either be computed ahead of time
using calibration tools or at runtime based on the actual minimum and
maximum values of a tensor. Either method can be used in conjunction
with oneDNN static quantization, as long as the quantization
parameters are passed as input to the oneDNN primitives at execution
time.


### Dynamic quantization

The only formula for dynamic quantization currently supported by
oneDNN is with scales computed following the [1],
namely:

\f[
x_{f32}[:] = scale_{x} \cdot x_{quant}[:] 
\f]

where \f$x_{f32}\f$ and \f$x_{quant}\f$ are the non-quantized and
quantized representation of \f$x\f$ respectively, and \f$scale_{x}\f$ is a
*scaling factor*:
- in e8m0 format,
- computed for each group of size 32 (see [set_scales](@ref dnnl::primitive_attr::set_scales)),
- and computed as the largest power-of-two less than or equal to the
  maximum absolute value of the group divided by the largest
  power-of-two representable in the \f$x_{quant}\f$ data type
  (e.g. \f$E8M0(amax(x_quant[:])) / E8M0(MAX\_QUANT\_DT) \f$).


## General numerical behavior notes

Primitive implementations are allowed to convert inputs to wider
datatypes (e.g. int8 to int16 or int32), when those conversions do not
impact accuracy.

During execution, primitives implementations avoid integer overflows
and maintain integer accuracy by using wider datatypes (e.g. int32)
for intermediate values and accumulators. 

Results are then converted as
necessary before the result is written to the output memory objects.

The scales are applied in single precision floating point data type
(#dnnl::memory::data_type::f32) before downconversion to the
destination datatype. When converting to integral datatypes,
implementations typically saturate, whereas for floating-point
datatypes, underflow/overflow can occur. To force saturation in
floating-point datatypes use @ref
dev_guide_attributes_post_ops_eltwise with clip algorithm. Rounding
happens according to [rounding mode attribute](@ref dev_guide_attributes_rounding_mode).

@warning
Depending on the architecture, the behavior of int8 computations might slightly
vary. For more details, refer to @ref dev_guide_int8_computations.

When multiple operations are fused in a single primitive using the
[post ops attribute](@ref dev_guide_attributes_post_ops), those are assumed to be
computed in f32 precision. As a result the destination quantization
parameters are applied after the post-ops as follows:

\f[
   \dst[:] = post\_ops(OP(src[:], weights[:], ...)) / scale_{\dst} + zp_{\dst}

\f]

Quantizing/dequantizing values between post-operations can be achieved
using one of [eltwise](@ref dev_guide_attributes_post_ops_eltwise),
[binary](@ref dev_guide_attributes_post_ops_binary), or the scale
parameter of the appropriate post-operation.


## API

oneDNN provides the following APIs to set scales:
- C: @ref dnnl_primitive_attr_set_scales
- C++: @ref dnnl::primitive_attr::set_scales

and the following APIs to set zero-points:
- C: @ref dnnl_primitive_attr_set_zero_points
- C++: @ref dnnl::primitive_attr::set_zero_points

Those take five parameters:
- an argument index, to specify which argument is having its
  quantization parameter description set.
- a mask, to specify along which axis the quantization parameters are
  applied. If the argument we are specifying is a \f$D_0 \times
  ... \times D_{n-1}\f$ tensor and we want to have scales per \f$d_i\f$
  dimension (where \f$0 \le d_i < n\f$), then the mask should be set to
  \f$mask = \sum \limits_{d_i} 2^{d_i}\f$, and the number of scales
  should be \f$\prod\limits_{d_i}D_{d_i}\f$.
- an array of group sizes, that specify the number of consecutive
  elements a single scale/zero-point applies to for each axis along
  which the quantization parameters apply,
- a scale/zero-point data type. It is f32 by default for scales and
  int32 for zero-points
- a quantization mode, which specifies how the scales are computed
  (e.g. static or dynamic).


### Special Case: Host-side Scalar Scale and Zero-point

When using the GPU engine and a single scale/zero-point is used for an
argument (mask=0), oneDNN supports passing those from the host to
reduce overheads of copying data from host to device or allocating
extra device memory. The host scale or zero-point attributes should be
set at creation time using the following API:

~~~cpp
dnnl::primitive_attr attr;
attr.set_host_scale(DNNL_ARG_DST,
           memory::data_type::f32);

attr.set_host_zero_point(DNNL_ARG_DST,
           memory::data_type::s32);
~~~

The corresponding memory objects for scale or zero-point host value
should be created as a host-side scalar (see @ref
dev_guide_host_side_scalars for details) and passed to the primitive
execution function.

## Examples of quantization workflow

### Convolution Quantization Workflow

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


#### Per-Channel Scaling

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

#### Preparing the weights with per-output-channel scaling

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
   dnnl::convolution_forward::primitive_desc conv_pd(/* see the next example */);

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

#### Create the convolution with per-output-channel quantization

This example is complementary to the previous example (which should ideally be
the first one). Let's say we want to create an int8 convolution with per-output
channel scaling.

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

### Matmul with weights-only quantization

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

### Matmul with precomputed reductions and advanced quantization

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


[1]: [Open Compute Project Microscaling specification version 1](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
