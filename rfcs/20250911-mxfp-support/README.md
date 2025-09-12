# Open Compute Microscaling (MX) datatype suport

There is growing interest in using block scaling formats in recent
LLMs. In an effort to standardize hardware support for those, the Open
Compute Platform (OCP) Microscaling standard (MX) [^1] defines
interchange formats as well as operations (dot-product and
conversions).  In particular, MXFP4 is currently used in gpt-oss [^2]
for weights compression.

In this RFC, we go over what are the MX formats, and what oneDNN need
to support those.

## MX formats and requirements to oneDNN

Even though though they are sometimes refered to as datatypes, MX
formats are actually a quantized blocked format to represent tensors.

Here are the elements oneDNN need to support for MX spec [^1] conformance:
- block element types. All types documented in MXFP spec but fp6 ones
  are already supported by oneDNN API (namely, int8, f8_e5m2, f8_e4m3,
  f4_e2m1).
- block scaling with 1d groups of 32. This is supported by oneDNN
  through `set_scales` API.
- e8m0 block scale datatype. This is supported in oneDNN.
- support dynamic quantization of output. This is the missing piece
  and main point of discussion of this RFC.

## MX Dynamic quantization definition

How the scales are computed is defined by the MXFP spec. For a given
1d block of size 32 (denoted $`X_i`$), its scale and quantized values
are defined as:

$$ S = E8M0(amax(X)) / E8M0(max(dst_dt)) $$

With `E8M0` a conversion to E8M0 datatype, by rounding down to the
closest power of two below the argument value.

Note that the division operation happens _after_ rounding down to the
scale datatype (E8M0). This allows to implement this division as a
subtraction of the exponent fields. However, this differs from other
dynamic scaling formulas, for example:
- traditional int8 dynamic quantization applies the division _before_
  conversion to scale datatype
- cuDNN [^3] and cublas [^4] formula compute the division in f32 _before_ the
  conversion to scale datatype

## Proposals

First of all, this RFC focuses on MXFP formats supports but might
cover some generalizations to allow extension to other formats and
recipes.  We will also assume that scales and data are not interleaved
in memory, similarly to current quantization support.

### Opetion 1 (Recommended): Through a new attribute (`set_dynamic_scales`)

Here we propose to expose a new `dynamic_scales` attribute, similar to
`scales` attribute except that:
- it would be applicable only to output memory,
- it implies that scales are computed by oneDNN, and not provided by user,
- it implies a new output memory to store the computed scales.

In order to support MXFP formats, it needs to support groups of 32,
and a way to express along which axis the grouping occurs. It also
needs to support E8M0 format for scales. We propose to use the same
semantics as with `set_scales` method:
- grouping factor can be made explicit with mask, ndims and group_dims
  arguments,
- E8M0 scale datatype can be set explicitely with a scale data_type
  argument.
- regarding the scale computation formula, we would not expose a knob
  to configure it, and support only MXFP spec formula.

This will result in the following new entrypoint in C API:
```C
dnnl_status_t DNNL_API dnnl_primitive_attr_set_dynamic_scales(
        dnnl_primitive_attr_t attr, int arg, int mask, int ndims,
        const dnnl_dims_t group_dims, dnnl_data_type_t data_type);
```

And the associated symbol in C++ API:
```c++
    void set_dynamic_scales(int arg, int mask, const memory::dims &groups,
            memory::data_type data_type = memory::data_type::f32);
```

Finally, the existing `scales/zero_points` attributes and the new
`dynamic_scales` attribute would be mutually exclusive. Hence only one
quantization scheme would be accepted for a given memory argument.

A new argument kind `DNNL_ARG_ATTR_DYNAMIC_SCALES`, will be used to
specify the memory argument for writting computed scales. We recommend
to make this argument kind a mask (similar to `DNNL_ARG_ATTR_SCALES`),
to align with non-dynamic scales, and allow to easily support
extension to different output (e.g. `DNNL_ARG_DIFF_SRC` in case of
training,

### Option 2: Through new datatypes 

Another option could be to expose new datatype (e.g. `mxfp8_e4m3`,
`mxfp8_e5m2`, `mxfp4_e2m1`, ...). Those would encode group element
datatype, but also scale datatype, group size (32 in case of MX
types), and axis along which scales apply (e.g. innermost physical
dimension).

A new argument kind `DNNL_ARG_MX_SCALES`, will be used to
specify the memory argument for reading/writting scales.

This option is not recommended for a couple of reasons:
- it is not aligned with existing quantization support in oneDNN,
  providing two very different ways to specify quantization of a
  memory object.
- it lacks flexibility: user cannot specify along which axis
  quantization would happen (it would be only physical innermost
  axis). Furthermore, any new combination of scale datatype / group
  size / group element type would require a new datatype in oneDNN.

### Testing

benchdnn will be extended according to the proposal adopted:
- if new attribute, a new `--attr-dynamic-scales` knob will be
  exposed.
- if new datatype, a new datatype will be available to be passed to
  `--dt` knob.

## Summary

oneDNN already suport MXFP inputs, as it supports base datatypes, as
well as grouped scales of type e8m0. To support MXFP outputs, we
recommend introducing a new attribute (set_dynamic_scales), which
would allow the user to specify that oneDNN will compute the scaling
factors and apply them upon output conversion to MXFP type. oneDNN
will also provide a new argument kind for user to collect the scales
computed by oneDNN. The formula used to compute the scales will not be
configurable, and will match MXFP spec formula when scale type is
e8m0.

# References:


[^1]: [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
[^2]: [GPT-OSS description](https://huggingface.co/openai/gpt-oss-20b)
[^3]: [cudnn block scaling support](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/BlockScaling.html#block-scale-quantize)
[^4]: [cublas block scaling support](https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization)
