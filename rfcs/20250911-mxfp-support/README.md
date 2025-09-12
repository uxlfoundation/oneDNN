# Open Compute Microscaling (MX) datatype support

There is growing interest in using block scaling formats in recent
LLMs. In an effort to standardize hardware support for them, the Open
Compute Platform (OCP) Microscaling standard (MX) [^1] defines
interchange formats as well as operations (dot-product and
conversions). In particular, MXFP4 is currently used in gpt-oss [^2]
for weight compression.

In this RFC, we go over what the MX formats are and what oneDNN needs
to support them.

## MX formats and requirements for oneDNN

Even though they are sometimes referred to as datatypes, MX formats
are actually a quantized blocked format for representing tensors.

Here are the elements oneDNN needs to support for MX spec [^1] conformance:
- Block element types. All types documented in the MXFP spec except `fp6`
  are already supported by the oneDNN API (namely, `s8`, `f8_e5m2`,
  `f8_e4m3`, and `f4_e2m1`).
- Block scaling with 1D groups of 32. This is supported by oneDNN
  through the `set_scales` API.
- The `e8m0` block scale datatype. This is supported in oneDNN.
- Dynamic quantization of the output. This is the missing piece and the
  main point of discussion in this RFC.

## MX Dynamic quantization definition

How the scales are computed is defined by the MXFP spec. For a given
1d block of size 32 (denoted $`X`$), its scale `S` and quantized values `Q`
are defined as:

$$ S = E8M0(amax(X)) / E8M0(MAX\\_DST\\_DT) $$
$$ Q = dst\\_dt(X / S) $$

With `E8M0` being a conversion to the `E8M0` datatype by rounding down to
the closest power of two below the argument value, `MAX_DST_DT` the maximum
representable value in the `dst` type, and `dst_dt` a conversion to the
`dst` type.

Note that the division operation happens _after_ rounding down to the
scale datatype (`E8M0`). This allows implementing the division as a
subtraction of the exponent fields. However, this differs from other
dynamic scaling formulas, for example:
- Traditional int8 dynamic quantization applies the division _before_
  conversion to the scale datatype.
- The cuDNN [^3] and cuBLAS [^4] formulas compute the division in `f32`
  _before_ the conversion to the scale datatype.

## Proposals

First, this RFC focuses on MXFP format support but may also cover some
generalizations to allow extension to other formats and recipes. We also
assume that scales and data are not interleaved in memory, similarly to
the current quantization support.

> **Status:** Option 1.b was selected and productized. The subsequent
> sections preserve the original analysis; see the [Summary](#summary) for
> the final, shipped API.

### Option 1.a (not adopted): Through a new attribute (`set_dynamic_scales`)

Here we propose exposing a new `dynamic_scales` attribute, similar to the
`scales` attribute except that:
- It applies only to output memory.
- It implies that scales are computed by oneDNN rather than provided by the
  user.
- It implies a new output memory to store the computed scales.

In order to support MXFP formats, it needs to support groups of 32 and a
way to express along which axis the grouping occurs. It also needs to
support the `E8M0` format for scales. We propose to use the same semantics
as the `set_scales` method:
- The grouping factor can be made explicit with the mask, ndims, and
  group_dims arguments.
- The `E8M0` scale datatype can be set explicitly with a scale data_type
  argument.
- Regarding the scale computation formula, we would not expose a knob to
  configure it and would support only the MXFP spec formula.

This results in the following new entry point in the C API:
```c
dnnl_status_t DNNL_API dnnl_primitive_attr_set_dynamic_scales(
        dnnl_primitive_attr_t attr, int arg, int mask, int ndims,
        const dnnl_dims_t group_dims, dnnl_data_type_t data_type);
```

And the associated symbol in the C++ API:
```cpp
void set_dynamic_scales(int arg, int mask, const memory::dims &groups,
        memory::data_type data_type = memory::data_type::e8m0);
```

Finally, the existing `scales`/`zero_points` attributes and the new
`dynamic_scales` attribute would be mutually exclusive. Hence, only one
quantization scheme would be accepted for a given memory argument. This
also allows reusing the `DNNL_ARG_ATTR_SCALES` argument kind to pass the
dynamic scales buffer to the `execute` function.

If they were not mutually exclusive, we would need to expose a new argument
kind `DNNL_ARG_ATTR_DYNAMIC_SCALES` for users to specify the memory
argument for writing computed scales.

To summarize, here is an example of this new API with MXFP4:
```cpp
    // Element type for all inputs/outputs is e2m1 for MXFP4.
    memory::desc a_md({M, K}, memory::data_type::f4_e2m1, {K, 1}); // M x K layout
    memory::desc b_md({K, N}, memory::data_type::f4_e2m1, {1, K}); // N x K layout
    memory::desc c_md({M, N}, memory::data_type::f4_e2m1, {N, 1}); // M x N layout

    // Create attributes and set static scales for inputs,
    // with type e8m0 and groups of 32 along K.
    primitive_attr attr;
    attr.set_scales(DNNL_ARG_SRC,
            /* mask */ (1 << 0) + (1 << 1), {1, 32}, memory::data_type::e8m0);
    attr.set_scales(DNNL_ARG_WEIGHTS,
            /* mask */ (1 << 0) + (1 << 1), {32, 1}, memory::data_type::e8m0);

    // Set dynamic scales for the output, with type e8m0 and groups of 32 along K.
    attr.set_dynamic_scales(DNNL_ARG_DST,
            /* mask */ (1 << 0) + (1 << 1), {1, 32}, memory::data_type::e8m0);

    // Create a MatMul primitive descriptor.
    matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
```

### Option 1.b (recommended, productized): extend the scales attribute (`set_scales`)

Same principle as option 1.a, except that instead of exposing a new
attribute, `set_scales` is extended to support dynamic scaling. The main
benefits are:
- The API itself makes it clear that static and dynamic quantization are
  mutually exclusive.
- It allows extension to other quantization formulas, such as static
  quantization with a floating-point zero-point expressed as
  $`x_{f32} = scale * x_q + zp`$ instead of $`x_{f32} = scale * (x_q - zp)`$,
  or dynamic quantization formulas where the conversion happens after the
  division.

To do so, a new enum class is introduced and a new argument is added to
`set_scales`. In the C API, this requires a v3 method as follows:

```c
typedef enum {
    /// used for unspecified quantization kind
    dnnl_quantization_mode_undef,
    /// static quantization mode: the quantization parameter is computed
    /// ahead of time, with the scale applied after the zero-point
    /// (x_f32 = scale * (x_quant - zp)), and passed to oneDNN as an input.
    dnnl_quantization_mode_static_sazp,
    /// dynamic quantization mode following the OCP MX spec: the quantization
    /// parameter is computed by oneDNN following the OCP MX spec formula and
    /// written as an output.
    dnnl_quantization_mode_dynamic_mx,
    /// dynamic quantization mode where the quantization parameter is computed
    /// by oneDNN as scale_dt(amax(X) / max(dst_dt)) in f32, then converted to
    /// the scale type and written as an output.
    dnnl_quantization_mode_dynamic_fp,
} dnnl_quantization_mode_t;

dnnl_status_t DNNL_API dnnl_primitive_attr_set_scales_v3(
        dnnl_primitive_attr_t attr, int arg, int mask, int ndims,
        const dnnl_dims_t group_dims, dnnl_data_type_t data_type,
        int is_on_host, dnnl_quantization_mode_t qmode);
```

In the C++ API, a matching enum type and `set_scales` extension are
introduced:

```cpp
enum class quantization_mode {
    undef = dnnl_quantization_mode_undef,
    static_sazp = dnnl_quantization_mode_static_sazp,
    dynamic_mx = dnnl_quantization_mode_dynamic_mx,
    dynamic_fp = dnnl_quantization_mode_dynamic_fp,
};

void set_scales(int arg, int mask, const memory::dims &groups,
        memory::data_type data_type = memory::data_type::f32,
        bool is_on_host = false,
        quantization_mode qmode = quantization_mode::static_sazp);
```

To summarize, here is the same MXFP4 example as for option 1.a, expressed
with the extended `set_scales` API:
```cpp
    // Element type for all inputs/outputs is e2m1 for MXFP4.
    memory::desc a_md({M, K}, memory::data_type::f4_e2m1, {K, 1}); // M x K layout
    memory::desc b_md({K, N}, memory::data_type::f4_e2m1, {1, K}); // N x K layout
    memory::desc c_md({M, N}, memory::data_type::f4_e2m1, {N, 1}); // M x N layout

    // Create attributes and set static scales for inputs,
    // with type e8m0 and groups of 32 along K.
    primitive_attr attr;
    attr.set_scales(DNNL_ARG_SRC,
            /* mask */ (1 << 0) + (1 << 1), {1, 32}, memory::data_type::e8m0,
            /* is_on_host */ false, quantization_mode::static_sazp);
    attr.set_scales(DNNL_ARG_WEIGHTS,
            /* mask */ (1 << 0) + (1 << 1), {32, 1}, memory::data_type::e8m0,
            /* is_on_host */ false, quantization_mode::static_sazp);

    // Set dynamic scales for the output: oneDNN computes e8m0 scales with
    // groups of 32 following the OCP MX spec and writes them as an output.
    attr.set_scales(DNNL_ARG_DST,
            /* mask */ (1 << 0) + (1 << 1), {1, 32}, memory::data_type::e8m0,
            /* is_on_host */ false, quantization_mode::dynamic_mx);

    // Create a MatMul primitive descriptor.
    matmul::primitive_desc(eng, a_md, b_md, c_md, attr);

    // ... After execution, the computed destination scales are retrieved
    // from the memory passed via DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST.
```

The main drawback of this option is compatibility with the zero-point
attribute, which is compatible with scales only when they are set with the
static quantization mode. That behavior is somewhat more complex to document
and understand from the user's perspective.

**Resolution:** This option was selected and productized. Compared to the
original proposal, the enum was renamed from `quantization_kind` to
`quantization_mode`; the static value was renamed to `static_sazp` (scale
applied after zero-point) to make the applied formula explicit; and a
`dynamic_fp` mode was added for the dynamic formula that computes the
division in `f32` before converting to the scale type (matching the cuDNN
and cuBLAS behavior noted above). As the scales attribute is extended
in place, the computed scales buffer is passed through the existing
`DNNL_ARG_ATTR_SCALES | arg` argument, so no new argument kind is required.
The implementation originated as a POC in
[PR3978](https://github.com/uxlfoundation/oneDNN/pull/3978).

### Option 2 (not adopted): Through new datatypes

Another option would be to expose new datatypes (e.g. `mxfp8_e4m3`,
`mxfp8_e5m2`, `mxfp4_e2m1`, and so on). These would encode the group
element datatype as well as the scale datatype, the group size (32 in the
case of MX types), and the axis along which scales apply (e.g. the
innermost physical dimension).

A new argument kind, `DNNL_ARG_MX_SCALES`, would be used to specify the
memory argument for reading/writing scales.

This option is not recommended for a couple of reasons:
- It is not aligned with the existing quantization support in oneDNN,
  providing two very different ways to specify the quantization of a memory
  object.
- It lacks flexibility: the user cannot specify along which axis
  quantization happens (it would only be the physical innermost axis).
  Furthermore, any new combination of scale datatype, group size, and group
  element type would require a new datatype in oneDNN.

### Testing

benchdnn was extended according to the adopted proposal (option 1.b): the
existing `--attr-scales` knob gained two additional policy values on top of
`per_tensor` grouping — `mx`, which marks the destination scales as
computed by the primitive following the OCP MX specification, and
`dynamic_fp`, which uses the `to_scale_dt(amax(x) / max(dst_dt))` formula.
Both apply to destination scales only.

## Summary

oneDNN already supports MXFP inputs, as it supports the base datatypes as
well as grouped scales of type `e8m0`. To support MXFP outputs, the
`set_scales` attribute was extended (option 1.b) with a `quantization_mode`
argument, allowing the user to specify that oneDNN computes the scaling
factors and applies them when converting the output to an MXFP type. The
computed scales are collected through the existing
`DNNL_ARG_ATTR_SCALES | arg` argument, so no new argument kind is needed.
The formula used to compute the scales is not freely configurable; it is
selected through the quantization mode: `dynamic_mx` follows the OCP MX
spec formula (used when the scale type is `e8m0`), while `dynamic_fp`
computes the division in `f32` before converting to the scale type.

## References

[^1]: [OCP MX spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
[^2]: [GPT-OSS description](https://huggingface.co/openai/gpt-oss-20b)
[^3]: [cuDNN block scaling support](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/BlockScaling.html#block-scale-quantize)
[^4]: [cuBLAS block scaling support](https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization)
