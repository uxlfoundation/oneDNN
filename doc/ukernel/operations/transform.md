Data transformation {#dev_guide_ukernel_transform}
=======================================

>
> [API Reference](@ref dnnl::ukernel::transform)
>

## General

The [BRGeMM ukernel](@ref dev_guide_ukernel_brgemm) might require the B tensor
in a specific memory layout depending on target data types and the machine
architecture. Check the requirement by calling the
[get_B_pack_type()](@ref dnnl::ukernel::brgemm::get_B_pack_type) function.
The return value specifies one of the following scenarios:
* If the call returns the [undef](@ref dnnl::ukernel::pack_type::undef) type, it
  indicates ukernel API is not supported on the target system.
* If the call returns the [no_trans](@ref dnnl::ukernel::pack_type::no_trans)
  type, it indicates the packing is not required.
* Depending on the backend and packing requirements, the value indicating these
  requirements can be different. For x64 backend the call returns the
  [pack32](@ref dnnl::ukernel::pack_type::pack32) type.

The transform ukernel allows the conversion of data from the original layout,
which is described as either
[non-transposed](@ref dnnl::ukernel::pack_type::no_trans) or
[transposed](@ref dnnl::ukernel::pack_type::trans) to the layout requested by
the BRGeMM ukernel.

The only supported output packing type is `pack32`.

This is an out-of-place operation.

## Data Types

The transform ukernel does not allow data type conversion.

## Data Representation

| src     | dst     |
|:------- |:------- |
| f32     | f32     |
| f16     | f16     |
| bf16    | bf16    |
| f8_e4m3 | f8_e4m3 |
| f8_e5m2 | f8_e5m2 |
| s8      | s8      |
| u8      | u8      |

## Attributes

No attribute is supported for transform ukernel.

## Implementation limitations

- Destination leading dimension, or `out_ld`, must be one of the following
  values: `16`, `32`, `48`, or `64`. This is the implementation limitation,
  there are no efficient kernels supported for other leading dimension values.

## Examples

[BRGeMM ukernel example](@ref cpu_brgemm_example_cpp)

@copydetails cpu_brgemm_example_cpp
