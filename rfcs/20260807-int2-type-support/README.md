# Support for 2-bit unsigned integer type

## Introduction

One of the ways of fitting large models on client devices with limited RAM is
using low-precision weight representations. oneDNN already supports 4-bit
integer types (`s4`/`u4`), but some users are moving to 2-bit compression for
further footprint reduction.

On an internal customer model, switching weights from 4-bit to 2-bit provided:

| Metric               | Baseline (`u4`) | With `u2` |
|----------------------|-----------------|-----------|
| Peak memory usage    | 1.00x           | 0.88x     |
| Throughput (PTL[^1]) | 1.00x           | 1.33x     |

Model conformance (accuracy) was sustained at the target level. Measurement
configuration: weights-only quantization, `f16` activations, `f16` scales,
group size 64.

This RFC proposes adding a 2-bit unsigned integer data type, `dnnl_u2`.

OpenVINO included 2-bit unsigned integer data type support into their core [^2]
based on NNCF recipes.

[^1]: PTL = Panther Lake (client GPU platform).
[^2]: https://github.com/openvinotoolkit/openvino/pull/36170

## Proposal

### API change

The only public API change is a new value appended to the data type enum:

```c
typedef enum {
    ...
    /// 2-bit unsigned integer.
    dnnl_u2 = 17,

    ...
} dnnl_data_type_t;
```

A matching `memory::data_type::u2` is added to the C++ API.

### Storage scheme

`u2` values are packed 4 per byte, versus 2 per byte for the 4-bit types. This
lets `u2` reuse the existing sub-byte machinery with a different packing
factor. The intra-byte ordering is the following:

```
byte:    [ v3 | v2 | v1 | v0 ]
bits:      7 6  5 4  3 2  1 0
```

Element `i` of the logical (unpacked, contiguous) tensor occupies bit field
`(i mod 4)`, i.e. the first logical element is stored in the least-significant
2 bits. This mirrors the existing nibble ordering used for `s4`/`u4`.

Because storage is packed, the innermost (physical) dimension of a `u2` tensor
must be a multiple of 4 to remain byte-aligned (the 4-bit types require a
multiple of 2).

### Semantics

`u2` is an integer type with value range `[0, 3]`. It is a storage/quantized
type only: it participates in matmul as quantized weights and as zero-points,
and is up-converted to the compute type (`f16`/`f32`) via the standard
scale/zero-point dequantization path:

```
dequant(w) = (u2_to_int(w) - zp) * scale
```

Group (micro-scaling) quantization reuses the existing grouped scales/zero-
points mechanism; no new scaling scheme is introduced. The quantization group
size (e.g. 64) is independent of the 4-values-per-byte packing factor.

### Scope

Covered functionality focuses on weights-only-quantization cases with f16
activations with f16 scales and optional zero-points grouped by 64 elements.

Cases of interest can be expressed with these benchdnn lines:
```
benchdnn --matmul --engine=gpu --dt=f16:u2:f16 --wtag=ba --attr-scales=wei:per_oc:f16 --attr-zero-points=wei:per_oc:u2 --attr-fpmath=f16:true 1x5120:5120x7680
benchdnn --matmul --engine=gpu --dt=f16:u2:f16 --wtag=ba --attr-scales=wei:3:f16:64x1 --attr-zero-points=wei:3:u2:64x1 --attr-fpmath=f16:true 1x5120:5120x7680
```

## Open Questions

- **Prefill / activation quantization.** The customer confirmed the decoding
  stage but not prefill. If prefill is covered, do activations need to be
  quantized too? This needs accuracy research to determine whether the model
  still conforms and what the quantization limits are.

- **Weights format.** Short-term delivery supports only `ba` (transposed). A
  blocked layout such as `Ab64a` (2D) may be needed for better deployment
  performance. Open: is that the only additional format required, what is the
  performance uplift, and how it would be enforced?

* **`s2` support** is currently out-of-scope due to lack of evidence it is
  useful at the time this document is written. It's open if it will ever be
  required in the future.
