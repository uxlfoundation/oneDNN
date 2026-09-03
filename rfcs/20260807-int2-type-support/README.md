# Support for 2-bit integer types

## Introduction

One of the ways of fitting large models on client devices with limited RAM is
using low-precision weight representations. oneDNN already has support for
4-bit integer types (s4/u4), but newest models use int2 compression to further
reduce their footprint.

## Proposal

The proposal is to add new data types, dnnl_s2 and dnnl_u2, to oneDNN.

### Model support and performance implications

A user interest in shipping models with u2 weights has been recorded. For one
example model provided to Intel developers, a transition from u4 to u2
compressed weights reduced peak memory utilization during inference by 14%
and improved token throughput on a PTL GPU by 33% without breaking model
conformance. The model in question is expected to be shipped to the public,
and further interest in ultra-low precisions is expected.

### Current state: int4

At present, 4-bit types are treated as a special case in gemmstone, requiring
explicit carve-outs with bespoke code paths. To implement int2 and leave space
open for possible efforts at representing int3 and ternary, a refactor of the
gemmstone data type representation is required.

### New gemmstone type ID

The gemmstone type ID is currently a loosely formatted 32-bit integer value.
To accommodate the addition of a sub-byte type with a size other than 4 bits,
as well as to create a possibility of extension for blocked int3 representations
and dense 5-trits-per-byte ternary, the following structure is proposed instead:

* nibble 0: type meta-information
    - 0 - fp=1, int=0
    - 1 - signed=1, unsigned=0
    - 2-3 - reserved
* nibble 1: complexity meta-information
    - 4 - complex=1, real=0
    - 5 - split complex=1
    - 6-7 - reserved
* nibble 2-4: size meta-information
    - 8-11 - size of the block (units based on following flags)
    - 12-15 - number of values per block
    - 16 - block size in bits=0, block size in bytes=1
    - 17 - log2 block size=0, actual block size=1
    - 18-19 - reserved
* nibble 5-6: index in ngen mapping table
    - 20-27 - ngen reference
* nibble 7: vector meta-information
    - 28-31 - vector component number

The nibble-wise structuring is intended to improve human readability of the
type information, with values like sizes and ngen references being represented
by separate hexadecimal characters.

The representation in nibbles 2-4 introduces a new concept of a block of values.
Previously, each value was assumed to be represented by a number of discrete
adjacent bits. The new representation allows definitions of blocked
representations. For example, a hypothetical ternary type could use the
following type ID:

```
0x10015002
```

This signifies that the type is represented by blocks of 5 values each taking
up 1 byte.

At present, no type with more than one element per block is being introduced.

### New types: u2/s2

For the specific needs of int2 support, the types being added are densely
packed 2-bit integer values, stored contiguously with 4 values per byte. The
type ID encoding in this case recognizes this dense packing as having 1 value
per block with a bit-scale block size, as opposed to cases where values are
stored non-contiguously or where they cannot be directly taken from specific
bits. The following type IDs are used for the new types:

```cpp
        u2       = 0x11001100,
        s2       = 0x11101102,
```

Value ranges are [0; 3] and [-2; 1] for u2 and s2 respectively.

### Immediate use case information

The following client requirements are driving the initial implementation:

- Weight decompression only in f16:u2:f16 matmul scenarios
- Optional f16 scale and u2 zero point support, aggregated per output feature
or in groups of 64x1
- "ba" weight layout
- Main focus is the GPU engine, utilizing existing gemmstone code base

The following benchdnn commands represent target use cases:

```
benchdnn.exe --matmul --mode=C --engine=gpu --dt=f16:u2:f16 --wtag=ba --attr-scales=wei:per_oc:f16 --attr-zero-points=wei:per_oc:u2 --attr-fpmath=f16:true 1x5120:5120x7680
benchdnn.exe --matmul --mode=C --engine=gpu --dt=f16:u2:f16 --wtag=ba --attr-scales=wei:3:f16:64x1 --attr-zero-points=wei:3:u2:64x1 --attr-fpmath=f16:true 1x5120:5120x7680
```
