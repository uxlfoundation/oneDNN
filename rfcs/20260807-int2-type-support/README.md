# Support for 2-bit integer types

## Introduction

One of the ways of fitting large models on client devices with limited RAM is
using low-precision weight representations. oneDNN already has support for
4-bit integer types (s4/u4), but newest models use int2 compression to further
reduce their footprint.

## Proposal

The proposal is to add new data types, dnnl_s2 and dnnl_u2, to oneDNN.

### Current state: int4

At present, 4-bit integer types are treated as a special case in gemmstone,
requiring explicit carve-outs with bespoke code paths. To implement int2 and
leave space open for a possible int3 effort, a refactor of the gemmstone data
type representation is required.

### New gemmstone type ID

The gemmstone type ID representation is to be changed to a 32-bit integer
interpreted in the following way:

- 0 - int/fp
- 1 - (if int) u/s
- 2 - byte/bit size
- 3 - reserved
- 4-7 - log2 of size in bytes / bit size (depends on flag in bit 2)
- 8-13 - ngen reference table
- 14-15 - reserved
- 16 - real/complex
- 17 - 1 if split complex
- 18-19 - reserved
- 20-23 - vector component number
- 24-31 - reserved

This new representation allows a unified way of expressing 4-bit and 2-bit
values. Additionally, the value fields are moved to align to nibbles in the
overall 32-bit value, making it more human-readable in a hexidecimal
representation.

### Limitation: int3

This new representation allows for later introduction of ternary types, but
does not support them very well. It still carries the assumption that a value
occupies a whole number of bits in memory, which does not align with int3
compression of 5 trits per byte. This may be handled by using one of the
reserved bits (e.g. bit 3) as a flag that would function similarly to the prior
int4 flag, but that is outside the scope of the int2 implementation effort.
