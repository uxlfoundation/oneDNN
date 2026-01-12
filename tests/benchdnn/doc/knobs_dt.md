# Data Types

**Benchdnn** supports the same data types as the library does (memory::data_type
enum). If an unsupported data type is specified, an error will be reported.
The following data types are supported:

| Data type | Description
| :---      | :---
| f32       | standard float
| bf16      | 2-byte float (1 sign bit, 8 exp bits, 7 mantissa bits)
| f16       | 2-byte float (1 sign bit, 5 exp bits, 10 mantissa bits)
| s8/u8     | standard signed/unsigned char or int8_t/uint8_t
| s4/u4     | signed/unsigned 4-bit integer
| s32       | standard int or int32_t
| f8_e5m2   | 1-byte float (1 sign bit, 5 exp bits, 2 mantissa bits)
| f8_e4m3   | 1-byte float (1 sign bit, 4 exp bits, 3 mantissa bits)
| e8m0      | 1-byte float (8 exp bits), supported for scales only
| f4_e2m1   | half-byte float (1 sign bit, 2 exp bits, 1 mantissa bits)
| f4_e3m0   | half-byte float (1 sign bit, 3 exp bits, 0 mantissa bits)
| f64       | double precision float

