Dropout {#dev_guide_op_dropout}
===============================

## General

Dropout operation is a regularization technique that randomly zeroes elements of
the input tensor during training, scaling the outputs so that the expected sum
of output elements matches the sum of input elements. It depends on a
deterministic PRNG (current implementation uses a variation of Philox algorithm)
and transforms the values as follows:

\f[
    \mathrm{mask}[:] = (\mathrm{PRNG}(seed, offset, :) > P) \\
    \mathrm{dst}[:] = \mathrm{mask}[:] \cdot {{\mathrm{src}[:]} \over {1 - P}}
\f]

where:

* \f$\mathrm{mask}\f$ values may be either 0 if the corresponding value in
  \f$\mathrm{dst}\f$ got zeroed (a.k.a. dropped out) or 1, otherwise.
* \f$seed, offset\f$ are the seed and the offset for the PRNG
  algorithm. The seed initializes the PRNG state, while the offset allows
  generating different random sequences from the same seed, ensuring
  reproducibility across executions.
* \f$P\f$ is the probability for any given value to get dropped out,
  \f$0 \leq P \leq 1\f$.

**Forward (Training)** applies the dropout mask and scaling as described above.
**Forward (Inference)** passes the input directly to the output without modification.
**Backward** applies the same mask and scaling to the gradient
\f$\mathrm{diff\_dst}\f$ to compute \f$\mathrm{diff\_src}\f$.

## Execution arguments

The inputs and outputs must be provided according to the following index order
when constructing an operation.

### Inputs

| Index | Argument Name | Description                | Required or Optional |
|:------|:--------------|:---------------------------|:---------------------|
| 0     | `src`         | Input tensor               | Required             |
| 1     | `seed`        | Random seed                | Required             |
| 2     | `offset`      | Random offset              | Required             |
| 3     | `probability` | Dropout probability        | Required             |

### Outputs

| Index | Argument Name | Description                | Required or Optional |
|:------|:--------------|:---------------------------|:---------------------|
| 0     | `dst`         | Output tensor              | Required             |

## Supported data types

Dropout operation supports the following data type combinations.

| src   | dst   | mask | seed  | offset | probability |
|:------|:------|:-----|:------|:-------|:------------|
| f32   | f32   | u8   | u64   | u64    | f32         |
| bf16  | bf16  | u8   | u64   | u64    | f32         |
| f16   | f16   | u8   | u64   | u64    | f32         |

## Implementation Limitations

1. The `seed`, `offset`, and `probability` inputs only support logical tensors
   with property type set to `host_scalar`.
