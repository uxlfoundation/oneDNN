# Graph Driver

The benchdnn graph driver reads a serialized oneDNN Graph from a JSON file,
optionally rewrites selected graph properties, partitions the graph, and runs
the resulting partitions. It supports correctness and performance testing on
CPU and GPU engines.

Refer to @ref dev_guide_graph_dump for information about generating serialized
graph files at runtime.

## Usage

``` sh
  ./benchdnn --graph [benchdnn-knobs] [graph-knobs] --case=JSON_FILE
  ./benchdnn --graph [benchdnn-knobs] --batch=BATCH_FILE
```

Options may appear before or after `--graph`. Common benchdnn options, such as
`--mode`, `--engine`, and verbosity options, are described in the general
benchdnn documentation. The graph-specific knobs are described below.

`JSON_FILE` may be an absolute path or a path relative to the current working
directory. If it is not found there, benchdnn also searches its configured
input directories. Consequently, files installed under the graph input
directory can be specified relative to `tests/benchdnn/inputs/graph`, for
example `--case=op/f32/conv_2d.json`.

Graph JSON files can be visualized with [Netron](https://netron.app/). The
visualized graph shows operation and logical tensor IDs, which are useful when
constructing command-line rewrites.

## Graph Options

Options that accept alternative values describe their comma-separated syntax
below. Alternatives specified by different options form a Cartesian product.

### `--mb=INT[,INT...]`

Overrides the minibatch dimension of graph inputs for which the driver
recognizes a minibatch dimension. The default is `0`, which preserves the value
from the JSON file. The option has no effect on operations without a minibatch
concept. Comma-separated values run separate tests; for example, `--mb=1,2,3`
runs three tests.

When an input is also named by `--in-shapes`, `--mb` replaces only the first
dimension of the supplied shape; the remaining dimensions and layout come from
`--in-shapes`.

### `--in-shapes=REWRITE[,REWRITE...]`

Overrides the shape or layout of graph boundary tensors. Despite the option
name, both graph inputs and graph outputs may be named; internal tensors may
not. Internal and unspecified output shapes are inferred after rewriting.

Comma-separated `REWRITE` values run separate tests. Each `REWRITE` has one or
more `ID:VALUE` entries joined with `+` as `ID:VALUE[+ID:VALUE]`. Each
`ID:VALUE` entry can be defined as:

```text
ID:SHAPE
ID:*TAG
ID:*STRIDES
ID:SHAPE*TAG
ID:SHAPE*STRIDES
```

- `ID` is a logical tensor ID from the JSON file.
- `SHAPE` is a dimension list separated by `x`, such as `2x64x112x112`. `scalar`
  specifies a rank-zero tensor. A value such as `0` is a rank-one tensor whose
  dimension is zero.
- `STRIDES` is an explicit stride list with the same rank as the tensor.
- `TAG` is a permutation of letters from `a` through the letter corresponding
  to the tensor rank. For example, `abcd` represents dense row-major layout for
  a four-dimensional tensor, while `abdc` swaps the last two dimensions.

The `*` is required when specifying a tag or explicit strides. Omitting a shape,
as in `0:*abdc`, preserves the current shape. Specifying only a shape assigns a
dense layout of the corresponding rank.

Examples:

``` sh
# Run two tests, each rewriting the shapes of tensors 0 and 1.
./benchdnn --mode=C --graph \
  --in-shapes=0:2x64x112x112+1:32x64x2x2,0:4x64x56x56+1:32x64x1x1 \
  --case=pattern/f32/conv_post_ops_fusion.json

# Rewrite two input shapes in one test.
./benchdnn --mode=C --graph \
  --in-shapes=0:2x64x112x112+1:32x64x2x2 \
  --case=pattern/f32/conv_post_ops_fusion.json

# Preserve the shape and rewrite only the layout or explicit strides.
./benchdnn --mode=C --graph --in-shapes=0:*dcba \
  --case=pattern/f32/conv_post_ops_fusion.json
./benchdnn --mode=C --graph --in-shapes=0:*401408x12544x112x1 \
  --case=pattern/f32/conv_post_ops_fusion.json

# Rewrite shape and layout together.
./benchdnn --mode=C --graph \
  --in-shapes=0:2x64x112x112*acdb+1:32x64x2x2*abcd \
  --case=pattern/f32/conv_post_ops_fusion.json

# Rewrite a scalar and a one-dimensional zero-size tensor.
./benchdnn --mode=C --graph --in-shapes=0:scalar --case=op/f32/add.json
./benchdnn --mode=C --graph --in-shapes=0:0+1:0 --case=op/f32/add.json
```

### `--op-attrs=REWRITE[,REWRITE...]`

Overrides operation attributes. Each `REWRITE` has the form:

```text
OP_ID:ATTR_NAME:ATTR_VALUE[*ATTR_NAME:ATTR_VALUE...]
[+OP_ID:ATTR_NAME:ATTR_VALUE[*ATTR_NAME:ATTR_VALUE...]...]
```

Use `*` for multiple attributes of one operation, `+` for multiple operations
in one test, and `,` for alternative tests. An empty comma-separated entry
selects the attributes from the JSON file; for example,
`--op-attrs=,0:auto_broadcast:numpy` tests both the original attributes and the
override. Specify `ATTR_NAME:-` to remove an attribute from the operation.
Attribute names, value types, and constraints are defined by the corresponding
oneDNN Graph operation.

``` sh
./benchdnn --mode=C --graph \
  --op-attrs=0:auto_pad:SAME_LOWER*kernel:2x2 \
  --case=op/f32/avgpool.json

# Change qtype and remove group_shape from operation 34107656704.
./benchdnn --mode=C --graph \
  --op-attrs=34107656704:qtype:per_tensor*group_shape:- \
  --in-shapes=1:1+2:1 \
  --case=complex_fusion/mha/sdpa-compressed-k-int8-gs32.json
```

### `--expected-n-partitions=INT`

Checks that graph partitioning produces exactly `INT` partitions. The default
is `1`. Set the value to `0` to disable the partition-count check. `INT` must be
non-negative.

### `--dt=DT[,DT...]`

Rewrites all logical tensor data types in a floating-point graph. Supported
values are `undef`, `f32`, `bf16`, and `f16`. `undef`, the default, preserves
the JSON data types. This global form is intended for graphs whose tensors have
floating-point data types; operation data type constraints still apply.

### `--dt=ID:DT[+ID:DT...][,ID:DT...]`

Rewrites individual logical tensors. `ID` may identify an input or output of
an operation. Use `+` to rewrite multiple tensors together and `,` to specify
alternative rewrite sets. Every ID must exist in the graph, and the resulting
data types must satisfy the operation schemas.

The global and ID-based `--dt` forms are mutually exclusive within one set of
driver settings.

``` sh
./benchdnn --mode=C --graph --dt=f16 --case=op/f32/conv_2d.json
./benchdnn --mode=C --graph --dt=0:f32+1:f32 \
  --case=complex_fusion/mha/sdpa-plain-training-forward-bf16-f32.json
```

### `--op-kind=REWRITE[,REWRITE...]`

Overrides operation kinds using `OP_ID:KIND` entries. Join multiple operation
changes in one test with `+`, and use `,` for alternative tests. Kind
substitution is supported for compatible binary and eltwise operations.

Setting `KIND` to `undef` removes the operation. A removed unary operation is
bypassed. For a binary operation, the driver bypasses the input produced by
another operation; removal fails if that input cannot be determined
unambiguously.

``` sh
# Run the same graph as Add, Divide, Maximum, and Subtract alternatives.
./benchdnn --mode=C --graph \
  --op-kind=0:Add,0:Divide,0:Maximum,0:Subtract \
  --case=op/f32/add.json
```

### `--tensor-property=REWRITE[,REWRITE...]`

Overrides logical tensor properties using `ID:PROPERTY` entries. Join multiple
changes in one test with `+`, and use `,` for alternative tests. `PROPERTY`
must be one of `undef`, `variable`, `constant`, or `host_scalar`, and every ID
must exist in the graph.

``` sh
./benchdnn --mode=C --graph \
  --tensor-property=3:undef,3:host_scalar \
  --case=complex_fusion/mha/sdpa-plain-simplified-f32.json
```

### `--attr-fpmath=MODE[:APPLY_TO_INT][,MODE[:APPLY_TO_INT]...]`

Overrides the graph floating-point math mode. `MODE` is one of `strict`,
`bf16`, `f16`, `tf32`, or `any`. `APPLY_TO_INT` is a Boolean and defaults to
`false`. Comma-separated values create alternative tests.

``` sh
./benchdnn --mode=C --graph \
  --attr-fpmath=strict:false,bf16:false,tf32:false \
  --case=pattern/int8/int8_bf16_matmul.json
```

## Batch Files

`--batch=BATCH_FILE` reads benchdnn arguments from a text file. Batch files may
include other batch files. The maintained graph suites under
`tests/benchdnn/inputs/graph` are composed this way; for example,
`test_graph_ci` includes the operation, pattern, and complex-fusion CI
harnesses.

Run the CI suite on CPU or GPU with:

``` sh
./benchdnn --mode=C --graph --batch=test_graph_ci
./benchdnn --mode=C --engine=gpu --graph --batch=test_graph_ci
```

## Essence of Testing

The graph driver submits the serialized graph to the oneDNN Graph API, checks
the number of generated partitions, and compiles and executes each supported
partition. In correctness mode, benchdnn constructs a reference path from the
primitive-specific reference implementations for the operations in each
partition. It initializes equivalent input data for the graph and reference
paths and compares each partition's outputs with the reference results.

## Examples

Run a single correctness test:

``` sh
./benchdnn --mode=C --graph --case=op/f32/conv_2d.json
```

Run performance testing and print graph input IDs, shapes, and partition
information:

``` sh
./benchdnn --mode=P -v1 --graph --mb=1,2,3 --case=op/f32/conv_2d.json
```

The graph driver does not support bitwise mode (`--mode=B`) or parallel test
object creation (`--mode-modifier=P`). Other modes, including fast performance
mode (`--mode=F`), are supported. The no-reference-memory modifier
(`--mode-modifier=M`) is supported with modes that do not require reference
results.
