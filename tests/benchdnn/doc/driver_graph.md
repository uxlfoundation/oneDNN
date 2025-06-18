# Graph Driver

## Usage

``` sh
    ./benchdnn --graph [benchdnn-knobs] [graph-knobs] [graph-case] ...
```

* [graph-knobs] can have the following attributes:

  - `--mb=INT` -- Override minibatch size specified in the JSON file default
      case. When set to `0`, use batch size as defined by the
      individual test case. The option doesn't take effect for
      operations that don't support the `mb` concept. The default is `0`.

  - `--in-shapes=ID:SHAPE[*STRIDES][+ID:SHAPE*STRIDES+...]` -- Override a shape
    and strides of a graph input tensor that includes `ID` in a graph with
    `SHAPE` and `STRIDES` values. The `STRIDES` values and `SHAPE` are separated
    by a `*`. Multiple inputs may be specified using the `+` delimiter.

    `STRIDES` can be specified in two different formats: 
      - Fully specified through a vector, of which the size should be the same
      as that of `SHAPE`.
      - Represented as a `Tag`. `Tag` should be provided in a string, of which
      the length should be the same as that of `SHAPE`. For instance, for 4D
      tensor, the `Tag` should contain 4 letters from `a` to `d`, which
      indicates the first, second, third, and fourth dimensions, respectively.
      The `Tag` is used as as a convenience shortcut to the alternative, full
      specification method via tensor with a similar look to shapes which
      implies the dimensions. For instance, for a tensor with shape `[1,32,4,2]`
      and strides `[256,8,2,1]`, the `Tag` of which can be represented as
      `abcd`. If the user wants to modify the strides to `[256,8,1,4]`, the
      `Tag` should be provided as `abdc`.

    Some notes:
      - If both `--mb` and `--in-shapes` are set, `--mb` takes precedence over
      `--in-shapes`. If only `SHAPE` rewriting is provided, the input tensors
      will be in the dense format, such as `abcd`.
      - An asterisk `*` is mandatory to allow modification of `STRIDES` without
      shape modification.
      - The shape of internal tensors and graph output tensors is inferred by
      the graph driver automatically. By default, the option value is empty,
      meaning values are taken from the original graph.

    Examples are provided below.

  - `--op-attrs=ID:ATTR_STRING[+ID:ATTR_STRING]` -- Override a series attributes
              value of op with `ID` in the graph with `ATTR_STRING` values.
              `ATTR_STRING` is `ATTR_NAME:ATTR_VALUE[*ATTR_NAME:ATTR_VALUE]`.
              Multiple attributes value changes may be specified using the `*`
              delimeter. Multiple ops modification may be specified using the `+`
              delimeter. By default, the option value is empty, meaning values are taken from original graph.
  - `--expected-n-partitions=INT` -- Specify the number of expected partitions
      returned from the graph. `INT` is a non-negative integer value. When `INT`
      value is `0`, the check is skipped. By default, the value is `1` which means
      the graph should be fused as one partition.
  - `--dt={undef [default], f32, bf16, f16}` -- Specify the data types in the
    input JSON file. Currently, you can define data types for pure floating-point
    input graph only. For example, you can specify `--dt=f16` for an `f32` graph
    and then test it in `f16`. It has the same effect as changing the data type
    field of all logical tensors in the input JSON file from `f32` to `f16`. If
    `--dt` is not specified or specified as `undef`, the original data types
    contained in the input JSON file will be used for testing.
    - `--dt=ID:DT[+ID:DT]` -- Another format to specify the data types in the
      input JSON file. `ID` specifies the input or output tensor of an operation
      in the JSON file. `DT` is the target data type. To specify the data types of
      multiple tensors, use `+` to concatenate the `ID` and `DT` pairs. An error
      will occur if `ID` is not contained in the JSON file. oneDNN operations have
      restrictions for input and output tensor data types. Changing the data type
      of a tensor may lead to graph construction failures, for example, failure to
      perform the `add_op()` operation in the graph.
  - `--op-kind=ID:KIND[+ID:KIND]` -- Override a series of operation kinds in the
    input JSON file. `ID` specifies the operations in the graph, and `KIND`
    indicates the target operation kind. To specify the kind of multiple
    operations, use `+` to concatenate the `ID` and `KIND` pairs. An error will
    occur if `ID` is not contained in the JSON file. Currently, this override
    behavior is only allowed for binary and eltwise operations. 

* [graph-case] is a JSON file which is dumped by a library or created from
  scratch. It must be passed to the graph driver as `--case=JSON_FILE`. Refer to
  the JSON file example at the end of this document.

The oneDNN Graph serialization feature to dump JSON files at runtime may be enabled
by using the `-DONEDNN_ENABLE_GRAPH_DUMP=ON` build time switch. By default, dump is
disabled. When the build option is on, and the `ONEDNN_GRAPH_DUMP=subgraph` environment
variable is specified, the library generates JSON files with partitions
returned.

## Limitations

* Graph driver doesn't support `--mode-modifier=M` or `--mode=F` (which contains
  `--mode-modifier=M` in it).

## Example

Run the demo `conv_post_ops_fusion` partition
[pattern/f32/conv_post_ops_fusion.json](../inputs/graph/pattern/f32/conv_post_ops_fusion.json)
with default input shapes and op attributes:

```shell
./benchdnn --mode=P --graph --case=./tests/benchdnn/inputs/graph/pattern/f32/conv_post_ops_fusion.json
```

If the JSON file is under `tests/benchdnn/inputs/graph`, we can use the relative
path as shown below,

```shell
./benchdnn --mode=P --graph --case=pattern/f32/conv_post_ops_fusion.json
```

Run the demo pattern with new input shapes by using `--in-shapes`:

```shell
# rewrite input shape only
./benchdnn --mode=C --graph --in-shapes=0:2x64x112x112+1:32x64x2x2 --case=pattern/f32/conv_post_ops_fusion.json
# rewrite strides with tag
./benchdnn --mode=C --graph --in-shapes=0:*dcba+1:*dcba --case=pattern/f32/conv_post_ops_fusion.json
# rewrite with fully specified strides
./benchdnn --mode=C --graph --in-shapes=0:*802816x25088x112x1 --case=pattern/f32/conv_post_ops_fusion.json
# rewrite both shape and strides with tag
./benchdnn --mode=C --graph --in-shapes=0:2x64x112x112*dcba+1:32x64x2x2*dcba --case=pattern/f32/conv_post_ops_fusion.json
# rewrite both shape and strides with full specification
./benchdnn --mode=C --graph --in-shapes=0:2x64x112x112*802816x12544x112x1 --case=pattern/f32/conv_post_ops_fusion.json
# rewrite rank
./benchdnn --mode=C --graph --in-shapes=0:2x64x112x112x112+1:32x64x2x2x2 --op-attrs=0:strides:1x1x1*pads_begin:0x0x0*pads_end:0x0x0*dilations:1x1x1 --case=pattern/f32/conv_post_ops_fusion.json
# rewrite rank and stride
./benchdnn --mode=C --graph --in-shapes=0:2x64x112x112x112*edcba+1:32x64x2x2x2*edcba --op-attrs=0:strides:1x1x1*pads_begin:0x0x0*pads_end:0x0x0*dilations:1x1x1 --case=pattern/f32/conv_post_ops_fusion.json
# rewrite rank to 0 rank with shape []
./benchdnn --mode=C --graph --in-shapes=0:- --case=op/f32/add.json
# rewrite to 1D tensor with shape [0]
./benchdnn --mode=C --graph --in-shapes=0:0+1:0 --case=op/f32/add.json
```

Run the demo `conv` op
[op/f32/conv_2d.json](../inputs/graph/op/f32/conv_2d.json) with new strides
attribute by using `--op-attrs`:

```shell
./benchdnn --mode=P --graph --op-attrs=,0:strides:4x4 --case=op/f32/conv_2d.json
```

Run a graph demo batch file [test_graph_ci](../inputs/graph/test_graph_ci):

```shell
./benchdnn --mode=P --graph --batch=test_graph_ci
```

Run same demo batch file on the GPU engine:

```shell
./benchdnn --mode=P --engine=gpu --graph --batch=test_graph_ci
```

Use `-v1` to get more test information, such as graph inputs id and shape,
partition numbers, and so on.

```shell
./benchdnn --mode=P -v1 --graph --mb=1,2,3 --case=op/f32/conv_2d.json
```

Use `--mode=C` or `--mode=c` for correctness testing:

```shell
./benchdnn --mode=C --graph --case=op/f32/conv_2d.json
```

## Demo Cases

Demo JSON files are located in [inputs/graph](../inputs/graph), including
partitions (FP32 MLP partition) and single op (Convolution). Different
data type folders for ops and patterns are available. In general, a JSON file is named as
`workload-pattern_name-additional_info.json`. In this scheme, `workload` stands
for workload name, `pattern_name` stands for the fusion pattern returned by the
library, and `additional_info` differentiates cases based on other settings.
A single op JSON file was named with the op name directly.

## JSON File Example
<details>
    <summary>Conv JSON</summary>

~~~json
{
  "version": "0.5.0",
  "engine_kind": "cpu",
  "fpmath_mode": "strict",
  "graph": [
    {
      "id": 0,
      "name": "Convolution",
      "kind": "Convolution",
      "attrs": {
        "strides": {
          "type": "s64[]",
          "value": [
            2,
            2
          ]
        },
        "pads_begin": {
          "type": "s64[]",
          "value": [
            0,
            0
          ]
        },
        "auto_pad": {
          "type": "string",
          "value": "None"
        },
        "data_format": {
          "type": "string",
          "value": "NCX"
        },
        "pads_end": {
          "type": "s64[]",
          "value": [
            -1,
            -1
          ]
        },
        "groups": {
          "type": "s64",
          "value": 1
        },
        "dilations": {
          "type": "s64[]",
          "value": [
            1,
            1
          ]
        },
        "weights_format": {
          "type": "string",
          "value": "OIX"
        }
      },
      "inputs": [
        {
          "id": 0,
          "dtype": "f32",
          "shape": [
            28,
            512,
            28,
            28
          ],
          "stride": [
            401408,
            1,
            14336,
            512
          ],
          "layout_type": "strided",
          "property_type": "undef"
        },
        {
          "id": 1,
          "dtype": "f32",
          "shape": [
            1024,
            512,
            1,
            1
          ],
          "stride": [
            512,
            1,
            1,
            1
          ],
          "layout_type": "strided",
          "property_type": "constant"
        }
      ],
      "outputs": [
        {
          "id": 2,
          "dtype": "f32",
          "shape": [
            28,
            1024,
            14,
            14
          ],
          "stride": [
            200704,
            1,
            14336,
            1024
          ],
          "layout_type": "strided",
          "property_type": "undef"
        }
      ]
    }
  ]
}
~~~
</details>
