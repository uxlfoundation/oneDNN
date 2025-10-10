# RFC: Graph Verbose Pipeline for SDPA Debugging and Reproducibility

## Introduction

This RFC proposes enhancements to the oneDNN Graph verbose logging mechanism to
better support PyTorch framework in reporting and debugging Scaled Dot-Product
Attention (SDPA) issues. The primary goal is to restore the usability and
reproducibility contract previously provided by primitive verbose logs, now for
the Graph API.

Currently, PyTorch leverages both the Graph API and Primitive API, but
reproducing issues requires different workflows:

- **Primitive API**: Users set the `ONEDNN_VERBOSE` environment variable to
  generate verbose logs, convert those logs to benchdnn command lines using a
  converter tool, and then run benchdnn for validation or debugging.
- **Graph API**: Users must enable an additional CMake option
  (`ONEDNN_ENABLE_GRAPH_DUMP`) and set the `ONEDNN_GRAPH_DUMP` environment
  variable to produce graph JSON files, which are then used with benchdnn.

This inconsistency complicates the debugging and reproducibility process for
users. Since graph JSON files are already adopted in benchdnn and effective for
reproducing issues, bridging the gap between verbose logs and graph JSONs will
streamline workflows and maintain the established usability contract.

## Proposal

### Option 1: Extend Verbose Logs with Minimal Fields

This option proposes enhancing the graph verbose log format to include only the
essential fields needed to convert logs into JSONs for SDPA cases. The goal is
to keep the logs concise and maintain compatibility with the original graph
verbose format.

For example, comparing the JSON file
([sdpa-plain-simplified-f16-f32.json](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/inputs/graph/complex_fusion/mha/sdpa-plain-simplified-f16-f32.json))
and its corresponding verbose log:

```shell
onednn_verbose,v1,graph,exec,gpu,100002,sdp,matmul_qk;scale_div;mask_add;softmax;matmul_v,,in0_f16:1:strided:undef:1x16x384x64:393216s24576s64s1 in1_f16:2:strided:undef:1x16x384x64:393216s24576s64s1 in2_f16:4:strided:constant:1:1 in3_f16:5:strided:undef:1x1x384x384:147456s147456s384s1 in4_f16:3:strided:undef:1x16x384x64:393216s24576s64s1 out0_f16:6:strided:undef:1x16x384x64:393216s24576s64s1,fpm:strict,sdp_primitive_v1_kernel_t,dnnl_backend,0.874023
```

**Fields currently covered in the verbose log:**

- engine_kind
- fpmath_mode
- input_ports
- output_ports
- op_name
- graph input/output logical tensors (id, dtype, shape, stride, layout, property_type)

**Fields missing from the verbose log:**

- oneDNN version
- fpmath_mode_apply_to_int
- op_id
- op_kind
- op_attrs
- graph intermediate logical tensors (id, dtype, shape, stride, layout, property_type)
- op connections

#### Minimal Viable Product (MVP) for SDPA Reproducibility

- Replace op_name with op_kind in verbose logs
- Add graph intermediate logical tensor ids and dtypes to represent op connections
- Internally generate remaining missing fields (version,
  fpmath_mode_apply_to_int, op_ids, op_attrs, shape, stride, layout,
  property_type)

**Example:**

```shell
onednn_verbose,v1,graph,exec,cpu,100002,sdp,MatMul:1xf16+2xf16:101xf32;Divide:101xf32+4xf16:102xf32;Add:102xf32+5xf16:103xf32;SoftMax:103xf32:104xf16;MatMul:104xf16+3xf16:6xf16,,,in0_f16:1:strided:undef:1x16x384x64:393216s24576s64s1 in1_f16:2:strided:undef:1x16x384x64:393216s24576s64s1 in2_f16:4:strided:constant:1:1 in3_f16:5:strided:undef:1x1x384x384:147456s147456s384s1 in4_f16:3:strided:undef:1x16x384x64:393216s24576s64s1 out0_f16:6:strided:undef:1x16x384x64:393216s24576s64s1,fpm:strict,larger_partition_kernel_t,dnnl_backend,18.2351
```

The key difference in this verbose log is the explicit encoding of each
operation as `op_kind`:`input_idx`x`dtype`:`output_id`x`dtype`, which clarifies
the connections and data types between graph operations.

**Pros:**

- Minimal changes to the existing format
- Short verbose lines

**Cons:**

- Some fields must be inferred or generated internally, which requires a verbose
  converter tool.
- The minimal format may omit details needed for advanced SDPA patterns, such as
  specific op attributes, quantization modes (per-tensor vs. per-channel), or
  SoftMax configurations (none vs. inf_as_zero). In these cases, the full graph
  dump is recommended for complete reproducibility.
- For current PyTorch SDPA usage, the pattern structures are limited and can be
  reproduced with a small set of arguments and fixed patterns. However,
  supporting the full flexibility of the Graph API would require handling much
  more dynamic and complex pattern structures.

### Option 2: Serialize Full Graph JSON in Verbose Logs

This option proposes directly serializing the full graph JSON and printing it as
part of the verbose log. This approach provides all the information required for
debugging and reproducibility, eliminating the need for a separate converter
tool.

**Example:**

```shell
onednn_verbose,v1,graph,info,serialize graph,
{"version": "3.10.0","engine_kind": "cpu","fpmath_mode": "strict","fpmath_mode_apply_to_int": "false","input_ports": [1,2,4,5,3],"output_ports": [6],"graph": [{"id": 0,"name": "matmul_qk","kind": "MatMul","attrs": {"transpose_a": {"type": "bool","value": 0},"transpose_b": {"type": "bool","value": 1}},"inputs": [{"id": 1,"dtype": "f16","shape": [1,16,384,64],"stride": [393216,24576,64,1],"layout_type": "strided","property_type": "undef"},{"id": 2,"dtype": "f16","shape": [1,16,384,64],"stride": [393216,24576,64,1],"layout_type": "strided","property_type": "undef"}],"outputs": [{"id": 101,"dtype": "f32","shape": [1,16,384,384],"stride": [2359296,147456,384,1],"layout_type": "strided","property_type": "undef"}]},{"id": 1,"name": "scale_div","kind": "Divide","attrs": {"auto_broadcast": {"type": "string","value": "numpy"}},"inputs": [{"id": 101,"dtype": "f32","shape": [1,16,384,384],"stride": [2359296,147456,384,1],"layout_type": "strided","property_type": "undef"},{"id": 4,"dtype": "f16","shape": [1],"stride": [1],"layout_type": "strided","property_type": "constant"}],"outputs": [{"id": 102,"dtype": "f32","shape": [1,16,384,384],"stride": [2359296,147456,384,1],"layout_type": "strided","property_type": "undef"}]},{"id": 2,"name": "mask_add","kind": "Add","attrs": {"auto_broadcast": {"type": "string","value": "numpy"}},"inputs": [{"id": 102,"dtype": "f32","shape": [1,16,384,384],"stride": [2359296,147456,384,1],"layout_type": "strided","property_type": "undef"},{"id": 5,"dtype": "f16","shape": [1,1,384,384],"stride": [147456,147456,384,1],"layout_type": "strided","property_type": "undef"}],"outputs": [{"id": 103,"dtype": "f32","shape": [1,16,384,384],"stride": [2359296,147456,384,1],"layout_type": "strided","property_type": "undef"}]},{"id": 3,"name": "softmax","kind": "SoftMax","attrs": {"axis": {"type": "s64","value": -1},"mode": {"type": "string","value": "inf_as_zero"}},"inputs": [{"id": 103,"dtype": "f32","shape": [1,16,384,384],"stride": [2359296,147456,384,1],"layout_type": "strided","property_type": "undef"}],"outputs": [{"id": 104,"dtype": "f16","shape": [1,16,384,384],"stride": [2359296,147456,384,1],"layout_type": "strided","property_type": "undef"}]},{"id": 4,"name": "matmul_v","kind": "MatMul","attrs": {"transpose_a": {"type": "bool","value": 0},"transpose_b": {"type": "bool","value": 0}},"inputs": [{"id": 104,"dtype": "f16","shape": [1,16,384,384],"stride": [2359296,147456,384,1],"layout_type": "strided","property_type": "undef"},{"id": 3,"dtype": "f16","shape": [1,16,384,64],"stride": [393216,24576,64,1],"layout_type": "strided","property_type": "undef"}],"outputs": [{"id": 6,"dtype": "f16","shape": [1,16,384,64],"stride": [393216,24576,64,1],"layout_type": "strided","property_type": "undef"}]}]}
```

**Pros:**

- Provides complete and explicit information for reproducibility
- Removes the need for a verbose converter tool

**Cons:**

- The output deviates from the original graph verbose format
- Verbose logs can become lengthy and less human-readable, especially for large
  graphs. Possible mitigations include omitting default attributes, abbreviating
  field names, or using enums for layout/property types, but some
  post-processing may still be required for downstream tools

### Option 3: Enable `ONEDNN_ENABLE_GRAPH_DUMP` by Default

This option proposes enabling the `ONEDNN_ENABLE_GRAPH_DUMP` CMake option by
default, so users do not need to manually configure it for graph debugging. By
default, no graph JSON files are generated unless the
`ONEDNN_GRAPH_DUMP` environment variable is set, so performance remains
unaffected. When `ONEDNN_GRAPH_DUMP=subgraph` is specified, graph JSON files are
dumped during partition compilation (calling the partition::compile API), and
corresponding verbose lines are printed:

```shell
onednn_verbose,v1,graph,info,serialize graph to a json file graph-100002-9333857264675079303.json
```

The file name format is `graph`-`partition id`-`partition hash key`.json, where
the partition id matches the one in the verbose log, allowing users to identify
relevant cases:

```shell
onednn_verbose,v1,graph,exec,gpu,100002,sdp,matmul_qk;scale_div;mask_add;softmax;matmul_v,,in0_f16:1:strided:undef:1x16x384x64:393216s24576s64s1 in1_f16:2:strided:undef:1x16x384x64:393216s24576s64s1 in2_f16:4:strided:constant:1:1 in3_f16:5:strided:undef:1x1x384x384:147456s147456s384s1 in4_f16:3:strided:undef:1x16x384x64:393216s24576s64s1 out0_f16:6:strided:undef:1x16x384x64:393216s24576s64s1,fpm:strict,sdp_primitive_v1_kernel_t,dnnl_backend,0.874023
```

**Pros:**

- Removes manual CMake configuration for graph dumps, which simplifies
  generation of graph JSON files for debugging
- Minimal changes required to the library

**Cons:**

- The environment variable required for graph dumps does not match the primitive
  API, so full consistency is not achieved. This could be mitigated by
  automatically setting `ONEDNN_GRAPH_DUMP=subgraph` when
  `ONEDNN_VERBOSE=profile_exec` is enabled
- Possible increase in disk usage if dumps are enabled frequently
- Frameworks need to provide both verbose logs and JSON files for issue reproduction

## Scope

This RFC targets the graph API, specifically the `partition::compile` and
`compiled_partition::execute` interfaces. Any usage of internal interfaces (eg.
OpenVINO) are not guaranteed to get this information.
