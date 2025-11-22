# Grouped GEMM for Mixture of Experts (MoE)

## Definition of the Operation

Grouped GEMM is a computational kernel used in Mixture of Experts (MoE) models, where tokens are dynamically routed to different expert networks.

Grouped GEMM processes variable-sized batches (subsets of tokens) per expert, which is a key difference from batched GEMM operations that assume uniform batch sizes across groups.

For each expert `i`:
```
output[i] = (src[i] * weights[i]) + bias[i]
```
Plus scaling factors, zero-point adjustments, etc. (as in regular matrix multiplication).

Expert `i` has dimensions: `src[Mi x K]`, `weights[K x N]`, `dst[Mi x N]`.

**Terms to be used and other considerations of MoE:**
- **Number of experts**: Total number of experts in the MoE layer.
- **TOP-K routing**: A routing strategy where each token is assigned to the top K experts according to some scoring mechanism.
- **Number of active experts OR group size**: Number of experts that received tokens after routing.
- **M dimension is Dynamic dimension**: M dimension (i.e., token count per expert) varies dynamically based on routing decisions.
And depending on the framework design, this information may only be available on the device side after routing.
- **Uniform K and N**: All experts share the same K and N dimensions, so weight shapes are the same.
- **Uniform scaling configurations**: All experts use the same scaling pattern (e.g., per-tensor, per-row), but different values.

## Frameworks Implementations

### PyTorch Grouped Scaled MM (`torch._scaled_mm_grouped`)

#### API

PyTorch provides [`torch/nn/functional.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py) API for grouped scaled matrix multiplication.
The `torch._scaled_grouped_mm` API is used internally:

```python
torch._scaled_mm_grouped(
    mat_a,                           # Input tensor A (concatenated groups)
    mat_b,                           # Input tensor B (concatenated groups)
    scale_a,                         # Scaling factors for mat_a (list[Tensor])
    scale_recipe_a,                  # Scaling type for mat_a (list[ScalingType])
    scale_b,                         # Scaling factors for mat_b (list[Tensor])
    scale_recipe_b,                  # Scaling type for mat_b (list[ScalingType])
    swizzle_a=None,                  # Swizzle pattern for scale_a (list[SwizzleType])
    swizzle_b=None,                  # Swizzle pattern for scale_b (list[SwizzleType])
    bias=None,                       # Optional bias tensor
    offs=None,                       # Offsets into source tensors (list[int])
    output_dtype=torch.bfloat16,     # Output data type
    contraction_dim=None,            # Which dims are K in matmul (list[int])
    use_fast_accum=False             # Enable fast accumulation (Nvidia GPUs)
)
```

| Parameter | Type | Description | Supported Values/Formats (*) |
|-----------|------|-------------|------------------------------|
| `mat_a`, `mat_b` | `Tensor` | Concatenated input matrices (all groups packed sequentially) | `float8_e4m3fn` (weights), `float8_e5m2` (activations), `int8`, or mixed FP8 |
| `scale_a`, `scale_b` | `list[Tensor]` | Scaling factors for dequantization | FP32 tensors or `float8_e8m0fnu` (for MXFP8) |
| `scale_recipe_a`, `scale_recipe_b` | `list[ScalingType]` | Scaling method enum | `PER_TENSOR` (single scale), `PER_ROW` (M-dim), `PER_COLUMN` (N/K-dim), `AXISWISE` (per-channel) |
| `swizzle_a`, `swizzle_b` | `list[SwizzleType]` | Memory access pattern control (optional) | Nvidia GPU-specific |
| `bias` | `Tensor` or `None` | Optional bias added to output | Not supported |
| `offs` | `list[int]` or `None` | Offsets for group boundaries | `int32` tensor: `[M0, M0+M1, M0+M1+M2, ...]`. First group starts at 0, `offs[i]` = end of group `i` |
| `output_dtype` | `torch.dtype` | Output tensor data type | `torch.bfloat16` (default), `torch.float16`, `torch.float32` |
| `contraction_dim` | `list[int]` or `None` | Dimensions representing K in matmul operation | |
| `use_fast_accum` | `bool` | Tensor-core fast accumulation | Nvidia GPU-specific |

(*) Based on initial investigation; support may have changed.

[See also TorchAO Drop-In Replacement](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training)
that supports various quantization schemes.

#### Memory Layouts

- All groups are concatenated into contiguous buffers per expert:
```python
mat_a:  [Group 0: M0xK][Group 1: M1xK][Group 2: M2xK] # [(M0 + M1 + M2) x K]
offs:    ^              ^              ^
         0              M0             M0 + M1        # [0, M0, M0 + M1]
```
- `offs` data is provided on the Device side.
- All expert weights present in memory regardless of token routing.

### OpenVINO MoE

OpenVINO implements a complete MoE operation in [PR #32469](https://github.com/openvinotoolkit/openvino/pull/32469):

| Primitive | Purpose | Description |
|-----------|---------|-------------|
| `moe_mask_gen` | Routing metadata generation | Generates per-expert token assignments and offsets from Top-K routing decisions |
| `moe_gather` | Token reorganization | Gathers tokens by expert assignment for contiguous memory access |
| `moe_gemm` | Grouped matrix multiplication | Performs expert-specific GEMMs with INT4/INT8/FP16 quantization support |
| `moe_scatter_reduction` | Result aggregation | Scatters expert outputs back to original token positions with weighted reduction |

**Data Types and Quantization Support:**
TBD

#### Memory Layouts

OpenVINO uses **per-expert indexing** with contiguous buffers organized by expert:

```cpp
// Inputs organized after moe_gather
float* input_data;                    // Contiguous by active experts (i.e., only experts with tokens)
int *expert_id;                       // Size = number of active experts (i.e., dynamic size), ids of only experts with tokens
int* experts_info_start_idx;          // Size = number of active experts (i.e., dynamic size), start offset for each expert's tokens
int* tokens_lens_per_expert;          // Size = number of active experts (i.e., dynamic size), number of tokens per expert

// Weights organized per expert
float* weight_data;                   // Contiguous: [Expert0_weights | Expert1_weights | ...]

// Per-expert scales for quantization
// TBD
```

**Data access pattern:**
```cpp
for (int i = 0; batch < total_token_assignments; batch++) {
    int expert_id = expert_ids[batch]; // Which expert to use

    // Fetch data
    int input_offset = expert_start_idx[expert_id];
    int weight_offset = expert_id * K * N; // Since weights shape is uniform across experts

    int4 weights = weight_data[weight_offset + ...];
    int8 input = input_data[input_offset + ...];

    // Scales, zero-points
    // TBD

    // Process...
}
```

- Offsets and sizes have dynamic size based on active experts.
- All expert weights present in memory regardless of token routing, we need to index into them based on `expert_id`.
- Current version provides offsets and sizes information on the Host.
However, we should expect this data to be coming from the Device, as previous gather/shuffle operation is most likely should be performed on the Device.

### vLLM MoE

Source code: [vllm-xpu/csrc/quantization/w8a8/cutlass/moe/](https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu/tree/main/csrc/quantization/w8a8/cutlass/moe)

#### API

Grouped Gemm API: [`vllm/_custom_ops.py::cutlass_blockwise_scaled_grouped_mm`](https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu/blob/main/vllm/_custom_ops.py#L717-L728)

```python
cutlass_blockwise_scaled_grouped_mm(
    output,           # Output tensor [total_tokens, N] (device memory, preallocated)
    a,                # Input tensor [total_tokens, K] (device memory, contiguous)
    b,                # Weight tensor [num_experts, N, K] (device memory, 3D)
    scales_a,         # Activation scales [total_tokens, K//128] (FP8 block-wise)
    scales_b,         # Weight scales [num_experts, N//128, K//128] (FP8 block-wise)
    problem_sizes,    # Problem size tensor [num_experts, 3] (device memory, int32)
    expert_offsets    # Cumulative offsets [num_experts + 1] (device memory, int32)
)
```

| Parameter | Shape | Data Type (*) | Description |
|-----------|-------|-----------|-------------|
| `a` | `[total_tokens, K]` | FP8/INT8 | Input activations with expert-grouped tokens: `[E0_tokens \| E1_tokens \| ...]` (device, contiguous) |
| `b` | `[num_experts, N, K]` | FP8/INT8 | Weight tensor, all experts present (even if unused) (device, 3D) |
| `output` | `[total_tokens, N]` | BF16/FP16 | Preallocated output buffer matching input token order (device, contiguous) |
| `scales_a` | `[total_tokens, K//128]` | FP32 | Block-wise (128x128) activation scales (device) |
| `scales_b` | `[num_experts, N//128, K//128]` | FP32 | Block-wise weight scales per expert (device) |
| `problem_sizes` | `[num_experts, 3]` | INT32 | Per-expert `[M, N, K]` where `M` varies, `N` and `K` are uniform (device) |
| `expert_offsets` | `[num_experts + 1]` | INT32 | Cumulative token counts: `[0, M0, M0+M1, ..., sum(Mi)]` (device) |

(*) Based on initial investigation; support may have changed.

#### Memory Layout

- Similar to PyTorch, all groups are concatenated into contiguous buffers per expert.
- `problem_sizes` and `expert_offsets` are Device-side tensors:

```python
problem_sizes[expert_id] = [M_i, N, K]
# M_i: Number of tokens assigned to expert_id (can be 0!)
# N, K: Uniform across all experts

expert_offsets = [0, 2, 2, 2, 5]  # Cumulative: experts with M=0 contribute 0 offset delta
```

- All expert weights present in memory regardless of token routing.

## oneDNN Grouped GEMM Support

### Proposal #1: Pointer-Based API and a New Primitive

- **Pointer-based API**: Use arrays of pointers to oneDNN memory objects for per-expert data,
    so that each expert has separate memory descriptors.
    This approach follows the pattern used in concat and sum, where oneDNN
    accepts multiple input memory objects via the `DNNL_ARG_MULTIPLE_SRC` argument
    (e.g., `DNNL_ARG_MULTIPLE_SRC + i` for input `i`).
- **Per-expert attributes**: Scaling factors are then configured via `DNNL_ARG_MULTIPLE_*` arguments as well.
- **New primitive**: matmul's API accepts single memory descriptors for src/weights/dst.
    The new `grouped_gemm` primitive would accept vectors of memory descriptors and handle per-expert execution
    (in C API we would also need to provide group count).
    There are additional challenges with handling device-side runtime dimensions in this approach, so a
    new primitive may be more suitable.

Note:
Frameworks with contiguous memory layouts could be mapped to pointer-based API:

```cpp
// Build pointer array (zero-copy)
for (int i = 0; i < total_number_of_experts; i++) {
    expert_ptrs[i] = (void*)contiguous_data + offsets_per_expert[i];
}
```

Moreover, if each expert's data was initially allocated using some `M_max`,
then there is never a need to recalculate pointers.

If expert's data is continuously stored/repacked into contiguous buffers,
then pointers need to be recalculated at each execution as well.

#### Memory Descriptors

```cpp
// Expert i has dimensions: src[Mi x K], weights[K x N], dst[Mi x N]
// Mi varies per expert (runtime dimension), K and N are fixed

// Declare runtime dimension using DNNL_RUNTIME_DIM_VAL
memory::dims src_shape = {DNNL_RUNTIME_DIM_VAL, K};  // M varies per expert
memory::dims weight_shape = {K, N};                  // Fixed dimensions
memory::dims bias_shape = {N};                       // 1D bias
memory::dims dst_shape = {DNNL_RUNTIME_DIM_VAL, N};  // M varies per expert

// Create memory descriptors (one per expert)
memory::desc src_md(src_shape, memory::data_type::f32, memory::format_tag::ab);
memory::desc weight_md(weight_shape, memory::data_type::f32, memory::format_tag::ab);
memory::desc bias_md(bias_shape, memory::data_type::f32, memory::format_tag::a);
memory::desc dst_md(dst_shape, memory::data_type::f32, memory::format_tag::ab);

// Build vectors for all experts
std::vector<memory::desc> src_mds(num_experts, src_md);
std::vector<memory::desc> weight_mds(num_experts, weight_md);
std::vector<memory::desc> bias_mds(num_experts, bias_md);
std::vector<memory::desc> dst_mds(num_experts, dst_md);
```

#### Scales Configuration

Scales are configured via `primitive_attr` before primitive descriptor creation:

```cpp
primitive_attr attr;

// Per-tensor scaling (mask = 0):
attr.set_scales_mask(DNNL_ARG_MULTIPLE_SRC + expert_id, 0);
attr.set_scales_mask(DNNL_ARG_MULTIPLE_WEIGHTS + expert_id, 0);
attr.set_scales_mask(DNNL_ARG_MULTIPLE_DST + expert_id, 0);

// Row-wise scaling for source (mask = 1):
attr.set_scales_mask(DNNL_ARG_MULTIPLE_SRC + expert_id, 1);

// Column-wise scaling for weights (mask = 2):
attr.set_scales_mask(DNNL_ARG_MULTIPLE_WEIGHTS + expert_id, 2);
```

#### Primitive Creation

```cpp
// Create primitive descriptor
// Here we still do not possess actual M_per_expert values, but the group count is known
auto gemm_pd = grouped_gemm::primitive_desc(
    eng,                    // Engine
    src_mds,                // Vector of source memory descriptors
    weight_mds,             // Vector of weight memory descriptors
    bias_mds,               // Vector of bias memory descriptors
    dst_mds,                // Vector of destination memory descriptors
    attr                    // Primitive attributes (optional, e.g., with scales)
);

// Create primitive
auto gemm_prim = grouped_gemm(gemm_pd);
```

> **Note:** The C++ API infers `num_experts` from the vector sizes.
    The C API will require an explicit `int group_count` parameter.

#### Execution

Simple case with `Mi` known on host, see below for device-side runtime dimensions support.

```cpp
// Host has M_per_expert values
std::vector<dim_t> M_per_expert = {64, 82, 57, 53};  // From routing

// Create memory objects with resolved dimensions per expert
std::vector<memory> src_mem, weight_mem, bias_mem, dst_mem;
for (int i = 0; i < num_experts; i++) {
    src_mem.push_back(
        memory({{M_per_expert[i], K_dim}, memory::data_type::f32,
                memory::format_tag::ab},
               eng, (void*)input_ptrs[i]));
    weight_mem.push_back(
        memory({{K_dim, N_dim}, memory::data_type::f32,
                memory::format_tag::ab},
               eng, (void*)weight_ptrs[i]));
    bias_mem.push_back(
        memory({{N_dim}, memory::data_type::f32, memory::format_tag::a},
               eng, (void*)bias_ptrs[i]));
    dst_mem.push_back(
        memory({{M_per_expert[i], N_dim}, memory::data_type::f32,
                memory::format_tag::ab},
               eng, (void*)output_ptrs[i]));
}

std::unordered_map<int, memory> args;

// Per-expert input/output memory
for (int i = 0; i < num_experts; ++i) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, src_mem[i]});
    args.insert({DNNL_ARG_MULTIPLE_WEIGHTS + i, weight_mem[i]});
    args.insert({DNNL_ARG_MULTIPLE_BIAS + i, bias_mem[i]});
    args.insert({DNNL_ARG_MULTIPLE_DST + i, dst_mem[i]});
}

// Per-expert scale memory (if configured)
// ... scales code ...

// Execute
gemm_prim.execute(stream, args);
stream.wait();
```

#### Scales Support

Scales support are very straightforward with pointer-based API,
just need to provide per-expert scale memory objects via `DNNL_ARG_MULTIPLE_*` arguments:

```cpp
// Per-expert scale memory
// Assume per-tensor scales for simplicity
if (scales_configured) {
    std::vector<memory> scale_src_mem(num_experts);
    std::vector<memory> scale_wei_mem(num_experts);
    std::vector<memory> scale_dst_mem(num_experts);

    for (int i = 0; i < num_experts; ++i) {
        scale_src_mem[i] = memory({{1}, memory::data_type::f32,
                                   memory::format_tag::x},
                                  eng, scale_src_data[i].data());
        scale_wei_mem[i] = memory({{1}, memory::data_type::f32,
                                   memory::format_tag::x},
                                  eng, scale_wei_data[i].data());
        scale_dst_mem[i] = memory({{1}, memory::data_type::f32,
                                   memory::format_tag::x},
                                  eng, scale_dst_data[i].data());

        args.insert({DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_SRC + i),
            scale_src_mem[i]});
        args.insert({DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_WEIGHTS + i),
            scale_wei_mem[i]});
        args.insert({DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_DST + i),
            scale_dst_mem[i]});
    }
}
```

#### Device-Side Runtime Dimensions Support

In MoE workloads with GPU-based routing, `M_per_expert` values are computed on
device and may not be transferred to the host.

As an option, runtime dimensions could be passed as an input argument during execution.

##### Primitive Creation

```cpp
// Use DNNL_RUNTIME_DIM_VAL in memory descriptors
memory::dims src_shape = {DNNL_RUNTIME_DIM_VAL, K};
memory::dims dst_shape = {DNNL_RUNTIME_DIM_VAL, N};

memory::desc src_md(src_shape, dt::f32, tag::ab);
memory::desc dst_md(dst_shape, dt::f32, tag::ab);

std::vector<memory::desc> src_mds(num_experts, src_md);
std::vector<memory::desc> dst_mds(num_experts, dst_md);

// Create descriptor for runtime dimensions input
memory::desc runtime_dims_md({num_experts}, dt::s32, tag::x);

// Create primitive descriptor
// If runtime dims are not provided, we should assume memory sizes would be resolved at execution time
auto gemm_pd = grouped_gemm::primitive_desc(
    eng, src_mds, weight_mds, bias_mds, dst_mds,
    runtime_dims_md,  // Runtime dimensions descriptor
    attr);
```

##### Execution

```cpp
// Device buffer: [M0, M1, M2, ..., M_{num_experts-1}]
int32_t* device_m_per_expert;

// Create memory object for runtime dimensions
memory runtime_dims_mem(runtime_dims_md, eng, device_m_per_expert);

std::unordered_map<int, memory> args;

// Pass runtime dimensions as input argument
args.insert({DNNL_ARG_ATTR_RUNTIME_DIMS, runtime_dims_mem});

// Per-expert data arguments
for (int i = 0; i < num_experts; ++i) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, src_mem[i]});
    args.insert({DNNL_ARG_MULTIPLE_WEIGHTS + i, weight_mem[i]});
    args.insert({DNNL_ARG_MULTIPLE_DST + i, dst_mem[i]});
}

gemm_prim.execute(stream, args);
```

> **Note:** Currently, oneDNN resolves memory object sizes
> at memory creation time. With device-side runtime
> dimensions, memory sizes cannot be known on the host, therefore the primitive
> will have to supply this information and resolve sizes during execution using the provided dimension buffer,
> which is a big change from current usage model.

### Proposal #2: New Grouped Memory Descriptor with Existing Matmul Primitive

This proposal introduces a new grouped memory descriptor that can be used with
the existing matmul primitive without additional API changes.

Idea: View Input Grouped GEMM Memory as "Sparse" Block-Diagonal Structure

Grouped GEMM inputs can be viewed as a Block-Diagonal structure:

```
(num_experts x num_experts blocks of sizes Mi x K):

│Expert0|   0   |  0    |  0    │
│[M0xK] |       |       |       │
|-------------------------------|
│  0    |Expert1|  0    |  0    │
│       |[0xK]  |       |       │ E.g., no tokens were assigned to Expert1
|-------------------------------|
│  0    |   0   |Expert2|  0    │
│       |       |[M2xK] |       │
|-------------------------------|
│  0    |   0   |  0    |Expert3│
│       |       |       |[M3xK] │

Layout in memory (contiguous):
[Expert0: M0xK | Expert2: M2xK | Expert3: M3xK]

Blocks offsets (array of size num_experts + 1):
block_offsets = [0, M0, M0(*), M0 + M2, M0 + M2 + M3]

(*) Expert1 starts from M0, Expert2 starts at M0 too since
    Expert1 has zero rows.
```

Properties of MoE that we know:
- Number of total experts: `num_experts`
- Block heights `Mi` vary dynamically (could be `0`)
- Block widths `K` uniform across all groups
- Total number of elements is `num_input_tokens x TOP_K x K`

Therefore this memory representation could be defined as:
- Number of blocks/groups: `num_experts`
    - known at creation time
- Total element count: `num_input_tokens x TOP_K x K`
    - known at creation time
- Block dimensions: `[Mi x K]`
    - `Mi` values known only at execution time (device-side), `K` known at creation time
- Block layout: Row-major
    - known at creation time
- Values buffer: Contiguous data for all groups, total size is `sum(Mi x K)` (same as `num_input_tokens x TOP_K x K`)
    - known at execution time (after routing and reshuffle happen)
- Offsets buffer: Cumulative row offsets `[0, M0, M0 + M1, ...]`, size is `num_experts + 1`
    - known at execution time (after routing happen)
- Data types for values and offsets
    - known at creation time

**Key Point:** This memory representation is very similar to the CSR format already supported in oneDNN.
Just as CSR describes sparse matrices using:
- Number of rows (provided for memory descriptor)
- Number of non-zero elements (provided for memory descriptor)
- Data types (provided for memory descriptor)
- Values buffer (resolved at memory creation)
- Row pointers and column indices (resolved at memory creation)

The grouped memory descriptor describes block-diagonal structure using:
- Number of groups/blocks (known at creation)
- Total element count (known at creation)
- Data types (known at creation)
- Values buffer (provided at execution)
- Offsets buffer marking block boundaries (provided at execution)

The offsets allow skipping empty blocks (experts with zero tokens), making
this a Sparse Block-Diagonal Matrix by Dense Matrix of Weight multiplication.

The structure of this "sparse" matrix is known at primitive creation time,
while the actual runtime dimensions and memory layout are resolved at execution time
using device-side offsets and values buffers.

#### Memory Descriptor API

The grouped memory descriptor is created similarly to CSR descriptors:

```cpp
static memory::desc grouped(
    dim_t group_count,               // Number of groups (or blocks)
    const memory::dims &group_dims,  // Per-group dims (RUNTIME_DIM_VAL for `Mi`)
    data_type dtype,                 // Elements data type
    dim_t total_elements,            // Total elements count
    format_tag tag = tag::ab,        // Layout within each group (default row-major)
    data_type offset_dtype = s32);   // Offset buffer dtype (default int32)
```

The descriptor specifies a memory object with 2 buffers:
- Buffer 0: Values, contiguous data of size `total_elements` organized as groups
- Buffer 1: Offsets, cumulative row offsets of size `group_count + 1`

Example:

```cpp
// Src memory
auto src_md = memory::desc::grouped(
    num_experts,                    // Group count
    {DNNL_RUNTIME_DIM_VAL, K},      // Per-group [Mi x K]
    dt::f32,                        // Data type
    total_tokens * K,               // Total elements
    tag::ab,                        // Row-major layout
    dt::s32);                       // int32 offsets
```

#### Memory Object Creation

Memory objects are created with 2 buffer pointers:

```cpp
// Buffers are coming from Framework side (could be device memory)
float* values;       // total_tokens * K contiguous data
int32_t* offsets;    // num_experts + 1 offsets

// Create memory object with 2 buffers
memory src_mem(src_md, eng, {values, offsets});
```

#### Primitive Descriptor Creation

Use the existing matmul primitive descriptor with grouped memory descriptor for src and dst:

```cpp
// Known at creation time
int total_tokens = num_input_tokens * TOP_K;
int total_elements_src = total_tokens * K;
int total_elements_dst = total_tokens * N;

// Create grouped memory descriptors for src and dst
auto src_md = memory::desc::grouped(
    num_experts,                    // Group count
    {DNNL_RUNTIME_DIM_VAL, K},      // Per-group [Mi x K]
    dt::f32,                        // Data type
    total_elements_src,             // Total elements
    tag::ab,                        // Row-major layout
    dt::s32);                       // Offset data type

auto dst_md = memory::desc::grouped(
    num_experts,                    // Group count
    {DNNL_RUNTIME_DIM_VAL, N},      // Per-group [Mi x N]
    dt::f32,                        // Data type
    total_elements_dst,             // Total elements
    tag::ab,                        // Row-major layout
    dt::s32);                       // Offset data type

// Weights: 3D dense tensor [num_experts, K, N]
auto weights_md = memory::desc({num_experts, K, N}, dt::f32, tag::abc);

// Bias: 2D dense tensor [num_experts, N]
auto bias_md = memory::desc({num_experts, N}, dt::f32, tag::ab);

// Create matmul primitive descriptor (no changes)
auto matmul_pd = matmul::primitive_desc(
    eng,
    src_md,                 // Grouped src descriptor
    weights_md,             // Dense 3D weights
    bias_md,                // Dense 2D bias (optional)
    dst_md,                 // Grouped dst descriptor
    attr);

// Create matmul primitive (no changes)
auto matmul_prim = matmul(matmul_pd);
```

#### Primitive Execution

```cpp
// Grouped GEMM data coming from Framework side
float* input_data;
float* output_data;
float* weights_data;
float* bias_data;

// Offsets data coming from routing kernel
int32_t* device_offsets;

// Create memory objects (resolve values and offsets buffers)
memory src_mem(src_md, eng, {input_data, device_offsets});
memory dst_mem(dst_md, eng, {output_data, device_offsets});
// Create "regular" dense memory objects for weights and bias
memory weights_mem(weights_md, eng, weights_data);
memory bias_mem(bias_md, eng, bias_data);

// Configure matmul arguments
std::unordered_map<int, memory> args;
args.insert({DNNL_ARG_SRC, src_mem});
args.insert({DNNL_ARG_WEIGHTS, weights_mem});
args.insert({DNNL_ARG_BIAS, bias_mem});
args.insert({DNNL_ARG_DST, dst_mem});

matmul_prim.execute(stream, args);
stream.wait();
```

#### Scales Support

Note, that scaling pattern is expected to be the same across all experts
(i.e., all experts use either per-tensor, per-row, per-column, or block-wise scaling,
no mixing is allowed).

How scale memory descriptors could be configured depending on scaling granularity:

- Per-tensor for src: regular 1D descriptor `[num_experts]`
```cpp
auto scales_md = memory::desc({num_experts}, dt::f32, tag::x);
// Memory: [E0: s0 | E1: s1 | E2: s2 | ...] - one scale per expert concatenated
```

- Per-row for src: grouped descriptor `[num_experts x Mi]`
```cpp
auto scales_md = memory::desc::grouped(
    num_experts, {DNNL_RUNTIME_DIM_VAL}, dt::f32, total_tokens,
    tag::x, dt::s32);
// Memory: 2 buffers (values + offsets)
// Values: [E0_row_scales | E1_row_scales | E2_row_scales | ...] - contiguous
// Offsets: [0, M0, M0 + M1, ...] - same as src offsets (!)
```

- Per-column for weights: regular 2D descriptor `[num_experts x N]`
```cpp
auto scales_md = memory::desc({num_experts, N}, dt::f32, tag::ab);
// Memory: [E0: N scales | E1: N scales | ...]
```

- Block-wise: grouped descriptor
```cpp
// For src [total_tokens x K] with block_size=32 along K
auto scales_md = memory::desc::grouped(
    num_experts, {DNNL_RUNTIME_DIM_VAL, K / 32}, dt::f32,
    total_tokens * (K / 32), tag::ab, dt::s32);
// Memory: 2 buffers (values + offsets)
// Values: [E0: M0 x (K/32) | E1: M1 x (K/32) | ...] - contiguous blocks
// Offsets: [0, M0, M0 + M1, ...] - same as src offsets (!)
```

