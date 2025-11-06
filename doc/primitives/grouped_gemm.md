# Grouped GEMM for Mixture of Experts (MoE)

## Definition of the Operation

Grouped GEMM is a computational kernel used in Mixture of Experts (MoE) models, where tokens are dynamically routed to different expert networks.

Grouped GEMM processes variable-sized batches (subsets of tokens) per expert, which is a key difference from batched GEMM operations that assume uniform batch sizes across groups.

For each expert `i`:
```
output[i] = (src[i] * weights[i]) + bias[i]
```
Plus scaling factors, zero-point adjustments, etc. (as in regular matrix multiplication).

Expert `i` has dimensions: `src[Mi × K]`, `weights[K × N]`, `dst[Mi × N]`.

**Terms to be used and other considerations of MoE:**
- **Number of experts**: Total number of experts in the MoE layer.
- **TOP-K routing**: A routing strategy where each token is assigned to the top K experts according to some scoring mechanism.
- **Number of active experts OR group size**: Number of experts that received tokens after routing.
- **M dimension is Dynamic dimension**: M dimension (i.e., token count per expert) varies dynamically based on routing decisions. And depending on the framework design, this information may only be available on the Device side after routing.
- **Uniform K and N**: All experts share the same K and N dimensions, so weight shapes are the same.
- **Uniform scaling configurations**: All experts use the same scaling pattern (e.g., per-tensor, per-row), but different values.

## PyTorch Grouped Scaled MM (`torch._scaled_mm_grouped`)

### API

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
| `bias` | `Tensor` or `None` | Optional bias added to output | Not yet supported |
| `offs` | `list[int]` or `None` | Offsets for group boundaries | `int32` tensor: `[M0, M0+M1, M0+M1+M2, ...]`. First group starts at 0, `offs[i]` = end of group `i` |
| `output_dtype` | `torch.dtype` | Output tensor data type | `torch.bfloat16` (default), `torch.float16`, `torch.float32` |
| `contraction_dim` | `list[int]` or `None` | Dimensions representing K in matmul operation | |
| `use_fast_accum` | `bool` | Tensor-core fast accumulation | Nvidia GPU-specific |

(*) Based on initial investigation; support may have changed.

[See also TorchAO Drop-In Replacement](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training)
that supports various quantization schemes.

### Memory Layouts

- All groups are concatenated into contiguous buffers per expert:
```python
mat_a:  [Group 0: M0×K][Group 1: M1×K][Group 2: M2×K] # [(M0 + M1 + M2) × K]
offs:    ^              ^              ^
         0              M0             M0 + M1        # [0, M0, M0 + M1]
```
- `offs` data is provided on the Device side.
- All expert weights present in memory regardless of token routing.

## OpenVINO MoE

OpenVINO implements a complete MoE operation in [PR #32469](https://github.com/openvinotoolkit/openvino/pull/32469):

| Primitive | Purpose | Description |
|-----------|---------|-------------|
| `moe_mask_gen` | Routing metadata generation | Generates per-expert token assignments and offsets from Top-K routing decisions |
| `moe_gather` | Token reorganization | Gathers tokens by expert assignment for contiguous memory access |
| `moe_gemm` | Grouped matrix multiplication | Performs expert-specific GEMMs with INT4/INT8/FP16 quantization support |
| `moe_scatter_reduction` | Result aggregation | Scatters expert outputs back to original token positions with weighted reduction |

**Data Types and Quantization Support:**
TBD

### Memory Layouts

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
However, we should expect this data to be coming from the Device, as previous gather/shuffle operation is most likely performed on the Device.

## vLLM MoE

Source code: [vllm-xpu/csrc/quantization/w8a8/cutlass/moe/](https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu/tree/main/csrc/quantization/w8a8/cutlass/moe)

### API

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
| `scales_a` | `[total_tokens, K//128]` | FP32 | Block-wise (128×128) activation scales (device) |
| `scales_b` | `[num_experts, N//128, K//128]` | FP32 | Block-wise weight scales per expert (device) |
| `problem_sizes` | `[num_experts, 3]` | INT32 | Per-expert `[M, N, K]` where `M` varies, `N` and `K` are uniform (device) |
| `expert_offsets` | `[num_experts + 1]` | INT32 | Cumulative token counts: `[0, M0, M0+M1, ..., sum(Mi)]` (device) |

(*) Based on initial investigation; support may have changed.

### Memory Layout

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

### Proposal #1

- **Pointer-based API**: Uses arrays of pointers to oneDNN memory objects for per-expert data,
    so that each expert has separate memory descriptors.
- **Per-expert attributes**: Scaling factors configured via `DNNL_ARG_MULTIPLE_*` arguments.

"+":
- If each expert's data was initially allocated using some `M_max`, then there is never a need to recalculate pointers.
- Natural support of scaling factor, zero-points and transpose operation per expert.

"-":
- If expert's data is stored/repacked into contiguous buffers, then pointers need to be recalculated at each execution.

### Memory Layouts Compatibility

Contiguous memory layouts could be mapped to pointer-based API:

```cpp
// Build pointer array (zero-copy)
for (int i = 0; i < total_number_of_experts; i++) {
    expert_ptrs[i] = (void*)contiguous_data + offsets_per_expert[i];
}
```

### Memory Descriptors

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

> **Note:** Each expert memory descriptor is identical based on MoE case, so here we create a vector with repeated entries.
    An alternative approach could use a special memory object representing all experts.

### Scales Configuration

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

### Primitive Creation

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

### Execution

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

#### Scales Memory

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

