# Grouped GEMM for Mixture of Experts (MoE)

## Definition of the Operation

Grouped GEMM is a computational kernel used in Mixture of Experts (MoE) models, where tokens are dynamically routed to different expert networks.
Grouped GEMM processes variable-sized batches (subsets of tokens) per expert, which is a key difference from batched GEMM operations that assume uniform batch sizes across groups.

## PyTorch Grouped Scaled MM (`torch._scaled_mm_grouped`)

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
| `swizzle_a`, `swizzle_b` | `list[SwizzleType]` | Memory access pattern control (optional) | `SwizzleType` enum values for GPU implementations |
| `bias` | `Tensor` or `None` | Optional bias added to output | Not yet supported |
| `offs` | `list[int]` or `None` | Offsets for group boundaries | `int32` tensor: `[M0, M0+M1, M0+M1+M2, ...]`. First group starts at 0, `offs[i]` = end of group `i` |
| `output_dtype` | `torch.dtype` | Output tensor data type | `torch.bfloat16` (default), `torch.float16`, `torch.float32` |
| `contraction_dim` | `list[int]` or `None` | Dimensions representing K in matmul operation | Typically `[-1, -2]` for standard matrix multiply |
| `use_fast_accum` | `bool` | Tensor-core fast accumulation | Nvidia GPU-specific optimization |

(*) Based on initial investigation; support may have changed.

**Memory Layout:** All groups are **concatenated** along the batch dimension.

```python
mat_a:  [Group 0: M0×K][Group 1: M1×K][Group 2: M2×K] # [(M0 + M1 + M2) × K]
offs:    ^              ^              ^
         0              M0             M0 + M1        # [0, M0, M0 + M1]
```

## OpenVINO MoE with Grouped GEMM

OpenVINO implements a complete MoE pipeline in [PR #32469](https://github.com/openvinotoolkit/openvino/pull/32469):

| Primitive | Purpose | Description |
|-----------|---------|-------------|
| `moe_mask_gen` | Routing metadata generation | Generates per-expert token assignments and offsets from Top-K routing decisions |
| `moe_gather` | Token reorganization | Gathers tokens by expert assignment for contiguous memory access |
| `moe_gemm` | Grouped matrix multiplication | Performs expert-specific GEMMs with INT4/INT8/FP16 quantization support |
| `moe_scatter_reduction` | Result aggregation | Scatters expert outputs back to original token positions with weighted reduction |

**Data Types and Quantization Support:**
- INT4 weights
- INT8 activations with per-channel scales
- Group-wise quantization
- Asymmetric quantization with zero-points
- MXFP8 (E4M3/E5M2)

### Memory Layout and Indexing

OpenVINO uses **per-expert indexing** with contiguous buffers organized by expert:

```cpp
// Inputs organized after moe_gather
float* input_data;                    // Contiguous: [Expert0_tokens | Expert1_tokens | ...]
int* input_offset_per_expert;         // Start offset for each expert's tokens
int* tokens_lens_per_expert;          // Number of tokens per expert

// Weights organized per expert
float* weight_data;                   // Contiguous: [Expert0_weights | Expert1_weights | ...]
int* weight_offset_per_expert;        // Start offset for each expert's weights

// Per-expert scales for quantization
float* scales_per_expert;             // Dequantization scales
float* zero_points_per_expert;        // Asymmetric quantization offsets
```

**Execution Pattern:**
```cpp
for (int batch = 0; batch < total_token_assignments; batch++) {
    int expert_id = expert_ids[batch];           // Which expert to use
    int input_offset = input_offset_per_expert[expert_id];
    int weight_offset = weight_offset_per_expert[expert_id];

    // Fetch quantized data
    int4 weights = weight_data[weight_offset + ...];
    int8 input = input_data[input_offset + ...];

    // Process...
}
```

## oneDNN Grouped GEMM Primitive

### Overview

**Operation:** For each expert `i`:
```
output[i] = (src[i] * weights[i]) + bias[i]
```
Plus scaling factors, zero-point adjustments, etc. (as in regular matrix multiplication).

Expert `i` has dimensions: `src[Mi × K]`, `weights[K × N]`, `dst[Mi × N]`.

**Context that we know:**
- **Group size**: Number of experts is known in advance as it is a part of model strategy for expert routing
- **Runtime dimensions**: M dimension uses `DNNL_RUNTIME_DIM_VAL` during primitive creation, resolved at execution time,
since it corresponds to token counts per expert that vary dynamically
- **Uniform K and N**: All experts share the same `K` and `N` dimensions, so weight shapes are the same
- **Uniform scaling configurations**: All experts use the same scaling pattern (e.g., per-tensor, per-row), but different values

**API choices:**
- **Pointer-based API**: Uses arrays of pointers to oneDNN memory objects for per-expert data,
    so that each expert has separate memory descriptors
- **Per-expert attributes**: Scaling factors configured via `DNNL_ARG_MULTIPLE_*` arguments

### Memory Layout Compatibility

Both PyTorch and OpenVINO use contiguous memory layouts
that can be mapped to pointer-based API:

```cpp
// Build pointer array (zero-copy)
for (int i = 0; i < num_experts; i++) {
    expert_ptrs[i] = (void*)contiguous_data + offsets_per_expert[i];
    // offset_per_expert[i] is `offs` from PyTorch or
    // `input_offset_per_expert[i]` for inputs and
    // `weight_offset_per_expert[i]` for weights from OpenVINO
}
```

### Memory Descriptors

```cpp
// Expert i has dimensions: src[Mi × K], weights[K × N], dst[Mi × N]
memory::dims src_shape = {DNNL_RUNTIME_DIM_VAL, K};  // M resolved at execution
memory::dims weight_shape = {K, N};                  // Fixed dimensions
memory::dims bias_shape = {N};                       // 1D bias
memory::dims dst_shape = {DNNL_RUNTIME_DIM_VAL, N};  // M resolved at execution

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

### Scale Configuration

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

At execution time, actual `M_per_expert` values are known,
so here we create memory objects and provide scaling factors:

```cpp
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

// Execute
gemm_prim.execute(stream, args);
stream.wait();
```
