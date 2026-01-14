# Add Support for Grouped GEMM for Mixture of Experts (MoE)

## Definition of the Operation

Grouped GEMM is a computational kernel used in Mixture of Experts (MoE).

Typical MoE workflow:
- Input tokens are analyzed by a router
- Router assigns each token to top-K experts based on learned scores (*)
- Tokens are regrouped by their assigned experts (**)
- Each expert processes its assigned tokens independently (this is where Grouped GEMM appears)
- Expert outputs are combined back into the original token order

(*) Routing creates variable-sized batches of tokens per expert. Some experts may
receive many tokens while others receive few or none (distribution could be expected to be very unbalanced).
The number of tokens per expert (`M` dimension) is only known after routing completes.

(**) Regrouping involves gathering all tokens assigned to each expert
and creating a contiguous block of memory for each expert's input.

Grouped GEMM processes variable-sized batches (subsets of tokens) per expert,
which is a key difference from batched GEMM operations that assume uniform batch sizes across groups.

For each expert `i` (dimensions are: `src[Mi x K]`, `weights[K x N]`, `dst[Mi x N]`) we need to compute:
```
output[i] = (src[i] * weights[i]) + bias[i]
```
Plus scaling factors, zero-point adjustments, etc. (as in regular matrix multiplication).

To reiterate **terms used/considerations of MoE:**
- **Number of experts**: Total number of experts in the MoE layer.
- **TOP-K routing**: A routing strategy where each token is assigned to the top K experts according to some scoring mechanism.
- **Number of active experts**: Number of experts that received tokens after routing.
- **M dimension is Dynamic dimension**: M dimension (i.e., token count per expert) varies dynamically based on routing decisions.
And depending on the framework design, this information may only be available on the device side after routing.
- **Uniform K and N**: All experts share the same K and N dimensions, so weight shapes are the same.
- **Uniform scaling configurations**: All experts use the same scaling pattern (e.g., per-tensor, per-row), but different values.

See also:
- [Useful resource to learn about MoE in PyTorch with examples of MoE workflow and challenges](https://pytorchconference.sched.com/event/27QE0/pytorch-apis-for-high-performance-moe-training-and-inference-daniel-vega-myhre-ke-wen-natalia-gimelshein-meta)

## Frameworks Implementations

### PyTorch Grouped Scaled MM

PyTorch provides [`torch/nn/functional.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py) API for grouped scaled matrix multiplication.
The `torch._scaled_grouped_mm` API is used internally:

```python
torch._scaled_mm_grouped(
    mat_a,                           # Input tensor A (concatenated groups)
    mat_b,                           # Input tensor B (concatenated groups)
    scale_a,                         # Scaling factors for mat_a
    scale_recipe_a,                  # Scaling type for mat_a, choises are: PER_TENSOR, PER_ROW, PER_COLUMN, AXISWISE
    scale_b,                         # Scaling factors for mat_b
    scale_recipe_b,                  # Scaling type for mat_b
    swizzle_a=None,                  # Swizzle pattern for scale_a
    swizzle_b=None,                  # Swizzle pattern for scale_b
    bias=None,                       # Optional bias tensor
    offs=None,                       # Device-side offsets for source tensor
    output_dtype=torch.bfloat16,     # Output data type
    contraction_dim=None,            # Which dims are K in matmul
    use_fast_accum=False             # Nvidia GPUs specifics
)
```
[See also TorchAO](https://github.com/pytorch/ao/tree/main/torchao/prototype/moe_training)
for drop-in replacement with various quantization schemes.

Memory layout:
```
mat_a:  [Expert0: M0xK | Expert1: M1xK | Expert2: M2xK]
offs:    ^              ^              ^
         0              M0             M0 + M1
Weights: [num_experts x K x N]
```

### OpenVINO MoE

OpenVINO implements a complete MoE operation in [PR #32469](https://github.com/openvinotoolkit/openvino/pull/32469):

| Functionality | Description |
|---------------|-------------|
| `moe_mask_gen` | Generates per-expert token assignments and offsets from Top-K routing |
| `moe_gather` | Gathers tokens by expert assignment for contiguous memory access |
| `moe_gemm` | Performs expert-specific GEMMs with INT4/INT8/FP16 quantization |
| `moe_scatter_reduction` | Scatters expert outputs back to original positions with weighted reduction |

Memory layout uses per-expert indexing with concatenated buffers:

```cpp
float* input_data;              // Contiguous by active experts only
int* expert_id;                 // Dynamic size, IDs of experts with tokens
int* experts_info_start_idx;    // Start offset for each expert's tokens
int* tokens_lens_per_expert;    // Number of tokens per expert
float* weight_data;             // [Expert0_weights | Expert1_weights | ...]
```

- Offsets and sizes have dynamic size based on active ((!) not total) experts (therefore extra array is needed),
  that is a bit different from the PyTorch/proposed oneDNN design, but compatible.

## oneDNN Grouped GEMM Support

The request is to add support for grouped GEMM operation in oneDNN.
The main target is GPU execution, since MoE workloads are primarily run on GPUs,
and ultimately we want to reduce number of kernel launches and be able to handle
the runtime variable dimensions per expert located on device side after routing.

**Note:**  Due to MoE and Grouped GEMM evolving and frameworks experimenting
with various designs, current proposal for oneDNN is to introduce changes
as experimental feature (`-DONEDNN_EXPERIMENTAL_GROUPED_GEMM=ON`), so that we can iterate
on feedback before stabilizing the API.

### Proposal #1: Via New Grouped Memory Descriptor (Recommended)

This proposal adds a new grouped memory descriptor that works with the
existing matmul primitive.

The idea is to represent grouped GEMM data as a new encoding somewhat similar
to the support of CSR format.

Data is stored as concatenated blocks where each expert processes a different number of tokens.

Representation is:
```
Contiguous memory: [Expert0 | Expert1 | Expert2 | Expert3]
Block sizes:       [M0 x K  | M1 x K  | M2 x K  | M3 x K ]

Offsets array is for tracking where each expert starts:
offsets = [0, M0, M0+M1, M0+M1+M2, M0+M1+M2+M3]

In the case of M_i equals zero, the next expert starts at the same offset, similar to CSR format row pointers.
Example: if M1 = 0, then offsets = [0, M0, M0, M0 + M2, M0 + M2 + M3]
```

The grouped memory descriptor defines:
- Block count: `num_experts` (known at creation)
- Total elements: `num_input_tokens x TOP_K x K` (known at creation)
- Data types: values and offsets (known at creation)
- Values buffer: Contiguous data sum(Mi x K) (provided at execution)
- Offsets buffer: `[0, M0, M0 + M1, ...]` size `num_experts + 1` (at execution)

This representation is analogous to CSR sparse format:
- CSR uses row pointers and column indices to describe sparse structure, buffers are resolved
when memory object is created
- Similarly, the grouped descriptor uses offsets to mark boundaries of each expert's data,
so that experts with zero tokens are skipped via offsets.
  - Structure is known at primitive creation, while runtime dimensions and buffers
    are resolved at execution time using device-side data.

**Note:** Due to these similarities, the proposal is to incorporate grouped encoding under
`sparse_encoding` enum. Moreover, the suggestion is to rename `sparse_encoding` to `multi_buffer_encoding`
(or similar) in the next major release to better reflect its expanded purpose.

#### Memory Descriptor API

The grouped memory descriptor API is:

```cpp
static memory::desc grouped(
    const dims &adims,           // Overall tensor dims [total_M, K]
    data_type adata_type,        // Data type for values
    int variable_dim_idx,        // Which dimension varies (0 for M dimension)
    dim group_count,             // Number of groups (all experts, not just active)
    data_type offsets_dt = s32   // Offset data type (default int32)
);
```

The descriptor specifies a memory object with 2 buffers:
- Buffer 0: Values, contiguous data organized as groups
- Buffer 1: Offsets, cumulative row offsets of size `group_count + 1`

Example:
```cpp
auto src_md = memory::desc::grouped(
    {total_tokens, K},      // Overall shape
    dt::f32,                // Data type
    0,                      // M dimension varies
    num_experts);           // Group count
```

**Note on Layout:** The grouped descriptor does not allow explicit row-major
or column-major layout specification because the layout is tied to the variable dimension.
When `variable_dim_idx = 0` (M varies), the layout is implicitly row-major.
When `variable_dim_idx = 1` (K varies), the layout is implicitly column-major.
This constraint ensures the variable dimension corresponds to the outer dimension.
Support for `variable_dim_idx = 1` is for the backward pass of MoE layers.
In the backward pass, gradients are computed with respect to the input,
which involves transposed matrix operations where the K dimension becomes variable
across experts.

This is a simple version of the grouped descriptor, that could be extended in the future
to support more use cases (e.g. explicit padding or strides information) if needed.
Currently offsets could accommodate padding between experts if this is required.

#### Memory Object Creation

Memory objects are created with 2 buffer pointers:

```cpp
// Buffers are coming from Framework side (could be device memory)
float* values;       // total_tokens * K contiguous data
int32_t* offsets;    // num_experts + 1 offsets

// Create memory object with 2 buffers
memory src_mem(src_md, eng, {values, offsets});
memory dst_mem(dst_md, eng, {output_values, offsets});
```

#### Primitive Descriptor Creation

Use the existing matmul primitive with grouped memory descriptor for src and dst:

```cpp
// Known at creation time
int total_tokens = num_input_tokens * TOP_K;

// Create grouped memory descriptors for src and dst
auto src_md = memory::desc::grouped(
    {total_tokens, K},      // Overall shape
    dt::f32,                // Data type
    0,                      // M dimension varies
    num_experts);           // Group count

auto dst_md = memory::desc::grouped(
    {total_tokens, N},
    dt::f32,
    0,
    num_experts);

// Weights: 3D dense tensor [num_experts, K, N]
auto weights_md = memory::desc({num_experts, K, N}, dt::f32, tag::abc);

// Bias: 2D tensor [num_experts, N]
auto bias_md = memory::desc({num_experts, N}, dt::f32, tag::ab);

// Create matmul primitive descriptor
auto matmul_pd = matmul::primitive_desc(
    eng,
    src_md,                 // Grouped src descriptor
    weights_md,             // Dense 3D weights
    bias_md,                // Dense 2D bias
    dst_md);                // Grouped dst descriptor

// Create matmul primitive (no changes)
auto matmul_prim = matmul(matmul_pd);
```

#### Primitive Execution

```cpp
// Grouped GEMM data coming from Framework side
float* input_data;
float* output_data;
float* weights_data;

// Offsets data coming from routing kernel
int32_t* device_offsets;

// Create memory objects (resolve values and offsets buffers)
memory src_mem(src_md, eng, {input_data, device_offsets});
memory dst_mem(dst_md, eng, {output_data, device_offsets});
memory weights_mem(weights_md, eng, weights_data); // regular 3D weights
memory bias_mem(bias_md, eng);

// Configure matmul arguments (no changes)
std::unordered_map<int, memory> args;
args.insert({DNNL_ARG_SRC, src_mem});
args.insert({DNNL_ARG_WEIGHTS, weights_mem});
args.insert({DNNL_ARG_BIAS, bias_mem});
args.insert({DNNL_ARG_DST, dst_mem});

matmul_prim.execute(stream, args);
```

#### Scales Support

Note, that scaling pattern is expected to be the same across all experts
(i.e., all experts use either per-tensor, per-row, per-column, or block-wise scaling,
no mixing), therefore setting up scales is similar to regular matmul usage,
meaning we need to specify scaling mask via attribute as if for individual expert.

**Scales Memory Options:** Suggestion is to keep scales as plain memory descriptors,
since scales `offsets` buffer could be derived from src/dst.

Snippets below cover most commonly used patterns:

- Row-wise for src: 1D descriptor `[total_tokens]`
```cpp
auto scales_md = memory::desc({total_tokens}, dt::f32, tag::x);
// Values: [E0_row_scales | E1_row_scales | E2_row_scales | ...]
// Offsets are derived from src offsets

primitive_attr attr;
attr.set_scales(DNNL_ARG_SRC,
                1 << 0,    // Row-wise
                {},        // No groups
                dt::f32);
```

- Column-wise for weights: 2D descriptor `[num_experts x N]`
```cpp
auto scales_md = memory::desc({num_experts, N}, dt::f32, tag::ab);
// Values: [E0: N scales | E1: N scales | ...]

primitive_attr attr;
attr.set_scales(DNNL_ARG_WEIGHTS,
                (1 << 0) | (1 << 2),  // Dimensions 0 (experts) and 2 (N)
                {},                   // No groups
                dt::f32);
```

- Grouped scales for weights: 3D descriptor `[num_experts x num_groups_K x N]`
```cpp
int group_size = 32;

auto scales_md = memory::desc(
    {num_experts, K / group_size, N},
    dt::f32,
    tag::abc);
// Values: [E0: (K/32) x N | E1: (K/32) x N | ...]

primitive_attr attr;
attr.set_scales(DNNL_ARG_WEIGHTS,
                (1 << 0) | (1 << 1) | (1 << 2),  // Dimensions 0 (experts), 1 (K), 2 (N) all vary
                {1, group_size, 1},              // Grouping along K
                dt::f32);
```

### Proposal #2: Via New Attribute for Offsets

This proposal relies on regular 2D descriptors for concatenated input/output
buffers and a 3D descriptor for weights.
Matmul primitive will require new attribute for offsets with passed as a runtime argument.

#### Memory Descriptors

Same memory layout as in Proposal #1:
```
Input memory (contiguous 2D):
[Expert0: M0xK | Expert1: M1xK | Expert2: M2xK | Expert3: M3xK]
Shape: [total_tokens x K] where total_tokens = M0 + M1 + M2 + M3

Offsets array (on device):
[0, M0, M0+M1, M0+M1+M2, M0+M1+M2+M3]

Weights (3D dense):
[num_experts x K x N]
```

```cpp
int total_tokens = num_input_tokens * TOP_K;

// Input: concatenated tokens [total_tokens x K]
auto src_md = memory::desc(
    {total_tokens, K},      // 2D shape
    dt::f32,
    tag::ab);

// Output: concatenated results [total_tokens x N]
auto dst_md = memory::desc(
    {total_tokens, N},      // 2D shape
    dt::f32,
    tag::ab);

// Weights: [num_experts x K x N]
auto weights_md = memory::desc(
    {num_experts, K, N},    // 3D shape
    dt::f32,
    tag::abc);
```

#### Primitive Creation

Configure the primitive to operate in grouped mode via attribute, then create
the matmul primitive descriptor:

```cpp
primitive_attr attr;

// Configure grouped GEMM offsets for SRC argument
attr.set_grouped_offsets(
    DNNL_ARG_SRC,       // Argument to which offsets apply
    num_experts,        // Number of groups in concatenated data
    dt::s32);           // Offset data type (int32)

// Offset values will then be provided at the execution time via
// DNNL_ARG_ATTR_GROUPED_OFFSETS | DNNL_ARG_SRC

// Create matmul primitive descriptor with grouped attribute
auto matmul_pd = matmul::primitive_desc(
    eng,
    src_md,         // [total_tokens x K] concatenated input
    weights_md,     // [num_experts x K x N]
    dst_md,         // [total_tokens x N] concatenated output
    attr);

// Create primitive
auto matmul_prim = matmul(matmul_pd);
```

#### Primitive Execution

```cpp
// Device memory coming from Framework side
float* input_data;      // [total_tokens x K]
float* output_data;     // [total_tokens x N]
float* weights_data;    // [num_experts x K x N]

// Offsets array that is a device memory (from routing kernel)
int32_t* device_offsets;  // [num_experts + 1]

// Create memory objects
memory src_mem(src_md, eng, input_data);
memory dst_mem(dst_md, eng, output_data);
memory weights_mem(weights_md, eng, weights_data);

// Create memory object for offsets (to be passed at execution)
memory::desc offsets_md({num_experts + 1}, dt::s32, tag::x);
memory offsets_mem(offsets_md, eng, device_offsets);

// Configure arguments
std::unordered_map<int, memory> args;
args.insert({DNNL_ARG_SRC, src_mem});
args.insert({DNNL_ARG_WEIGHTS, weights_mem});
args.insert({DNNL_ARG_DST, dst_mem});

// Pass offset values
args.insert({DNNL_ARG_ATTR_GROUPED_OFFSETS | DNNL_ARG_SRC, offsets_mem});

// Execute
matmul_prim.execute(stream, args);
```

#### Scales Support

Scaling configuration follows the same patterns as Proposal #1.

**Note:** As in Proposal #1, for row-wise or block-wise scales on src, the same
offset array used for src data is reused to interpret the scale memory
layout. The offsets define **token** boundaries which apply to both the data and
related scales.

**Row-wise scaling for src:**

```cpp
// Row scales for all tokens: [total_tokens]
// Memory layout is concatenated per-expert:
// [E0_row_scales[M0] | E1_row_scales[M1] | E2_row_scales[M2] | ...]
auto scales_src_md = memory::desc({total_tokens}, dt::f32, tag::x);

attr.set_scales_mask(DNNL_ARG_SRC, 1);
```

**Block-wise scaling:**

```cpp
int block_size = 128;

// For src [total_tokens x K]:
// Memory layout on device is [E0: M0 x (K/128) | E1: M1 x (K/128) | E2: M2 x (K/128) | ...]
// Total elements is total_tokens * (K/128)
// Uses the SAME offset array as src since offsets define row boundaries
auto scales_src_md = memory::desc(
    {total_tokens, K / block_size},
    dt::f32,
    tag::ab);

attr.set_scales_mask(DNNL_ARG_SRC, 3);

// For weights [num_experts x K x N]
// Memory layout on device is [E0: (K/128) x (N/128) | E1: (K/128) x (N/128) | ...]
// No variable dimensions here, so no offsets needed
auto scales_wei_md = memory::desc(
    {num_experts, K / block_size, N / block_size},
    dt::f32,
    tag::abc);
```

### Proposal #3: Additionally Considered: Pointer-Based API

- **Pointer-based API**: Use arrays of pointers to oneDNN memory objects
    for per-expert data, so that each expert has separate memory descriptors.
    This approach then follows the pattern used in concat and sum, where oneDNN
    accepts multiple input memory objects via `DNNL_ARG_MULTIPLE_SRC`
    argument (i.e., `DNNL_ARG_MULTIPLE_SRC + i` for input i).
- **Per-expert attributes**: Scaling factors are configured via
    `DNNL_ARG_MULTIPLE_*`.
- **Runtime dimensions**: M per expert values can be provided at execution
    time via device-side buffer.
- **New primitive**: Likely will need a new grouped_gemm primitive,
    that accepts vectors of memory descriptors and runtime dimensions.

#### Memory Descriptors

```cpp
// Expert i has dimensions: src[Mi x K], weights[K x N], dst[Mi x N]
// Mi varies per expert (runtime dimension), K and N are fixed

// Use DNNL_RUNTIME_DIM_VAL for variable M dimension
memory::dims src_shape = {DNNL_RUNTIME_DIM_VAL, K};
memory::dims weight_shape = {K, N};
memory::dims dst_shape = {DNNL_RUNTIME_DIM_VAL, N};

memory::desc src_md(src_shape, memory::data_type::f32,
                    memory::format_tag::ab);
memory::desc weight_md(weight_shape, memory::data_type::f32,
                       memory::format_tag::ab);
memory::desc dst_md(dst_shape, memory::data_type::f32,
                    memory::format_tag::ab);

std::vector<memory::desc> src_mds(num_experts, src_md);
std::vector<memory::desc> weight_mds(num_experts, weight_md);
std::vector<memory::desc> dst_mds(num_experts, dst_md);
```

#### Scales Configuration

```cpp
primitive_attr attr;

attr.set_scales_mask(DNNL_ARG_MULTIPLE_SRC + expert_id, 0);      // Per-tensor
attr.set_scales_mask(DNNL_ARG_MULTIPLE_SRC + expert_id, 1);      // Per-row
attr.set_scales_mask(DNNL_ARG_MULTIPLE_WEIGHTS + expert_id, 2);  // Per-column
```

#### Primitive Creation

```cpp
// Device-side buffer with M values for each expert
memory::desc runtime_dims_md({num_experts}, dt::s32, tag::x);

auto gemm_pd = grouped_gemm::primitive_desc(
    eng, src_mds, weight_mds, dst_mds, runtime_dims_md, attr);

auto gemm_prim = grouped_gemm(gemm_pd);
```

#### Execution

Runtime dimensions are provided via device-side buffer during execution:

```cpp
int32_t* device_m_per_expert;  // [M0, M1, ..., M_{num_experts-1}]
memory runtime_dims_mem(runtime_dims_md, eng, device_m_per_expert);

std::vector<memory> src_mem(num_experts);
std::vector<memory> weight_mem(num_experts);
std::vector<memory> dst_mem(num_experts);

// Warning: it is not possible to resolve M dimension at memory creation time,
// that is a change from current usage model
for (int i = 0; i < num_experts; ++i) {
    src_mem[i] = memory(src_mds[i], eng, src_ptrs[i]);
    weight_mem[i] = memory(weight_mds[i], eng, weight_ptrs[i]);
    dst_mem[i] = memory(dst_mds[i], eng, dst_ptrs[i]);
}

std::unordered_map<int, memory> args;

// Pass runtime dimensions as input argument
args.insert({DNNL_ARG_RUNTIME_DIMS, runtime_dims_mem});

for (int i = 0; i < num_experts; ++i) {
    args.insert({DNNL_ARG_MULTIPLE_SRC + i, src_mem[i]});
    args.insert({DNNL_ARG_MULTIPLE_WEIGHTS + i, weight_mem[i]});
    args.insert({DNNL_ARG_MULTIPLE_DST + i, dst_mem[i]});
}

gemm_prim.execute(stream, args);
```

#### Scales Support

Scales support is straightforward with pointer-based API. Provide
per-expert scale memory objects via DNNL_ARG_MULTIPLE_* arguments:

```cpp
std::vector<memory> scale_src_mem(num_experts);
std::vector<memory> scale_wei_mem(num_experts);
std::vector<memory> scale_dst_mem(num_experts);

for (int i = 0; i < num_experts; ++i) {
    scale_src_mem[i] = memory({{1}, dt::f32, tag::x}, eng,
                              scale_src_data[i].data());
    scale_wei_mem[i] = memory({{1}, dt::f32, tag::x}, eng,
                              scale_wei_data[i].data());
    scale_dst_mem[i] = memory({{1}, dt::f32, tag::x}, eng,
                              scale_dst_data[i].data());

    args.insert({DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_SRC + i),
                 scale_src_mem[i]});
    args.insert({DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_WEIGHTS + i),
                 scale_wei_mem[i]});
    args.insert({DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_DST + i),
                 scale_dst_mem[i]});
}
```

#### Frameworks Compatibility

Frameworks with contiguous memory layouts can be mapped to pointer-based
API:

```cpp
// Build pointer array (zero-copy)
for (int i = 0; i < total_number_of_experts; i++) {
    expert_ptrs[i] = (void*)contiguous_data + offsets_per_expert[i];
}
```

**Warning:**
- If each expert data was initially allocated using some maximum M value,
then there is never a need to recalculate pointers.
- If expert data is continuously repacked into contiguous buffers after
routing, then pointers need to be recalculated at each execution.

### Proposals Comparisons

| Point to consider | Proposal 1 (Grouped MD) | Proposal 2 (Offset Attr) | Proposal 3 (Pointer API) |
|-------------------|-------------------------|--------------------------|--------------------------|
| Known Frameworks integration | Direct mapping | Direct mapping | Must build pointer arrays for contiguous data |
| Handling of variable dimensions | Part of memory descriptor | Runtime argument | Implicit by building pointer array |
| Complexity for the user and similarity to existing concept | Similar to CSR sparse format | Regular 2D/3D descriptors. Usage is similar to scales | New concept with per-expert descriptors and memory objects, potentially new primitive. Similar to concat/sum multi-src but with extra details |
| Extensibility | New "memory" support could be extended to other ops (no known requests) | Stays as matmul specifics | Changes limited to new grouped_gemm primitive |
| Scales, ZPs setup and usage | Grouped or regular descriptors based on variable dimensions, require offsets | Regular descriptors, require offsets | Regular descriptors consolidated into vector, require `DNNL_ARG_MULTIPLE_SRC` |
| Weight layout | Dense 3D tensor | Dense 3D tensor | Per-expert 2D |
| Drawbacks Worth Considering | New memory descriptor concept requiring changes to memory object creation and special handling in primitives | Offsets (equivalent to sub-matrix lengths) are probably more naturally suited to be a memory layout property rather than a matmul attribute | More complex API with pointer array creation and recalculation overhead |

Summary:
- **Proposal 1** (Recommended) enables cleaner abstraction
- **Proposal 2** is simplest for users / developers
- **Proposal 3** adds most complexity but may suit certain use cases better (e.g., when frameworks already use pointer arrays or when experts have very different K/N dimensions, although no known examples)
