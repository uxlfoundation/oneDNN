Grouped Memory Format for Variable-Size Batching {#dev_guide_grouped_mem}
=========================================================================

@note This is an [experimental feature](@ref dev_guide_experimental). Build oneDNN
with `ONEDNN_EXPERIMENTAL_GROUPED_GEMM=ON` to enable grouped memory support.

Grouped memory format handles data with variable-sized groups, commonly used in
Mixture-of-Experts (MoE). Unlike regular batching where all in the batch share the same size,
grouped format allows one dimension to vary per group while keeping other dimensions uniform.

## Memory Layout

Grouped format is described by two buffers and an indication of
which dimension varies across groups (`variable_dim_idx`):

| Buffer | Description | Data Type |
|:-------|:------------|:----------|
| 0 | Concatenated data from all the groups        | User-specified |
| 1 | Cumulative offsets defining group boundaries | s32            |

### Variable Dimension

The `variable_dim_idx` parameter specifies which dimension varies in size across groups.
For example, in a 2D tensor `[M, K]`:
- If `variable_dim_idx=0`, dimension M varies per group while K stays constant.
- If `variable_dim_idx=1`, dimension K varies per group while M stays constant.

The value specified for the variable dimension in the memory descriptor represents
the **total size** summed across all groups. Individual group sizes are determined
by the offsets buffer at execution time.

### Offsets Buffer

The offsets buffer contains cumulative counts that define group boundaries, that allows
to locate the starting position of each group in the data buffer.
For groups with sizes `[M_0, M_1, M_2, ..., M_{num_groups-1}]`, the offsets array is
`[M_0, M_0+M_1, M_0+M_1+M_2, ..., sum(M)]` with length equal to `num_groups`.

Note, that empty groups (size `0`) are valid and common in MoE when no tokens are routed to an expert.
In this case, consecutive offsets will be equal (i.e. `offsets[i] == offsets[i+1]`).

## Grouped Memory Descriptor API

To create a grouped memory descriptor, use the following C++ API:

~~~cpp
static memory::desc memory::desc::grouped(
    const dims &dims,              // Tensor dimensions (for variable dim use total size)
    data_type dtype,               // Data type (e.g., f32, s8)
    int variable_dim_idx,          // Index of dimension that varies per group
    int num_groups);               // Number of groups
~~~

## Creating and Using Grouped Memory

For a 2D grouped tensor `[total_M, K]` with `variable_dim_idx=0`:
- `total_M` = sum of all group sizes in dimension `0`
- `K` remains constant across all groups
- Data is concatenated for all groups in buffer `0`

~~~cpp
// Example setup:
// 8 groups with varying sizes in dimension 0
// tokens_per_group = {800, 600, 700, 500, 650, 450, 550, 750}
// total_tokens = 5000
// offsets = {800, 1400, 2100, 2600, 3250, 3700, 4250, 5000}

const int num_groups = 8;
const int total_M = 5000;
const int K = 512;

// Create grouped memory descriptor
auto grouped_md = memory::desc::grouped(
    {total_M, K},              // dims: [total_M, K]
    memory::data_type::f32,    // data type
    0,                         // variable_dim_idx: dimension 0 varies
    num_groups);               // number of groups

// Prepare data buffers
std::vector<float> values(total_M * K);
std::vector<int32_t> offsets(num_groups);

// Create memory object with both buffers
memory grouped_mem(grouped_md, engine, {values.data(), offsets.data()});

// Access individual buffers
void* values_handle = grouped_mem.get_data_handle(0);
void* offsets_handle = grouped_mem.get_data_handle(1);
size_t values_size = grouped_mem.get_size(0);  // total_M * K * sizeof(float)
size_t offsets_size = grouped_mem.get_size(1); // num_groups * sizeof(int32_t)
~~~

## Examples

See @ref matmul_grouped_cpp for an example of using grouped memory format
with MatMul primitive.
