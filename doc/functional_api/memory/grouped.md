Grouped Memory Format for Variable-Size Batching {#dev_guide_grouped_mem}
=========================================================================

@note This is an [experimental feature](@ref dev_guide_experimental). Build oneDNN
with `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON` to enable grouped memory support.

Grouped memory holds `G` matrices (groups) that share one dimension while the
other one varies. A typical case is Mixture-of-Experts (MoE): each expert gets a
different number of tokens, but the hidden size is the same for all of them.

## Memory Descriptor

The descriptor is 3D. Dimension 0 is `G`, and the dimension that varies is set
to #DNNL_VARIABLE_DIM_VAL, written as `*`:
- `{G, *, K}`: group `g` is `M_g x K`.
- `{G, M, *}`: group `g` is `M x K_g`, for example, a weight gradient where `K`
  is along the tokens.

Note, that `*` stays in the dimensions and is never replaced by a number.

Descriptor also requires:
- `total`: the size of the variable dimension summed over all groups, for
  example, the number of tokens routed to all experts in MoE.
- `strides`: optional, see Layout section below.
- The offsets data type: only s32 is supported.

## Buffers

- Buffer 0: the values of all groups, one after another along the variable
  dimension.
- Buffer 1: the offsets, one s32 value per group. `offsets[g]` is where group
  `g` ends: for group sizes `{S_0, S_1, ...}` they are `{S_0, S_0 + S_1, ...}`.
  The last offset must not exceed `total`. Empty groups are allowed.

Offsets are read at execution time, so one descriptor could work for different
distributions of the same `total`.

## Layout

The groups are joined along the variable dimension into one matrix: `[total, K]`
for `{G, *, K}` and `[M, total]` for `{G, M, *}`.

Strides (in elements) are expected to be as follows:
- `strides[0]` is 0, since the offsets place each group.
- Of the other two strides, one is 1 and the other one is the leading dimension.
- Without strides specified, the variable dimension is assumed to be the outermost one.

Possible combinations:
| Dimensions  | Strides               | Layout                                                            |
|:------------|:----------------------|:------------------------------------------------------------------|
| `{G, *, X}` | `{0, X, 1}` (default) | `[total, X]`, row-major: each group is exactly one block          |
| `{G, *, X}` | `{0, 1, total}`       | `[total, X]`, column-major: each column has a part of every group |
| `{G, X, *}` | `{0, 1, X}` (default) | `[X, total]`, column-major: each group is one block               |
| `{G, X, *}` | `{0, total, 1}`       | `[X, total]`, row-major: each row has a part of every group       |

The layouts that a primitive supports are listed on its page, for example, see
[grouped matmul](@ref dev_guide_matmul_grouped_gemm).

## API

~~~cpp
static memory::desc memory::desc::grouped(const dims &adims,
        data_type adata_type, dim total, const dims &astrides = {},
        data_type offsets_dt = data_type::s32, bool allow_empty = false);
~~~

The C API is `dnnl_memory_desc_create_with_grouped_encoding()`.

See @ref matmul_grouped_cpp for an example of querying a grouped descriptor
with `get_dims()`, `get_strides()`, and `get_size()`.

## Examples

~~~cpp
// MoE: 8 experts with {800, 600, 700, 500, 650, 450, 550, 750} tokens,
// 5000 in total, hidden size 512, default layout
auto src_md = memory::desc::grouped(
        {8, DNNL_VARIABLE_DIM_VAL, 512}, memory::data_type::f32, 5000);
std::vector<float> values(5000 * 512);
std::vector<int32_t> offsets = {800, 1400, 2100, 2600, 3250, 3700, 4250, 5000};
memory src_mem(src_md, eng, {values.data(), offsets.data()});

// Weight gradient with K-major operands, as in PyTorch _scaled_grouped_mm
// with 2D x 2D inputs: [M, total_K] row-major and [total_K, N] column-major
auto a_md = memory::desc::grouped({G, M, DNNL_VARIABLE_DIM_VAL},
        memory::data_type::f8_e4m3, total_K, {0, total_K, 1});
auto b_md = memory::desc::grouped({G, DNNL_VARIABLE_DIM_VAL, N},
        memory::data_type::f8_e4m3, total_K, {0, 1, total_K});
~~~
