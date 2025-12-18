/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_COMPUTE_DISPATCH_REUSABLE_HPP
#define GPU_INTEL_COMPUTE_DISPATCH_REUSABLE_HPP

#include <string>
#include <vector>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/serialization.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/block_manipulation.hpp"
#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/compute/types_interop.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

// How many buffers can be registered simultaneously
#define MAX_REGISTERED_BUFFERS 16
#define MAX_EXPR_TERMS 128

enum class name_id_t : uint64_t {
    // Common buffer names
    src = 1,
    wei = uint64_t(1) << 1,
    dst = uint64_t(1) << 2,
    diff_src = uint64_t(1) << 3,
    diff_wei = uint64_t(1) << 4,
    diff_dst = uint64_t(1) << 5,
    stat = uint64_t(1) << 6,

    // Common dimension names
    a = uint64_t(1) << 12,
    b = uint64_t(1) << 13,
    c = uint64_t(1) << 14,
    d = uint64_t(1) << 15,
    e = uint64_t(1) << 16,
    f = uint64_t(1) << 17,

    // Implementation Specific Names
    original = uint64_t(1) << 24,
    ss = uint64_t(1) << 25,
    buffer = uint64_t(1) << 26,
    atomic = uint64_t(1) << 27,
    local = uint64_t(1) << 28,
    ic_dim = uint64_t(1) << 29,

};

inline const std::string &to_string(name_id_t value) {
    static const std::unordered_map<name_id_t, std::string> names {
            {name_id_t::src, "SRC"},
            {name_id_t::wei, "WEI"},
            {name_id_t::dst, "DST"},
            {name_id_t::diff_src, "DIFF_SRC"},
            {name_id_t::diff_wei, "DIFF_WEI"},
            {name_id_t::diff_dst, "DIFF_DST"},
            {name_id_t::a, "A"},
            {name_id_t::b, "B"},
            {name_id_t::c, "C"},
            {name_id_t::d, "D"},
            {name_id_t::e, "E"},
            {name_id_t::f, "F"},
            {name_id_t::stat, "STAT"},
            {name_id_t::original, "ORIGINAL"},
            {name_id_t::ss, "SS"},
            {name_id_t::buffer, "BUFFER"},
            {name_id_t::atomic, "ATOMIC"},
            {name_id_t::local, "LOCAL"},
            {name_id_t::ic_dim, "IC_DIM"},
    };
    auto ret = names.find(value);
    gpu_assert(ret != names.end());
    return ret->second;
}

GPU_DEFINE_BIT_MASK_ENUM_OPS(name_id_t);
constexpr dim_idx_t dim_not_found = std::numeric_limits<dim_idx_t>::max();

struct named_buffer_t : public memory_desc_t {
    struct dim_info_t {
        dim_info_t(dim_idx_t idx, int64_t size) : idx(idx), size(size) {}
        dim_idx_t idx = dim_idx::invalid;
        int64_t size = 0;
    };

    named_buffer_t(name_id_t name_id, const memory_desc_t &md,
            const std::vector<dim_idx_t> &dims)
        : memory_desc_t(md), name_id(name_id), dim_ids(dims) {
        gpu_assert(format_kind == format_kind::blocked);
        gpu_assert(static_cast<size_t>(md.ndims) <= dim_ids.size());
    }
    named_buffer_t(name_id_t name_id, const std::vector<dim_info_t> &dims = {})
        : name_id(name_id) {
        format_kind = format_kind::blocked;
        for (auto &d : dims)
            insert(d.idx, d.size);
    }
    named_buffer_t(name_id_t name_id, const memory_desc_t &md)
        : named_buffer_t(name_id, md, default_dims(md.ndims)) {};

    // Copy the named_buffer_t, while changing the name
    named_buffer_t(name_id_t name_id, const named_buffer_t &buf)
        : memory_desc_t(buf), name_id(name_id), dim_ids(buf.get_dim_ids()) {};

    static std::vector<dim_idx_t> default_dims(int ndims) {
        std::vector<dim_idx_t> dims(ndims);
        for (int i = 0; i < ndims; i++)
            dims[i] = i;
        return dims;
    }
    dim_t nelems(bool with_padding = false) const {
        return memory_desc_wrapper(static_cast<memory_desc_t>(*this))
                .nelems(with_padding);
    }

    uint64_t size(int index = 0, bool include_additional_size = true,
            bool include_offset0 = false) const {
        return memory_desc_wrapper(static_cast<memory_desc_t>(*this))
                .size(index, include_additional_size, include_offset0);
    }

    const std::string &name() const { return to_string(get_name_id()); }
    const name_id_t &get_name_id() const { return name_id; }
    const std::vector<dim_idx_t> &get_dim_ids() const { return dim_ids; }

    void remove_dim(dim_idx_t dim, bool update_strides = true) {
        dim_idx_t dim_idx = get_dim_idx(dim);
        if (dim_idx == dim_not_found) return;

        remove_blocking(dim_idx);

        auto &blk = format_desc.blocking;
        dim_t dim_stride = blk.strides[dim_idx];
        dim_t dim_size = padded_dims[dim_idx];

        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            if (update_strides && blk.strides[i] > dim_stride) {
                blk.strides[i] /= dim_size;
            }

            // Shift dims down
            if (i > dim_idx) {
                blk.strides[i - 1] = blk.strides[i];
                dims[i - 1] = dims[i];
                padded_dims[i - 1] = padded_dims[i];
            }
        }

        // Reindex blocks to reflect the shift
        for (size_t blk_idx = 0; blk_idx < static_cast<size_t>(blk.inner_nblks);
                blk_idx++) {
            if (static_cast<size_t>(blk.inner_idxs[blk_idx]) > dim_idx)
                blk.inner_idxs[blk_idx]--;
        }

        // Remove the dimension label
        dim_ids.erase(dim_ids.begin() + static_cast<dim_t>(dim_idx));

        // Decrement the number of dimensions
        ndims--;
    }

    // Inserts the given dimension with the given size as the innermost dimension.
    void insert(dim_idx_t dim, dim_t size) {
        auto &blk = format_desc.blocking;

        size_t dim_idx = get_dim_idx(dim);
        if (dim_idx == dim_not_found) {
            // Add a new dimension
            assert(ndims < DNNL_MAX_NDIMS - 1);
            dims[ndims] = 1;
            padded_dims[ndims] = 1;
            blk.strides[ndims] = 1;
            dim_idx = static_cast<size_t>(ndims++);
            dim_ids.emplace_back(dim);
        }

        // Update the dimension size
        dims[dim_idx] *= size;
        padded_dims[dim_idx] *= size;
        blk.strides[dim_idx] = [&]() {
            auto ret = 1;
            for (int i = 0; i < blk.inner_nblks; i++) {
                ret *= blk.inner_blks[i];
            }
            return ret;
        }();

        // Update the strides
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            if (i == dim_idx) continue;
            blk.strides[i] *= size;
        }
    }

    void remove_blocking() {
        auto &blk = format_desc.blocking;
        while (blk.inner_nblks > 0) {
            remove_blocking(blk.inner_idxs[0]);
        }
    }

    dim_idx_t get_dim_idx(dim_idx_t dim) const {
        for (dim_idx_t i = 0; i < into<dim_idx_t>(dim_ids.size()); i++) {
            if (dim_ids[i] == dim) { return i; }
        }
        return dim_not_found;
    }

    block_layout_t layout() const {
        // Create the block layout and reindex to the canonical dimension indexing
        block_layout_t layout(*this);
        for (auto &block : layout) {
            // Re-index the layout according to the included dims
            block.dim_idx = get_dim_ids()[static_cast<size_t>(block.dim_idx)];
        }
        return layout;
    }

private:
    name_id_t name_id;
    std::vector<dim_idx_t> dim_ids;

    void remove_blocking(dim_idx_t dim_idx) {
        // Remove the inner blocks
        auto &blk = format_desc.blocking;
        size_t n_blks = 0;
        dim_t block_size = 1;
        for (size_t i = 0; i < static_cast<size_t>(blk.inner_nblks); i++) {
            if (blk.inner_idxs[i] == dim_idx) {
                block_size *= blk.inner_blks[i];
                continue;
            }
            blk.inner_idxs[n_blks] = blk.inner_idxs[i];
            blk.inner_blks[n_blks++] = blk.inner_blks[i];
        }
        blk.inner_nblks = static_cast<int>(n_blks);

        // Update strides
        dim_t outer_stride = blk.strides[dim_idx];
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            if (blk.strides[i] > outer_stride) continue;
            blk.strides[i] /= block_size;
        }
    }
};

// The reusable dispatcher interface involves a number of terms like (idx / stride % max) * block,
// and a mapping from several equations into these terms. Equations can share terms,
// and generally correspond to offset calculation for a buffer or dimension index
// calculation. As long as the sharing of terms is reasonably generic, the compiled
// parameters encode just the block structure and therefore are able to be reused.
struct dispatch_compile_params_t {
    dispatch_compile_params_t() = default;
#if __cplusplus >= 202002L
    bool operator==(const dispatch_compile_params_t &) const = default;
#endif

    void def_kernel_macros(
            kernel_ctx_t &kernel_ctx, const char *suffix = "DEFAULT") const;
    bool has_padding() const;

    std::string str() const;

    name_id_t buffer_set = {}; // Bitset representing supported buffers
    uint8_t subgroup_size = 0;
    bool use_int32_offset = false;
    bool require_stateless_addressing = true;
    uint8_t padding[3] = {0};

    // Offset into exprs where buffer offset expression is stored
    uint8_t gws_overflow_expr = {};
    uint8_t in_padding_expr = {};
    uint8_t buffer_off_expr[MAX_REGISTERED_BUFFERS] = {};
    uint8_t exprs[MAX_EXPR_TERMS] = {};
    data_type_t buffer_types[MAX_REGISTERED_BUFFERS] = {data_type::undef};
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(dispatch_compile_params_t);

class dispatch_runtime_params_t {
public:
    dispatch_runtime_params_t() = default;
    dispatch_runtime_params_t(const nd_range_t &nd_range, bool use_int32_offset,
            const std::vector<int64_t> &term_list)
        : nd_range(nd_range)
        , use_int32_offset(use_int32_offset)
        , num_terms(term_list.size()) {
        for (size_t i = 0; i < num_terms; i++) {
            if (use_int32_offset)
                rt32.params[i] = into<int32_t>(term_list[i]);
            else
                rt64.params[i] = term_list[i];
        }
    }
    dispatch_gws_rt_params64_t get64() const {
        gpu_assert(!use_int32_offset);
        return rt64;
    }
    dispatch_gws_rt_params32_t get32() const {
        gpu_assert(use_int32_offset);
        return rt32;
    }

    std::string str() const {
        stringstream_t ss;
        ss << "<dispatch_runtime_params_t: ";
        for (size_t i = 0; i < num_terms; i++) {
            if (i > 0) ss << ", ";
            ss << "term[" << std::to_string(i)
               << "]: " << (use_int32_offset ? rt32.params[i] : rt64.params[i]);
        }
        ss << ">";
        return ss.str();
    }

    nd_range_t nd_range;
    bool use_int32_offset = false;

private:
    size_t num_terms = 0;
    union {
        dispatch_gws_rt_params64_t rt64;
        dispatch_gws_rt_params32_t rt32;
    };
};

inline void set_rt_params(compute::kernel_arg_list_t &arg_list, int index,
        const dispatch_runtime_params_t &params) {
    if (params.use_int32_offset)
        arg_list.set(index, params.get32());
    else
        arg_list.set(index, params.get64());
}

inline void append_rt_params(compute::kernel_arg_list_t &arg_list,
        const dispatch_runtime_params_t &params) {
    if (params.use_int32_offset)
        arg_list.append(params.get32());
    else
        arg_list.append(params.get64());
}

struct lws_strategy_t {
    struct dim_info_t {
        dim_info_t() = default;
        dim_info_t(dim_idx_t idx, int64_t size) : idx(idx), size(size) {}
        dim_idx_t idx = dim_idx::invalid;
        int64_t size = 0;
    };

    virtual ~lws_strategy_t() = default;
    virtual range_t create_lws(range_t &gws) const = 0;

    virtual const dim_info_t &subgroup() const {
        static dim_info_t ret;
        return ret;
    }

    virtual const std::array<dim_info_t, 3> &local() const {
        static std::array<dim_info_t, 3> ret;
        return ret;
    }
};

// Balance lws size with occupation
struct default_lws_strategy_t : public lws_strategy_t {
    default_lws_strategy_t(gpu_arch_t arch) : arch(arch) {};
    default_lws_strategy_t(const impl::engine_t *engine)
        : default_lws_strategy_t(
                  utils::downcast<const intel::engine_t *>(engine)
                          ->device_info()
                          ->gpu_arch()) {};
    range_t create_lws(range_t &gws) const override {
        range_t lws = get_optimal_lws(gws, -1, arch);
        return lws;
    }
    gpu_arch_t arch;
};

struct explicit_lws_strategy_t : public lws_strategy_t {
    explicit_lws_strategy_t(
            int subgroup_size, const std::array<dim_info_t, 3> &local)
        : subgroup_(subgroup_size
                          ? dim_info_t(local[0].idx, int64_t(subgroup_size))
                          : dim_info_t())
        , local_(local) {}

    range_t create_lws(range_t &gws) const override {
        std::vector<int> lws;
        for (size_t i = 0; i < gws.ndims(); i++) {
            if (local()[i].idx == dim_idx::invalid) {
                lws.emplace_back(1);
                continue;
            }
            lws.emplace_back(local()[i].size);
            gws[i] = utils::rnd_up(gws[i], local()[i].size);
        }
        return lws;
    }
    const dim_info_t &subgroup() const override { return subgroup_; }
    const std::array<dim_info_t, 3> &local() const override { return local_; }

private:
    dim_info_t subgroup_;
    std::array<dim_info_t, 3> local_;
};

class reusable_dispatch_t {
public:
    reusable_dispatch_t() = default;
    reusable_dispatch_t(const compute::nd_range_t &nd_range,
            int64_t subgroup_size, const std::vector<named_buffer_t> &buffers,

            uint8_t gws_overflow, uint8_t in_padding,
            const std::vector<uint8_t> &buffer_off_exprs,
            const std::vector<uint8_t> &expr_data,
            const std::vector<int64_t> &term_list) {

        compile_params.gws_overflow_expr = gws_overflow;
        compile_params.in_padding_expr = in_padding;
        for (size_t i = 0; i < expr_data.size(); i++)
            compile_params.exprs[i] = expr_data[i];
        for (size_t i = 0; i < buffer_off_exprs.size(); i++)
            compile_params.buffer_off_expr[i] = buffer_off_exprs[i];

        auto &buffer_set = compile_params.buffer_set;
        for (size_t i = 0; i < buffers.size(); i++)
            buffer_set = buffer_set | buffers[i].get_name_id();

        int term_idx = 0;
        dim_t max_buffer_size = 0;
        for (auto &buffer : buffers) {
            compile_params.buffer_types[term_idx] = buffer.data_type;
            max_buffer_size = std::max(max_buffer_size, buffer.nelems(true));
            term_idx++;
        }

        compile_params.use_int32_offset = max_buffer_size <= INT32_MAX;
        compile_params.subgroup_size = into<uint8_t>(subgroup_size);

        // Set runtime params
        runtime_params = dispatch_runtime_params_t(
                nd_range, compile_params.use_int32_offset, {term_list});
    }

    const dispatch_compile_params_t &get_compile_params() const {
        return compile_params;
    }
    const dispatch_runtime_params_t &get_runtime_params() const {
        return runtime_params;
    }

private:
    dispatch_compile_params_t compile_params;
    dispatch_runtime_params_t runtime_params;
};

class reusable_dispatch_config_t {
public:
    reusable_dispatch_config_t(
            std::vector<dim_idx_t> dims, const lws_strategy_t &lws_strategy)
        : dispatched_dims(std::move(dims)), lws_strategy(lws_strategy) {};
    status_t generate(reusable_dispatch_t &dispatch);
    status_t register_buffer(const named_buffer_t &buffer);
    status_t define_dim_index(name_id_t dim_name, dim_idx_t dim_id, dim_t size);

    struct dim_size_t {
        dim_t size;
        dim_t padded_size;
    };

private:
    std::vector<named_buffer_t> buffers;
    std::vector<dim_idx_t> dispatched_dims;
    std::map<dim_idx_t, dim_size_t> dim_sizes;
    const lws_strategy_t &lws_strategy;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
