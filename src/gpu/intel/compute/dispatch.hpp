/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_COMPUTE_DISPATCH_HPP
#define GPU_INTEL_COMPUTE_DISPATCH_HPP

#include <cassert>
#include <string>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

range_t get_optimal_lws(range_t &gws, const dim_idx_t mapped_vec_dim_idx,
        const gpu_arch_t gpu_arch);

class compute_engine_t;

class dispatch_t {
public:
    static constexpr int min_nesting_level = -1;
    static constexpr dim_idx_t dim_not_found = -1;

    // md - memory descriptor hint to extract nesting levels based on the layout.
    dispatch_t(const compute_engine_t *engine = nullptr,
            const memory_desc_t *md = nullptr);

    nd_range_t nd_range() const {
        assert(generate_called && "generate() must be called.");
        return nd_range_;
    }

    std::string str() const;

    void define_dim(const std::string &name, dim_idx_t md_hint_idx, dim_t size,
            dim_t block = 1) {
        define_dim_with_md_hint(name, md_hint_idx, size, block);
    }

    void define_dim(const std::string &name, dim_t size) {
        define_dim_with_nesting_level(name, min_nesting_level, size);
    }

    void define_dim_with_nesting_level(const std::string &name,
            int nesting_level, dim_t size, dim_t block = 1);
    status_t vectorize_dim(const std::string &name, int vector_size);

    void def_kernel_macros(kernel_ctx_t &kernel_ctx) const;

    // Attribute suffix is only required to support multiple kernels within a
    // single kernel context.
    void set_kernel_attr_suffix(const std::string &suffix) {
        attr_suffix_ = suffix;
    }

    void generate(bool generate_lws = true);

    void generate_override(
            const range_t &grange, const range_t &lrange = range_t());
    void set_lws(const range_t &lrange);

    // Dimension information necessary for mapping to global work IDs.
    struct dim_info_t {
        // Dimension name to access from a kernel as GWS_GET_<name>().
        std::string name;

        // Size of the dimension.
        dim_t size;

        // Block size that the kernel uses for the dimension. With blocking,
        // every kernel instance handles a block of indices. Possible values:
        //     0: flexible blocking
        //     1: no blocking
        //     > 1: fixed blocking
        dim_t block;

        // Outermost dimension has the min value.
        // Innermost dimension has the max value.
        int nesting_level;

        // -1: no vectorization; at most one dimension may be vectorized.
        int vector_size;

        // Either of [0, 1, 2] - the ID that the dimension maps to.
        int gws_index;
    };

protected:
    void define_dim_with_md_hint(const std::string &name,
            dim_idx_t md_hint_index, dim_t size, dim_t block = 1);

    dim_idx_t find_vectorized_dim() const {
        dim_idx_t vec_dim_idx = dim_not_found;
        for (dim_idx_t i = 0; i < ndims_; ++i) {
            if (dims_[i].vector_size != 1) {
                assert(vec_dim_idx == dim_not_found);
                assert(dims_[i].block > 0);
                vec_dim_idx = i;
            }
        }
        return vec_dim_idx;
    }

    dim_t get_gws_stride(int idx) const {
        dim_t s = 1;
        for (int i = 0; i < idx; ++i) {
            if (dims_[i].gws_index == dims_[idx].gws_index) {
                s *= utils::div_up(dims_[i].size, dims_[i].block);
            }
        }
        return s;
    }

    const compute_engine_t *engine_;

    dim_idx_t md_ndims_ = 0;
    int md_nesting_levels_[DNNL_MAX_NDIMS];

    dim_idx_t ndims_ = 0;
    dim_info_t dims_[DNNL_MAX_NDIMS];

    std::string attr_suffix_ = "DEFAULT";
    nd_range_t nd_range_;
    bool generate_called = false;
};

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
