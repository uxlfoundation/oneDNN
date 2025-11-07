/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_KAI_WINO_UTILS_HPP
#define CPU_AARCH64_KAI_WINO_UTILS_HPP

#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/kai_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace kai_wino_utils {

inline dim_t n_matrices(const wino_desc_t &wd) {
    return static_cast<dim_t>(wd.alpha) * wd.alpha;
}

inline memory_desc_t make_packed_weights_desc(data_type_t data_type,
        dim_t matrices, dim_t ic, dim_t oc, int ic_block, int oc_block) {
    memory_desc_t md = {};
    md.ndims = 3;
    md.dims[0] = md.padded_dims[0] = matrices;
    md.dims[1] = md.padded_dims[1] = ic;
    md.dims[2] = md.padded_dims[2] = oc;
    md.data_type = data_type;
    md.format_kind = format_kind::blocked;
    md.format_desc.blocking = blocking_desc_t {};

    constexpr dim_t matrix_dim = 0;
    constexpr dim_t ic_dim = 1;
    constexpr dim_t oc_dim = 2;

    md.format_desc.blocking.strides[ic_dim] = oc_block * ic_block;
    md.padded_dims[ic_dim] = utils::rnd_up(md.dims[ic_dim], ic_block);

    md.format_desc.blocking.strides[oc_dim] = oc_block * md.padded_dims[ic_dim];
    md.padded_dims[oc_dim] = utils::rnd_up(md.dims[oc_dim], oc_block);

    md.format_desc.blocking.strides[matrix_dim]
            = md.padded_dims[ic_dim] * md.padded_dims[oc_dim];

    if (oc_block > 1) {
        md.format_desc.blocking.inner_nblks = 1 + (ic_block > 1);
        md.format_desc.blocking.inner_idxs[0] = oc_dim;
        md.format_desc.blocking.inner_blks[0] = oc_block;
        if (ic_block > 1) {
            md.format_desc.blocking.inner_idxs[1] = ic_dim;
            md.format_desc.blocking.inner_blks[1] = ic_block;
        }
    }

    return md;
}

inline memory_desc_t make_packed_weights_desc(const memory_desc_t &wino_md) {
    const auto &wd = wino_md.format_desc.wino_desc;
    return make_packed_weights_desc(wino_md.data_type, n_matrices(wd), wd.ic,
            wd.oc, wd.ic_block, wd.oc_block);
}

inline void init_packed_weights_desc(memory_desc_t &md,
        const kai::ops::WeightFormat weight_format, int r, int alpha, int ic,
        int oc) {
    const int ic_block = kai_utils::block_by(weight_format);
    const int oc_block = kai_utils::interleave_by(weight_format);
    const auto packed_md = make_packed_weights_desc(md.data_type,
            static_cast<dim_t>(alpha) * alpha, ic, oc, ic_block, oc_block);

    md.format_kind = format_kind::wino;
    md.format_desc.wino_desc = wino_desc_t {};
    auto &wd = md.format_desc.wino_desc;
    wd.wino_format = wino_memory_format_t::wino_wei_aaIOoi;
    wd.r = r;
    wd.alpha = alpha;
    wd.ic = ic;
    wd.oc = oc;
    wd.ic_block = ic_block;
    wd.oc_block = oc_block;
    wd.ic2_block = 0;
    wd.oc2_block = 0;
    wd.adj_scale = 1.f;
    wd.size = memory_desc_wrapper(&packed_md).size();
}

inline int packed_ld(const memory_desc_t &wino_md) {
    const auto packed_md = make_packed_weights_desc(wino_md);
    return static_cast<int>(packed_md.format_desc.blocking.strides[2]);
}

inline int packed_multi_stride(const memory_desc_t &wino_md) {
    const auto packed_md = make_packed_weights_desc(wino_md);
    return static_cast<int>(packed_md.format_desc.blocking.strides[0]);
}

} // namespace kai_wino_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
