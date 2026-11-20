/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include <cstring>

#include "common/dnnl_thread.hpp"

#include "cpu/simple_concat.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;

status_t simple_concat_t::execute(const exec_ctx_t &ctx) const {
    const auto &scratchpad = ctx.get_scratchpad_grantor();
    auto iptrs = scratchpad.template get<const char *>(key_concat_iptrs);
    auto optrs = scratchpad.template get<char *>(key_concat_optrs);
    auto nelems_to_copy = scratchpad.template get<dim_t>(key_concat_nelems);
    auto is = scratchpad.template get<strides_t>(key_concat_istrides);

    const int num_arrs = pd()->n_inputs();
    const int *perm = pd()->perm_, *iperm = pd()->iperm_;
    const int concat_dim = pd()->concat_dim();
    const size_t dt_size = types::data_type_size(pd()->src_md(0)->data_type);
    auto o_base_ptr = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    if (o_base_ptr == nullptr) return status::success;

    for (int a = 0; a < num_arrs; ++a) {
        const memory_desc_wrapper i_d(pd()->src_md(a));
        const memory_desc_wrapper o_d(pd()->src_image_md(a));
        const auto iptr = CTX_IN_MEM(const char *, DNNL_ARG_MULTIPLE_SRC + a);
        if (iptr == nullptr) {
            iptrs[a] = nullptr;
            nelems_to_copy[a] = 0;
            continue;
        }
        iptrs[a] = iptr + i_d.blk_off(0) * dt_size;
        optrs[a] = o_base_ptr + o_d.blk_off(0) * dt_size;
        nelems_to_copy[a] = pd()->nelems_to_concat(i_d);
        for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
            if (i < perm[concat_dim])
                is[a][i] = size_t(i_d.blocking_desc().strides[iperm[i]]);
            else
                is[a][i] = 0;
        }
    }

    const memory_desc_wrapper o_d(pd()->dst_md(0));

    strides_t os = {0};
    bool has_outer_loop = false;
    for (int i = 0; i < perm[concat_dim]; i++) {
        os[i] = o_d.blocking_desc().strides[iperm[i]];
        // CAVEAT: if this impl supports not matching stag and dtag, strides
        // should be taken into account for this condition.
        if (o_d.padded_dims()[iperm[i]] != 1) has_outer_loop = true;
    }

    // Applies when concat axis is the outermost dimension, e.g. concat_axis = 0
    // or concat_axis = 1, and dims[0] = 1;
    if (!has_outer_loop) {
        int nthr = dnnl_get_max_threads();
        parallel(nthr, [=](int ithr, int nthr) {
            for (int a = 0; a < num_arrs; ++a) {
                dim_t start {0}, end {0};
                balance211(nelems_to_copy[a], nthr, ithr, start, end);

                const char *i = iptrs[a] + start * dt_size;
                char *o = optrs[a] + start * dt_size;

                std::memcpy(o, i, (end - start) * dt_size);
            }
        });

        return status::success;
    }

    dims_t phys_dims;
    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        if (i < perm[concat_dim])
            phys_dims[i]
                    = o_d.padded_dims()[iperm[i]] / pd()->blocks_[iperm[i]];
        else
            phys_dims[i] = 1;
    }

    const auto L1_size = platform::get_per_core_cache_size(1);
    UNUSED(L1_size); // for Windows

    parallel_nd(phys_dims[0], phys_dims[1], phys_dims[2], phys_dims[3],
            phys_dims[4], num_arrs,
            [=](dim_t n0, dim_t n1, dim_t n2, dim_t n3, dim_t n4, dim_t a) {
        // check if zero memory
        if (iptrs[a] == nullptr) return;

        // XXX: this code may access uninitialized values in is[*][0-4] --
        // that's why we have to set them to zero although this is
        // probably benign
        size_t in_off = is[a][0] * n0 + is[a][1] * n1 + is[a][2] * n2
                + is[a][3] * n3 + is[a][4] * n4;
        size_t out_off = os[0] * n0 + os[1] * n1 + os[2] * n2 + os[3] * n3
                + os[4] * n4;
        const char *i = iptrs[a] + in_off * dt_size;
        char *o = optrs[a] + out_off * dt_size;

#if defined(__GNUC__)
        // Heuristic:
        // memcpy works generally faster for data sizes not
        // exceeding L1 cache.
        if (nelems_to_copy[a] * dt_size > L1_size) {
            // The code below performs data copying: o[e] = i[e]
            // and uses a workaround to make GNU compilers optimize it
            uint8_t *ptro = reinterpret_cast<uint8_t *>(o);
            const uint8_t *ptri = reinterpret_cast<const uint8_t *>(i);

            const size_t head_part = sizeof(uint32_t)
                    - reinterpret_cast<uint64_t>(ptro) % sizeof(uint32_t);
            const size_t main_part = (nelems_to_copy[a] - head_part / dt_size)
                    * dt_size / sizeof(uint32_t);
            const size_t tail_part = (nelems_to_copy[a] * dt_size) - head_part
                    - (main_part * sizeof(uint32_t));
            for (size_t e = 0; e < head_part; ++e) {
                *ptro = *ptri;
                ++ptro;
                ++ptri;
            }
            PRAGMA_OMP_SIMD()
            for (size_t e = 0; e < main_part; ++e) {
                *(reinterpret_cast<uint32_t *>(ptro))
                        = *(reinterpret_cast<const uint32_t *>(ptri));
                ptro += sizeof(uint32_t);
                ptri += sizeof(uint32_t);
            }
            for (size_t e = 0; e < tail_part; ++e) {
                *ptro = *ptri;
                ++ptro;
                ++ptri;
            }
        } else {
            std::memcpy(o, i, nelems_to_copy[a] * dt_size);
        }
#else
        std::memcpy(o, i, nelems_to_copy[a] * dt_size);
#endif
    });

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
