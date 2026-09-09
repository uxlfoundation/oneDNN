/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#include <cmath>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/ref_dynamic_quantize.hpp"
#include "cpu/ref_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {

constexpr float scale_eps = 1.0e-30f;

float symmetric_scale(float absmax) {
    return nstl::max(absmax / 127.0f, scale_eps);
}

void set_src_pos_from_scale_pos(
        dims_t src_pos, const dims_t scale_pos, int ndims, dim_t m, dim_t n) {
    for (int d = 0; d < ndims - 2; ++d)
        src_pos[d] = scale_pos[d];
    src_pos[ndims - 2] = m;
    src_pos[ndims - 1] = n;
}

void get_group_bounds(const memory_desc_wrapper &src_mdw,
        const memory_desc_wrapper &scale_mdw, const dims_t scale_pos,
        dim_t &m_start, dim_t &m_end, dim_t &n_start, dim_t &n_end) {
    const int ndims = src_mdw.ndims();
    const dim_t M = src_mdw.dims()[ndims - 2];
    const dim_t N = src_mdw.dims()[ndims - 1];
    const dim_t sm = scale_mdw.dims()[ndims - 2];
    const dim_t sn = scale_mdw.dims()[ndims - 1];
    const dim_t scale_m = scale_pos[ndims - 2];
    const dim_t scale_n = scale_pos[ndims - 1];

    if (sm == 1 && sn == 1) {
        m_start = 0;
        m_end = M;
        n_start = 0;
        n_end = N;
    } else if (sm == M && sn == 1) {
        m_start = scale_m;
        m_end = scale_m + 1;
        n_start = 0;
        n_end = N;
    } else if (sm == 1 && sn == N) {
        m_start = 0;
        m_end = M;
        n_start = scale_n;
        n_end = scale_n + 1;
    } else if (sn == N) {
        const dim_t rows_per_group = M / sm;
        m_start = scale_m * rows_per_group;
        m_end = m_start + rows_per_group;
        n_start = scale_n;
        n_end = scale_n + 1;
    } else {
        const dim_t cols_per_group = N / sn;
        m_start = scale_m;
        m_end = scale_m + 1;
        n_start = scale_n * cols_per_group;
        n_end = n_start + cols_per_group;
    }
}

} // namespace

status_t ref_dynamic_quantize_reduction_t::execute_ref(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    auto scale = CTX_OUT_MEM(void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const memory_desc_wrapper src_mdw(pd()->src_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());
    const memory_desc_wrapper scale_mdw(pd()->scale_md());

    const int ndims = src_mdw.ndims();
    const bool compute_only = pd()->is_compute_only();

    parallel_nd(scale_mdw.nelems(), [=](dim_t scale_l_offset) {
        dims_t scale_pos;
        utils::l_dims_by_l_offset(
                scale_pos, scale_l_offset, scale_mdw.dims(), ndims);

        dim_t m_start, m_end, n_start, n_end;
        get_group_bounds(
                src_mdw, scale_mdw, scale_pos, m_start, m_end, n_start, n_end);

        float absmax = 0.0f;

        dims_t src_pos;
        for (dim_t m = m_start; m < m_end; ++m) {
            for (dim_t n = n_start; n < n_end; ++n) {
                set_src_pos_from_scale_pos(src_pos, scale_pos, ndims, m, n);
                const dim_t src_off = src_mdw.off_v(src_pos);
                const float val = io::load_float_value(
                        src_mdw.data_type(), src, src_off);
                if (!std::isfinite(val)) continue;
                absmax = nstl::max(absmax, nstl::abs(val));
            }
        }

        const float scale_val = symmetric_scale(absmax);

        const dim_t scale_off = scale_mdw.off_v(scale_pos);
        io::store_float_value(
                scale_mdw.data_type(), scale_val, scale, scale_off);

        if (compute_only) return;

        for (dim_t m = m_start; m < m_end; ++m) {
            for (dim_t n = n_start; n < n_end; ++n) {
                set_src_pos_from_scale_pos(src_pos, scale_pos, ndims, m, n);
                const dim_t src_off = src_mdw.off_v(src_pos);
                const dim_t dst_off = dst_mdw.off_v(src_pos);
                const float val = io::load_float_value(
                        src_mdw.data_type(), src, src_off);

                float q = 0.0f;
                if (std::isfinite(val)) {
                    q = nstl::max(-127.0f,
                            nstl::min(127.0f, nearbyintf(val / scale_val)));
                }
                io::store_float_value(dst_mdw.data_type(), q, dst, dst_off);
            }
        }
    });

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
