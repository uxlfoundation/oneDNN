/******************************************************************************
* Copyright 2025 ZTE Corporation
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
******************************************************************************/

#include <assert.h>
#include <math.h>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/rv64/jit_rvv_batch_normalization_kernel.hpp"
#include "cpu/rv64/rvv_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

static inline void bn_fwd_kernel_f32(const void *s_base, void *d_base,
        size_t len, const float *mean, const float *sm, const float *sv,
        bool per_elem_params, bool with_relu) {
    jit_rvv_batch_normalization_apply_f32(static_cast<const float *>(s_base),
            static_cast<float *>(d_base), static_cast<dim_t>(len), mean, sm, sv,
            per_elem_params, with_relu);
}

} // namespace

status_t rvv_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    const memory_desc_wrapper data_d(pd()->src_md());
    const auto dtsrc = pd()->src_md()->data_type;
    const int ndims = data_d.ndims();

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();

    const float eps = pd()->desc()->batch_norm_epsilon;

    void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const float *mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
    const float *var = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    const float *scale = pd()->use_scale()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SCALE)
            : nullptr;
    const float *shift = pd()->use_shift()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SHIFT)
            : nullptr;

    const bool with_relu = pd()->fused_relu_in_kernel()
            || pd()->attr()->post_ops_.find(primitive_kind::eltwise) != -1;

    auto off = [&](dim_t n, dim_t c, dim_t d, dim_t h, dim_t w) -> size_t {
        switch (ndims) {
            case 3: return data_d.off(n, c, w);
            case 4: return data_d.off(n, c, h, w);
            case 5: return data_d.off(n, c, d, h, w);
            default: assert(!"unsupported ndims"); return dim_t(0);
        }
    };

    const bool channels_dense = data_d.blocking_desc().strides[1] == 1;

    if (!channels_dense) {
        // abx data tag: vectorize over W for fixed channel
        parallel_nd(C, N, D, H, [&](dim_t c, dim_t n, dim_t d, dim_t h) {
            const float vmean = mean[c];
            const float inv_std = 1.0f / sqrtf(var[c] + eps);
            const float vscale = scale ? scale[c] : 1.0f;
            const float vshift = shift ? shift[c] : 0.0f;
            const float sm = vscale * inv_std;
            const float sv = vshift;
            size_t base_off = off(n, c, d, h, 0);

            switch (dtsrc) {
                case data_type::f32: {
                    const size_t data_size
                            = types::data_type_size(data_type::f32);
                    const void *s_ptr = reinterpret_cast<const void *>(
                            reinterpret_cast<const char *>(src)
                            + base_off * data_size);
                    void *d_ptr = reinterpret_cast<void *>(
                            reinterpret_cast<char *>(dst)
                            + base_off * data_size);
                    const float mean_b[1] = {vmean};
                    const float sm_b[1] = {sm};
                    const float sv_b[1] = {sv};
                    bn_fwd_kernel_f32(s_ptr, d_ptr, static_cast<size_t>(W),
                            mean_b, sm_b, sv_b, /*per_elem_params=*/false,
                            with_relu);
                    break;
                }
                default:
                    assert(!"Unsupported data type for RVV batch "
                            "normalization");
            }
        });
    } else {
        // axb data tag: vectorize across channels
        auto &grantor = ctx.get_scratchpad_grantor();
        float *sm_arr = grantor.template get<float>(
                memory_tracking::names::key_bnorm_tmp_mean);
        float *sv_arr = grantor.template get<float>(
                memory_tracking::names::key_bnorm_tmp_var);
        for (dim_t c = 0; c < C; ++c) {
            const float inv_std = 1.0f / sqrtf(var[c] + eps);
            const float vscale = scale ? scale[c] : 1.0f;
            const float vshift = shift ? shift[c] : 0.0f;
            sm_arr[static_cast<size_t>(c)] = vscale * inv_std;
            sv_arr[static_cast<size_t>(c)] = vshift;
        }

        parallel_nd(N, D, H, W, [&](dim_t n, dim_t d, dim_t h, dim_t w) {
            switch (dtsrc) {
                case data_type::f32: {
                    const size_t data_size
                            = types::data_type_size(data_type::f32);
                    size_t base_off = off(n, 0, d, h, w);
                    const void *s_ptr = reinterpret_cast<const void *>(
                            reinterpret_cast<const char *>(src)
                            + base_off * data_size);
                    void *d_ptr = reinterpret_cast<void *>(
                            reinterpret_cast<char *>(dst)
                            + base_off * data_size);

                    bn_fwd_kernel_f32(s_ptr, d_ptr, static_cast<size_t>(C),
                            mean, sm_arr, sv_arr,
                            /*per_elem_params=*/true, with_relu);
                    break;
                }
                default:
                    assert(!"Unsupported data type for RVV batch "
                            "normalization");
            }
        });
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
