/******************************************************************************
* Copyright 2025
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
#include <riscv_vector.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/rv64/rvv_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

// If per_elem_params is false, uses broadcast scalars mean/sm/sv (mean[0], sm[0], sv[0]).
// If true, loads per-element mean/sm/sv from the provided arrays.
static inline void bn_fwd_kernel_f32(const void *s_base, void *d_base,
        size_t len, const float *mean, const float *sm, const float *sv,
        bool per_elem_params, const rv64::rvv_postops_t &po) {
    const size_t data_size = types::data_type_size(data_type::f32);
    for (size_t i = 0; i < len;) {
        size_t vl = __riscv_vsetvl_e32m1(len - i);

        const float *s_ptr = reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(s_base) + i * data_size);
        float *d_ptr = reinterpret_cast<float *>(
                reinterpret_cast<char *>(d_base) + i * data_size);

        vfloat32m1_t vx = __riscv_vle32_v_f32m1(s_ptr, vl);

        vfloat32m1_t vmean_v;
        vfloat32m1_t vsm_v;
        vfloat32m1_t vsv_v;
        if (per_elem_params) {
            vmean_v = __riscv_vle32_v_f32m1(mean + i, vl);
            vsm_v = __riscv_vle32_v_f32m1(sm + i, vl);
            vsv_v = __riscv_vle32_v_f32m1(sv + i, vl);
        } else {
            vmean_v = __riscv_vfmv_v_f_f32m1(mean[0], vl);
            vsm_v = __riscv_vfmv_v_f_f32m1(sm[0], vl);
            vsv_v = __riscv_vfmv_v_f_f32m1(sv[0], vl);
        }

        vfloat32m1_t vtmp = __riscv_vfsub_vv_f32m1(vx, vmean_v, vl);
        vfloat32m1_t vout = __riscv_vfmul_vv_f32m1(vtmp, vsm_v, vl);
        vout = __riscv_vfadd_vv_f32m1(vout, vsv_v, vl);
        vout = po.apply(vout, vl);

        __riscv_vse32_v_f32m1(d_ptr, vout, vl);
        i += vl;
    }
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

    rv64::rvv_postops_t po = pd()->fused_relu_in_kernel()
            ? rv64::rvv_postops_t(alg_kind::eltwise_relu)
            : rv64::rvv_postops_t(pd()->attr()->post_ops_);

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
                            mean_b, sm_b, sv_b, /*per_elem_params=*/false, po);
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
                            /*per_elem_params=*/true, po);
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
