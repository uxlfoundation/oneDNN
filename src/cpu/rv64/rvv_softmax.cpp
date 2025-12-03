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

#include <math.h>
#include <riscv_vector.h>

#include "cpu/rv64/rvv_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {

// f32 compute kernel
void compute_softmax_f32_rvv(
        const float *src, float *dst, dim_t len, bool is_logsoftmax) {
    float max_val = -INFINITY;
    for (dim_t i = 0; i < len; ++i)
        max_val = src[i] > max_val ? src[i] : max_val;

    if (is_logsoftmax) {
        float sum_exp = 0.f;
        for (dim_t i = 0; i < len; ++i) {
            sum_exp += expf(src[i] - max_val);
        }
        const float log_sum = logf(sum_exp);

        for (dim_t i = 0; i < len;) {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(len - i));
            vfloat32m1_t vx = __riscv_vle32_v_f32m1(src + i, vl);
            vfloat32m1_t vdelta = __riscv_vfsub_vf_f32m1(vx, max_val, vl);
            vfloat32m1_t vy = __riscv_vfsub_vf_f32m1(vdelta, log_sum, vl);
            __riscv_vse32_v_f32m1(dst + i, vy, vl);
            i += (dim_t)vl;
        }
    } else {
        float sum_exp = 0.f;
        for (dim_t i = 0; i < len; ++i) {
            float e = expf(src[i] - max_val);
            dst[i] = e;
            sum_exp += e;
        }

        const float inv_sum = 1.0f / sum_exp;
        for (dim_t i = 0; i < len;) {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(len - i));
            vfloat32m1_t vy = __riscv_vle32_v_f32m1(dst + i, vl);
            vy = __riscv_vfmul_vf_f32m1(vy, inv_sum, vl);
            __riscv_vse32_v_f32m1(dst + i, vy, vl);
            i += (dim_t)vl;
        }
    }
}

} // namespace

status_t rvv_softmax_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    const auto &rsp = pd()->rsp_;

    switch (rsp.data_type) {
        case data_type::f32: {
            const float *src_f32 = static_cast<const float *>(src);
            float *dst_f32 = static_cast<float *>(dst);

            const dim_t outer_stride = pd()->axis_size(true) * rsp.inner_size;
            const int nthr = pd()->nthr_;

            if (rsp.inner_size == 1) {
                parallel_nd(rsp.outer_size, [&](dim_t outer) {
                    const dim_t base = outer * outer_stride;
                    compute_softmax_f32_rvv(src_f32 + base, dst_f32 + base,
                            rsp.axis_size, rsp.is_logsoftmax);
                });
            } else {
                auto scratch = ctx.get_scratchpad_grantor().template get<char>(
                        memory_tracking::names::key_softmax_interim_store);

                parallel(nthr, [&](int ithr, int nthr) {
                    float *tmp = reinterpret_cast<float *>(scratch)
                            + static_cast<size_t>(ithr)
                                    * static_cast<size_t>(rsp.axis_size);

                    const dim_t work_amount = rsp.outer_size * rsp.inner_size;
                    dim_t start {0}, end {0};
                    balance211(work_amount, nthr, ithr, start, end);
                    for (dim_t idx = start; idx < end; ++idx) {
                        const dim_t outer = idx / rsp.inner_size;
                        const dim_t i = idx % rsp.inner_size;
                        const dim_t base = outer * outer_stride + i;

                        // gather -> tmp
                        for (dim_t a = 0; a < rsp.axis_size; ++a)
                            tmp[a] = src_f32[base + a * rsp.inner_size];

                        // contiguous kernel (in-place)
                        compute_softmax_f32_rvv(
                                tmp, tmp, rsp.axis_size, rsp.is_logsoftmax);

                        // write back
                        for (dim_t a = 0; a < rsp.axis_size; ++a)
                            dst_f32[base + a * rsp.inner_size] = tmp[a];
                    }
                });
            }
        } break;
        default: return status::unimplemented;
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
