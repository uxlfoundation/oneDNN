/******************************************************************************
* Copyright 2025 ZTE Corporation
* Copyright 2026 SpacemiT Corporation
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

#include "common/float16.hpp"
#include "common/nstl.hpp"

#include "cpu/rv64/jit_rvv_softmax_kernel.hpp"
#include "cpu/rv64/rvv_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

rvv_softmax_fwd_t::rvv_softmax_fwd_t(const pd_t *apd) : primitive_t(apd) {
    if (pd()->use_jit_) {
        affine_kernel_.reset(new jit_rvv_softmax_affine_kernel_t());
    }
}

namespace {

// f32 compute kernel. jit_exp selects the vectorized exp/sub/sum kernel for the
// reduction; the caller gates it on axis length and stride so that very short
// or memory-bound (large-stride) reductions keep the scalar exp path.
void compute_softmax_f32_rvv(const float *src, float *dst, dim_t len,
        bool is_logsoftmax, bool is_softmax_inf_as_zero,
        const jit_rvv_softmax_affine_kernel_t *affine_kernel, bool jit_exp) {
    float max_val = -INFINITY;
    for (dim_t i = 0; i < len; ++i)
        max_val = src[i] > max_val ? src[i] : max_val;

    if (is_logsoftmax) {
        float sum_exp = 0.f;
        if (jit_exp) {
            jit_rvv_softmax_f32_exp_sub_sum(
                    src, dst, len, max_val, &sum_exp, false);
        } else {
            for (dim_t i = 0; i < len; ++i) {
                sum_exp += expf(src[i] - max_val);
            }
        }
        const float log_sum = logf(sum_exp);

        if (affine_kernel) {
            jit_rvv_softmax_affine_kernel_t::call_params_t p;
            p.src = src;
            p.dst = dst;
            p.len = len;
            p.sub = max_val + log_sum;
            p.mul = 1.f;
            (*affine_kernel)(&p);
        } else {
            for (dim_t i = 0; i < len; ++i)
                dst[i] = src[i] - max_val - log_sum;
        }
    } else {
        float sum_exp = 0.f;
        const bool all_minus_inf
                = is_softmax_inf_as_zero && (max_val == -INFINITY);
        if (jit_exp) {
            if (all_minus_inf) {
                for (dim_t i = 0; i < len; ++i)
                    dst[i] = 0.f;
            } else {
                jit_rvv_softmax_f32_exp_sub_sum(
                        src, dst, len, max_val, &sum_exp, true);
            }
        } else {
            for (dim_t i = 0; i < len; ++i) {
                float e = all_minus_inf ? 0.f : expf(src[i] - max_val);
                dst[i] = e;
                sum_exp += e;
            }
        }

        const float inv_sum = sum_exp ? (1.0f / sum_exp) : 1.0f;
        if (affine_kernel) {
            jit_rvv_softmax_affine_kernel_t::call_params_t p;
            p.src = dst;
            p.dst = dst;
            p.len = len;
            p.sub = 0.f;
            p.mul = inv_sum;
            (*affine_kernel)(&p);
        } else {
            for (dim_t i = 0; i < len; ++i)
                dst[i] *= inv_sum;
        }
    }
}

#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
void compute_softmax_f16_scalar(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, bool is_logsoftmax,
        float max_val) {
    float sum_exp = 0.f;

    for (dim_t i = 0; i < len; ++i) {
        const float D = (float)src[i] - max_val;
        sum_exp += expf(D);
    }

    if (is_logsoftmax) {
        const float log_sum = logf(sum_exp);
        for (dim_t i = 0; i < len; ++i)
            dst[i] = (float)src[i] - max_val - log_sum;
    } else {
        const float inv_sum = sum_exp ? (1.0f / sum_exp) : 1.0f;
        for (dim_t i = 0; i < len; ++i)
            dst[i] = expf((float)src[i] - max_val) * inv_sum;
    }
}

// f16 compute kernel
void compute_softmax_f16_rvv(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, bool is_logsoftmax,
        bool is_softmax_inf_as_zero, float *scratchpad) {

    float max_val = 0.f;
    uint32_t has_nan = 0;
    // Vectorized pass that replaces the former scalar loop with per-element
    // f16->f32 widening (vfwcvt) and a vector max reduction. The seed and the
    // NaN semantics reproduce the scalar `val > max_val` loop: a NaN never
    // updates the max, and has_nan is set for any (raw & 0x7fff) > 0x7c00.
    jit_rvv_softmax_f16_reduce_max(src, len, &max_val, &has_nan);
    const bool max_is_pos_inf = max_val == INFINITY;

    if (len == 1 && isfinite((float)src[0])) {
        dst[0] = is_logsoftmax ? 0.f : 1.f;
        return;
    }

    if (has_nan || max_is_pos_inf) {
        compute_softmax_f16_scalar(src, dst, len, is_logsoftmax, max_val);
        return;
    }

    if (is_logsoftmax) {
        float sum_exp = 0.f;

        jit_rvv_softmax_f16_exp_sub_sum(
                src, scratchpad, len, max_val, &sum_exp);
        const float log_sum = logf(sum_exp);

        const float sub = max_val + log_sum;
        jit_rvv_softmax_f16_affine_from_f16(src, dst, len, sub, 1.0f);
    } else {
        float sum_exp = 0.f;
        const bool all_minus_inf
                = is_softmax_inf_as_zero && (max_val == -INFINITY);

        if (all_minus_inf) {
            for (dim_t i = 0; i < len; ++i)
                scratchpad[i] = 0.f;
        } else {
            jit_rvv_softmax_f16_exp_sub_sum(
                    src, scratchpad, len, max_val, &sum_exp);
        }
        const float inv_sum = sum_exp ? (1.0f / sum_exp) : 1.0f;

        jit_rvv_softmax_f16_affine_from_f32(
                scratchpad, dst, len, 0.0f, inv_sum);
    }
}
#endif

} // namespace

status_t rvv_softmax_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    const auto &rsp = pd()->rsp_;
    const bool is_softmax_inf_as_zero
            = pd()->alg_kind() == alg_kind::softmax_accurate_inf_as_zero;

    switch (rsp.data_type) {
        case data_type::f32: {
            const float *src_f32 = static_cast<const float *>(src);
            float *dst_f32 = static_cast<float *>(dst);

            const dim_t outer_stride = pd()->axis_size(true) * rsp.inner_size;
            const int nthr = pd()->nthr_;

            // Use the vectorized exp kernel only when it pays off: the axis
            // must fill at least one LMUL=4 e32 vector (16 elements at VLEN=128;
            // on larger VLEN the kernel still runs a correct partial vector),
            // and a strided axis must have a moderate stride so the reduction is
            // not dominated by the gather/scatter memory traffic.
            constexpr dim_t exp_jit_min_len = 16;
            constexpr dim_t exp_jit_max_inner = 8192;
            const bool jit_exp = affine_kernel_.get()
                    && rsp.axis_size >= exp_jit_min_len
                    && rsp.inner_size <= exp_jit_max_inner;

            if (rsp.inner_size == 1) {
                parallel_nd(rsp.outer_size, [&](dim_t outer) {
                    const dim_t base = outer * outer_stride;
                    compute_softmax_f32_rvv(src_f32 + base, dst_f32 + base,
                            rsp.axis_size, rsp.is_logsoftmax,
                            is_softmax_inf_as_zero, affine_kernel_.get(),
                            jit_exp);
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
                        compute_softmax_f32_rvv(tmp, tmp, rsp.axis_size,
                                rsp.is_logsoftmax, is_softmax_inf_as_zero,
                                affine_kernel_.get(), jit_exp);

                        // write back
                        for (dim_t a = 0; a < rsp.axis_size; ++a)
                            dst_f32[base + a * rsp.inner_size] = tmp[a];
                    }
                });
            }
        } break;
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
        case data_type::f16: {
            const auto *src_f16
                    = static_cast<const dnnl::impl::float16_t *>(src);
            auto *dst_f16 = static_cast<dnnl::impl::float16_t *>(dst);

            const dim_t outer_stride = pd()->axis_size(true) * rsp.inner_size;
            const int nthr = pd()->nthr_;
            auto reduction = ctx.get_scratchpad_grantor().template get<float>(
                    memory_tracking::names::key_softmax_reduction);

            if (rsp.inner_size == 1) {
                parallel(nthr, [&](int ithr, int nthr) {
                    float *tmp = reduction
                            + static_cast<size_t>(ithr)
                                    * static_cast<size_t>(rsp.axis_size);
                    dim_t start {0}, end {0};
                    balance211(rsp.outer_size, nthr, ithr, start, end);
                    for (dim_t outer = start; outer < end; ++outer) {
                        const dim_t base = outer * outer_stride;
                        compute_softmax_f16_rvv(src_f16 + base, dst_f16 + base,
                                rsp.axis_size, rsp.is_logsoftmax,
                                is_softmax_inf_as_zero, tmp);
                    }
                });
            } else {
                auto scratch = ctx.get_scratchpad_grantor().template get<char>(
                        memory_tracking::names::key_softmax_interim_store);

                parallel(nthr, [&](int ithr, int nthr) {
                    auto *tmp
                            = reinterpret_cast<dnnl::impl::float16_t *>(scratch)
                            + static_cast<size_t>(ithr)
                                    * static_cast<size_t>(rsp.axis_size);
                    float *reduction_tmp = reduction
                            + static_cast<size_t>(ithr)
                                    * static_cast<size_t>(rsp.axis_size);

                    const dim_t work_amount = rsp.outer_size * rsp.inner_size;
                    dim_t start {0}, end {0};
                    balance211(work_amount, nthr, ithr, start, end);

                    dim_t stride_bytes
                            = rsp.inner_size * sizeof(dnnl::impl::float16_t);

                    for (dim_t idx = start; idx < end; ++idx) {
                        const dim_t outer = idx / rsp.inner_size;
                        const dim_t i = idx % rsp.inner_size;
                        const dim_t base = outer * outer_stride + i;

                        jit_rvv_softmax_f16_gather(src_f16 + base, tmp,
                                rsp.axis_size, stride_bytes);

                        compute_softmax_f16_rvv(tmp, tmp, rsp.axis_size,
                                rsp.is_logsoftmax, is_softmax_inf_as_zero,
                                reduction_tmp);

                        jit_rvv_softmax_f16_scatter(tmp, dst_f16 + base,
                                rsp.axis_size, stride_bytes);
                    }
                });
            }
        } break;
#endif
        default: return status::unimplemented;
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
