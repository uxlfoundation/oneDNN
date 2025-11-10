/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2025 Arm Ltd. and affiliates
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

#include <cstdlib>
#include <memory>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/verbose.hpp"
#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

#if DNNL_X64
#include "cpu/x64/jit_gemm_x8s8s32x_convolution_utils.hpp"
#endif

#include "cpu/gemm_x8s8s32x_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace gemm_x8s8s32x_convolution_utils {

#define VCHECK_PO_BOOL(cond, msg) \
    VCONDCHECK(primitive, create, check, gemm_x8s8s32x, cond, false, msg);

template <typename dst_data_t>
struct ref_pp_ker_t : pp_ker_t {
    ref_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
        : pp_ker_t(pd, jcp), dst_md_(pd->dst_md()) {}

    using acc_data_t = pp_ker_t::acc_data_t;

    void operator()(void *dst, const acc_data_t *acc, const char *bias,
            const float *scales, float dst_scale, float sum_scale,
            float signed_scale, int g, int mb, size_t start, size_t end,
            const zero_point_call_params_t &zp,
            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
            const exec_ctx_t &ctx, const memory_desc_t &dst_md,
            const single_gemm_conv_chunk_desc_t &chunk_desc) const override;

    status_t create_kernel() override {
        if (this->jcp_.with_eltwise || this->jcp_.with_binary) {
            ref_post_ops_
                    = utils::make_unique<ref_post_ops_t>(this->jcp_.post_ops);
            if (!ref_post_ops_) return status::out_of_memory;
            return ref_post_ops_->init(dst_md_);
        }
        return status::success;
    }

private:
    std::unique_ptr<ref_post_ops_t> ref_post_ops_;
    const memory_desc_t *dst_md_;
};

template <typename dst_data_t>
void ref_pp_ker_t<dst_data_t>::operator()(void *void_dst, const acc_data_t *acc,
        const char *bias, const float *scales, float dst_scale, float sum_scale,
        float signed_scale, int g, int mb, size_t start, size_t end,
        const zero_point_call_params_t &zp,
        const void * /* post_ops_binary_rhs_arg_vec */,
        const void * /* dst_orig */, const exec_ctx_t &ctx,
        const memory_desc_t &dst_md,
        const single_gemm_conv_chunk_desc_t &chunk_desc) const {

    if (end <= start) return;

    assert(data_traits_t<dst_data_t>::data_type == jcp_.dst_data_type);

    const lldiv_t dv_start = std::div((long long)start, (long long)jcp_.oc);
    const lldiv_t dv_end = std::div((long long)(end - 1), (long long)jcp_.oc);
    const size_t first_oc = dv_start.rem;
    const size_t last_oc = dv_end.rem;
    const size_t first_os = dv_start.quot;
    const size_t last_os = dv_end.quot;
    const int32_t zp_dst_val = jcp_.zp.dst_exists ? *(zp.dst) : 0;

    ref_post_ops_t::args_t args;
    args.ctx = &ctx;
    args.dst_md = &dst_md;

    for (size_t os = first_os; os <= last_os; os++) {
        const size_t start_oc = (os == first_os) ? first_oc : 0;
        const size_t end_oc = (os == last_os) ? last_oc : jcp_.oc - 1;
        for (size_t oc = start_oc; oc <= end_oc; oc++) {
            const size_t acc_off = os * jcp_.oc + oc;
            const size_t dst_off = os * jcp_.dst_os_stride + oc;

            int32_t data_s32 = acc[acc_off];

            if (jcp_.zp.src_exists) {
                const auto oc_offset = g * jcp_.oc + oc;
                data_s32 += zp.src_comp[oc_offset];
            }

            float data = static_cast<float>(data_s32);

            if (jcp_.signed_input) data *= signed_scale;

            // dequantize data
            data *= scales[(g * jcp_.oc + oc) * jcp_.scale_idx_mult];

            if (jcp_.with_bias) {
                const float b = io::load_float_value(
                        jcp_.bias_data_type, bias, g * jcp_.oc + oc);
                data += b;
            }

            if (jcp_.with_sum)
                data += sum_scale
                        * io::load_float_value(
                                jcp_.sum_data_type, void_dst, dst_off);
            if (jcp_.with_eltwise || jcp_.with_binary) {
                args.l_offset = ((mb * jcp_.ngroups + g) * jcp_.oc + oc)
                                * jcp_.os * jcp_.od
                        + os;
                ref_post_ops_->execute(data, args);
            }

            // quantize data
            if (jcp_.with_dst_scale) data *= dst_scale;
            if (jcp_.zp.dst_exists) data += static_cast<float>(zp_dst_val);

            io::store_float_value(jcp_.dst_data_type, data, void_dst, dst_off);
        }
    }
}

// Interface section

pp_ker_t::pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
    : jcp_(jcp) {}

pp_ker_t *pp_ker_t::create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
#if DNNL_X64
    auto *res
            = x64::gemm_x8s8s32x_convolution_utils::jit_pp_ker_create(pd, jcp);
    if (res) return res;
#endif
    switch (pd->dst_md()->data_type) {
        case data_type::f32: return new ref_pp_ker_t<float>(pd, jcp);
        case data_type::bf16: return new ref_pp_ker_t<bfloat16_t>(pd, jcp);
        case data_type::s32: return new ref_pp_ker_t<int32_t>(pd, jcp);
        case data_type::s8: return new ref_pp_ker_t<int8_t>(pd, jcp);
        case data_type::u8: return new ref_pp_ker_t<uint8_t>(pd, jcp);
        default: assert(!"unexpected data type");
    }
    return nullptr;
}

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_wrapper *dst_d) {
// TODO: Align x64 and non-x64 support-scopes to avoid implementation surprises.
#if DNNL_X64
    return x64::gemm_x8s8s32x_convolution_utils::post_ops_ok(post_ops, dst_d);
#endif

    for (size_t po_index = 0; po_index < post_ops.entry_.size(); ++po_index) {
        const auto &post_op = post_ops.entry_[po_index];

        bool alg_kind_ok = post_op.is_eltwise() || post_op.is_sum()
                || post_op.is_prelu()
                || (post_op.is_binary()
                        && !post_op.is_binary_with_ternary_op());

        VCHECK_PO_BOOL(alg_kind_ok, "unsupported post-op alg kind");

        if (post_op.is_prelu()) {
            VCHECK_PO_BOOL(post_op.prelu.mask <= 3, "unsupported PReLU mask");

        } else if (post_op.is_binary()) {
            // The intent here is to limit the MASK_INPUT parameter of
            // binary attr-post-ops to {0, 1, 2, 3}. However, there is no
            // standard utility function for doing this, so we use the
            // one provided for broadcasting_strategy_t because the
            // logic is the same as it would be for MASK_INPUT.
            //
            // See the discussion in:
            // https://github.com/uxlfoundation/oneDNN/issues/3803
            const auto &bcast_type = get_rhs_arg_broadcasting_strategy(
                    post_op.binary.src1_desc, *dst_d,
                    {broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_mb,
                            broadcasting_strategy_t::per_oc,
                            broadcasting_strategy_t::spatial});

            VCHECK_PO_BOOL(bcast_type != broadcasting_strategy_t::unsupported,
                    "unsupported binary post-op mask")
        } else if (post_op.is_sum()) {
            VCHECK_PO_BOOL(
                    po_index == 0, "unsupported position for sum post-op")
        }
    }

    return true;
}

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_t *dst_d) {
    const auto dst_md = memory_desc_wrapper(dst_d);
    return post_ops_ok(post_ops, &dst_md);
}

bool mayiuse_jit_pp_kernel(data_type_t dst_dt) noexcept {
#if DNNL_X64
    return x64::gemm_x8s8s32x_convolution_utils::mayiuse_jit_pp_kernel(dst_dt);
#else
    return false;
#endif
}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
