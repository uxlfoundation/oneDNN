/*******************************************************************************
* Copyright 2016-2025 Intel Corporation
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

#ifndef CPU_RV64_RVV_GEMM_CONVOLUTION_HPP
#define CPU_RV64_RVV_GEMM_CONVOLUTION_HPP

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/gemm/gemm.hpp"
#include "cpu/primitive_attr_postops.hpp"
#include "cpu/rv64/rvv_gemm_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct riscv_gemm_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, riscv_gemm_convolution_fwd_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using namespace data_type;

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);

            if (with_bias()) {
                VDISPATCH_CONV(expect_data_types(f32, f32, f32, f32, f32),
                        VERBOSE_UNSUPPORTED_DT_CFG);
            } else {
                VDISPATCH_CONV(
                        expect_data_types(f32, f32, data_type::undef, f32, f32),
                        VERBOSE_UNSUPPORTED_DT_CFG);
            }

            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_CONV(
                    attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, f32),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);

            auto scratchpad = scratchpad_registry().registrar();

            // TODO: make `init_conf` assign initialized object to `jcp_`
            jcp_ = conv_gemm_conf_t();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md_, weights_md_, dst_md_, bias_md_, attr_,
                    dnnl_get_max_threads());
        }

        conv_gemm_conf_t jcp_ = utils::zero<decltype(jcp_)>();

    protected:
        bool post_ops_ok() const {
            auto const &po = attr()->post_ops_;
            auto is_sum_ok = [&](int idx) {
                return IMPLICATION(po.entry_[idx].kind == primitive_kind::sum,
                        idx == 0 && po.entry_[idx].is_sum());
            };
            auto is_binary
                    = [&](int idx) { return po.entry_[idx].is_binary(); };
            auto is_prelu = [&](int idx) { return po.entry_[idx].is_prelu(); };
            auto is_binary_or_prelu_supported = [&](int idx) {
                bool ok = dnnl::impl::get_rhs_arg_broadcasting_strategy(
                                  binary_injector_utils::get_src1_desc(
                                          po.entry_[idx], dst_md_),
                                  dst_md_,
                                  {broadcasting_strategy_t::scalar,
                                          broadcasting_strategy_t::per_oc})
                        != broadcasting_strategy_t::unsupported;
                return ok;
            };

            if (!ref_post_ops_t::post_ops_ok(attr()->post_ops_)) return false;

            for (int idx = 0; idx < po.len(); idx++) {
                bool ok = is_sum_ok(idx)
                        && IMPLICATION(is_binary(idx) || is_prelu(idx),
                                is_binary_or_prelu_supported(idx));
                if (!ok) return false;
            }

            return true;
        }
    };

    riscv_gemm_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), post_ops_(nullptr) {}

    status_t init(engine_t *engine) override {
        const auto &jcp = pd()->jcp_;

        if (jcp.with_eltwise || jcp.with_binary) {
            CHECK(safe_ptr_assign(post_ops_, new ref_post_ops_t(jcp.post_ops)));
            CHECK(post_ops_->init(pd()->dst_md()));
        }
        return status::success;
    }

    using data_t = typename prec_traits_t<data_type::f32>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        bool is_nspc = pd()->jcp_.is_nspc;
        return is_nspc ? execute_forward_nspc(ctx) : execute_forward_ncsp(ctx);
    }

private:
    status_t execute_forward_ncsp(const exec_ctx_t &ctx) const;
    status_t execute_forward_nspc(const exec_ctx_t &ctx) const;
    status_t execute_forward_thr_nspc(const exec_ctx_t &ctx, const int ithr,
            const int nthr, const data_t *src_base, const data_t *wei_base,
            const data_t *bia_base, data_t *dst_base,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<ref_post_ops_t> post_ops_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
