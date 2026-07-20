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

#ifndef CPU_AARCH64_KAI_WINO_CONVOLUTION_HPP
#define CPU_AARCH64_KAI_WINO_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/aarch64/post_ops_fallback.hpp"
#include "cpu/cpu_convolution_pd.hpp"

namespace kai::ops {
struct ConvolutionArgs;
struct GemmConfig;
struct IGemmCommon;
namespace winograd {
struct WinogradImpl;
} // namespace winograd
} // namespace kai::ops

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kai_wino_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                impl_name(), kai_wino_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);
        std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm() const;

        std::shared_ptr<kai::ops::GemmConfig> cfg_ = nullptr;
        std::shared_ptr<kai::ops::ConvolutionArgs> conv_args_ = nullptr;
        std::shared_ptr<kai::ops::winograd::WinogradImpl> wino_impl_ = nullptr;
        bool fixed_format_ = false;
        bool run_weight_reorder_ = false;
        bool src_channels_last_ = true;
        bool dst_channels_last_ = true;
        bool use_src_reorder_ = false;
        bool use_dst_reorder_ = false;
        size_t working_size_ = 0;
        bool has_post_ops_fallback_ = false;
        post_ops_fallback_t post_ops;
        memory_desc_t tmp_src_md_ {};
        memory_desc_t tmp_dst_md_ {};
        std::shared_ptr<primitive_desc_t> src_reorder_pd_;
        std::shared_ptr<primitive_desc_t> dst_reorder_pd_;

    private:
        const char *impl_name() const {
            return has_post_ops_fallback_ ? "wino:kai+post_ops_fallback"
                                          : "wino:kai";
        }

        bool set_default_formats();
    };

    using data_t = typename prec_traits_t<data_type::f32>::type;

    kai_wino_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    std::shared_ptr<primitive_t> src_reorder_;
    std::shared_ptr<primitive_t> dst_reorder_;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
