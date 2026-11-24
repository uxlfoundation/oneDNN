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

#include "cpu/aarch64/post_ops_fallback.hpp"
#include "cpu/cpu_convolution_pd.hpp"

namespace kai {
namespace ops {
// NOLINTBEGIN(readability-identifier-naming)
struct ConvolutionArgs;
struct GemmArgs;
class IGemmCommon;
namespace winograd {
struct WinogradImpl;
} // namespace winograd
// NOLINTEND(readability-identifier-naming)
} // namespace ops
} // namespace kai

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kai_wino_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                impl_name(), kai_wino_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(const engine_t *engine);
        std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm() const;
        const kai::ops::GemmArgs &gemm_args() const;
        kai::ops::ConvolutionArgs convolution_args() const;

        std::shared_ptr<kai::ops::winograd::WinogradImpl> wino_impl_;
        post_ops_fallback_t post_ops;
        std::shared_ptr<primitive_desc_t> src_reorder_pd_;
        std::shared_ptr<primitive_desc_t> dst_reorder_pd_;

    private:
        const char *impl_name() const {
            return post_ops.len() > 0 ? "wino:kai+post_ops_fallback"
                                      : "wino:kai";
        }
    };

    kai_wino_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    static status_t execute_reorder(const exec_ctx_t &ctx,
            const primitive_t &reorder, const void *src, void *dst,
            memory_tracking::key_t nested_key);

    std::shared_ptr<primitive_t> src_reorder_;
    std::shared_ptr<primitive_t> dst_reorder_;
    post_ops_fallback_t post_ops_;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
