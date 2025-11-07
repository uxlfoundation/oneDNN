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

#ifndef CPU_AARCH64_KAI_WINO_REORDER_HPP
#define CPU_AARCH64_KAI_WINO_REORDER_HPP

#include <memory>

#include "common/dnnl_thread.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace kai::ops {
struct ConvolutionArgs;
struct GemmConfig;
namespace winograd {
struct WinogradImpl;
} // namespace winograd
} // namespace kai::ops

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kai_wino_reorder_t : public primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("kai_wino_reorder", kai_wino_reorder_t);

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine);

        std::shared_ptr<kai::ops::GemmConfig> cfg_;
        std::shared_ptr<kai::ops::ConvolutionArgs> conv_args_;
        std::shared_ptr<kai::ops::winograd::WinogradImpl> wino_impl_;
        memory_desc_t packed_md_ {};
        size_t plain_weights_size_ = 0;
        size_t wino_weights_size_ = 0;

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        void init_scratchpad();
        friend dnnl::impl::impl_list_item_t;
    };

    kai_wino_reorder_t(const pd_t *apd) : primitive_t(apd) {}

private:
    status_t execute(const exec_ctx_t &ctx) const override;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
