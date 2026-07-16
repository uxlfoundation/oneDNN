/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_AARCH64_JIT_UNI_LAYER_NORMALIZATION_HPP
#define CPU_AARCH64_JIT_UNI_LAYER_NORMALIZATION_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/cpu_layer_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct stat_and_data_kernel_iface_t {
    struct ker_args_t {
        const void *src;
        void *dst;
        const float *scale;
        const float *shift;
        const float *mean;
        const float *var;
        const void *src_scales;
        const void *dst_scales;
        const void *post_ops_binary_rhs_arg_vec;
        size_t block_size_bytes;
    };

    virtual ~stat_and_data_kernel_iface_t() = default;

    virtual void operator()(const ker_args_t &args) const = 0;

    virtual status_t create_kernel() = 0;
};

template <cpu_isa_t isa>
class jit_uni_layer_normalization_fwd_t : public primitive_t {
public:
    class pd_t : public cpu_layer_normalization_fwd_pd_t {
    public:
        using cpu_layer_normalization_fwd_pd_t::
                cpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_lnorm:", isa, ""),
                jit_uni_layer_normalization_fwd_t);

        status_t init(const engine_t *engine);

        bool use_tmp_stats() const { return reorder_pd_ || stats_are_tmp(); }

        std::shared_ptr<primitive_desc_t> reorder_pd_;
        memory_desc_t reordered_stat_md_;
        int nthr_ {};

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (use_tmp_stats()) {
                if (!skip_mean()) {
                    scratchpad.template book<float>(
                            key_lnorm_tmp_mean, across_axis());
                }
                scratchpad.template book<float>(
                        key_lnorm_tmp_var, across_axis());
            }
            if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
                scratchpad.book(key_nested, reorder_pd_->scratchpad_registry());
            }
            if (!attr()->scales_.has_default_values(DNNL_ARG_DST)) {
                scratchpad.book(key_lnorm_dst_scales,
                        static_cast<size_t>(nthr_) * sizeof(float), 64);
            }
        }
    };

    jit_uni_layer_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    void reorder_stat(const exec_ctx_t &ctx, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<stat_and_data_kernel_iface_t> stat_and_data_kernel_;
    std::shared_ptr<primitive_t> reorder_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
