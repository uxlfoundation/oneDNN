/*******************************************************************************
* Copyright 2025 Intel Corporation
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
#ifndef CPU_RV64_RVV_LAYER_NORMALIZATION_HPP
#define CPU_RV64_RVV_LAYER_NORMALIZATION_HPP

#include <memory>

#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"

#include "cpu/cpu_layer_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_layer_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_fwd_pd_t {
        using cpu_layer_normalization_fwd_pd_t::
                cpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("RISCV64GCV", rvv_layer_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_LNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            VDISPATCH_LNORM(
                    (src_md()->data_type == f32), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    (dst_md()->data_type == f32), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(
                    (stat_md()->data_type == f32), VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_LNORM((check_scale_shift_data_type()),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale or shift data type");

            VDISPATCH_LNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_LNORM(!(desc()->flags & dnnl_rms_norm),
                    VERBOSE_UNSUPPORTED_FEATURE, "RMSNorm not supported");

            VDISPATCH_LNORM(
                    (set_default_formats_common()), VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_LNORM((src_d.is_plain() && src_d.is_dense(true)),
                    VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_LNORM((dst_d.is_plain() && dst_d.is_dense(true)),
                    VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_LNORM((src_d == dst_d), VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_LNORM(
                    (src_d.blocking_desc().strides[src_d.ndims() - 1] == 1),
                    VERBOSE_BLOCKING_FAIL,
                    "last logical dimension's stride is not 1");

            CHECK(fill_compatible_stats_md(*src_md(), reordered_stat_md_));
            if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
                CHECK(reorder_primitive_desc_create(reorder_pd_, engine,
                        stats_are_src() ? stat_md() : &reordered_stat_md_,
                        stats_are_src() ? &reordered_stat_md_ : stat_md()));
            }

            init_scratchpad();
            return status::success;
        }

        bool use_tmp_stats() const { return reorder_pd_ || stats_are_tmp(); }

        std::shared_ptr<primitive_desc_t> reorder_pd_;
        memory_desc_t reordered_stat_md_;

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();

            bool tmp_stats = use_tmp_stats();
            if (tmp_stats) {
                scratchpad.template book<float>(
                        key_lnorm_tmp_mean, across_axis());
                scratchpad.template book<float>(
                        key_lnorm_tmp_var, across_axis());
            }

            if (reorder_pd_ && !stats_are_tmp()) {
                scratchpad.book(key_nested, reorder_pd_->scratchpad_registry());
            }
        }
    };

    status_t init(engine_t *engine) override {
        if (pd()->reorder_pd_)
            pd()->reorder_pd_->create_primitive(reorder_, engine);
        return status::success;
    }

    rvv_layer_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    void reorder_stat(const exec_ctx_t &ctx, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const {
        using namespace memory_tracking::names;

        exec_args_t r_args;
        r_args[DNNL_ARG_SRC] = in;
        r_args[DNNL_ARG_DST] = out;
        exec_ctx_t r_ctx(ctx, std::move(r_args));

        auto *nested_grantor
                = create_nested_grantor(ctx.get_scratchpad_grantor(),
                        key_nested, reorder_->pd()->scratchpad_registry());
        r_ctx.set_scratchpad_grantor(nested_grantor);

        reorder_->execute(r_ctx);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        using namespace memory_tracking::names;
        engine_t *engine = ctx.stream()->engine();
        auto scratchpad = ctx.get_scratchpad_grantor();

        std::unique_ptr<memory_t, memory_deleter_t> mean;
        auto mean_mem = scratchpad.get_memory_storage(key_lnorm_tmp_mean);
        CHECK(safe_ptr_assign(mean,
                new memory_t(engine, &(pd()->reordered_stat_md_),
                        std::move(mean_mem))));

        auto variance_mem = scratchpad.get_memory_storage(key_lnorm_tmp_var);
        std::unique_ptr<memory_t, memory_deleter_t> variance;
        CHECK(safe_ptr_assign(variance,
                new memory_t(engine, &(pd()->reordered_stat_md_),
                        std::move(variance_mem))));

        if (pd()->stats_are_src() && reorder_) {
            reorder_stat(ctx, engine, ctx.args().at(DNNL_ARG_MEAN),
                    {mean.get(), false});
            reorder_stat(ctx, engine, ctx.args().at(DNNL_ARG_VARIANCE),
                    {variance.get(), false});
        }

        status_t status = execute_forward(ctx);
        if (status != status::success) return status;

        if (!pd()->stats_are_src() && reorder_) {
            reorder_stat(ctx, engine, {mean.get(), true},
                    ctx.args().at(DNNL_ARG_MEAN));
            reorder_stat(ctx, engine, {variance.get(), true},
                    ctx.args().at(DNNL_ARG_VARIANCE));
        }

        return status::success;
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<primitive_t> reorder_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
