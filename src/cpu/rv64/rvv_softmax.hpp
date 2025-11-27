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

#ifndef CPU_RV64_RVV_SOFTMAX_HPP
#define CPU_RV64_RVV_SOFTMAX_HPP

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "cpu/cpu_softmax_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_softmax_conf_t {
    data_type_t data_type;
    dim_t inner_size;
    dim_t axis_size;
    dim_t outer_size;
    bool is_logsoftmax;
};

struct rvv_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;
        DECLARE_COMMON_PD_T("RISCV64GCV", rvv_softmax_fwd_t);

        rvv_softmax_conf_t rsp_;

        status_t init(engine_t *engine) {
            UNUSED(engine);

            VDISPATCH_SOFTMAX(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            rsp_.is_logsoftmax = is_logsoftmax();
            rsp_.data_type = src_md()->data_type;
            rsp_.inner_size = src_d.blocking_desc().strides[axis()];
            rsp_.axis_size = axis_size();
            rsp_.outer_size
                    = src_d.nelems(true) / (rsp_.inner_size * axis_size(true));

            VDISPATCH_SOFTMAX(rsp_.data_type == data_type::f32
                            && dst_md()->data_type == rsp_.data_type,
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    check_layouts(src_d, dst_d), VERBOSE_UNSUPPORTED_TAG);

            init_scratchpad();

            return status::success;
        }

        bool check_layouts(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &dst_d) const {
            if (!src_d.is_plain() || !dst_d.is_plain()) return false;
            if (!src_d.is_dense(true) || !dst_d.is_dense(true)) return false;
            if (!(src_d == dst_d)) return false;
            return true;
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            nthr_ = rsp_.inner_size > 1 ? dnnl_get_max_threads() : 1;
            if (rsp_.inner_size > 1) {
                scratchpad.template book<char>(
                        memory_tracking::names::key_softmax_interim_store,
                        static_cast<size_t>(axis_size(true)) * sizeof(float)
                                * static_cast<size_t>(nthr_));
            }
        }

        int nthr_ = 0;
    };

    rvv_softmax_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_SOFTMAX_HPP
