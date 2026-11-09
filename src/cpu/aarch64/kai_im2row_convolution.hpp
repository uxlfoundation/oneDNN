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

#ifndef CPU_AARCH64_KAI_IM2ROW_CONVOLUTION_HPP
#define CPU_AARCH64_KAI_IM2ROW_CONVOLUTION_HPP

#include <vector>

#include "cpu/aarch64/kai_convolution_base.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct jit_kai_nchw_im2row_t;

struct kai_im2row_convolution_fwd_t : public kai_convolution_fwd_base_t {
    struct pd_t : public kai_convolution_fwd_base_t::pd_t {
        using kai_convolution_fwd_base_t::pd_t::pd_t;

        DECLARE_COMMON_PD_T(name_.c_str(), kai_im2row_convolution_fwd_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(const engine_t *engine);
        static data_type_t src_preparation_dt(
                const cpu_convolution_fwd_pd_t &pd);

        struct src_preparation_t {
            enum class strategy_t {
                // Copy contiguous chunks of channel dimension so that
                // convolution becomes a simple strided matrix multiplication
                // useful if we have padding or non-trivial strides
                nhwc_copy,
                // We need nhwc to do conv via GEMM, for some cases, we can just
                // do a transpose
                nchw_reorder,
                // Some more complicated cases (e.g strides) require a custom
                // im2row kernel
                nchw_im2row,
            };
            strategy_t strategy = strategy_t::nhwc_copy;
            std::shared_ptr<primitive_desc_t> reorder_pd;
            struct region_t {
                size_t src_offset = 0, dst_offset = 0;
                dim_t first_h = 0, first_w = 0, src_w = 0;
                dim_t height = 0, width = 0;
            };
            std::vector<region_t> regions;
            bool zero_pad = false;
        } src_preparation_;

    private:
        status_t init_src_preparation(const engine_t *engine);
        void book_datapath_scratchpad(
                memory_tracking::registrar_t &scratchpad) const;
    };

    kai_im2row_convolution_fwd_t(const pd_t *apd)
        : kai_convolution_fwd_base_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    std::shared_ptr<primitive_t> nchw_reorder_;
    std::shared_ptr<jit_kai_nchw_im2row_t> nchw_im2row_jit_kernel_;
    void copy_nhwc_im2row(const kernel_call_args_t &args, char *im2row) const;
    status_t prepare_src(const kernel_call_args_t &args, char *im2row) const;
    status_t setup_kernel_arrays(const kernel_call_args_t &args) const;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
