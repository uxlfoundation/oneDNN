/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_LRN_LRN_AVX512_NHWC_EXECUTOR_HPP
#define CPU_X64_LRN_LRN_AVX512_NHWC_EXECUTOR_HPP

#include "common/lrn_pd.hpp"

#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_nhwc.hpp"
#include "cpu/x64/lrn/jit_avx512_common_lrn_fwd_nhwc.hpp"
#include "cpu/x64/lrn/lrn_executor.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

class lrn_avx512_nhwc_executor_fwd_t : public i_lrn_executor_t {
public:
    lrn_avx512_nhwc_executor_fwd_t(const lrn_pd_t *pd)
        : ker_(utils::make_unique<lrn::jit_avx512_common_lrn_kernel_fwd_nhwc_t>(
                  pd->src_md()->data_type, static_cast<unsigned>(pd->C()),
                  pd->desc()->prop_kind,
                  pd->desc()->lrn_alpha / pd->desc()->local_size,
                  pd->desc()->lrn_beta, pd->desc()->lrn_k,
                  static_cast<int>(pd->desc()->local_size)))
        , dt_size_(types::data_type_size(pd->src_md()->data_type))
        , N_(pd->MB())
        , C_(pd->C())
        , H_(pd->H())
        , W_(pd->W()) {}

    status_t create_kernel() override { return ker_->create_kernel(); }

    status_t execute(const exec_ctx_t &ctx) const override {
        status_t status = status::success;
        const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
        const auto dst = CTX_OUT_CLEAN_MEM(char *, DNNL_ARG_DST, status);
        CHECK(status);
        const auto ws = CTX_OUT_CLEAN_MEM(char *, DNNL_ARG_WORKSPACE, status);
        CHECK(status);

        const auto ker = ker_.get();
        parallel_nd(
                N_, H_ * W_, [= COMPAT_THIS_CAPTURE](dim_t n, dim_t pixel_id) {
            lrn::jit_avx512_common_lrn_kernel_fwd_t::jit_args_fwd_t args;
            const auto offset = n * C_ * H_ * W_ + pixel_id * C_;
            const auto ws_offset0 = offset * 2;
            const auto ws_offset1 = ws_offset0 + C_;

            args.src = src + offset * dt_size_;
            args.dst = dst + offset * dt_size_;
            args.ws0 = ws ? ws + ws_offset0 * dt_size_ : nullptr;
            args.ws1 = ws ? ws + ws_offset1 * dt_size_ : nullptr;

            (*ker)(&args);
        });

        return status::success;
    }

    ~lrn_avx512_nhwc_executor_fwd_t() override = default;

private:
    std::unique_ptr<jit_avx512_common_lrn_kernel_fwd_nhwc_t> ker_;
    const size_t dt_size_;
    const dim_t N_;
    const dim_t C_;
    const dim_t H_;
    const dim_t W_;
};
class lrn_avx512_nhwc_executor_bwd_t : public i_lrn_executor_t {
public:
    lrn_avx512_nhwc_executor_bwd_t(const lrn_pd_t *pd)
        : ker_ {utils::make_unique<
                  lrn::jit_avx512_common_lrn_kernel_bwd_nhwc_t>(
                  pd->src_md()->data_type, static_cast<unsigned>(pd->C()),
                  pd->desc()->lrn_alpha / pd->desc()->local_size,
                  pd->desc()->lrn_beta,
                  static_cast<int>(pd->desc()->local_size))}
        , dt_size_(types::data_type_size(pd->src_md()->data_type))
        , N_(pd->MB())
        , C_(pd->C())
        , H_(pd->H())
        , W_(pd->W()) {}

    status_t create_kernel() override { return ker_->create_kernel(); }

    status_t execute(const exec_ctx_t &ctx) const override {
        status_t status = status::success;
        auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
        auto diff_src = CTX_OUT_CLEAN_MEM(char *, DNNL_ARG_DIFF_SRC, status);
        CHECK(status);
        auto diff_dst = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
        auto ws = CTX_IN_MEM(const char *, DNNL_ARG_WORKSPACE);

        const auto ker = ker_.get();
        parallel_nd(
                N_, H_ * W_, [= COMPAT_THIS_CAPTURE](dim_t n, dim_t pixel_id) {
            lrn::jit_avx512_common_lrn_kernel_bwd_nhwc_t::jit_args_bwd_t args;
            const auto offset = n * C_ * H_ * W_ + pixel_id * C_;
            const auto ws_offset0 = offset * 2;
            const auto ws_offset1 = ws_offset0 + C_;

            args.src = src + offset * dt_size_;
            args.diff_dst = diff_dst + offset * dt_size_;
            args.ws0 = ws + ws_offset0 * dt_size_;
            args.ws1 = ws + ws_offset1 * dt_size_;
            args.diff_src = diff_src + offset * dt_size_;

            (*ker)(&args);
        });

        return status::success;
    }

    ~lrn_avx512_nhwc_executor_bwd_t() override = default;

private:
    std::unique_ptr<jit_avx512_common_lrn_kernel_bwd_nhwc_t> ker_;
    const size_t dt_size_;
    const dim_t N_;
    const dim_t C_;
    const dim_t H_;
    const dim_t W_;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
