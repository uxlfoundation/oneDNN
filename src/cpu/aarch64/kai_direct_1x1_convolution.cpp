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

#include "cpu/aarch64/kai_direct_1x1_convolution.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

// Flatten H and W into GEMM M when the required source pixels are
// contiguous across rows.
bool can_flatten_spatial_dims(
        const kai_direct_1x1_convolution_fwd_t::pd_t &pd) {
    return pd.KSH() == 1 && pd.KSW() == 1 && pd.OH() == pd.IH()
            && pd.OW() == pd.IW();
}

} // namespace

status_t kai_direct_1x1_convolution_fwd_t::pd_t::init(const engine_t *engine) {
    name_ = "direct_1x1:kai";
    CHECK(init_common(engine));
    VDISPATCH_CONV(is_1x1(*this), "only supports 1x1 kernels");
    VDISPATCH_CONV(has_zero_top_left_padding(*this),
            "only supports zero top and left padding");
    VDISPATCH_CONV(OH() > 0 && OW() > 0, VERBOSE_EMPTY_TENSOR, "dst");
    VDISPATCH_CONV(source_pixels_in_bounds(*this),
            "requires every 1x1 patch to lie within the source image");

    const bool flatten_spatial = can_flatten_spatial_dims(*this);
    // Dense NHWC rows are also contiguous across images. Expose the full M
    // dimension so GEMM can partition work without image boundaries.
    init_gemm_args(flatten_spatial ? MB() * OH() * OW() : OW(), IC(), 1,
            flatten_spatial ? 1 : OH(), flatten_spatial ? 1 : MB(), false);
    CHECK(init_gemm(engine));
    return status::success;
}

status_t kai_direct_1x1_convolution_fwd_t::execute(
        const exec_ctx_t &ctx) const {
    auto kernel = pd()->create_kai_gemm();
    if (!kernel) return status::runtime_error;
    kernel_call_args_t args {ctx, *pd(), *kernel};
    CHECK(prepare(args));
    CHECK(setup_kernel_arrays(args));
    return execute_kernel(args);
}

status_t kai_direct_1x1_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd
            = static_cast<const kai_direct_1x1_convolution_fwd_t::pd_t &>(
                    args.pd);

    const void *A = args.src_base;
    int lda, A_batch_stride, A_multi_stride, C_batch_stride, C_multi_stride;
    // Canonical dense NHWC strides also cover singleton dimensions whose
    // descriptor strides need not match the flattened GEMM view.
    if (can_flatten_spatial_dims(pd)) {
        lda = static_cast<int>(pd.IC());
        A_batch_stride = 0;
        A_multi_stride = 0;
        C_batch_stride = 0;
        C_multi_stride = 0;
    } else {
        lda = static_cast<int>(pd.KSW() * pd.IC());
        A_batch_stride = static_cast<int>(pd.KSH() * pd.IW() * pd.IC());
        A_multi_stride = static_cast<int>(pd.IH() * pd.IW() * pd.IC());
        C_batch_stride = static_cast<int>(pd.OW() * pd.OC());
        C_multi_stride = static_cast<int>(pd.OH() * pd.OW() * pd.OC());
    }

    args.kernel.set_arrays_generic(A, lda, A_batch_stride, A_multi_stride,
            args.B(), args.ldb(), args.B_multi_stride(), args.C(), args.ldc(),
            C_batch_stride, C_multi_stride, args.bias(),
            args.bias_multi_stride());
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
