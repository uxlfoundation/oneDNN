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

#include "cpu/aarch64/kai_indirect_convolution.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t kai_indirect_convolution_fwd_t::pd_t::init(const engine_t *engine) {
    name_ = "indirect_gemm:kai";
    CHECK(init_common(engine));
    init_gemm_args(OH() * OW(), IC(), KH() * KW(), MB(), 1, true);
    CHECK(init_gemm(engine));
    return status::success;
}

status_t kai_indirect_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    auto kernel = pd()->create_kai_gemm();
    if (!kernel) return status::runtime_error;
    kernel_call_args_t args {ctx, *pd(), *kernel};
    CHECK(prepare(args));
    CHECK(setup_kernel_arrays(args));
    return execute_kernel(args);
}

status_t kai_indirect_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd = static_cast<const kai_indirect_convolution_fwd_t::pd_t &>(
            args.pd);

    args.kernel.set_convolution_parameters(
            kai::ops::ConvolutionParameters {pd.IW(), pd.IH(), pd.IC(), pd.KW(),
                    pd.KH(), pd.OW(), pd.OH(), pd.KSW(), pd.KSH(), pd.KDW() + 1,
                    pd.KDH() + 1, pd.padT(), pd.padL(), 0.f});
    const void *A = args.src_base;
    // The convolver calculates actual jumps via set_convolution_parameters
    const int lda = static_cast<int>(pd.IC());
    const int A_batch_stride = static_cast<int>(pd.IH() * pd.IW() * pd.IC());
    const int A_multi_stride = 0;
    const int C_batch_stride = static_cast<int>(pd.OH() * pd.OW() * pd.OC());
    const int C_multi_stride = 0;

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
