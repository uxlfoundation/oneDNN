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

#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/kai_utils.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

bool uses_flattened_src(const kai_direct_1x1_convolution_fwd_t::pd_t &pd) {
    return pd.KSH() == 1 && pd.KSW() == 1 && pd.OH() == pd.IH()
            && pd.OW() == pd.IW();
}

} // namespace

status_t kai_direct_1x1_convolution_fwd_t::pd_t::init_datapath(
        engine_t *engine) {
    VDISPATCH_CONV(
            direct_1x1_src_layout_ok(), VERBOSE_UNSUPPORTED_TAG_S, "src");
    VDISPATCH_CONV(direct_1x1_kernel_ok(), "only supports 1x1 kernels");
    VDISPATCH_CONV(
            direct_1x1_padding_ok(), "only supports zero top and left padding");
    VDISPATCH_CONV(OH() > 0 && OW() > 0, VERBOSE_EMPTY_TENSOR, "dst");
    VDISPATCH_CONV(direct_1x1_output_samples_in_bounds(),
            "only supports output samples inside src bounds");
    const bool pad_ow = !uses_flattened_src(*this)
            && kai_utils::is_bf16(*src_md(), *weights_md(), *attr());
    if (pad_ow) {
        auto padded_ow = static_cast<double>(
                utils::rnd_up(OW(), simd_elems(data_type::bf16)));
        VDISPATCH_CONV(padded_ow < 1.05 * OW(),
                "too much wasted work due to padded OW, falling to indirect");
    }
    return status::success;
}

unsigned int kai_direct_1x1_convolution_fwd_t::pd_t::gemm_m() const {
    if (uses_flattened_src(*this))
        return kai_convolution_fwd_base_t::pd_t::gemm_m();
    return static_cast<unsigned int>(OW());
}

unsigned int kai_direct_1x1_convolution_fwd_t::pd_t::gemm_n_batches() const {
    if (uses_flattened_src(*this))
        return kai_convolution_fwd_base_t::pd_t::gemm_n_batches();
    return static_cast<unsigned int>(OH());
}

unsigned int kai_direct_1x1_convolution_fwd_t::pd_t::gemm_n_multi() const {
    if (uses_flattened_src(*this))
        return kai_convolution_fwd_base_t::pd_t::gemm_n_multi();
    return static_cast<unsigned int>(MB());
}

status_t kai_direct_1x1_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd
            = static_cast<const kai_direct_1x1_convolution_fwd_t::pd_t &>(
                    args.pd);

    if (uses_flattened_src(pd)) {
        args.kernel.set_arrays_generic(args.src_base, args.ld_src, 0, 0,
                args.wei_base, args.ld_wei, 0, args.kernel_dst_base,
                args.ld_dst, args.dst_batch_stride, 0, args.bias_base, 0);
        return status::success;
    }

    args.kernel.set_arrays_generic(args.src_base,
            static_cast<int>(pd.KSW()) * args.ld_src,
            static_cast<int>(pd.KSH()) * args.src_h_stride,
            args.src_batch_stride, args.wei_base, args.ld_wei, 0,
            args.kernel_dst_base, args.ld_dst, args.dst_h_stride,
            args.dst_batch_stride, args.bias_base, 0);
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
