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

#include <algorithm>
#include <cstring>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

void kai_indirect_convolution_fwd_t::pd_t::book_datapath_scratchpad(
        memory_tracking::registrar_t &scratchpad, size_t src_dt_size) const {

    const size_t kernel_points = static_cast<size_t>(KH() * KW());
    const size_t output_points = static_cast<size_t>(OH() * OW());
    const size_t outer_count = static_cast<size_t>(MB()) * kernel_points;
    const size_t inner_count = outer_count * output_points;
    const size_t outer_bytes = outer_count * sizeof(const void *const *);
    const size_t inner_bytes = inner_count * sizeof(const void *);
    const size_t pad_offset = utils::rnd_up(
            outer_bytes + inner_bytes, std::max<size_t>(1, src_dt_size));
    const size_t total_bytes
            = pad_offset + static_cast<size_t>(IC()) * src_dt_size;

    scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer, total_bytes, 1,
            64, 64);
}

unsigned int kai_indirect_convolution_fwd_t::pd_t::gemm_m() const {
    return static_cast<unsigned int>(OH() * OW());
}

unsigned int kai_indirect_convolution_fwd_t::pd_t::gemm_k() const {
    return static_cast<unsigned int>(IC());
}

unsigned int kai_indirect_convolution_fwd_t::pd_t::gemm_k_sections() const {
    return static_cast<unsigned int>(KH() * KW());
}

unsigned int kai_indirect_convolution_fwd_t::pd_t::gemm_n_batches() const {
    return static_cast<unsigned int>(MB());
}

status_t kai_indirect_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd = static_cast<const kai_indirect_convolution_fwd_t::pd_t &>(
            args.pd);

    args.kernel.set_convolution_parameters(
            kai::ops::ConvolutionParameters {pd.IW(), pd.IH(), pd.IC(), pd.KW(),
                    pd.KH(), pd.OW(), pd.OH(), pd.KSW(), pd.KSH(), pd.KDW() + 1,
                    pd.KDH() + 1, pd.padT(), pd.padL(), 0.f});
    args.kernel.set_arrays_generic(args.src_base, args.ld_src,
            args.src_batch_stride, 0, args.wei_base, args.ld_wei, 0,
            args.kernel_dst_base, args.ld_dst, args.dst_batch_stride, 0,
            args.bias_base, 0);
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
