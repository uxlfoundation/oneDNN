/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed tos in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "cpu/aarch64/ref_eltwise_lut.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// bf16 specialization

template <>
status_t ref_eltwise_lut_fwd_t<::dnnl::impl::data_type::bf16>::execute(
        const exec_ctx_t &ctx) const {
    using namespace ::dnnl::impl;

    if (pd()->desc()->alg_kind != alg_kind::eltwise_gelu_erf)
        return status::unimplemented;

    // build LUT table lazily.
    maybe_build_gelu_bf16_lut_();

    const memory_desc_wrapper src_mdw(pd()->src_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());
    if (src_mdw.has_zero_dim()) return status::success;

    const dim_t n = src_mdw.nelems();

    auto *src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto *dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    static_assert(sizeof(data_t) == 2, "bf16 element must be 2 bytes");

    // for dense layout
    if (src_mdw.is_dense(true) && dst_mdw.is_dense(true)) {
        // Read bf16 payloads as 16-bit indices; assign data_t directly.
        const uint16_t *src_u16 = reinterpret_cast<const uint16_t *>(src);
        parallel_nd(n, [&](dim_t i) {
            // Index LUT by raw bf16 payload; write a data_t (bf16) value.
            dst[i] = gelu_bf16_lut_[src_u16[i]];
        });
        return status::success;
    }

    // works for arbitrary layouts
    parallel_nd(n, [&](dim_t i) {
        const dim_t s_off = src_mdw.off_l(i);
        const dim_t d_off = dst_mdw.off_l(i);
        uint16_t raw = 0;
        std::memcpy(&raw, &src[s_off], sizeof(uint16_t));

        dst[d_off] = gelu_bf16_lut_[raw];
    });

    return status::success;
}

// Explicit instantiation for bf16 data type.
template struct ref_eltwise_lut_fwd_t<::dnnl::impl::data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl