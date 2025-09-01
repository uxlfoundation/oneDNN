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
#include "cpu/aarch64/lut/lut_tables.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
// bf16 specialization

template <>
status_t ref_eltwise_lut_fwd_t<::dnnl::impl::data_type::bf16>::execute(
        const exec_ctx_t &ctx) const {
    using namespace ::dnnl::impl;

    auto ret = status::unimplemented;

    const memory_desc_wrapper src_mdw(pd()->src_md());
    const memory_desc_wrapper dst_mdw(pd()->dst_md());
    if (src_mdw.has_zero_dim()) return status::success;

    const dim_t n = src_mdw.nelems();

    auto *src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto *dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    if (pd()->desc()->alg_kind == alg_kind::eltwise_gelu_erf) {
        dnnl::impl::parallel(0, [&](int ithr, int nthr) {
            dim_t begin = 0, end = 0;
            dnnl::impl::balance211(n, nthr, ithr, begin, end);
            if (begin == end) return;
            uint16_t raw = 0;
            for (dim_t i = begin; i < end; ++i) {
                std::memcpy(&raw, &src[i], sizeof(uint16_t));
                dst[i] = lut_eltwise_gelu_erf_bf16_value(raw);
            }
        });
        ret = status::success;
    }

    return ret;
}

// Explicit instantiation for bf16 data type.
template struct ref_eltwise_lut_fwd_t<::dnnl::impl::data_type::bf16>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl