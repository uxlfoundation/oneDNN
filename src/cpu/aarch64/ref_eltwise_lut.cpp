/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#include "common/dnnl_thread.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// bf16 specialization

template <>
status_t ref_eltwise_lut_fwd_t<::dnnl::impl::data_type::bf16>::execute(
        const exec_ctx_t &ctx) const {
    using namespace ::dnnl::impl;

    if (pd()->bf16_lut_.empty()) return status::runtime_error;
    const auto *lut = pd()->bf16_lut_.data();

    const memory_desc_wrapper data_d(pd()->src_md());
    if (data_d.has_zero_dim()) return status::success;

    const dim_t n = data_d.nelems(true);

    const auto *src_u8 = CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto *dst_u8 = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DST);

    const auto offset_bytes = types::elements_to_bytes(
            pd()->src_md()->data_type, data_d.offset0());
    src_u8 += offset_bytes;
    dst_u8 += offset_bytes;

    const auto *src = reinterpret_cast<const data_t *>(src_u8);
    auto *dst = reinterpret_cast<data_t *>(dst_u8);

    static_assert(sizeof(data_t) == sizeof(bfloat16_t),
            "bf16 element must be 2 bytes");
    dnnl::impl::parallel(0, [&](int ithr, int nthr) {
        dim_t begin = 0, end = 0;
        dnnl::impl::balance211(n, nthr, ithr, begin, end);
        if (begin == end) return;
        for (dim_t i = begin; i < end; ++i) {
            dst[i] = lut[src[i].raw_bits_];
        }
    });

    return status::success;
}

// Explicit instantiation for bf16 data type.
template struct ref_eltwise_lut_fwd_t<::dnnl::impl::data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
