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

#include "cpu/aarch64/eltwise_lut.hpp"

#include <cstdint>

#include "common/dnnl_thread.hpp"

#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t eltwise_lut_fwd_t::init(engine_t * /*engine*/) {
    using namespace ::dnnl::impl;

    constexpr uint32_t lut_size = 1u << 16;
    lut_.resize(lut_size);

    for (uint32_t x_u16 = 0; x_u16 < lut_size; ++x_u16) {
        const bfloat16_t x_bf16(static_cast<uint16_t>(x_u16),
                /*ignored=*/true);
        const float x = static_cast<float>(x_bf16);
        const float y = compute_eltwise_scalar_fwd(pd()->desc()->alg_kind, x,
                pd()->desc()->alpha, pd()->desc()->beta);
        lut_[x_u16] = bfloat16_t(y);
    }

    return status::success;
}

status_t eltwise_lut_fwd_t::execute(const exec_ctx_t &ctx) const {
    using namespace ::dnnl::impl;

    if (lut_.empty()) return status::runtime_error;
    const auto *lut = lut_.data();

    const memory_desc_wrapper data_d(pd()->src_md());
    if (data_d.has_zero_dim()) return status::success;

    const dim_t n = data_d.nelems(true);

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    src += data_d.offset0();
    dst += data_d.offset0();

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

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
