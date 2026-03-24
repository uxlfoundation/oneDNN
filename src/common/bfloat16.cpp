/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "common/bfloat16.hpp"

namespace dnnl {
namespace impl {

bfloat16_t &bfloat16_t::operator=(float f) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    if (try_cvt_float_to_bfloat16(this, &f)) { return *this; }
#endif
    const uint32_t f_raw = utils::bit_cast<uint32_t>(f);

    switch (std::fpclassify(f)) {
        case FP_SUBNORMAL:
        case FP_ZERO:
            // sign-preserving zero (denormals flush to zero)
            raw_bits_ = static_cast<uint16_t>(f_raw >> 16) & 0x8000;
            break;
        case FP_INFINITE: raw_bits_ = static_cast<uint16_t>(f_raw >> 16); break;
        case FP_NAN:
            // truncate and set MSB of mantissa to force QNAN
            raw_bits_ = static_cast<uint16_t>(f_raw >> 16) | (1 << 6);
            break;
        case FP_NORMAL: {
            // round to nearest even, then truncate
            const uint16_t result_lsb = (f_raw >> 16) & 0x1u;
            const uint32_t rounding_bias = 0x00007FFFu + result_lsb;
            raw_bits_ = static_cast<uint16_t>((f_raw + rounding_bias) >> 16);
            break;
        }
    }

    return *this;
}

bfloat16_t::operator float() const {
    return utils::bit_cast<float>(static_cast<uint32_t>(raw_bits_) << 16);
}

} // namespace impl
} // namespace dnnl
