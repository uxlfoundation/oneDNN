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

#ifndef CPU_X64_LRN_JIT_AVX512_COMMON_LRN_UTILS_HPP
#define CPU_X64_LRN_JIT_AVX512_COMMON_LRN_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

#define IRB_LOOP(statement) \
    for (int irb = 0; irb < loop_size; irb++) { \
        statement; \
    }

enum class direction { forward, backward };

enum class across_version : char { First, Middle, Last, Single };

// NOLINTBEGIN(readability-identifier-naming)
struct nChw16c_across_t {
    int H, W;
    across_version version;
    constexpr nChw16c_across_t(int h, int w, across_version version)
        : H(h), W(w), version(version) {}
};
// NOLINTEND(readability-identifier-naming)

enum class tail_mode { NoTail, NextTail, CurrentTail };

template <data_type_t d_type>
cpu_isa_t get_isa() {
    return utils::map(true, avx512_core,
            d_type == data_type::bf16 && mayiuse(avx512_core_bf16),
            avx512_core_bf16,
            d_type == data_type::bf16 && !mayiuse(avx512_core_bf16),
            bf16_emulation_t::get_isa(), d_type == data_type::f16,
            avx512_core_fp16);
}

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
