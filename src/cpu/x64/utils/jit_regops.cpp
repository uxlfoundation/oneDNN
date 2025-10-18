/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "common/z_magic.hpp"

#include "cpu/x64/utils/jit_regops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace regops {

void horizontal_add_ps(
        jit_generator_t *code, Xbyak::Xmm src, Xbyak::Xmm workspace) {
    UNUSED(workspace);
    if (code->is_valid_isa(avx)) {
        code->vhaddps(src, src, src);
        code->vhaddps(src, src, src);
    } else {
        code->haddps(src, src);
        code->haddps(src, src);
    }
}

void horizontal_add_ps(
        jit_generator_t *code, Xbyak::Ymm src, Xbyak::Ymm workspace) {
    const Xbyak::Xmm xmm_ws {workspace.getIdx()};
    const Xbyak::Xmm xmm_src {src.getIdx()};

    code->vextractf128(xmm_ws, src, 1);
    code->vaddps(xmm_src, xmm_src, xmm_ws);
    horizontal_add_ps(code, xmm_src);
}

void horizontal_add_ps(
        jit_generator_t *code, Xbyak::Zmm src, Xbyak::Zmm workspace) {
    const Xbyak::Ymm ymm_ws {workspace.getIdx()};
    const Xbyak::Ymm ymm_src {src.getIdx()};

    // Extract upper 256 bits and add to lower 256 bits
    code->vextractf32x8(ymm_ws, src, 1);
    code->vaddps(ymm_src, ymm_src, ymm_ws);

    const Xbyak::Xmm xmm_ws {workspace.getIdx()};
    const Xbyak::Xmm xmm_src {src.getIdx()};

    // Add upper 128 bits to lower 128 bits within the YMM
    code->vextractf32x4(xmm_ws, ymm_src, 1);
    code->vaddps(xmm_src, xmm_src, xmm_ws);

    // Horizontal add within 128 bits - swap 64-bit lanes and add
    code->vshufps(
            xmm_ws, xmm_src, xmm_src, 0x4E); // swap 64-bit lanes: [2,3,0,1]
    code->vaddps(xmm_src, xmm_src, xmm_ws);

    // Horizontal add within 64 bits - swap 32-bit elements and add
    code->vpshufd(
            xmm_ws, xmm_src, 0xB1); // swap adjacent 32-bit elements: [1,0,3,2]
    code->vaddps(xmm_src, xmm_src, xmm_ws);
}

} // namespace regops
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
