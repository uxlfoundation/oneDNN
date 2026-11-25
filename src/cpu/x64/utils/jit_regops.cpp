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
        jit_generator_t *code, Xbyak::Ymm src, Xbyak::Ymm workspace) {
    const Xbyak::Xmm xmm_ws {workspace.getIdx()};
    const Xbyak::Xmm xmm_src {src.getIdx()};

    // vextractf128 is VEX-only and cannot encode ymm16-31. Use the EVEX
    // vextractf32x4 for high registers.
    const bool requires_evex = src.getIdx() >= 16 || workspace.getIdx() >= 16;
    if (requires_evex)
        code->vextractf32x4(xmm_ws, src, 1);
    else
        code->vextractf128(xmm_ws, src, 1);

    code->vaddps(xmm_src, xmm_src, xmm_ws);

    // vshufps + vaddps rather than two vhaddps: vhaddps is 3 uops with a
    // 6-cycle latency on Intel cores, the shuffle pair is 2 single-uop
    // instructions per step. It also encodes with EVEX, which vhaddps does
    // not, so the same sequence serves xmm16-31.
    code->vshufps(xmm_ws, xmm_src, xmm_src, 0xB1); // [1,0,3,2]
    code->vaddps(xmm_src, xmm_src, xmm_ws);
    code->vshufps(xmm_ws, xmm_src, xmm_src, 0x4E); // [2,3,0,1]
    code->vaddps(xmm_src, xmm_src, xmm_ws);
}

void horizontal_add_ps(
        jit_generator_t *code, Xbyak::Zmm src, Xbyak::Zmm workspace) {
    const Xbyak::Ymm ymm_ws {workspace.getIdx()};
    const Xbyak::Ymm ymm_src {src.getIdx()};

    code->vextractf64x4(ymm_ws, src, 1);
    code->vaddps(ymm_src, ymm_src, ymm_ws);
    horizontal_add_ps(code, ymm_src, ymm_ws);
}

} // namespace regops
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
