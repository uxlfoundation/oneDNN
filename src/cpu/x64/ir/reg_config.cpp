/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <algorithm>

#include "cpu/x64/ir/reg_config.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

reg_config_t make_reg_config(cpu_isa_t isa, int param_reg, int rsp_reg,
        const std::vector<int> &gpr_scratch,
        const std::vector<int> &vec_scratch,
        const std::vector<int> &mask_scratch) {
    reg_config_t rc;
    rc.param_reg = param_reg;
    rc.gpr_scratch = gpr_scratch;
    rc.vec_scratch = vec_scratch;

    // TODO: enable Intel APX.
    const int n_gpr = 16;
    const int n_vec = isa_num_vregs(isa);

    auto contains = [](const std::vector<int> &v, int i) {
        return std::find(v.begin(), v.end(), i) != v.end();
    };

    // GPR file includes every gpr except the stack pointer (`rsp`), the
    // argument pointer, and the scratch registers. The spill slot size is
    // 8 bytes.
    reg_file_t gpr_file;
    gpr_file.slot_size = 8;
    for (int i = 0; i < n_gpr; i++) {
        if (i != rsp_reg && i != param_reg && !contains(gpr_scratch, i))
            gpr_file.regs.push_back(i);
    }

    // Vector file includes every vector register except the scratch registers.
    // On AVX2* a mask is a vector register, so masks allocate from this same
    // file (see `kind_to_file` below). Spill slot is the vector size 32 for
    // ymm and 64 for zmm.
    reg_file_t vec_file;
    vec_file.slot_size = isa_max_vlen(isa);
    for (int i = 0; i < n_vec; i++) {
        if (!contains(vec_scratch, i)) vec_file.regs.push_back(i);
    }

    rc.pools.files = {gpr_file, vec_file};

    // Map each register kind to a file. `reg_kind_t` order is
    // { gpr, vec, mask }.
    //  * gpr: file 0
    //  * vec: file 1
    //  * mask: file 1 on AVX2*, where a mask is a vector register, and file 2
    //    on AVX-512, which has a dedicated k-register file.
    if (is_superset(isa, avx512_core)) {
        // Mask file holds `k1` to `k7` minus `mask_scratch`. `k0` is excluded
        // because it cannot encode a write mask. A spill slot is 8 bytes, the
        // width of an opmask, although the emitter never spills a mask (see
        // `set_mask_imm` in `emitter.cpp`).
        reg_file_t mask_file;
        mask_file.slot_size = 8;
        for (int i = 1; i < 8; i++) {
            if (!contains(mask_scratch, i)) mask_file.regs.push_back(i);
        }

        rc.pools.files.push_back(mask_file);
        rc.pools.kind_to_file = {0, 1, 2};
    } else {
        rc.pools.kind_to_file = {0, 1, 1};
    }

    return rc;
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
