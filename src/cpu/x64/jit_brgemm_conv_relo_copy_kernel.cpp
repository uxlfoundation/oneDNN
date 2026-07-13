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

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_brgemm_conv_relo_copy_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

void jit_brgemm_relo_copy_to_wbuffer_t::generate() {

    const bool is_xf16 = one_of(wjcp.wei_dt, data_type::bf16, data_type::f16);
    const bool is_f32 = wjcp.wei_dt == data_type::f32;

    // required for use of VPERMB instruction
    assert(IMPLICATION(
            !(is_xf16 || is_f32), cpu().has(Xbyak::util::Cpu::tAVX512_VBMI)));
    assert(wjcp.inp_oc_block == 16);

    preamble();

    const int vnni_width = data_type_vnni_granularity(wjcp.wei_dt);
    const auto wei_dsz = types::data_type_size(wjcp.wei_dt);
    const auto inp_ocb_size = wjcp.inp_oc_block * vnni_width * wei_dsz;
    const auto out_ocb_size = wjcp.out_oc_block * vnni_width * wei_dsz;
    const auto oc_chunks = wjcp.out_oc_block / wjcp.inp_oc_block;
    const auto has_ocb_tail = (oc_chunks != wjcp.last_occ_to_copy);
    auto nb_rd = div_up(wjcp.rd, vnni_width);
    if (wjcp.is_rd_padded_to_block) nb_rd = rnd_up(nb_rd, 16);
    const auto rtail = (wjcp.rd % vnni_width) * wjcp.inp_oc_block;
    const auto has_rdb_tail = (nb_rd - 1) * vnni_width >= wjcp.rd;

    auto copy_zmm = [&](bool rd_tail) {
        auto zmm_src_tmp = (rd_tail) ? zmm_src | kmask_load | T_z : zmm_src;
        if (is_xf16) {
            vmovdqu16(zmm_src_tmp, ptr[aux_reg_src]);
            vpermw(zmm_dst, zmm_idx, zmm_src);
            vmovdqu16(ptr[aux_reg_dst], zmm_dst);
        } else if (is_f32) {
            vmovdqu32(zmm_src_tmp, ptr[aux_reg_src]);
            vmovdqu32(ptr[aux_reg_dst], zmm_src_tmp);
        } else {
            vmovdqu8(zmm_src_tmp, ptr[aux_reg_src]);
            vpermb(zmm_dst, zmm_idx, zmm_src);
            vmovdqu8(ptr[aux_reg_dst], zmm_dst);
        }
    };

    auto rdb_loop = [&](bool last_ocb) {
        for (int rdb = 0; rdb < nb_rd; rdb++) {
            mov(aux_reg_src, reg_src);
            mov(aux_reg_dst, reg_dst);

            for (int occ = 0; occ < oc_chunks; occ++) {
                if ((rdb * vnni_width >= wjcp.rd)
                        || (last_ocb && occ >= wjcp.last_occ_to_copy)) {
                    if (is_xf16)
                        vmovdqu16(ptr[aux_reg_dst], zmm_zero);
                    else
                        vmovdqu8(ptr[aux_reg_dst], zmm_zero);
                } else if ((rdb + 1) * vnni_width > wjcp.rd)
                    copy_zmm(true);
                else
                    copy_zmm(false);

                add(aux_reg_src, wjcp.inp_ocb_offs);
                add(aux_reg_dst, inp_ocb_size);
            }
            add(reg_src, inp_ocb_size);
            add(reg_dst, out_ocb_size);
        }
    };

    if (rtail > 0) {
        uint64_t mask = (UINT64_C(1) << rtail) - 1;
        mov(reg_tmp, mask);
        kmovq(kmask_load, reg_tmp);
    }

    if (has_rdb_tail || has_ocb_tail) vpxord(zmm_zero, zmm_zero, zmm_zero);

    // load permute indices from data section
    Label full_ocb_label, finish_label, permute_index_table;
    if (!is_f32) {
        if (is_xf16)
            vmovdqu16(zmm_idx, ptr[rip + permute_index_table]);
        else
            vmovdqu8(zmm_idx, ptr[rip + permute_index_table]);
    }

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);
    if (has_ocb_tail) {
        mov(reg_tmp, ptr[param1 + GET_OFF(last_ocb)]);
        cmp(reg_tmp, 0);
        je(full_ocb_label, T_NEAR);
        rdb_loop(true);
        jmp(finish_label, T_NEAR);
    }

    L(full_ocb_label);
    rdb_loop(false);

    L(finish_label);
    postamble();

    align(64);
    L(permute_index_table);
    if (!is_f32) {
        const uint8_t no = 16; // 16o
        for (uint8_t o = 0; o < no; ++o) {
            for (uint8_t r = 0; r < static_cast<uint8_t>(vnni_width); r++) {
                const uint8_t index = o + r * no;
                if (is_xf16)
                    dw(index);
                else
                    db(index);
            }
        }
    }
}

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
