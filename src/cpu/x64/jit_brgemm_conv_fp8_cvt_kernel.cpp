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

#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_brgemm_conv_fp8_cvt_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

jit_brgemm_conv_fp8_cvt_kernel_t::jit_brgemm_conv_fp8_cvt_kernel_t(
        const cfg_t &ajcp)
    : jit_generator_t(jit_name(), ajcp.isa), wjcp(ajcp) {
    assert(utils::one_of(wjcp.inp_dt, data_type::f8_e5m2, data_type::f8_e4m3));
    if (wjcp.inp_dt == data_type::f8_e5m2)
        fp8_cvt_ = utils::make_unique<fp8_conversion_e5m2_t>(
                this, xmm_aux1, xmm_aux2, xmm_aux3, kmask_aux, reg64_cvt_aux);
    else
        fp8_cvt_ = utils::make_unique<fp8_conversion_e4m3_t>(this, xmm_aux1,
                xmm_aux2, xmm_aux3, xmm_aux4, xmm_aux5, reg64_cvt_aux);
}

jit_brgemm_conv_fp8_cvt_kernel_t::~jit_brgemm_conv_fp8_cvt_kernel_t() = default;

dim_t jit_brgemm_conv_fp8_cvt_kernel_t::offset(
        const dim_t dt_sz, const int rd, const int occ) {
    const auto rdb = rd / vnni_block;
    return dt_sz * (rdb * vnni_block * wjcp.oc_block + vnni_block * occ * 16);
}

void jit_brgemm_conv_fp8_cvt_kernel_t::load2cvt(const bool is_last_ocb) {

    //    const auto oc_block = is_last_ocb ? wjcp.last_oc_block : wjcp.oc_block;
    const auto oc_chunks = wjcp.oc_block / 16;
    const auto last_oc_chunks = wjcp.last_oc_block / 16;
    const auto cur_oc_chunks = is_last_ocb ? last_oc_chunks : oc_chunks;
    printf("cur_oc_chunks: %d\n", cur_oc_chunks);
    for_(int rd = 0; rd < wjcp.rd; rd += vnni_block)
    for (int occ = 0; occ < cur_oc_chunks; occ++) {
        //        const auto stride = static_cast<dim_t>(vec_elems) * (occ + (rd / 2) * oc_chunks);
        auto ymm_result = Ymm(zmm_result.getIdx());
        vmovups(ymm_result, ptr[reg_aux_src + offset(src_dsz, rd, occ)]);
        vmovups(ptr[reg_aux_dst + offset(src_dsz, rd, occ)], ymm_result);
        fp8_cvt_->vcvt_f8_to_f16(zmm_result, ptr[reg_aux_src + offset(src_dsz, rd, occ)]);
        //        fp8_cvt_->vcvt_f8_to_f16(zmm_result, ptr[reg_aux_src + offset(src_dsz, rd, occ)]);// stride * src_dsz]);
        //        vmovdqu16(ptr[reg_aux_dst + offset(dst_dsz, rd, occ)], zmm_result);
    }
}

void jit_brgemm_conv_fp8_cvt_kernel_t::kernel_loop(const bool is_last_ocb) {
    Label label_kernel_loop, label_kernel_end;
    mov(reg_kernel_size, wjcp.kernel_size);
    mov(reg_aux_src, reg_src);
    mov(reg_aux_dst, reg_dst);

    L_aligned(label_kernel_loop);
    {
        cmp(reg_kernel_size, 0);
        je(label_kernel_end, T_NEAR);

        load2cvt(is_last_ocb);

        add(reg_aux_src, wjcp.kernel_size_offs * src_dsz);
        add(reg_aux_dst, wjcp.kernel_size_offs * dst_dsz);

        dec(reg_kernel_size);
        jmp(label_kernel_loop, T_NEAR);
    }
    L_aligned(label_kernel_end);
}

void jit_brgemm_conv_fp8_cvt_kernel_t::generate() {
    preamble();

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);
    mov(reg_last_ocb, ptr[param1 + GET_OFF(last_ocb)]);
    printf("src dsz: %d, dst dsz: %d, last_oc_block: %d\n", src_dsz, dst_dsz,
            wjcp.last_oc_block);
    Label full_ocb_label, finish_label;
    if (wjcp.last_oc_block > 0) {
        mov(reg_tmp, ptr[param1 + GET_OFF(last_ocb)]);
        cmp(reg_tmp, 0);
        je(full_ocb_label, T_NEAR);
        kernel_loop(true);
        jmp(finish_label, T_NEAR);
    }

    L(full_ocb_label);
    kernel_loop(false);

    L(finish_label);

    postamble();

    fp8_cvt_->prepare_table();
}

#undef GET_OFF
#define GET_OFF(x) offsetof(jit_brgemm_conv_src_copy_kernel_t::ctx_t, x)

void jit_brgemm_conv_src_copy_kernel_t::generate() {
    preamble();

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);
    mov(reg_rows, ptr[param1 + GET_OFF(rows)]);
    mov(reg_row_bytes, ptr[param1 + GET_OFF(row_bytes)]);
    mov(reg_stride, ptr[param1 + GET_OFF(stride)]);

    // rcx (param1 on Windows) is free once all params are loaded
    mov(rcx, reg_row_bytes);
    and_(rcx, vlen - 1);
    mov(reg_tmp, 1);
    shl(reg_tmp, cl);
    sub(reg_tmp, 1);
    kmovq(ktail_mask, reg_tmp);

    Label row_loop, unroll_loop, vec_loop, col_tail, row_next, done;
    L(row_loop);
    {
        test(reg_rows, reg_rows);
        jz(done, T_NEAR);

        mov(reg_aux_src, reg_src);
        mov(reg_aux_dst, reg_dst);
        mov(reg_cnt, reg_row_bytes);

        L(unroll_loop);
        {
            cmp(reg_cnt, unroll * vlen);
            jl(vec_loop, T_NEAR);
            for (int i = 0; i < unroll; i++)
                vmovdqu8(Zmm(i), ptr[reg_aux_src + i * vlen]);
            for (int i = 0; i < unroll; i++)
                vmovdqu8(ptr[reg_aux_dst + i * vlen], Zmm(i));
            for (int i = 0; i < unroll; i++)
                vcvthf82ph(Zmm(i), Ymm(i));
            add(reg_aux_src, unroll * vlen);
            add(reg_aux_dst, unroll * vlen);
            sub(reg_cnt, unroll * vlen);
            jmp(unroll_loop, T_NEAR);
        }

        L(vec_loop);
        {
            cmp(reg_cnt, vlen);
            jl(col_tail, T_NEAR);
            vmovdqu8(Zmm(0), ptr[reg_aux_src]);
            vmovdqu8(ptr[reg_aux_dst], Zmm(0));
            add(reg_aux_src, vlen);
            add(reg_aux_dst, vlen);
            sub(reg_cnt, vlen);
            jmp(vec_loop, T_NEAR);
        }

        L(col_tail);
        test(reg_cnt, reg_cnt);
        jz(row_next, T_NEAR);
        vmovdqu8(Zmm(0) | ktail_mask | T_z, ptr[reg_aux_src]);
        vmovdqu8(ptr[reg_aux_dst] | ktail_mask, Zmm(0));

        L(row_next);
        add(reg_src, reg_stride);
        add(reg_dst, reg_stride);
        dec(reg_rows);
        jmp(row_loop, T_NEAR);
    }
    L(done);

    postamble();
}

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
