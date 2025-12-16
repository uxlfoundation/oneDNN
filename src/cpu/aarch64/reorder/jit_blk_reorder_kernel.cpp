/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
* Copyright 2022-2025 Arm Ltd. and affiliates
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

#include <cassert>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/reorder/jit_uni_reorder_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace tr {
using namespace Xbyak_aarch64;
using namespace dnnl::impl::types;

// Seperate class for no unroll/threading burden
bool jit_single_blk_kernel_t::applicable(const prb_t &p) {

    using namespace data_type;

    bool ok = p.ndims >= 2 && utils::one_of(get_max_cpu_isa(), sve_128, sve_256)
            && p.src_scale_type == scale_type_t::NONE
            && p.dst_scale_type == scale_type_t::NONE
            && utils::one_of(p.itype, f32) && utils::one_of(p.otype, f32)
            && utils::everyone_is(0, p.ioff, p.ooff) && p.beta == 0.f
            && prb_has_small_strides(p);
    if (!ok) return false;

    int64_t n0 = p.nodes[0].n;
    auto i0 = p.nodes[0].is;
    auto o0 = p.nodes[0].os;
    int64_t n1 = p.nodes[1].n;
    auto i1 = p.nodes[1].is;
    auto o1 = p.nodes[1].os;

    /*
         * for a transpose of plain to 8c case, nodes would be like:
         *     n    is   os
         *     m    1    8
         *     8    m    1
         * or
         *     8    m    1
         *     m    1    8
         */
    ok = (utils::one_of(n0, 4, 8, 16, 32, 64)
                 || utils::one_of(n1, 4, 8, 16, 32, 64))
            && ((i0 == 1 && o1 == 1 && n0 == i1 && o0 == n1)
                    || (o0 == 1 && i1 == 1 && n0 == o1 && i0 == n1));
    if (!ok) return false;

    // The 128-bit version only supports blocking of exactly 4, while the
    // 256-bit version only suppports the larger block sizes.
    if (get_max_cpu_isa() == sve_128) {
        if (n0 != 4 && n1 != 4) { return false; }
    } else if (get_max_cpu_isa() == sve_256) {
        if (n0 == 4 || n1 == 4) return false;
    }

    // Do not handle transpose of dimensions other than last 2
    for (int i = 2; i < p.ndims; ++i) {
        if (p.nodes[i].is != p.nodes[i].os) {
            ok = false;
            break;
        }
    }

    return ok;
}

jit_single_blk_kernel_t::jit_single_blk_kernel_t(const prb_t &prb)
    : prb_(prb)
    , itype_sz_(data_type_size(prb_.itype))
    , otype_sz_(data_type_size(prb_.otype))
    , block_sz(prb.nodes[0].n) {}

void jit_single_blk_kernel_t::preamble() {
    if (get_sve_length() == 32) {
        ptrue(p_lsb_256.b, VL32);
    } else {
        ptrue(p_lsb_256.b, VL16);
    }
}

void jit_single_blk_kernel_t::postamble() {
    ret();
}

void jit_single_blk_kernel_t::generate() {
    auto input_stride
            = prb_.nodes[0].is != 1 ? prb_.nodes[0].is : prb_.nodes[1].is;
    auto output_stride
            = prb_.nodes[0].os != 1 ? prb_.nodes[0].os : prb_.nodes[1].os;

    Label tail_processing;

    set_preg(p_tmp2.s, 4, X_TMP_0, X_TMP_1);
    rev(p_tmp1.s, p_tmp2.s);

    preamble();

    cmp(reg_ptr_tail, true);
    b(EQ, tail_processing);

    if (block_sz == 4) {
        gen_ker4x4(0, 0, input_stride, output_stride, 4, 4);
    } else if (block_sz == 8) {
        gen_ker8x8(0, 0, input_stride, output_stride, 8, 8);
    } else if (block_sz == 16) {
        gen_ker16x16_in_8x8(0, 0, input_stride, output_stride);
    } else if (block_sz == 32) {
        gen_ker32x32_in_16x16(0, 0, input_stride, output_stride);
    } else if (block_sz == 64) {
        gen_ker64x64_in_32x32(0, 0, input_stride, output_stride);
    } else {
        assert(!"unimplemented");
    }

    postamble();

    L(tail_processing);

    if (block_sz == 4) {
        auto i_tail = input_stride % 4 != 0 ? input_stride % 4 : 4;
        auto o_tail = output_stride % 4 != 0 ? output_stride % 4 : 4;
        auto t_mask = i_tail == 4 ? o_tail : i_tail;
        gen_setmask(t_mask);
        gen_ker4x4(0, 0, input_stride, output_stride, i_tail, o_tail);
    } else if (block_sz == 8) {
        auto i_tail = input_stride % 8 != 0 ? input_stride % 8 : 8;
        auto o_tail = output_stride % 8 != 0 ? output_stride % 8 : 8;
        if (i_tail != o_tail) {
            auto t_mask = i_tail == 8 ? o_tail : i_tail;
            gen_setmask(t_mask);
            gen_ker8x8(0, 0, input_stride, output_stride, i_tail, o_tail);
        }
    } else if (block_sz == 16) {
        auto i_tail = input_stride % 16 != 0 ? input_stride % 16 : 16;
        auto o_tail = output_stride % 16 != 0 ? output_stride % 16 : 16;
        if (i_tail != o_tail) {
            auto t_mask = i_tail == 16 ? o_tail : i_tail;
            t_mask %= 8;
            if (t_mask != 0) gen_setmask(t_mask);
            gen_ker16x16_in_8x8(
                    0, 0, input_stride, output_stride, i_tail, o_tail);
        }
    } else if (block_sz == 32) {
        auto i_tail = input_stride % 32 != 0 ? input_stride % 32 : 32;
        auto o_tail = output_stride % 32 != 0 ? output_stride % 32 : 32;
        if (i_tail != o_tail) {
            auto t_mask = i_tail == 32 ? o_tail : i_tail;
            t_mask %= 8;
            if (t_mask != 0) gen_setmask(t_mask);
            gen_ker32x32_in_16x16(
                    0, 0, input_stride, output_stride, i_tail, o_tail);
        }
    } else if (block_sz == 64) {
        auto i_tail = input_stride % 64 != 0 ? input_stride % 64 : 64;
        auto o_tail = output_stride % 64 != 0 ? output_stride % 64 : 64;
        if (i_tail != o_tail) {
            auto t_mask = i_tail == 64 ? o_tail : i_tail;
            t_mask %= 8;
            if (t_mask != 0) gen_setmask(t_mask);
            gen_ker64x64_in_32x32(
                    0, 0, input_stride, output_stride, i_tail, o_tail);
        }
    } else {
        assert(!"unimplemented");
    }

    postamble();
}

void jit_single_blk_kernel_t::gen_loadu(
        const ZRegS ymm, const XReg &addr, int size) {
    QReg xmm(ymm.getIdx());
    switch (size) {
        case 32: ld1w(ymm, p_lsb_256 / T_z, ptr(addr)); break;
        case 16: ldr(xmm, ptr(addr)); break;
        default: assert(!"unreachable");
    }
}

void jit_single_blk_kernel_t::gen_storeu(
        const XReg &addr, const ZRegS ymm, int size) {
    QReg xmm(ymm.getIdx());
    switch (size) {
        case 32: st1w(ymm, p_lsb_256, ptr(addr)); break;
        case 16: str(xmm, ptr(addr)); break;
        default: assert(!"unreachable");
    }
}

void jit_single_blk_kernel_t::gen_maskloadu(
        const ZRegS ymm, const XReg &addr, const PReg mask, int size) {
    switch (size) {
        case 32:
        case 16: ld1w(ymm, mask / T_z, ptr(addr)); break;
        default: assert(!"unreachable");
    }
}

void jit_single_blk_kernel_t::gen_maskstoreu(
        const XReg &addr, const ZRegS ymm, const PReg mask, int size) {
    switch (size) {
        case 32:
        case 16: st1w(ymm, mask, ptr(addr)); break;
        default: assert(!"unreachable");
    }
}

// Register allocation xmm0~11
void jit_single_blk_kernel_t::gen_transpose_8x8() {
    const uint64_t sveLen = get_sve_length();
    constexpr int lane = 8;

#if 0
        /* Debug code
	   z0:   7,  6,  5,  4,  3,  2,  1,  0
	   z1:  15, 14, 13, 12, 11, 10,  9,  8
	   ...
	   z17: 63, 62, 61, 60, 59, 58, 57, 56
	*/
	ptrue(P_ALL_ONE.b);
	ptrue(P_TMP.s, VL8);
	not_(P_TMP.b, P_ALL_ONE/T_z, P_TMP.b);
    index(z0.s, 0, 1);
    mov(z0.s, P_TMP/T_m, 0);
    mov(z_tmp_vec[0].s, 8);
    mov(z_tmp_vec[0].s, P_TMP/T_m, 0);
    for(uint32_t i=1; i<lane; i++)
        add(ZRegS{i}, ZRegS{i-1}, z_tmp_vec[0].s);
#endif

    ptrue(P_TMP.s, VL4);

    /* 1st turn */
    for (uint32_t i = 0; i < lane / 2; i++) {
        trn1(z_tmp_vec[i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
        trn2(z_tmp_vec[lane / 2 + i].s, ZRegS {2 * i}, ZRegS {2 * i + 1});
    }

    /* 2nd turn */
    trn1(z4.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
    trn1(z5.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
    trn2(z6.d, z_tmp_vec[0].d, z_tmp_vec[1].d);
    trn2(z7.d, z_tmp_vec[4].d, z_tmp_vec[5].d);
    trn1(z_tmp_vec[0].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
    trn1(z_tmp_vec[1].d, z_tmp_vec[6].d, z_tmp_vec[7].d);
    trn2(z_tmp_vec[2].d, z_tmp_vec[2].d, z_tmp_vec[3].d);
    trn2(z_tmp_vec[3].d, z_tmp_vec[6].d, z_tmp_vec[7].d);

    /* 3rd turn */
    for (uint32_t i = 0; i < lane / 2; i++) {
        mov(ZRegD {i}, ZRegD {lane / 2 + i});
        mov(z_tmp_vec[lane / 2 + i].d, z_tmp_vec[i].d);
    }

    /* 4th turn */
    for (uint32_t i = 0; i < lane / 2; i++) {
        ZRegB z {lane / 2 + i};
        ZRegB z_tmp = z_tmp_vec[lane / 2 + i].b;
        /* Move bit 0-127 to 128-255. */
        ext(z, z, 16);
        /* Move bit 128-255 to 0-127. */
        ext(z_tmp, z_tmp, sveLen - 16);
    }

    /* 5th turn */
    for (uint32_t i = 0; i < lane / 2; i++) {
        ZRegS z0 {i};
        ZRegS z1 {lane / 2 + i};
        sel(z0, P_TMP, z0, z_tmp_vec[lane / 2 + i].s);
        sel(z1, P_TMP, z1, z_tmp_vec[i].s);
    }
}

// keep order nchw -> nChw()C
// or nChw()C -> nchw
void jit_single_blk_kernel_t::gen_setmask(int mask) {
    set_preg(p_mask.s, mask, x_tmp_0, x_tmp_1);
}

void jit_single_blk_kernel_t::gen_transpose_4x4() {
    auto &z_tmp4 = z_tmp_vec[0];
    auto &z_tmp5 = z_tmp_vec[1];
    auto &z_tmp6 = z_tmp_vec[2];
    auto &z_tmp7 = z_tmp_vec[3];

    /* 1st turn */
    trn1(z_tmp4.s, z0.s, z1.s);
    trn1(z_tmp5.s, z2.s, z3.s);
    trn2(z_tmp6.s, z0.s, z1.s);
    trn2(z_tmp7.s, z2.s, z3.s);

    trn1(z0.d, z_tmp4.d, z_tmp5.d);
    trn1(z1.d, z_tmp6.d, z_tmp7.d);
    trn2(z2.d, z_tmp4.d, z_tmp5.d);
    trn2(z3.d, z_tmp6.d, z_tmp7.d);
}

void jit_single_blk_kernel_t::gen_tr4x4(int i_off, int o_off, int input_stride,
        int output_stride, int in_tail, int out_tail) {

    constexpr int lane = 4;

    if (in_tail == 0 || out_tail == 0) return;

    for (int i = 0; i < out_tail; ++i) {
        if (in_tail != lane) {
            add_imm(x_addr, reg_ptr_in_, i_off + i * input_stride * itype_sz_,
                    x_tmp_0);
            gen_maskloadu(ZRegS(i), x_addr, p_mask, lane * itype_sz_);
        } else {
            add_imm(x_addr, reg_ptr_in_, i_off + i * input_stride * itype_sz_,
                    x_tmp_0);
            gen_loadu(ZRegS(i), x_addr, lane * itype_sz_);
        }
    }

    gen_transpose_4x4();

    for (int i = 0; i < in_tail; ++i) {
        if (out_tail == lane) {
            add_imm(x_addr, reg_ptr_out_, o_off + i * output_stride * otype_sz_,
                    x_tmp_0);
            gen_storeu(x_addr, ZRegS(i), lane * otype_sz_);
        } else {
            add_imm(x_addr, reg_ptr_out_, o_off + i * output_stride * otype_sz_,
                    x_tmp_0);
            gen_maskstoreu(x_addr, ZRegS(i), p_mask, lane * otype_sz_);
        }
    }
}

void jit_single_blk_kernel_t::gen_ker4x4(int i_off, int o_off, int input_stride,
        int output_stride, int in_tail, int out_tail) {
    gen_tr4x4(i_off, o_off, input_stride, output_stride, in_tail, out_tail);
}

void jit_single_blk_kernel_t::gen_tr8x8(int i_off, int o_off, int input_stride,
        int output_stride, int in_tail, int out_tail) {

    constexpr int lane = 8;

    if (in_tail == 0 || out_tail == 0) return;

    for (int i = 0; i < out_tail; ++i) {
        if (in_tail != lane) {
            add_imm(x_addr, reg_ptr_in_, i_off + i * input_stride * itype_sz_,
                    x_tmp_0);
            gen_maskloadu(ZRegS(i), x_addr, p_mask, lane * itype_sz_);
        } else {
            add_imm(x_addr, reg_ptr_in_, i_off + i * input_stride * itype_sz_,
                    x_tmp_0);
            gen_loadu(ZRegS(i), x_addr, lane * itype_sz_);
        }
    }

    gen_transpose_8x8();

    for (int i = 0; i < in_tail; ++i) {
        if (out_tail == lane) {
            add_imm(x_addr, reg_ptr_out_, o_off + i * output_stride * otype_sz_,
                    x_tmp_0);
            gen_storeu(x_addr, ZRegS(i), lane * otype_sz_);
        } else {
            add_imm(x_addr, reg_ptr_out_, o_off + i * output_stride * otype_sz_,
                    x_tmp_0);
            gen_maskstoreu(x_addr, ZRegS(i), p_mask, lane * otype_sz_);
        }
    }
}

// tail: 0 ~ 8
// support: either in_tail or out_tail is not 8, but not both
void jit_single_blk_kernel_t::gen_ker8x8(int i_off, int o_off, int input_stride,
        int output_stride, int in_tail, int out_tail) {
    gen_tr8x8(i_off, o_off, input_stride, output_stride, in_tail, out_tail);
}

void jit_single_blk_kernel_t::gen_ker16x16_in_8x8(
        int i_off, int o_off, int input_stride, int output_stride) {
    const auto lane = 16;
    const auto sub_lane = lane / 2;

    i_off *= itype_sz_;
    o_off *= otype_sz_;

    gen_tr8x8(i_off, o_off, input_stride, output_stride, sub_lane, sub_lane);
    gen_tr8x8(i_off + input_stride * sub_lane * itype_sz_,
            o_off + sub_lane * otype_sz_, input_stride, output_stride, sub_lane,
            sub_lane);
    gen_tr8x8(i_off + sub_lane * itype_sz_,
            o_off + output_stride * sub_lane * otype_sz_, input_stride,
            output_stride, sub_lane, sub_lane);
    gen_tr8x8(i_off + (input_stride * sub_lane + sub_lane) * itype_sz_,
            o_off + (output_stride * sub_lane + sub_lane) * otype_sz_,
            input_stride, output_stride, sub_lane, sub_lane);
}

// tail can be 1 ~ 16, using sve2 for now
void jit_single_blk_kernel_t::gen_ker16x16_in_8x8(int i_off, int o_off,
        int input_stride, int output_stride, int in_tail, int out_tail) {
    constexpr auto lane = 16;
    constexpr auto sub_lane = lane / 2;
    auto tail = in_tail != lane ? in_tail : out_tail;

    const auto l_tail = tail < sub_lane ? tail : sub_lane;
    const auto u_tail = tail < sub_lane ? 0 : tail - sub_lane;

    i_off *= itype_sz_;
    o_off *= otype_sz_;

    if (tail == in_tail) {
        gen_tr8x8(i_off, o_off, input_stride, output_stride, l_tail, sub_lane);
        gen_tr8x8(i_off + input_stride * sub_lane * itype_sz_,
                o_off + sub_lane * otype_sz_, input_stride, output_stride,
                l_tail, sub_lane);
        gen_tr8x8(i_off + sub_lane * itype_sz_,
                o_off + output_stride * sub_lane * otype_sz_, input_stride,
                output_stride, u_tail, sub_lane);
        gen_tr8x8(i_off + itype_sz_ * (input_stride * sub_lane + sub_lane),
                o_off + otype_sz_ * (output_stride * sub_lane + sub_lane),
                input_stride, output_stride, u_tail, sub_lane);
    } else {
        gen_tr8x8(i_off, o_off, input_stride, output_stride, sub_lane, l_tail);
        gen_tr8x8(i_off + input_stride * sub_lane * itype_sz_,
                o_off + sub_lane * otype_sz_, input_stride, output_stride,
                sub_lane, u_tail);
        gen_tr8x8(i_off + sub_lane * itype_sz_,
                o_off + output_stride * sub_lane * itype_sz_, input_stride,
                output_stride, sub_lane, l_tail);
        gen_tr8x8(i_off + itype_sz_ * (input_stride * sub_lane + sub_lane),
                o_off + otype_sz_ * (output_stride * sub_lane + sub_lane),
                input_stride, output_stride, sub_lane, u_tail);
    }
}

void jit_single_blk_kernel_t::gen_ker32x32_in_16x16(
        int i_off, int o_off, int input_stride, int output_stride) {

    const auto lane = 32;
    const auto sub_lane = lane / 2;
    gen_ker16x16_in_8x8(i_off, o_off, input_stride, output_stride);
    gen_ker16x16_in_8x8(i_off + sub_lane * input_stride, o_off + sub_lane,
            input_stride, output_stride);
    gen_ker16x16_in_8x8(i_off + sub_lane, o_off + output_stride * sub_lane,
            input_stride, output_stride);
    gen_ker16x16_in_8x8(i_off + input_stride * sub_lane + sub_lane,
            o_off + output_stride * sub_lane + sub_lane, input_stride,
            output_stride);
}

void jit_single_blk_kernel_t::gen_ker32x32_in_16x16(int i_off, int o_off,
        int input_stride, int output_stride, int in_tail, int out_tail) {

    constexpr auto lane = 32;
    constexpr auto sub_lane = lane / 2;
    auto tail = in_tail != lane ? in_tail : out_tail;

    const auto l_tail = tail < sub_lane ? tail : sub_lane;
    const auto u_tail = tail < sub_lane ? 0 : tail - sub_lane;

    if (tail == in_tail) {
        gen_ker16x16_in_8x8(
                i_off, o_off, input_stride, output_stride, l_tail, sub_lane);
        gen_ker16x16_in_8x8(i_off + sub_lane * input_stride, o_off + sub_lane,
                input_stride, output_stride, l_tail, sub_lane);
        gen_ker16x16_in_8x8(i_off + sub_lane, o_off + output_stride * sub_lane,
                input_stride, output_stride, u_tail, sub_lane);
        gen_ker16x16_in_8x8(i_off + input_stride * sub_lane + sub_lane,
                o_off + output_stride * sub_lane + sub_lane, input_stride,
                output_stride, u_tail, sub_lane);
    } else {
        gen_ker16x16_in_8x8(
                i_off, o_off, input_stride, output_stride, sub_lane, l_tail);
        gen_ker16x16_in_8x8(i_off + sub_lane * input_stride, o_off + sub_lane,
                input_stride, output_stride, sub_lane, u_tail);
        gen_ker16x16_in_8x8(i_off + sub_lane, o_off + output_stride * sub_lane,
                input_stride, output_stride, sub_lane, l_tail);
        gen_ker16x16_in_8x8(i_off + input_stride * sub_lane + sub_lane,
                o_off + output_stride * sub_lane + sub_lane, input_stride,
                output_stride, sub_lane, u_tail);
    }
}

void jit_single_blk_kernel_t::gen_ker64x64_in_32x32(
        int i_off, int o_off, int input_stride, int output_stride) {

    const auto lane = 64;
    const auto sub_lane = lane / 2;
    gen_ker32x32_in_16x16(i_off, o_off, input_stride, output_stride);
    gen_ker32x32_in_16x16(i_off + sub_lane * input_stride, o_off + sub_lane,
            input_stride, output_stride);
    gen_ker32x32_in_16x16(i_off + sub_lane, o_off + output_stride * sub_lane,
            input_stride, output_stride);
    gen_ker32x32_in_16x16(i_off + input_stride * sub_lane + sub_lane,
            o_off + output_stride * sub_lane + sub_lane, input_stride,
            output_stride);
}

void jit_single_blk_kernel_t::gen_ker64x64_in_32x32(int i_off, int o_off,
        int input_stride, int output_stride, int in_tail, int out_tail) {
    constexpr auto lane = 64;
    constexpr auto sub_lane = lane / 2;
    auto tail = in_tail != lane ? in_tail : out_tail;

    const auto l_tail = tail < sub_lane ? tail : sub_lane;
    const auto u_tail = tail < sub_lane ? 0 : tail - sub_lane;

    if (tail == in_tail) {
        gen_ker32x32_in_16x16(
                i_off, o_off, input_stride, output_stride, l_tail, sub_lane);
        gen_ker32x32_in_16x16(i_off + sub_lane * input_stride, o_off + sub_lane,
                input_stride, output_stride, l_tail, sub_lane);
        gen_ker32x32_in_16x16(i_off + sub_lane,
                o_off + output_stride * sub_lane, input_stride, output_stride,
                u_tail, sub_lane);
        gen_ker32x32_in_16x16(i_off + input_stride * sub_lane + sub_lane,
                o_off + output_stride * sub_lane + sub_lane, input_stride,
                output_stride, u_tail, sub_lane);
    } else {
        gen_ker32x32_in_16x16(
                i_off, o_off, input_stride, output_stride, sub_lane, l_tail);
        gen_ker32x32_in_16x16(i_off + sub_lane * input_stride, o_off + sub_lane,
                input_stride, output_stride, sub_lane, u_tail);
        gen_ker32x32_in_16x16(i_off + sub_lane,
                o_off + output_stride * sub_lane, input_stride, output_stride,
                sub_lane, l_tail);
        gen_ker32x32_in_16x16(i_off + input_stride * sub_lane + sub_lane,
                o_off + output_stride * sub_lane + sub_lane, input_stride,
                output_stride, sub_lane, u_tail);
    }
}

} // namespace tr

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
