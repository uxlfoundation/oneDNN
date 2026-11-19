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

#include <numeric>
#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_base.hpp"
#include "cpu/x64/lrn/jit_avx512_common_lrn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using namespace data_type;

static constexpr int acc_size = sizeof(acc_data_t);
static constexpr int acc_bf_16_size = sizeof(acc_data_bf16_t);

const int32_t jit_avx512_common_lrn_kernel_bwd_t::jit_args_bwd_t::mask[48]
        = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0};

jit_avx512_common_lrn_kernel_bwd_t::jit_args_bwd_t::jit_args_bwd_t()
    : src(nullptr)
    , diff_dst(nullptr)
    , ws0(nullptr)
    , ws1(nullptr)
    , diff_src(nullptr)
    , mask_ptr(&mask[16]) {}

void jit_avx512_common_lrn_kernel_bwd_t::load_data(
        Xmm reg, const Address p, bool from_stack) {
    if (from_stack || dt_ == f32) {
        this->vmovups(reg, p);
    } else if (dt_ == bf16) {
        this->vpmovzxwd(reg, p);
        this->vpslld(reg, reg, 0x10);
    } else {
        this->vcvtph2psx(reg, p);
    }
}

void jit_avx512_common_lrn_kernel_bwd_t::store_data(
        bool non_temp_hint, const Address addr, Zmm zr) {
    if (dt_ == f32) {
        if (non_temp_hint)
            uni_vmovntps(addr, zr);
        else
            uni_vmovups(addr, zr);
        return;
    }
    if (dt_ == f16) {
        this->vcvtps2ph(addr, zr, jit_generator_t::_op_mxcsr);
        return;
    }

    const bool is_bf16_supported
            = mayiuse(avx512_core_bf16) || bf16_emu_ != nullptr;
    if (!is_bf16_supported) {
        assert(!"Failure in storing bf16 data.");
        return;
    }
    const Ymm yr = Ymm(zr.getIdx());
    if (mayiuse(avx512_core_bf16))
        vcvtneps2bf16(yr, zr);
    else
        bf16_emu_->vcvtneps2bf16(yr, zr);
    vmovdqu16(addr, yr);
}

void jit_avx512_common_lrn_kernel_bwd_t::load_tail(int tail_value, Reg64 src,
        int src_mem_offset, int dst_stack_offset,
        int tmp_load_to_stack_idx_tail) {
    // TODO: Investigate if this method can be simplified by using mask or
    // jit_generator_t load utilities.
    const int src_acc_size
            = utils::one_of(dt_, bf16, f16) ? acc_bf_16_size : acc_size;
    auto tmp_xreg = this->xreg(0, tmp_load_to_stack_idx_tail);

    const auto load_tail_simd = [&](Xmm tmp_reg, int vlen) {
        this->load_data(tmp_reg, this->EVEX_compress_addr(src, src_mem_offset));
        this->vmovups(this->EVEX_compress_addr(rsp, dst_stack_offset), tmp_reg);
        dst_stack_offset += vlen * acc_size;
        src_mem_offset += vlen * src_acc_size;
        tail_value -= vlen;
    };

    if (tail_value >= 8)
        load_tail_simd(this->yreg(0, tmp_load_to_stack_idx_tail), 8);
    if (tail_value >= 4) load_tail_simd(tmp_xreg, 4);

    for (int i = 0; i < tail_value; ++i) {
        if (dt_ == bf16) {
            this->movzx(this->imm_addr64_, word[src + src_mem_offset]);
            this->vmovq(tmp_xreg, this->imm_addr64_);
            this->vpslld(tmp_xreg, tmp_xreg, 0x10);
        } else if (dt_ == f16) {
            this->vxorps(tmp_xreg, tmp_xreg, tmp_xreg);
            this->vcvtsh2ss(tmp_xreg, tmp_xreg,
                    this->EVEX_compress_addr(src, src_mem_offset));
        } else
            this->vmovss(
                    tmp_xreg, this->EVEX_compress_addr(src, src_mem_offset));

        this->vmovss(ptr[rsp + dst_stack_offset], tmp_xreg);

        dst_stack_offset += acc_size;
        src_mem_offset += src_acc_size;
    }
}

void jit_avx512_common_lrn_kernel_bwd_t::store_tail(int tail_value, Zmm src,
        Reg64 dst, int dst_mem_offset, int tmp_stack_offset, int tmp_idx) {

    this->store_data(
            false, this->EVEX_compress_addr(rsp, tmp_stack_offset), src);

    if (dt_ == f32) {
        const auto store_tail_simd = [&](Xmm tmp_reg, int vlen) {
            this->vmovups(
                    tmp_reg, this->EVEX_compress_addr(rsp, tmp_stack_offset));
            this->vmovups(
                    this->EVEX_compress_addr(dst, dst_mem_offset), tmp_reg);
            tmp_stack_offset += vlen * acc_size;
            dst_mem_offset += vlen * acc_size;
            tail_value -= vlen;
        };

        if (tail_value >= 8) store_tail_simd(this->yreg(0, tmp_idx), 8);
        if (tail_value >= 4) store_tail_simd(this->xreg(0, tmp_idx), 4);

        for (int i = 0; i < tail_value;
                ++i, tmp_stack_offset += acc_size, dst_mem_offset += acc_size) {
            this->vmovss(this->xreg(0, tmp_idx),
                    this->EVEX_compress_addr(rsp, tmp_stack_offset));
            this->vmovss(this->EVEX_compress_addr(dst, dst_mem_offset),
                    this->xreg(0, tmp_idx));
        }
        return;
    }

    const auto res = std::div(tail_value, 4);

    for (int i = 0; i < res.quot; ++i, tmp_stack_offset += 4 * acc_bf_16_size,
             dst_mem_offset += 4 * acc_bf_16_size) {
        this->mov(this->imm_addr64_, qword[rsp + tmp_stack_offset]);
        this->mov(qword[dst + dst_mem_offset], this->imm_addr64_);
    }

    for (int i = 0; i < res.rem; ++i, tmp_stack_offset += acc_bf_16_size,
             dst_mem_offset += acc_bf_16_size) {
        this->mov(this->imm_addr16_, word[rsp + tmp_stack_offset]);
        this->mov(word[dst + dst_mem_offset], this->imm_addr16_);
    }
}

jit_avx512_common_lrn_kernel_bwd_t::jit_avx512_common_lrn_kernel_bwd_t(
        data_type_t dt, float alpha, float beta, int local_size,
        const char *name)
    : jit_generator_t(name, get_isa(dt))
    , dt_(dt)
    , local_size_ {local_size - !(local_size % 2)}
    , z_prev_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(), 3);
        return v;
    }()}
    , z_next_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(), 3 + this->local_size_ / 2);
        return v;
    }()}
    , nalphabeta_(-2 * alpha * beta)
    , emulateBfloat_(dt == bf16 && !mayiuse(avx512_core_bf16))
    , regs_used_per_block_ {std::max(this->local_size_ + 2, 7)}
    , reg_block_ {[this]() {
        const int max_possible_reg_block
                = (emulateBfloat_ ? 27 : 31) / this->regs_used_per_block_;
        return mayiuse(avx512_core) ? max_possible_reg_block
                                    : std::min(max_possible_reg_block, 2);
    }()}
    , vlen_(utils::one_of(dt, bf16, f16) ? 32 : 64) {
    if (emulateBfloat_) {
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                bf16_emu_reserv_1_, bf16_emu_reserv_2_, bf16_emu_reserv_3_,
                bf16_emu_scratch_, bf16_emu_reserv_4_);
    }
}

Zmm jit_avx512_common_lrn_kernel_bwd_t::zreg(int irb, int i) const {
    return Zmm(irb * regs_used_per_block_ + i);
}

Ymm jit_avx512_common_lrn_kernel_bwd_t::yreg(int irb, int i) const {
    return Ymm(irb * regs_used_per_block_ + i);
}

Xmm jit_avx512_common_lrn_kernel_bwd_t::xreg(int irb, int i) const {
    return Xmm(irb * regs_used_per_block_ + i);
}

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
