/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace io {

io_conf_t::io_conf_t(const bool nt_stores_enabled)
    : nt_stores_enabled_(nt_stores_enabled) {}

io_tail_conf_t::io_tail_conf_t(const std::size_t simd_w,
        const std::size_t tail_size, const Xbyak_aarch64::PReg &tail_opmask,
        const int tail_vmm_mask_idx, const Xbyak_aarch64::XReg &reg_tmp,
        const Xbyak_aarch64::XReg &reg_tmp1)
    : simd_w_(simd_w)
    , tail_size_(tail_size)
    , tail_opmask_(tail_opmask)
    , tail_vmm_mask_idx_(tail_vmm_mask_idx)
    , reg_tmp_(reg_tmp)
    , reg_tmp1_(reg_tmp1) {}

io_saturation_conf_t::io_saturation_conf_t(const int vreg_zero_saturation_idx,
        const int vreg_saturation_ubound_idx,
        const Xbyak_aarch64::XReg &reg_tmp)
    : vreg_zero_saturation_idx_(vreg_zero_saturation_idx)
    , vreg_saturation_ubound_idx_(vreg_saturation_ubound_idx)
    , reg_tmp_(reg_tmp) {}

io_gather_conf_t::io_gather_conf_t(const std::size_t simd_w,
        const Xbyak_aarch64::PReg &full_opmask, const int full_vmm_mask_idx,
        const Xbyak_aarch64::XReg &reg_tmp, const Xbyak_aarch64::XReg &reg_tmp1,
        const utils::optional_t<int> &vmm_tmp_idx)
    : simd_w_(simd_w)
    , full_opmask_(full_opmask)
    , full_vmm_mask_idx_(full_vmm_mask_idx)
    , reg_tmp_(reg_tmp)
    , reg_tmp1_(reg_tmp1)
    , vmm_tmp_idx_(vmm_tmp_idx) {}

template <typename Vmm>
jit_io_helper_t<Vmm>::jit_io_helper_t(jit_generator_t *host,
        const cpu_isa_t &isa, const data_type_t &data_type,
        const io_conf_t &io_conf,
        const utils::optional_t<io_tail_conf_t> &tail_conf,
        const utils::optional_t<io_saturation_conf_t> &saturation_conf,
        const utils::optional_t<io_gather_conf_t> &gather_conf)
    : host_(host)
    , isa_(isa)
    , data_type_(data_type)
    , io_conf_(io_conf)
    , tail_conf_(tail_conf)
    , saturation_conf_(saturation_conf)
    , gather_conf_(gather_conf) {

    assert(utils::one_of(data_type_, data_type::f32, data_type::s8,
                   data_type::u8, data_type::s32)
            && "Supported data types f32, s8, u8, s32");

    static constexpr bool is_zmm
            = std::is_same<Vmm, Xbyak_aarch64::ZReg>::value;
    MAYBE_UNUSED(is_zmm);
    assert(IMPLICATION(!is_superset(isa_, sve_128), !is_zmm)
            && "This architecture does not support z registers.");
}

template <typename Vmm>
jit_io_helper_t<Vmm>::~jit_io_helper_t() = default;

// That is ok for ASIMD paths because we handle tail manually with tail_conf_->tail_size_.
template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::prepare_opmask(
        const std::size_t how_many_bits_to_set,
        const Xbyak_aarch64::XReg &reg_tmp0,
        const Xbyak_aarch64::XReg &reg_tmp1, const Xbyak_aarch64::PReg &mask) {
    UNUSED(how_many_bits_to_set);
    UNUSED(reg_tmp0);
    UNUSED(reg_tmp1);
    UNUSED(mask);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_opmask(
        const std::size_t how_many_bits_to_set,
        const Xbyak_aarch64::XReg &reg_tmp0,
        const Xbyak_aarch64::XReg &reg_tmp1, const Xbyak_aarch64::PReg &mask) {
    host_->mov_imm(reg_tmp0, 0);

    host_->mov_imm(host_->X_TMP_2, how_many_bits_to_set);
    host_->whilelt(mask.s, reg_tmp0, host_->X_TMP_2);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_tail_mask() {
    assert(tail_conf_.has_value() && "Config for tail processing is not set.");

    if (!tail_conf_->tail_size_) return;

    assert(is_superset(isa_, asimd));

    prepare_opmask(tail_conf_->tail_size_, tail_conf_->reg_tmp_,
            tail_conf_->reg_tmp1_, tail_conf_->tail_opmask_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::prepare_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    assert(is_superset(isa_, asimd));

    prepare_opmask(gather_conf_->simd_w_, gather_conf_->reg_tmp_,
            gather_conf_->reg_tmp1_, gather_conf_->full_opmask_);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_full_mask() {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");

    if (isa_ == sve_256) {
        const Vmm vmm_mask = Vmm(gather_conf_->full_vmm_mask_idx_);
        host_->eor(Xbyak_aarch64::ZReg(vmm_mask.getIdx()).d,
                Xbyak_aarch64::ZReg(vmm_mask.getIdx()).d,
                Xbyak_aarch64::ZReg(vmm_mask.getIdx()).d);
        host_->mov(Xbyak_aarch64::ZReg(vmm_mask.getIdx()).s,
                host_->P_NOT_256 / Xbyak_aarch64::T_m, 0);
    }
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::init_saturate_f32() const {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    if (utils::one_of(data_type_, data_type::u8, data_type::s8, data_type::s32))
        host_->init_saturate_f32(
                Vmm(saturation_conf_->vreg_zero_saturation_idx_),
                Vmm(saturation_conf_->vreg_saturation_ubound_idx_),
                saturation_conf_->reg_tmp_, data_type::f32, data_type_);
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::gather(
        const Xbyak_aarch64::XReg &src_reg,
        const Xbyak_aarch64::VReg &indices_vmm,
        const Xbyak_aarch64::VReg &dst_vmm, const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const unsigned number_of_values_to_load = tail ? tail_conf_->tail_size_ : 4;

    host_->uni_clear(dst_vmm);

    switch (data_type_) {
        case data_type::f32:
        case data_type::s32:
            for (unsigned i = 0; i < number_of_values_to_load; i++) {
                // Calculate address of the i-th element to load: src_reg + indices_vmm[i]
                host_->mov(host_->W_TMP_0,
                        Xbyak_aarch64::VReg4S(indices_vmm.getIdx())[i]);
                host_->add(host_->X_DEFAULT_ADDR, src_reg, host_->X_TMP_0);
                // Load the i-th element to the i-th lane of dst_vmm
                host_->ld1(Xbyak_aarch64::VReg4S(dst_vmm.getIdx())[i],
                        Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
            }
            break;
        case data_type::s8:
        case data_type::u8:
            for (unsigned i = 0; i < number_of_values_to_load; i++) {
                host_->mov(host_->W_TMP_0,
                        Xbyak_aarch64::VReg4S(indices_vmm.getIdx())[i]);
                host_->add(host_->X_DEFAULT_ADDR, src_reg, host_->X_TMP_0);
                host_->ld1(Xbyak_aarch64::VReg16B(dst_vmm.getIdx())[i],
                        Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
            }
            if (data_type_ == data_type::s8) {
                // sign-extend 8x8-bit -> 8x16-bit
                host_->sxtl(dst_vmm.h8, dst_vmm.b8);
                // sign-extend low 4x16-bit -> 4x32-bit
                host_->sxtl(dst_vmm.s4, dst_vmm.h4);
            } else {
                // unsign-extend 8x8-bit -> 8x16-bit
                host_->uxtl(dst_vmm.h8, dst_vmm.b8);
                // unsign-extend low 4x16-bit -> 4x32-bit
                host_->uxtl(dst_vmm.s4, dst_vmm.h4);
            }
            break;
        default: assert(!"Unsupported data type.");
    }

    if (utils::one_of(data_type_, data_type::s8, data_type::u8))
        convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::gather(const Xbyak_aarch64::XReg &src_reg,
        const Vmm &indices_vmm, const Vmm &dst_vmm, const bool tail) {
    assert(gather_conf_.has_value() && "Config for loading with the use of gather instruction is not set.");
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    const Xbyak_aarch64::PReg mask
            = tail ? tail_conf_->tail_opmask_ : gather_conf_->full_opmask_;

    switch (data_type_) {
        case data_type::f32:
        case data_type::s32:
            host_->ld1w({dst_vmm.s}, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(
                            src_reg, indices_vmm.s, Xbyak_aarch64::SXTW));
            if (data_type_ == data_type::s32)
                convert_to_f32(dst_vmm, dst_vmm, data_type_);
            break;
        case data_type::s8:
            host_->ld1sb({dst_vmm.s}, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(
                            src_reg, indices_vmm.s, Xbyak_aarch64::SXTW));
            break;
        case data_type::u8:
            host_->ld1b({dst_vmm.s}, mask / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(
                            src_reg, indices_vmm.s, Xbyak_aarch64::SXTW));
            break;
        default: assert(!"Unsupported data type.");
    }

    if (data_type_ != data_type::f32)
        convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_raw_vmm, const bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");

    Xbyak_aarch64::PReg mask
            = tail ? tail_conf_->tail_opmask_ : host_->P_ALL_ONE;

    switch (data_type_) {
        case data_type::f32:
            load_f32(src_addr, offt, dst_raw_vmm, tail, mask);
            break;
        case data_type::s32:
            load_s32(src_addr, offt, dst_raw_vmm, tail, mask);
            break;
        case data_type::s8:
        case data_type::u8:
            load_i8(src_addr, offt, dst_raw_vmm, tail, mask);
            break;
        default: assert(!"Unsupported data type.");
    }
}

#if 0
/**
* load_bytes is the utility function to facilitate loading of
* load_size (0 <= load_size <= 32) many contiguous bytes into the Xmm/Ymm
* register from the memory referenced by ptr[reg + offset] address.
*
* Functionally, invocation of load_bytes is equivalent to
* the following loop:
*
* for (int idx = 0; idx < load_size; ++idx)
*     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
*
* TODO: Add an option to zero-out unloaded bytes in the Xmm register.
* TODO: Add an option for unsafe_load wherein one could read outside the
* provided memory buffer so as to minimize the total number of read
* memory instructions.
*/
static void load_bytes(jit_generator *host, const Xbyak_aarch64::VReg &vmm,
        const Xbyak_aarch64::XReg reg_addr, int load_size) {
    if (load_size == 32) {
        host->not_(host->P_TMP.b, host->P_ALL_ONE / Xbyak_aarch64::T_z,
                host->P_NOT_256.b);
        host->ld1w(Xbyak_aarch64::ZRegS(vmm.getIdx()),
                host->P_TMP / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(reg_addr));
        return;
    }
    int start_bytes = 0;
    int bytes_to_load = load_size;

    if (load_size > 16) {
        start_bytes = 16;
        bytes_to_load -= 16;
    }

    if (bytes_to_load >= 8 && bytes_to_load < 16) {
        host->add_imm(
                host->X_DEFAULT_ADDR, reg_addr, start_bytes, host->X_TMP_0);
        host->ldr(host->X_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
        host->mov(vmm.b16, vmm.b16);
        host->ins(Xbyak_aarch64::VReg2D(vmm.getIdx())[0], host->X_TMP_0);
    } else if (bytes_to_load == 16) {
        host->add_imm(
                host->X_DEFAULT_ADDR, reg_addr, start_bytes, host->X_TMP_0);
        host->ldr(Xbyak_aarch64::QReg(vmm.getIdx()),
                Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
    }

    host->add_imm(host->X_DEFAULT_ADDR, reg_addr, start_bytes, host->X_TMP_0);
    switch (bytes_to_load) {
        case 0: break;
        case 1:
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[0], host->W_TMP_0);
            break;
        case 2:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[0], host->W_TMP_0);
            break;
        case 3:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[0], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 2);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[2], host->W_TMP_0);
            break;
        case 4:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[0], host->W_TMP_0);
            break;
        case 5:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[0], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 4);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[4], host->W_TMP_0);
            break;
        case 6:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[0], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 4);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[2], host->W_TMP_0);
            break;
        case 7:
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_DEFAULT_ADDR));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[0], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 4);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[2], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 6);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[6], host->W_TMP_0);
            break;
        case 8: break;
        case 9:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[8], host->W_TMP_0);
            break;
        case 10:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[4], host->W_TMP_0);
            break;
        case 11:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[4], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 10);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[10], host->W_TMP_0);
            break;
        case 12:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[2], host->W_TMP_0);
            break;
        case 13:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[2], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 12);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[12], host->W_TMP_0);
            break;
        case 14:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[2], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 12);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[6], host->W_TMP_0);
            break;
        case 15:
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 8);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->ins(Xbyak_aarch64::VReg4S(vmm.getIdx())[2], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 12);
            host->ldr(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(Xbyak_aarch64::VReg8H(vmm.getIdx())[6], host->W_TMP_0);
            host->add(host->X_TMP_1, host->X_DEFAULT_ADDR, 14);
            host->ldrb(host->W_TMP_0, Xbyak_aarch64::ptr(host->X_TMP_1));
            host->mov(vmm.b16, vmm.b16);
            host->ins(vmm.b16[14], host->W_TMP_0);
            break;
        case 16: break;
        default: assert(!"improper load size");
    }

    if (load_size > 16) {
        host->str(host->z31,
                Xbyak_aarch64::ptr(
                        host->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
        const Xbyak_aarch64::ZReg z_vmm(vmm.getIdx());
        const Xbyak_aarch64::ZReg z_tmp(host->z31.getIdx());
        host->ptrue(host->P_TMP.d, Xbyak_aarch64::VL2);
        host->mov(z_tmp.d, z_vmm.d);
        host->splice(z_tmp.d, host->P_TMP.d, z_vmm.d);
        host->mov(z_vmm.d, z_tmp.d);
        host->mov(z_vmm.s, host->P_NOT_256 / Xbyak_aarch64::T_m, 0);
        host->ld1w(z_tmp.d, host->P_ALL_ONE / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(reg_addr));
        host->ptrue(host->P_TMP.d, Xbyak_aarch64::VL2);
        host->sel(z_vmm.d, host->P_TMP, z_tmp.d, z_vmm.d);
        host->mov(z_vmm.s, host->P_NOT_256 / Xbyak_aarch64::T_m, 0);
        host->ldr(host->z31,
                Xbyak_aarch64::ptr(
                        host->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
    }
}

/**
* load_bytes_to_dword_extension is the utility function to facilitate
* loading of load_size (0 <= load_size <= 16) many contiguous bytes in
* the Xmm register from the memory referenced by ptr[reg + offset]
* address and then do signed/zero extension of those to double words.
*
* Functionally, invocation of load_bytes_to_dword_extension is equivalent
* to the following:
*
* for (int idx = 0; idx < load_size; ++idx)
*     vpinsrb(xmm, xmm, ptr[reg + offset + idx], idx);
* if (is_signed) vpmovsxbd(vmm, vmm); else vpmovzxbd(vmm, vmm);
*
* Valid values for the load_size variable are:
* [0..4] for XMM version of the function
* [0..8] for YMM version of the function.
* TODO: Implement this routine for every ISA.
*/
static void load_bytes_to_dword_extension(jit_generator *host,
        const Xbyak_aarch64::VReg &vmm, const Xbyak_aarch64::XReg &reg_addr,
        bool is_signed, int load_size) {
    if (host->cpu_sveLen == Xbyak_aarch64::util::SVE_128) {
        assert(load_size >= 0 && load_size <= 8);
    } else if (host->cpu_sveLen == Xbyak_aarch64::util::SVE_256) {
        assert(load_size >= 0 && load_size <= 4);
    } else {
        assert(!"routine is not supported for the current isa");
    }
    host->str(host->z31,
            Xbyak_aarch64::ptr(
                    host->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
    // For load_size == 8/4, do load/extension in one go
    const Xbyak_aarch64::ZReg z_tmp(host->z31.getIdx());
    if (load_size == 8) {
        const Xbyak_aarch64::ZReg z_vmm(vmm.getIdx());
        if (is_signed) {
            host->ld1sb(z_tmp.s, host->P_NOT_256, Xbyak_aarch64::ptr(reg_addr));
        } else {
            host->ld1b(z_tmp.s, host->P_NOT_256, Xbyak_aarch64::ptr(reg_addr));
        }
    } else if (load_size == 4) {
        const Xbyak_aarch64::ZReg z_vmm(vmm.getIdx());
        if (is_signed) {
            host->ld1sb(z_tmp.s, host->P_NOT_128, Xbyak_aarch64::ptr(reg_addr));
        } else {
            host->ld1b(z_tmp.s, host->P_NOT_128, Xbyak_aarch64::ptr(reg_addr));
        }
    } else {
        load_bytes(host, vmm, reg_addr, load_size);
        if (is_signed) {
            host->mov(z_tmp.d, host->P_ALL_ONE,
                    Xbyak_aarch64::ZRegD(vmm.getIdx()));
            host->sxtl(Xbyak_aarch64::VReg8H(vmm.getIdx()),
                    Xbyak_aarch64::VReg8B(vmm.getIdx()));
            host->sxtl(Xbyak_aarch64::VReg4S(vmm.getIdx()),
                    Xbyak_aarch64::VReg4H(vmm.getIdx()));
            host->mov(Xbyak_aarch64::ZRegD(vmm.getIdx()), host->P_NOT_128,
                    z_tmp.d);
        } else {
            host->mov(z_tmp.d, host->P_ALL_ONE,
                    Xbyak_aarch64::ZRegD(vmm.getIdx()));
            host->uxtl(Xbyak_aarch64::VReg8H(vmm.getIdx()),
                    Xbyak_aarch64::VReg8B(vmm.getIdx()));
            host->uxtl(Xbyak_aarch64::VReg4S(vmm.getIdx()),
                    Xbyak_aarch64::VReg4H(vmm.getIdx()));
            host->mov(Xbyak_aarch64::ZRegD(vmm.getIdx()), host->P_NOT_128,
                    z_tmp.d);
        }
    }
    host->ldr(host->z31,
            Xbyak_aarch64::ptr(
                    host->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
}
template <typename Vmm>
void load_data(jit_generator *host, data_type_t type_in, const Vmm &vmm,
        const Xbyak_aarch64::XReg &src_addr, int load_size) {

    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            load_bytes(host, Xbyak_aarch64::VReg(vmm.getIdx()), src_addr,
                    sizeof(int32_t) * load_size);
            break;
        case data_type::s8:
        case data_type::u8:
            load_bytes_to_dword_extension(host,
                    Xbyak_aarch64::VReg(vmm.getIdx()), src_addr,
                    type_in == data_type::s8, load_size);
            break;
        default: assert(!"unsupported source data type");
    }
}
#endif

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_f32(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_vmm, const bool tail,
        const Xbyak_aarch64::PReg &mask) {

    host_->ld1w(
            dst_vmm.s, mask / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(src_addr));
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::load_f32(
        const Xbyak_aarch64::XReg &src_addr, const int offt,
        const Xbyak_aarch64::VReg &dst_vmm, const bool tail,
        const Xbyak_aarch64::PReg &mask) {

    if (tail && tail_conf_->tail_size_ > 0) {
        // tail_size_ = nelems % simd_w_, so it cannot be greater than (simd_w_ - 1), which is 4 for VReg.
        // Refer to binary_kernel_t::get_tail_size().
        const int SZ = sizeof(int32_t);
        switch (tail_conf_->tail_size_) {
            case 1:
                host_->ld1(dst_vmm.s[0], Xbyak_aarch64::ptr(src_addr));
                break;
            case 2:
                host_->ld1(dst_vmm.d[0], Xbyak_aarch64::ptr(src_addr));
                break;
            case 3:
                host_->ld1(dst_vmm.d[0], Xbyak_aarch64::ptr(src_addr));
                host_->add(src_addr, src_addr, SZ * 2);
                host_->ld1(dst_vmm.s[2], Xbyak_aarch64::ptr(src_addr));
                host_->sub(src_addr, src_addr, SZ * 2);
                break;
            default: assert(!"unreachable");
        }
    } else {
        host_->ld1(dst_vmm.s, Xbyak_aarch64::ptr(src_addr));
    }
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::load_s32(
        const Xbyak_aarch64::XReg &src_addr, const int offt,
        const Xbyak_aarch64::VReg &dst_vmm, const bool tail,
        const Xbyak_aarch64::PReg &mask) {
    load_f32(src_addr, offt, dst_vmm, tail, mask);
    host_->scvtf(dst_vmm.s, dst_vmm.s);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_s32(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_vmm, const bool tail,
        const Xbyak_aarch64::PReg &mask) {
    host_->ld1w(
            dst_vmm.s, mask / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(src_addr));
    host_->scvtf(dst_vmm.s, host_->P_TMP / Xbyak_aarch64::T_m, dst_vmm.s);
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::load_i8(
        const Xbyak_aarch64::XReg &src_addr, const int offt,
        const Xbyak_aarch64::VReg &dst_vmm, const bool tail,
        const Xbyak_aarch64::PReg &mask) {
    UNUSED(offt);
    UNUSED(mask);

    const unsigned number_of_values_to_load = tail ? tail_conf_->tail_size_ : 4;
    const auto &reg_tmp = tail_conf_->reg_tmp_;

    host_->movi(dst_vmm.s4, 0);

    for (size_t i = 0; i < number_of_values_to_load; ++i) {
        const Xbyak_aarch64::XReg addr
                = host_->addr_off(src_addr, i, reg_tmp, host_->X_TMP_0);
        if (data_type_ == data_type::s8) {
            host_->ldrsb(host_->W_TMP_0, Xbyak_aarch64::ptr(addr));
        } else {
            host_->ldrb(host_->W_TMP_0, Xbyak_aarch64::ptr(addr));
        }
        host_->ins(Xbyak_aarch64::VReg4S(dst_vmm.getIdx())[i], host_->W_TMP_0);
    }

    convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::load_i8(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_vmm, const bool tail,
        const Xbyak_aarch64::PReg &mask) {
    UNUSED(offt);
    UNUSED(tail);

    if (data_type_ == data_type::s8)
        host_->ld1sb(dst_vmm.s, mask / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(src_addr));
    else
        host_->ld1b(dst_vmm.s, mask / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(src_addr));

    convert_to_f32(dst_vmm, dst_vmm, data_type::s32);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store(const Vmm &src_raw_vmm,
        const Xbyak_aarch64::XReg &dst_raw_addr, const int offt,
        const bool tail) {
    assert(IMPLICATION(tail, tail_conf_.has_value())
            && "Config for tail processing is not set.");
    assert(!(tail && io_conf_.nt_stores_enabled_)
            && "Usage of non-temporal stores with tail leads to a general-protection exception.");

    Xbyak_aarch64::PReg mask
            = tail ? tail_conf_->tail_opmask_ : host_->P_ALL_ONE;
    const bool is_i8 = utils::one_of(data_type_, data_type::s8, data_type::u8);

    if (data_type_ == data_type::s32 || is_i8) saturate(src_raw_vmm);

    switch (data_type_) {
        case data_type::f32:
        case data_type::s32:
            store_f32(src_raw_vmm, dst_raw_addr, offt, tail, mask);
            break;
        case data_type::s8:
        case data_type::u8:
            store_i8(src_raw_vmm, dst_raw_addr, offt, tail, mask);
            break;
        default: assert(!"Unsupported data type.");
    }
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::saturate(
        const Xbyak_aarch64::VReg &vmm) {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    host_->saturate_f32(vmm,
            Xbyak_aarch64::VReg(saturation_conf_->vreg_zero_saturation_idx_),
            Xbyak_aarch64::VReg(saturation_conf_->vreg_saturation_ubound_idx_),
            data_type_, host_->P_ALL_ONE);
    host_->frintn(vmm.s, vmm.s); // Round to nearest even
    host_->fcvtzs(vmm.s, vmm.s); // Floating-point convert to signed integer
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::saturate(const Vmm &vmm) {
    assert(saturation_conf_.has_value() && "Config for saturation is not set.");

    host_->saturate_f32(vmm, Vmm(saturation_conf_->vreg_zero_saturation_idx_),
            Vmm(saturation_conf_->vreg_saturation_ubound_idx_), data_type_,
            host_->P_ALL_ONE);
    host_->frintn(vmm.s, host_->P_ALL_ONE / Xbyak_aarch64::T_m,
            vmm.s); // Round to nearest even
    host_->fcvtzs(vmm.s, host_->P_ALL_ONE / Xbyak_aarch64::T_m, vmm.s);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_f32(const Vmm &src_vmm,
        const Xbyak_aarch64::XReg &dst_addr, const int offt, const bool tail,
        const Xbyak_aarch64::PReg &mask) {
    if (io_conf_.nt_stores_enabled_) {
        host_->stnt1d(Xbyak_aarch64::ZRegD(src_vmm.getIdx()), mask,
                Xbyak_aarch64::ptr(dst_addr));
    } else {
        host_->st1w(Xbyak_aarch64::ZRegS(src_vmm.getIdx()), mask,
                Xbyak_aarch64::ptr(dst_addr));
    }
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::store_f32(
        const Xbyak_aarch64::VReg &src_vmm, const Xbyak_aarch64::XReg &dst_addr,
        const int offt, const bool tail, const Xbyak_aarch64::PReg &mask) {
    UNUSED(offt);
    UNUSED(mask);
    if (io_conf_.nt_stores_enabled_) {
        host_->stnt1d(Xbyak_aarch64::ZRegD(src_vmm.getIdx()), mask,
                Xbyak_aarch64::ptr(dst_addr));
    } else if (isa_ == asimd && tail && tail_conf_->tail_size_ > 0) {
        // tail_size_ = nelems % simd_w_, so it cannot be greater than (simd_w_ - 1), which is 4 for VReg.
        // Refer to binary_kernel_t::get_tail_size().
        const int SZ = sizeof(float);
        switch (tail_conf_->tail_size_) {
            case 1:
                host_->str(Xbyak_aarch64::SReg(src_vmm.getIdx()),
                        Xbyak_aarch64::ptr(dst_addr));
                break;
            case 2:
                host_->str(Xbyak_aarch64::DReg(src_vmm.getIdx()),
                        Xbyak_aarch64::ptr(dst_addr));
                break;
            case 3:
                host_->str(Xbyak_aarch64::DReg(src_vmm.getIdx()),
                        Xbyak_aarch64::ptr(dst_addr));
                host_->add(dst_addr, dst_addr, SZ * 2);
                host_->st1(Xbyak_aarch64::VReg4S(src_vmm.getIdx())[2],
                        Xbyak_aarch64::ptr(dst_addr));
                host_->sub(dst_addr, dst_addr, SZ * 2);
                break;
            default: assert(!"unreachable");
        }
    } else {
        host_->st1(src_vmm.s, Xbyak_aarch64::ptr(dst_addr));
    }
}

template <>
int jit_io_helper_t<Xbyak_aarch64::VReg>::allocate_temp_register(
        const Xbyak_aarch64::VReg &reg) {
    const int SIMD_SZ = 16; // SIMD register length in bytes

    for (size_t i = 0; i < 32; i++) {
        // Look for a temporary vector register whose index isn’t the same as lhs.
        if (reg.getIdx() != i) {
            // Allocate space on stack
            host_->sub(host_->X_SP, host_->X_SP, SIMD_SZ);
            // Store the scratch register into the stack so it's safe to clobber it.
            host_->str(Xbyak_aarch64::QReg(i), Xbyak_aarch64::ptr(host_->X_SP));
            // The temporary register was found, therefore there is no need to keep searching.
            return i;
        }
    }
    assert("cannot find temporary register to allocate");
    return -1;
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::deallocate_temp_register(
        const int idx) {
    const int SIMD_SZ = 16; // SIMD register length in bytes

    // Restore the scratch register's initial content from the stack.
    host_->ldr(Xbyak_aarch64::QReg(idx), Xbyak_aarch64::ptr(host_->X_SP));
    // Free allocated stack space
    host_->add(host_->X_SP, host_->X_SP, SIMD_SZ);
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::umin(
        Xbyak_aarch64::VReg &dst, const int32_t imm) {
    Xbyak_aarch64::VReg v_tmp(allocate_temp_register(dst));
    host_->mov_imm(host_->W_TMP_0, imm);
    host_->dup(v_tmp.s, host_->W_TMP_0);
    host_->umin(dst.s4, dst.s4, v_tmp.s4);
    deallocate_temp_register(v_tmp.getIdx());
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::smin(
        Xbyak_aarch64::VReg &dst, const int32_t imm) {
    Xbyak_aarch64::VReg v_tmp(allocate_temp_register(dst));
    host_->mov_imm(host_->W_TMP_0, imm);
    host_->dup(v_tmp.s, host_->W_TMP_0);
    host_->smin(dst.s4, dst.s4, v_tmp.s4);
    deallocate_temp_register(v_tmp.getIdx());
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::smax(
        Xbyak_aarch64::VReg &dst, const int32_t imm) {
    Xbyak_aarch64::VReg v_tmp(allocate_temp_register(dst));
    host_->mov_imm(host_->W_TMP_0, imm);
    host_->dup(v_tmp.s, host_->W_TMP_0);
    host_->smax(dst.s4, dst.s4, v_tmp.s4);
    deallocate_temp_register(v_tmp.getIdx());
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::store_i8_sdb(
        Xbyak_aarch64::XReg addr, const Xbyak_aarch64::VReg &src_vmm,
        const bool tail, const Xbyak_aarch64::PReg &mask) {
    UNUSED(mask);

    Xbyak_aarch64::VReg v_tmp(allocate_temp_register(src_vmm));
    host_->mov(v_tmp.b16, Xbyak_aarch64::VReg(src_vmm.getIdx()).b16);
    smin(v_tmp, 127);
    smax(v_tmp, -128);

    const int SZ = sizeof(int);
    if (tail && tail_conf_->tail_size_ > 0 && tail_conf_->tail_size_ < 4) {
        // tail_size_ = nelems % simd_w_, so it cannot be greater than (simd_w_ - 1), which is 4 for VReg.
        // Refer to binary_kernel_t::get_tail_size().
        switch (tail_conf_->tail_size_) {
            case 1: host_->st1(v_tmp.b16[0], Xbyak_aarch64::ptr(addr)); break;
            case 2:
                host_->st1(v_tmp.b16[0], Xbyak_aarch64::ptr(addr));
                host_->add(addr, addr, 1);
                host_->st1(v_tmp.b16[SZ], Xbyak_aarch64::ptr(addr));
                host_->sub(addr, addr, 1);
                break;
            case 3:
                host_->st1(v_tmp.b16[0], Xbyak_aarch64::ptr(addr));
                host_->add(addr, addr, 1);
                host_->st1(v_tmp.b16[SZ], Xbyak_aarch64::ptr(addr));
                host_->add(addr, addr, 1);
                host_->st1(v_tmp.b16[SZ * 2], Xbyak_aarch64::ptr(addr));
                host_->sub(addr, addr, 2);
                break;
            default: assert(!"unreachable");
        }
    } else {
        host_->st1(v_tmp.b16[0], Xbyak_aarch64::ptr(addr));
        host_->add(addr, addr, 1);
        host_->st1(v_tmp.b16[SZ], Xbyak_aarch64::ptr(addr));
        host_->add(addr, addr, 1);
        host_->st1(v_tmp.b16[SZ * 2], Xbyak_aarch64::ptr(addr));
        host_->add(addr, addr, 1);
        host_->st1(v_tmp.b16[SZ * 3], Xbyak_aarch64::ptr(addr));
        host_->sub(addr, addr, 3);
    }
    deallocate_temp_register(v_tmp.getIdx());
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_i8_sdb(Xbyak_aarch64::XReg addr,
        const Vmm &src_vmm, const bool tail, const Xbyak_aarch64::PReg &mask) {
    UNUSED(tail);
    host_->str(host_->z31,
            Xbyak_aarch64::ptr(
                    host_->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
    const Xbyak_aarch64::ZReg z_tmp(host_->z31.getIdx());
    host_->mov(z_tmp.d, Xbyak_aarch64::ZRegD(src_vmm.getIdx()));
    host_->smin(z_tmp.s, 127);
    host_->smax(z_tmp.s, -128);
    host_->st1b(z_tmp.s, mask, Xbyak_aarch64::ptr(addr));
    host_->ldr(host_->z31,
            Xbyak_aarch64::ptr(
                    host_->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
}

template <>
void jit_io_helper_t<Xbyak_aarch64::VReg>::store_i8_udb(
        Xbyak_aarch64::XReg addr, const Xbyak_aarch64::VReg &src_vmm,
        const bool tail, const Xbyak_aarch64::PReg &mask) {
    UNUSED(mask);

    const uint32_t idx = allocate_temp_register(src_vmm);
    Xbyak_aarch64::VReg v_tmp(idx);
    host_->mov(v_tmp.b16, Xbyak_aarch64::VReg16B(src_vmm.getIdx()));
    umin(v_tmp, 255);

    const int SZ = sizeof(int);
    if (tail && tail_conf_->tail_size_ > 0 && tail_conf_->tail_size_ < 4) {
        // tail_size_ = nelems % simd_w_, so it cannot be greater than (simd_w_ - 1), which is 4 for VReg.
        // Refer to binary_kernel_t::get_tail_size().
        switch (tail_conf_->tail_size_) {
            case 1: host_->st1(v_tmp.b16[0], Xbyak_aarch64::ptr(addr)); break;
            case 2:
                host_->st1(v_tmp.b16[0], Xbyak_aarch64::ptr(addr));
                host_->add(addr, addr, 1);
                host_->st1(v_tmp.b16[SZ], Xbyak_aarch64::ptr(addr));
                host_->sub(addr, addr, 1);
                break;
            case 3:
                host_->st1(v_tmp.b16[0], Xbyak_aarch64::ptr(addr));
                host_->add(addr, addr, 1);
                host_->st1(v_tmp.b16[SZ], Xbyak_aarch64::ptr(addr));
                host_->add(addr, addr, 1);
                host_->st1(v_tmp.b16[SZ * 2], Xbyak_aarch64::ptr(addr));
                host_->sub(addr, addr, 2);
                break;
            default: assert(!"unreachable");
        }
    } else {
        host_->st1(v_tmp.b16[0], Xbyak_aarch64::ptr(addr));
        host_->add(addr, addr, 1);
        host_->st1(v_tmp.b16[SZ], Xbyak_aarch64::ptr(addr));
        host_->add(addr, addr, 1);
        host_->st1(v_tmp.b16[SZ * 2], Xbyak_aarch64::ptr(addr));
        host_->add(addr, addr, 1);
        host_->st1(v_tmp.b16[SZ * 3], Xbyak_aarch64::ptr(addr));
        host_->sub(addr, addr, 3);
    }

    deallocate_temp_register(idx);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_i8_udb(Xbyak_aarch64::XReg addr,
        const Vmm &src_vmm, const bool tail, const Xbyak_aarch64::PReg &mask) {
    UNUSED(tail);
    host_->str(host_->z31,
            Xbyak_aarch64::ptr(
                    host_->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
    const Xbyak_aarch64::ZReg z_tmp(host_->z31.getIdx());
    host_->mov(z_tmp.d, Xbyak_aarch64::ZRegD(src_vmm.getIdx()));
    host_->umin(z_tmp.s, 255);
    host_->st1b(z_tmp.s, mask, Xbyak_aarch64::ptr(addr));
    host_->ldr(host_->z31,
            Xbyak_aarch64::ptr(
                    host_->X_TRANSLATOR_STACK, -1, Xbyak_aarch64::MUL_VL));
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::store_i8(const Vmm &src_vmm,
        const Xbyak_aarch64::XReg &dst_addr, const int offt, const bool tail,
        const Xbyak_aarch64::PReg &mask) {
    UNUSED(offt);
    using namespace std::placeholders;
    static constexpr bool is_zmm
            = std::is_same<Vmm, Xbyak_aarch64::ZReg>::value;

    auto store_i8_fn = data_type_ == data_type::s8
            ? std::bind(&jit_io_helper_t::store_i8_sdb, this, _1, _2, _3, _4)
            : std::bind(&jit_io_helper_t::store_i8_udb, this, _1, _2, _3, _4);

    if (io_conf_.nt_stores_enabled_ && is_zmm) {
        host_->not_(
                host_->P_TMP.b, mask / Xbyak_aarch64::T_z, host_->P_NOT_128.b);
        host_->stnt1d(Xbyak_aarch64::ZRegD(src_vmm.getIdx()), mask,
                Xbyak_aarch64::ptr(dst_addr));
    } else {
        store_i8_fn(dst_addr, src_vmm, tail, mask);
    }
}

void uni_expand_s8_to_s32(jit_generator_t *host_,
        const Xbyak_aarch64::VReg &dst, const Xbyak_aarch64::VReg &src) {
    host_->zip1(dst.b, src.b, src.b);
    host_->zip1(dst.h, dst.h, dst.h);
    // sign-extend 8x8-bit -> 8x16-bit
    host_->sxtl(dst.h8, dst.b8);
    // sign-extend low 4x16-bit -> 4x32-bit
    host_->sxtl(dst.s4, dst.h4);
}

template <typename Vmm>
void uni_expand_s8_to_s32(
        jit_generator_t *host_, const Vmm &dst, const Vmm &src) {
    Xbyak_aarch64::ZReg z_dst(dst.getIdx());
    Xbyak_aarch64::ZReg z_src(src.getIdx());
    host_->zip1(z_dst.b, z_src.b, z_src.b);
    host_->zip1(z_dst.h, z_dst.h, z_dst.h);
    host_->sxtb(z_dst.s, host_->P_ALL_ONE / Xbyak_aarch64::T_m, z_dst.s);
}

void uni_expand_u8_to_s32(jit_generator_t *host_,
        const Xbyak_aarch64::VReg &dst, const Xbyak_aarch64::VReg &src) {
    host_->zip1(dst.b, src.b, src.b);
    host_->zip1(dst.h, dst.h, dst.h);
    // zero-extend 8x8-bit -> 8x16-bit
    host_->uxtl(dst.h8, dst.b8);
    // zero-extend low 4x16-bit -> 4x32-bit
    host_->uxtl(dst.s4, dst.h4);
}

template <typename Vmm>
void uni_expand_u8_to_s32(
        jit_generator_t *host_, const Vmm &dst, const Vmm &src) {
    Xbyak_aarch64::ZReg z_dst(dst.getIdx());
    Xbyak_aarch64::ZReg z_src(src.getIdx());
    host_->zip1(z_dst.b, z_src.b, z_src.b);
    host_->zip1(z_dst.h, z_dst.h, z_dst.h);
    host_->uxtb(z_dst.s, host_->P_ALL_ONE / Xbyak_aarch64::T_m, z_dst.s);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::convert_to_f32(const Vmm &dst_vmm,
        const Vmm &src_vmm, const data_type_t src_data_type) {
    switch (src_data_type) {
        case data_type::f32: // Do nothing
            break;
        case data_type::s32: {
            assert(dst_vmm.getIdx() == src_vmm.getIdx());
            host_->uni_scvtf(dst_vmm.s, dst_vmm.s);
            break;
        }
        case data_type::s8: {
            uni_expand_s8_to_s32(host_, dst_vmm, src_vmm);
            host_->uni_scvtf(dst_vmm.s, dst_vmm.s);
            break;
        }
        case data_type::u8: {
            uni_expand_u8_to_s32(host_, dst_vmm, src_vmm);
            host_->uni_scvtf(dst_vmm.s, dst_vmm.s);
            break;
        }
        default: assert(!"Unsupported data type.");
    }
}

void uni_broadcast(jit_generator_t *host_, const Xbyak_aarch64::VReg &dst,
        const Xbyak_aarch64::XReg &src) {
    host_->ld1r(dst.s4, Xbyak_aarch64::ptr(src));
}

void uni_broadcast(jit_generator_t *host_, const Xbyak_aarch64::VReg &dst,
        const Xbyak_aarch64::VReg &src) {
    host_->dup(dst.s4, src.s4[0]);
}

template <typename Vmm>
void uni_broadcast(jit_generator_t *host_, const Vmm &dst,
        const Xbyak_aarch64::XReg &src) {
    uint8_t dstIdx = dst.getIdx();
    host_->ld1rw(Xbyak_aarch64::ZRegS(dstIdx),
            host_->P_ALL_ONE / Xbyak_aarch64::T_z, Xbyak_aarch64::ptr(src));
}

template <typename Vmm>
void uni_broadcast(jit_generator_t *host_, const Vmm &dst,
        const Xbyak_aarch64::VReg &src) {
    uint8_t dstIdx = dst.getIdx();
    uint8_t srcIdx = src.getIdx();
    host_->dup(Xbyak_aarch64::ZRegS(dstIdx), Xbyak_aarch64::ZRegS(srcIdx)[0]);
}

template <typename Vmm>
void jit_io_helper_t<Vmm>::broadcast(const Xbyak_aarch64::XReg &src_addr,
        const int offt, const Vmm &dst_vmm) {
    switch (data_type_) {
        case data_type::f32: uni_broadcast(host_, dst_vmm, src_addr); break;
        case data_type::s32: {
            if (is_superset(isa_, sve_512)) {
                if (host_->cpu_sveLen == sve_128) {
                    host_->ld1(Xbyak_aarch64::VReg(dst_vmm.getIdx()).s4,
                            Xbyak_aarch64::ptr(src_addr));
                } else {
                    host_->ld1w(Xbyak_aarch64::ZRegD(dst_vmm.getIdx()),
                            host_->P_ALL_ONE / Xbyak_aarch64::T_z,
                            Xbyak_aarch64::ptr(src_addr));
                    host_->mov(host_->P_TMP.b, host_->P_ALL_ONE.b);
                    host_->scvtf(Xbyak_aarch64::ZReg(dst_vmm.getIdx()).s,
                            host_->P_TMP / Xbyak_aarch64::T_m,
                            Xbyak_aarch64::ZReg(dst_vmm.getIdx()).s);
                    ;
                    if (host_->cpu_sveLen == sve_256)
                        host_->mov(Xbyak_aarch64::ZReg(dst_vmm.getIdx()).s,
                                host_->P_NOT_256 / Xbyak_aarch64::T_m, 0);
                }
            } else {
                uni_broadcast(host_, dst_vmm, src_addr);
                convert_to_f32(dst_vmm, dst_vmm, data_type_);
            }
            break;
        }
        case data_type::s8:
        case data_type::u8: {
            const Xbyak_aarch64::VReg dst_xmm {dst_vmm.getIdx()};
            host_->ldrb(host_->W_TMP_0, Xbyak_aarch64::ptr(src_addr));
            host_->ins(dst_xmm.b16[0], host_->W_TMP_0);
            convert_to_f32(dst_vmm, dst_vmm, data_type_);
            uni_broadcast(host_, dst_vmm, dst_xmm);

            break;
        }
        default: assert(!"Unsupported data type.");
    }
}

template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::jit_io_multi_dt_helper_t(jit_generator_t *host,
        const cpu_isa_t &isa, const data_types_t &data_types,
        const io_conf_t &io_conf,
        const utils::optional_t<io_tail_conf_t> &tail_conf,
        const std::map<data_type_t, io_saturation_conf_t> &saturation_confs,
        const utils::optional_t<io_gather_conf_t> &gather_conf) {
    assert(!data_types.empty());
    for (const auto &dt : data_types) {
        // can be replaced by try_emplace from C++17
        if (storage_.find(dt) == storage_.cend()) {

            const auto saturation_conf = saturation_confs.find(dt);
            const bool store_saturation_needed
                    = saturation_conf != saturation_confs.cend();

            storage_.emplace(dt,
                    std::make_shared<jit_io_helper_t<Vmm>>(host, isa, dt,
                            io_conf, tail_conf,
                            store_saturation_needed
                                    ? utils::optional_t<
                                              io_saturation_conf_t> {saturation_conf
                                                                             ->second}
                                    : utils::nullopt,
                            gather_conf));
        }
    }
}

template <typename Vmm>
std::shared_ptr<jit_io_helper_t<Vmm>> jit_io_multi_dt_helper_t<Vmm>::at(
        const data_type_t dt) const {
    const auto it = storage_.find(dt);
    if (it != storage_.cend()) return it->second;

    return nullptr;
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::prepare_tail_mask() {
    return storage_.cbegin()->second->prepare_tail_mask();
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::prepare_full_mask() {
    return storage_.cbegin()->second->prepare_full_mask();
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::init_saturate_f32(
        const data_types_t &store_data_types) {
    for (const auto &dt : store_data_types) {
        const auto it = storage_.find(dt);
        if (it != storage_.cend()) {
            if (it->second->saturation_conf_.has_value())
                it->second->init_saturate_f32();
        }
    }
}

template <typename Vmm>
void jit_io_multi_dt_helper_t<Vmm>::init_full_mask() {
    return storage_.cbegin()->second->init_full_mask();
}

template <typename Vmm>
jit_io_multi_dt_helper_t<Vmm>::~jit_io_multi_dt_helper_t() = default;

template class jit_io_helper_t<Xbyak_aarch64::ZReg>;
template class jit_io_multi_dt_helper_t<Xbyak_aarch64::ZReg>;
template class jit_io_helper_t<Xbyak_aarch64::VReg>;
template class jit_io_multi_dt_helper_t<Xbyak_aarch64::VReg>;

} // namespace io
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
