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
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/utils/jit_regops.hpp"

#include "cpu/x64/matmul/jit_brgemm_matmul_reduce.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

#define GET_OFF(field) offsetof(brgemm_kernel_reduce_args_t, field)

// Used by MatMul to implement the `reduce` attribute: reduces the source (`A`)
// tensor along the K dimension.
template <typename Vmm>
jit_brgemm_kernel_reduce_t<Vmm>::jit_brgemm_kernel_reduce_t(
        const brgemm_matmul_conf_t &bgmmc, const brgemm_desc_t &abrg)
    : jit_generator_t(jit_name())
    , brg_(abrg)
    , reduce_kind_(bgmmc.reduce_kind)
    // MatMul `src`.
    , in_dt_((bgmmc.isa == avx512_core_fp16 && bgmmc.use_buffer_a)
                      ? data_type::f32
                      : bgmmc.src_dt)
    // MatMul `reduce` buffer.
    , out_dt_(bgmmc.reduce_dt)
    , acc_dt_(bgmmc.acc_dt)
    , in_typesize_(static_cast<int>(types::data_type_size(in_dt_)))
    , out_typesize_(static_cast<int>(types::data_type_size(out_dt_)))
    , acc_typesize_(static_cast<int>(types::data_type_size(acc_dt_))) {
    // This kernel must be called after the copy A routine because it assumes
    // that fp16 data has already been upconverted to f32.
    // Only reduction for `src` is supported.
    assert(reduce_kind_ == matmul_reduce_kind::src);
    // `src` matrix is assumed to have a row major layout.
    assert(bgmmc.treat_A_as_plain || bgmmc.use_buffer_a);
}

template <typename Vmm>
Vmm jit_brgemm_kernel_reduce_t<Vmm>::vmm_mask(const Vmm vmm_in, bool mask_flag,
        bool store, Xbyak::Opmask ktail_mask) {
    return mask_flag && isa_has_masks(brg_.isa_impl)
            ? (store ? vmm_in | ktail_mask : vmm_in | ktail_mask | T_z)
            : vmm_in;
}

// Loads from the input and accumulates it into the reduction accumulator.
template <typename Vmm>
void jit_brgemm_kernel_reduce_t<Vmm>::accumulate(bool mask_flag) {

    auto vin = get_in_reg(0);
    auto vacc = get_acc_reg(0);

    auto vin_load = vmm_mask(vin, mask_flag, false, k_tail_mask);
    auto addr_in = ptr[aux_reg_in];

    if (in_dt_ == data_type::f16) {
        vpmovzxwd(vin_load, addr_in);
        vpermw(vin | k_f16_perm_mask | T_z, vreg_perm, vin);
        vcvtph2psx(vin, Vmm_lower_t(vin.getIdx()));
        vaddps(vacc, vacc, vin);
    } else if (in_dt_ == data_type::bf16) {
        vpmovzxwd(vin_load, addr_in);
        vdpbf16ps(vacc, vreg_unit, vin);
    } else if (in_dt_ == data_type::f32) {
        if (IMPLICATION(mask_flag, isa_has_masks(brg_.isa_impl)))
            vmovups(vin_load, addr_in);
        else
            vmaskmovps(vin_load, vmm_tail_mask, addr_in);
        vaddps(vacc, vacc, vin);
    } else
        assert(!"Unsupported data type");
}

template <typename Vmm>
void jit_brgemm_kernel_reduce_t<Vmm>::loop_by_K() {
    Xbyak::Label k_loop, init_zero, init_done, store_final, store_done;

    mov(aux_reg_in, reg_in);

    test(reg_flag, FLAG_REDUCE_FIRST);
    jnz(init_zero, T_NEAR);

    // Load data from the accumulator when reg_flag != FLAG_REDUCE_FIRST.
    auto vacc = get_acc_reg(0);
    auto addr_acc = ptr[reg_acc];
    uni_vmovss(vacc, addr_acc);
    jmp(init_done, T_NEAR);

    // Zero out the accumulator register.
    L(init_zero);
    uni_vxorps(vacc, vacc, vacc);
    L(init_done);

    const auto k_size = brg_.reduce_dim / brg_.ld_block;
    const auto k_tail = brg_.reduce_dim % brg_.ld_block;

    // Do reduction over K.
    if (k_size > 0) {
        mov(reg_k_iter, brg_.reduce_dim / brg_.ld_block);
        L(k_loop);
        {
            accumulate(false);
            add(aux_reg_in, in_typesize_ * brg_.ld_block);
            sub(reg_k_iter, 1);
            jnz(k_loop, T_NEAR);
        }
    }

    if (k_tail > 0) accumulate(true);

    // Do horizontal reduction.
    regops::horizontal_add_ps(this, vacc, get_workspace_reg());

    test(reg_flag, FLAG_REDUCE_LAST);
    jnz(store_final, T_NEAR);

    // Store intermediate results to accumulator.
    uni_vmovss(addr_acc, vacc);
    jmp(store_done, T_NEAR);

    L(store_final);

    // Convert and store final results.
    auto addr_out = ptr[reg_out];
    auto vacc_lower = get_acc_reg_lower(0);
    switch (out_dt_) {
        case data_type::bf16:
            vcvtneps2bf16(vacc_lower, vacc);
            vmovdqu16(addr_out, vacc | k_store_mask);
            break;
        case data_type::f16:
            vcvtps2ph(vacc_lower, vacc, 0x4);
            vmovdqu16(addr_out, vacc | k_store_mask);
            break;
        case data_type::f32: uni_vmovss(addr_out, vacc); break;
        default: assert(!"Unsupported reduce data type");
    }
    L(store_done);
}

template <typename Vmm>
void jit_brgemm_kernel_reduce_t<Vmm>::init_masks(int tail_length) {
    if (in_dt_ == data_type::f16) {
        const auto half_mask = size_t((1 << 16) - 1);
        mov(reg_mask, half_mask);
        kmovq(k_f16_perm_mask, reg_mask);

        vmovups(vreg_perm | k_f16_perm_mask | T_z, ptr[rip + f16_perm_table_]);
    }

    if (reduce_kind_ == matmul_reduce_kind::src
            && utils::one_of(out_dt_, data_type::f16, data_type::bf16)) {
        assert(isa_has_masks(brg_.isa_impl));
        mov(reg_mask, 1);
        kmovq(k_store_mask, reg_mask);
    }

    if (tail_length == 0) return;
    if (isa_has_masks(brg_.isa_impl)) {
        const auto full_mask = size_t {0xffffffffffffffff};
        const auto tail_mask = size_t((1 << tail_length) - 1);
        mov(reg_mask, full_mask);
        kmovq(k_full_mask, reg_mask);
        mov(reg_mask, tail_mask);
        kmovq(k_tail_mask, reg_mask);

    } else {
        vmovups(vmm_tail_mask, ptr[rip + mask_label_]);
    }
}

template <typename Vmm>
void jit_brgemm_kernel_reduce_t<Vmm>::generate_reduce() {

    mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]);
    mov(reg_acc, ptr[param1 + GET_OFF(ptr_acc)]);
    mov(reg_out, ptr[param1 + GET_OFF(ptr_out)]);
    mov(reg_flag, ptr[param1 + GET_OFF(flags)]);

    const int k_tail = brg_.reduce_dim % brg_.ld_block;
    init_masks(k_tail);

    for (int m = 0; m < brg_.load_dim; m++) {
        loop_by_K();
        add(reg_in, in_typesize_ * brg_.LDA);
        add(reg_out, out_typesize_);
        add(reg_acc, acc_typesize_);
    }
}

template <typename Vmm>
void jit_brgemm_kernel_reduce_t<Vmm>::generate() {
    preamble();

    if (in_dt_ == data_type::bf16) {
        auto reg_tmp = rax;
        auto reg_unit_val = reg_tmp.cvt16();
        mov(reg_unit_val, 0x3f80); // bf16 values of 1.
        vpbroadcastw(vreg_unit, reg_unit_val);
    }

    int tail = 0;

    if (reduce_kind_ == matmul_reduce_kind::src) {
        tail = brg_.reduce_dim % brg_.ld_block;
        generate_reduce();
    } else {
        assert(!"Unsupported reduce kind");
    }

    postamble();

    if (in_dt_ == data_type::f16) {
        // convert interleaved vnni data with holes to packed.
        const uint16_t f16_prm_array[16]
                = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
        align(64);
        L(f16_perm_table_);
        for (int i = 0; i < 16; ++i)
            dw(f16_prm_array[i]);
    }

    if (!isa_has_masks(brg_.isa_impl) && tail > 0) {
        align(32);
        L(mask_label_);
        for (int i = 0; i < tail; ++i)
            dd(~uint32_t(0));
        for (int i = tail; i < brg_.ld_block; ++i)
            dd(0);
    }
}

#undef GET_OFF

template struct jit_brgemm_kernel_reduce_t<Xbyak::Ymm>;
template struct jit_brgemm_kernel_reduce_t<Xbyak::Zmm>;

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
