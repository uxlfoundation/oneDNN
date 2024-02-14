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

#ifndef CPU_X64_JIT_AVX512_CORE_F16_DW_CONV_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_F16_DW_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_dw_conv_fwd_kernel_f16_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_dw_conv_fwd_kernel_f16_t)

    jit_avx512_dw_conv_fwd_kernel_f16_t(
            const jit_conv_conf_t &ajcp, const memory_desc_t &dst_md);

    jit_conv_conf_t jcp;

private:
    using reg64_t = const Xbyak::Reg64;
    using mask_t = const Xbyak::Opmask;

    // dw convolution
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t reg_kernel = r10;
    reg64_t aux_reg_kernel = r11;
    reg64_t reg_ch_blocks = r12;
    reg64_t reg_output = r13;
    reg64_t reg_bias = r14;
    reg64_t reg_kh = r15;
    reg64_t iter_kh = rax;
    reg64_t reg_oi = rbx;
    reg64_t aux_reg_ch_blocks = rsi;
    // fused convolution
    reg64_t reg_input_buffer_ptr = rdx;
    reg64_t aux_reg_input_buffer_ptr = rbp;
    reg64_t reg_iw_offset = reg_input; //Hack: clear reg_input early in kernel

    reg64_t reg_tmp = reg_ch_blocks;
    reg64_t reg_tail = rax;
    mask_t k_oc_tail_mask = Xbyak::Opmask(2);

    inline void load_src(int ur_ch_blocks, int ur_w, bool is_ch_tail);
    inline void compute_loop(int ur_w, int ur_ch_blocks, int pad_l, int pad_r);
    inline void ow_loop(int ur_ch_blocks);
    inline void apply_filter_unrolled(
            int ur_ch_blocks, int ur_w, int pad_l, int pad_r, bool is_ch_tail);
    inline void apply_postops(
            const int ur_ch_blocks, const int ur_w, const bool is_ch_tail);
    inline void store_dst(int ur_ch_blocks, int ur_w, bool is_ch_tail);

    inline Xbyak::Zmm get_ker_reg(int idx) { return Xbyak::Zmm(idx + 0); }
    inline Xbyak::Zmm get_src_reg(int idx) { return Xbyak::Zmm(idx + 1); }
    inline int get_acc_reg_idx(int idx) const {
        const int max_regs = 32;
        return idx + (max_regs - jcp.ur_w * jcp.nb_ch_blocking);
    }
    inline Xbyak::Zmm get_acc_reg(int idx) const {
        return Xbyak::Zmm(get_acc_reg_idx(idx));
    }

    int get_ow_start(int ki, int pad_l) const {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }

    int get_ow_end(int ur_w, int ki, int pad_r) const {
        return ur_w
                - nstl::max(0,
                        utils::div_up(
                                pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1),
                                jcp.stride_w));
    }

    inline bool is_src_layout_nxc() const {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_dst_layout_nxc() const {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }

    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
            postops_injector_;
    std::unique_ptr<io::jit_io_multi_dt_helper_t<Xbyak::Zmm>> io_;

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
