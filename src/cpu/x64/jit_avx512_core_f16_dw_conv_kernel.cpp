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

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx512_core_f16_dw_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_args_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace Xbyak;

jit_avx512_dw_conv_fwd_kernel_f16_t::jit_avx512_dw_conv_fwd_kernel_f16_t(
        const jit_conv_conf_t &ajcp, const memory_desc_t &dst_md)
    : jit_generator_t(jit_name()), jcp(ajcp) {
    const auto simd_w = cpu_isa_traits_t<avx512_core>::vlen / sizeof(float);
    const auto tail_size = jcp.oc_without_padding % simd_w;
    if (jcp.with_eltwise || jcp.with_binary) {
        using namespace binary_injector;
        static constexpr auto preserve_gpr = true;
        static constexpr auto preserve_zmm = false;
        static constexpr auto helper_zmm_idx = 31;
        static constexpr auto use_exact_tail_scalar_bcast = true;
        rhs_arg_static_params_t rhs_arg_static_params {helper_zmm_idx, r14, r15,
                r12, preserve_gpr, preserve_zmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(dst_orig),
                memory_desc_wrapper(dst_md), tail_size, k_oc_tail_mask,
                use_exact_tail_scalar_bcast};
        static_params_t static_params {this->param1, rhs_arg_static_params};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx512_core>>(
                this, jcp.post_ops, static_params);
    }

    const io::jit_io_multi_dt_helper_t<Xbyak::Zmm>::data_types_t data_types {
            jcp.src_dt, jcp.dst_dt};
    io_ = utils::make_unique<io::jit_io_multi_dt_helper_t<Xbyak::Zmm>>(this,
            jcp.isa, data_types, io::io_conf_t {},
            io::io_tail_conf_t {simd_w, tail_size, k_oc_tail_mask, 0, reg_tmp});
}

static bool check_if_tail(const bool is_ch_tail, const int c_tail, const int ch,
        const int ur_ch_blocks, const int simd_w) {
    return is_ch_tail && (ch + 1 == ur_ch_blocks) && simd_w > c_tail;
}

void jit_avx512_dw_conv_fwd_kernel_f16_t::load_src(
        int ur_ch_blocks, int ur_w, bool is_ch_tail) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
    const int simd_w = cpu_isa_traits_t<avx512_core>::vlen / sizeof(float);
    const int c_tail = jcp.oc % jcp.ch_block;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        const bool is_tail_load
                = check_if_tail(is_ch_tail, c_tail, ch, ur_ch_blocks, simd_w);
        if ((ch + 1 == ur_ch_blocks) && is_ch_tail && c_tail <= 0) continue;
        for (int ow = 0; ow < ur_w; ow++) {
            Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);

            if (jcp.with_bias) {
                const Zmm zmm_acc_msk = is_tail_load
                        ? zmm_acc | k_oc_tail_mask | T_z
                        : zmm_acc;
                const int b_off = ch * ch_blk;
                uni_vmovups(zmm_acc_msk, ptr[reg_bias + b_off * sizeof(float)]);
            } else {
                uni_vpxor(zmm_acc, zmm_acc, zmm_acc);
            }

            if (jcp.with_sum) {
                const int o_off = ch * ocb_stride + ow * ow_stride;
                // using ker_zmm as zmm_tmp as it is safe to do so.
                auto zmm_tmp = get_ker_reg(0);
                io_->at(jcp.src_dt)
                        ->load(ptr[reg_output + o_off * jcp.typesize_in],
                                zmm_tmp, is_tail_load);
                uni_vaddps(zmm_acc, zmm_acc, zmm_tmp);
            }
        }
    }
}

void jit_avx512_dw_conv_fwd_kernel_f16_t::apply_filter_unrolled(
        int ur_ch_blocks, int ur_w, int pad_l, int pad_r, bool is_ch_tail) {
    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto iw_stride = src_layout_nxc ? jcp.ngroups : ch_blk;
    const auto ih_stride = jcp.iw * iw_stride;
    const auto icb_stride = src_layout_nxc
            ? ch_blk
            : (jcp.is_fused_conv ? 1 : jcp.ih) * jcp.iw * ch_blk;
    const int simd_w = cpu_isa_traits_t<avx512_core>::vlen / sizeof(float);

    auto get_input_spatial_index = [=](int oi, int ki) {
        return (ki * dilate_w + oi * stride_w - pad_l);
    };

    auto get_input_offset = [&](int ii, int ci) {
        return (ci * icb_stride + ii * iw_stride) * jcp.typesize_in;
    };

    int ii_start = 0;
    int ii_end = -1;
    if (jcp.is_resrc_depthwise) {
        // find bounds of input spatial indices
        bool first = true;
        for (int ki = 0; ki < jcp.kw; ki++) {
            int oi_start = get_ow_start(ki, pad_l);
            int oi_end = get_ow_end(ur_w, ki, pad_r);
            for (int oi = oi_start; oi < oi_end; oi++) {
                int ii = get_input_spatial_index(oi, ki);
                if (first || ii < ii_start) ii_start = ii;
                if (first || ii > ii_end) ii_end = ii;
                first = false;
            }
        }
    }

    Label iter_exit_label;

    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label);
    {
        if (jcp.is_fused_conv) {
            mov(aux_reg_input, ptr[aux_reg_input_buffer_ptr]);
            add(aux_reg_input, reg_iw_offset);
        }
        const int c_tail = jcp.oc % jcp.ch_block;
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            const bool is_tail_load = check_if_tail(
                    is_ch_tail, c_tail, ch, ur_ch_blocks, simd_w);
            if ((ch + 1 == ur_ch_blocks) && is_ch_tail && c_tail <= 0) continue;
            if (jcp.is_resrc_depthwise) {
                // now we can load input once and reuse up to jcp.kw times
                for (int ii = ii_start; ii <= ii_end; ii++) {
                    Zmm zmm_src = get_src_reg(ii);
                    const int inp_off = get_input_offset(ii, ch);
                    io_->at(jcp.src_dt)
                            ->load(ptr[aux_reg_input + inp_off], zmm_src,
                                    is_tail_load);
                }
            }
            for (int kw = 0; kw < jcp.kw; kw++) {
                const int ker_off = ch * jcp.kh * jcp.kw * ch_blk + kw * ch_blk;

                Zmm zmm_ker = get_ker_reg(0);
                io_->at(jcp.src_dt)
                        ->load(ptr[aux_reg_kernel + ker_off * jcp.typesize_in],
                                zmm_ker, is_tail_load);
                int ow_start = get_ow_start(kw, pad_l);
                int ow_end = get_ow_end(ur_w, kw, pad_r);
                for (int ow = ow_start; ow < ow_end; ow++) {

                    const int ii = get_input_spatial_index(ow, kw);
                    Zmm zmm_src = jcp.is_resrc_depthwise ? get_src_reg(ii)
                                                         : get_src_reg(0);
                    if (!jcp.is_resrc_depthwise) {
                        const int inp_off = get_input_offset(ii, ch);
                        io_->at(jcp.src_dt)
                                ->load(ptr[aux_reg_input + inp_off], zmm_src,
                                        is_tail_load);
                    }
                    Zmm zmm_acc = get_acc_reg(ch * ur_w + ow);
                    const Zmm zmm_ker_msk = is_tail_load
                            ? zmm_ker | k_oc_tail_mask | T_z
                            : zmm_ker;
                    uni_vfmadd231ps(zmm_acc, zmm_src, zmm_ker_msk);
                }
            }
        }

        add(aux_reg_kernel, jcp.kw * ch_blk * jcp.typesize_in);
        if (jcp.is_fused_conv) {
            // Move to next row pointer in the buffer
            add(aux_reg_input_buffer_ptr, sizeof(void *));
        } else {
            add(aux_reg_input, ih_stride * dilate_h * jcp.typesize_in);
        }

        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }

    L(iter_exit_label);
}

template <typename F>
void iterate(const int ur_ch_blocks, const int ur_w, const bool mask_tail,
        const F &f) {
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        const bool mask_flag = mask_tail && ch + 1 == ur_ch_blocks;
        for (int ow = 0; ow < ur_w; ow++)
            f(ch, ow, mask_flag);
    }
}

template <typename F>
void iterate(const int ur_ch_blocks, const int ur_w, const F &f) {
    iterate(ur_ch_blocks, ur_w, false, f);
}

void jit_avx512_dw_conv_fwd_kernel_f16_t::apply_postops(
        const int ur_ch_blocks, const int ur_w, const bool is_ch_tail) {
    if (this->jcp.with_eltwise || this->jcp.with_binary) {
        injector_utils::vmm_index_set_t zmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params,
                    rhs_arg_params_tail;
            const auto dst_layout_nxc = is_dst_layout_nxc();
            const auto ch_blk = jcp.ch_block;
            const auto ocb_stride
                    = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
            const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
            const auto mask_tail_blocked_layout
                    = jcp.oc_without_padding % jcp.ch_block && !dst_layout_nxc;
            const int c_tail = jcp.oc_without_padding % jcp.ch_block;
            iterate(ur_ch_blocks, ur_w, mask_tail_blocked_layout,
                    [&](const int ch, const int ow,
                            const bool mask_flag_blocked_layout) {
                const int simd_w
                        = cpu_isa_traits_t<avx512_core>::vlen / sizeof(float);
                const bool is_tail_load = check_if_tail(
                        is_ch_tail, c_tail, ch, ur_ch_blocks, simd_w);
                if ((ch + 1 == ur_ch_blocks) && is_ch_tail && c_tail <= 0)
                    return;
                const size_t o_off
                        = jcp.typesize_out * (ch * ocb_stride + ow * ow_stride);
                const auto zmm_idx = get_acc_reg_idx(ch * ur_w + ow);
                zmm_idxs.emplace(zmm_idx);

                rhs_arg_params_tail.vmm_idx_to_out_reg.emplace(
                        zmm_idx, reg_output);
                rhs_arg_params_tail.vmm_idx_to_out_elem_off_val.emplace(
                        zmm_idx, o_off);
                if (mask_flag_blocked_layout || is_tail_load)
                    rhs_arg_params_tail.vmm_tail_idx_.emplace(zmm_idx);
            });
            rhs_arg_params = rhs_arg_params_tail;
            rhs_arg_params.vmm_tail_idx_.clear();

            Label postops_done;
            if (mask_tail_blocked_layout) {
                // mask_tail_blocked_layout approach of dynamic tail handling is
                // used in blocked layout only. TODO: may be unify?
                Label postops_no_tail;
                mov(reg_tmp, ptr[param1 + GET_OFF(load_work)]);
                cmp(reg_tmp, jcp.nb_ch_blocking * jcp.ch_block);
                jge(postops_no_tail, T_NEAR);
                postops_injector_->compute_vector_range(
                        zmm_idxs, rhs_arg_params_tail);
                jmp(postops_done, T_NEAR);
                L(postops_no_tail);
            } else if (is_ch_tail) {
                postops_injector_->compute_vector_range(
                        zmm_idxs, rhs_arg_params_tail);
            }
            if (!is_ch_tail) {
                postops_injector_->compute_vector_range(
                        zmm_idxs, rhs_arg_params);
                L(postops_done);
            }
        } else {
            iterate(ur_ch_blocks, ur_w,
                    [&](const int ch, const int ow, const bool) {
                zmm_idxs.emplace(get_acc_reg_idx(ch * ur_w + ow));
            });
            postops_injector_->compute_vector_range(zmm_idxs);
        }
    }
}

void jit_avx512_dw_conv_fwd_kernel_f16_t::store_dst(
        int ur_ch_blocks, int ur_w, bool is_ch_tail) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
    const int simd_w = cpu_isa_traits_t<avx512_core>::vlen / sizeof(float);
    const int c_tail = jcp.oc_without_padding % jcp.ch_block;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        const bool is_tail_load
                = check_if_tail(is_ch_tail, c_tail, ch, ur_ch_blocks, simd_w);
        if ((ch + 1 == ur_ch_blocks) && is_ch_tail && c_tail <= 0) continue;
        for (int ow = 0; ow < ur_w; ow++) {
            const int o_off = ch * ocb_stride + ow * ow_stride;
            Zmm zmm_dst = get_acc_reg(ch * ur_w + ow);
            io_->at(jcp.dst_dt)
                    ->store(zmm_dst, ptr[reg_output + o_off * jcp.typesize_out],
                            is_tail_load);
        }
    }
}

void jit_avx512_dw_conv_fwd_kernel_f16_t::compute_loop(
        int ur_w, int ur_ch_blocks, int pad_l, int pad_r) {

    const bool ch_loop = ur_ch_blocks > jcp.nb_ch_blocking;
    // ch_loop currently happen only when data layout is nxc. The strides are
    // calculated for this layout only.
    const size_t wei_ch_stride = (size_t)jcp.nb_ch_blocking * jcp.kh * jcp.kw
            * jcp.ch_block * jcp.typesize_in;
    const size_t inp_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_in;
    const size_t out_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_out;
    const size_t bias_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);

    auto compute = [&](int ur_ch_blocks, bool is_ch_tail) {
        if (jcp.is_fused_conv) {
            mov(aux_reg_input_buffer_ptr, reg_input_buffer_ptr);
        } else {
            mov(aux_reg_input, reg_input);
        }

        mov(aux_reg_kernel, reg_kernel);
        load_src(ur_ch_blocks, ur_w, is_ch_tail);
        apply_filter_unrolled(ur_ch_blocks, ur_w, pad_l, pad_r, is_ch_tail);
        apply_postops(ur_ch_blocks, ur_w, is_ch_tail);
        store_dst(ur_ch_blocks, ur_w, is_ch_tail);
    };

    mov(aux_reg_ch_blocks, reg_ch_blocks);
    if (ch_loop) {
        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int ch_block_tail = jcp.nb_ch
                - (utils::rnd_dn(jcp.oc / jcp.ch_block, jcp.nb_ch_blocking));
        const int ch_step = jcp.nb_ch_blocking * jcp.ch_block;

        push(reg_kernel);
        push(reg_input);
        push(reg_output);
        if (jcp.with_bias) push(reg_bias);

        if ((jcp.oc / jcp.ch_block) >= jcp.nb_ch_blocking) {
            if (ch_block_tail) {
                cmp(aux_reg_ch_blocks, ch_step);
                jl(ch_tail_label, T_NEAR);
            }

            L(ch_loop_label);
            {
                compute(jcp.nb_ch_blocking, false);
                add(reg_kernel, wei_ch_stride);
                add(reg_input, inp_ch_stride);
                add(reg_output, out_ch_stride);
                if (jcp.with_bias) add(reg_bias, bias_stride);
                sub(aux_reg_ch_blocks, ch_step);
                cmp(aux_reg_ch_blocks, ch_step);
                jge(ch_loop_label, T_NEAR);
            }
        }

        if (ch_block_tail) {
            // ch work range [1, jcp.nb_ch_blocking * ch_block)
            L(ch_tail_label);
            cmp(aux_reg_ch_blocks, 0);
            jle(skip_ch_tail_label, T_NEAR);
            compute(ch_block_tail, jcp.oc % jcp.ch_block);
            L(skip_ch_tail_label);
        }

        if (jcp.with_bias) pop(reg_bias);
        pop(reg_output);
        pop(reg_input);
        pop(reg_kernel);

    } else {
        compute(ur_ch_blocks, jcp.oc % jcp.ch_block);
    }
}

void jit_avx512_dw_conv_fwd_kernel_f16_t::ow_loop(int ur_ch_blocks) {

    int iw = jcp.iw;
    int ow = jcp.ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto dat_c_stride = src_layout_nxc ? jcp.ngroups : jcp.ch_block;
    size_t inp_shift = (size_t)jcp.typesize_in * ur_w * stride_w * dat_c_stride;
    size_t out_shift = (size_t)jcp.typesize_out * ur_w * dat_c_stride;

    int inp_shift_pad
            = jcp.typesize_in * (ur_w * stride_w - l_pad) * dat_c_stride;

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    assert(jcp.nb_ow <= 1);

    if (r_pad1 > 0) n_oi--;
    xor_(reg_oi, reg_oi);
    if (ow == ur_w) {
        compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad);
    } else {
        if (n_oi == 0) {
            compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad1);
            add(reg_input, inp_shift_pad);
            add(reg_output, out_shift);
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        } else {
            if (l_pad > 0) {
                compute_loop(ur_w, ur_ch_blocks, l_pad, 0);
                add(reg_input, inp_shift_pad);
                add(reg_output, out_shift);
                inc(reg_oi);
            }
            if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                Label ow_loop_label;
                L(ow_loop_label);
                {
                    compute_loop(ur_w, ur_ch_blocks, 0, 0);
                    add(reg_input, inp_shift);
                    add(reg_output, out_shift);

                    inc(reg_oi);
                    cmp(reg_oi, n_oi);
                    jl(ow_loop_label, T_NEAR);
                }
            }
            if (r_pad1 > 0) {
                compute_loop(ur_w, ur_ch_blocks, 0, r_pad1);
                add(reg_input, inp_shift);
                add(reg_output, out_shift);
            }
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        }
    }
}

void jit_avx512_dw_conv_fwd_kernel_f16_t::generate() {
    this->preamble();

    if (jcp.is_fused_conv) {
        mov(reg_input_buffer_ptr, ptr[this->param1 + GET_OFF(src)]);
        /* In case of fused depthwise convolution, `param.src` is not a pointer
        to input, instead it points to a buffer containing pointers to
        consecutive rows of input in format Cwc with blocking nb_ch_blocking.
        Example: [ptr_to_inp_row0, ptr_to_inp_row1, ptr_to_inp_row2].
        Traverse the data as
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row0 ...
            add(reg_input_buffer_ptr, sizeof(void*))
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row1 ...
            add(reg_input_buffer_ptr, sizeof(void*))
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row2 ...
        */
        xor_(reg_iw_offset, reg_iw_offset);
    } else {
        mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    }
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias) mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(load_work)]);

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    const auto oc_tail = jcp.oc_without_padding % jcp.ch_block;
    if (oc_tail != 0) {
        // Prepare masks for tailing
        // Not using io_->prepare_tail_mask() since mask needs shifting
        const int oc_tail_shift
                = jcp.ch_block - jcp.oc_without_padding % jcp.ch_block;
        static constexpr auto zmm_full_mask = ((1 << 16) - 1);
        Reg32 reg_tail_32 = reg_tail.cvt32();
        mov(reg_tail_32, (zmm_full_mask >> oc_tail_shift));
        kmovw(k_oc_tail_mask, reg_tail_32);
    }

    if (is_src_layout_nxc()) {
        ow_loop(jcp.nb_ch);
    } else {
        cmp(reg_ch_blocks, (jcp.nb_ch_blocking - 1) * jcp.ch_block);
        jle(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);

        ow_loop(jcp.nb_ch_blocking); // channel main loop

        if (ch_blocks_tail) {
            jmp(exit_label, T_NEAR);
            L(ch_blocks_tail_label);
            ow_loop(ch_blocks_tail); // channel tail loop
        }

        L(exit_label);
    }

    this->postamble();

    if (jcp.with_eltwise)
        postops_injector_->prepare_table(/* generate = */ true);
}

#undef GET_OFF

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
