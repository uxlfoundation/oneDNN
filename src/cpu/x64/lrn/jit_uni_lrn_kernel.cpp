/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include <array>
#include <cmath>
#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/lrn/jit_uni_lrn_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;

#define IRB_LOOP(statement) \
    for (int irb = 0; irb < reg_block; irb++) { \
        const int irb_off = irb * this->single_pixel_offset_; \
        statement; \
        MAYBE_UNUSED(irb_off); \
    }

using namespace Xbyak;

cpu_isa_t get_io_isa(cpu_isa_t isa, data_type_t d_type) {
    // re-using avx512_core instantiation for bf16
    return isa == avx512_core && d_type == data_type::bf16
                    && mayiuse(avx512_core_bf16)
            ? avx512_core_bf16
            : isa;
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_kernel_t<Derived<isa, d_type>>::jit_uni_lrn_kernel_t(
        const char *name)
    : jit_generator_t(name, isa)
    , emulate_bfloat_(d_type == data_type::bf16 && !mayiuse(avx512_core_bf16)
              && is_superset(isa, avx512_core))
    , bf16_emu_(emulate_bfloat_
                      ? utils::make_unique<bf16_emulation_t>(this,
                                bf16_emu_reserv_1_, bf16_emu_reserv_2_,
                                bf16_emu_reserv_3_, bf16_emu_scratch_,
                                bf16_emu_reserv_4_)
                      : nullptr)
    , io_(this, get_io_isa(isa, d_type), {d_type}, {},
              io::io_tail_conf_t {simd_w_, 0, this->k1, 0, this->reg_tmp_},
              io::io_emu_bf16_conf_t {bf16_emu_reserv_1_, bf16_emu_reserv_2_,
                      bf16_emu_reserv_3_, bf16_emu_scratch_,
                      bf16_emu_reserv_4_}) {}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_kernel_t<Derived<isa, d_type>>::jit_uni_lrn_kernel_t(
        const within_config_t &config, const char *name)
    : jit_uni_lrn_kernel_t(name) {
    if (config.dat_tag == nhwc)
        single_pixel_offset_
                = config.C * sizeof(typename prec_traits_t<d_type>::type);
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_kernel_t<Derived<isa, d_type>>::~jit_uni_lrn_kernel_t() = default;

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_kernel_t<Derived<isa, d_type>>::within_loop(
        const within_config_t &config, int max_reg_blocks, prop_kind_t pk) {
    const auto derived_ptr = static_cast<Derived<isa, d_type> *>(this);

    const int lower_bound = (config.size - 1) / 2,
              upper_bound = config.size - lower_bound - 1;

    int pixel_count = 0;

    for (int i = 0; i < lower_bound; ++i) {
        pixel_count = 0;
        for (int j = 0; j < lower_bound; ++j)
            derived_ptr->within_body(-i, upper_bound, -j, upper_bound, config.W,
                    pk, 1, pixel_count++ * this->single_pixel_offset_);
        derived_ptr->move_data_pointers(pixel_count, pk);

        within_body_reg_blocked(config.W - config.size + 1, max_reg_blocks, -i,
                upper_bound, -lower_bound, upper_bound, config.W, pk);

        pixel_count = 0;
        for (int j = config.W - upper_bound; j < config.W; ++j)
            derived_ptr->within_body(-i, upper_bound, -lower_bound,
                    config.W - 1 - j, config.W, pk, 1,
                    pixel_count++ * this->single_pixel_offset_);
        derived_ptr->move_data_pointers(pixel_count, pk);
    }

    this->mov(h_, config.H - config.size + 1);
    Label lrn_loop_h;
    this->L(lrn_loop_h);
    pixel_count = 0;
    for (int j = 0; j < lower_bound; ++j)
        derived_ptr->within_body(-lower_bound, upper_bound, -j, upper_bound,
                config.W, pk, 1, pixel_count++ * this->single_pixel_offset_);
    derived_ptr->move_data_pointers(pixel_count, pk);

    within_body_reg_blocked(config.W - config.size + 1, max_reg_blocks,
            -lower_bound, upper_bound, -lower_bound, upper_bound, config.W, pk);

    pixel_count = 0;
    for (int j = config.W - upper_bound; j < config.W; ++j)
        derived_ptr->within_body(-lower_bound, upper_bound, -lower_bound,
                config.W - 1 - j, config.W, pk, 1,
                pixel_count++ * this->single_pixel_offset_);
    derived_ptr->move_data_pointers(pixel_count, pk);

    this->dec(h_);
    this->cmp(h_, 0);
    this->jne(lrn_loop_h, T_NEAR);

    for (int i = config.H - upper_bound; i < config.H; ++i) {
        pixel_count = 0;
        for (int j = 0; j < lower_bound; ++j)
            derived_ptr->within_body(-lower_bound, config.H - 1 - i, -j,
                    upper_bound, config.W, pk, 1,
                    pixel_count++ * this->single_pixel_offset_);
        derived_ptr->move_data_pointers(pixel_count, pk);

        within_body_reg_blocked(config.W - config.size + 1, max_reg_blocks,
                -lower_bound, config.H - 1 - i, -lower_bound, upper_bound,
                config.W, pk);

        pixel_count = 0;
        for (int j = config.W - upper_bound; j < config.W; ++j)
            derived_ptr->within_body(-lower_bound, config.H - 1 - i,
                    -lower_bound, config.W - 1 - j, config.W, pk, 1,
                    pixel_count++ * this->single_pixel_offset_);
        derived_ptr->move_data_pointers(pixel_count, pk);
    }
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_kernel_t<Derived<isa, d_type>>::within_body_reg_blocked(
        int loop_count, int max_reg_blocks, int hoff, int Hoff, int woff,
        int Woff, int stride, prop_kind_t pk) {

    const auto derived_ptr = static_cast<Derived<isa, d_type> *>(this);
    Label reg_block_compute_loop;

    const auto res = std::div(loop_count, max_reg_blocks);
    if (res.quot) {
        this->mov(this->w_, res.quot);
        this->L(reg_block_compute_loop);
        derived_ptr->within_body(
                hoff, Hoff, woff, Woff, stride, pk, max_reg_blocks, 0);
        derived_ptr->move_data_pointers(max_reg_blocks, pk);
        this->dec(this->w_);
        this->cmp(this->w_, 0);
        this->jne(reg_block_compute_loop, T_NEAR);
    }
    if (res.rem) {
        derived_ptr->within_body(
                hoff, Hoff, woff, Woff, stride, pk, res.rem, 0);
        derived_ptr->move_data_pointers(res.rem, pk);
    }
}

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_kernel_t<Derived<isa, d_type>>::load_constant(
        float constant, const Vmm &v_constant, const Xbyak::Xmm &x_constant) {
    this->mov(this->imm_addr64_, float2int(constant));
    this->uni_vmovq(x_constant, this->imm_addr64_);
    this->vbroadcastss(v_constant, x_constant);
}

//////////////////////////////////////////////////////////////////////////////
// forward kernel
template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::within_body(int hoff, int Hoff,
        int woff, int Woff, int stride, prop_kind_t pk, const int reg_block,
        int pixel_offset) {

    static const std::array<Vmm, 5> vsum {
            {Vmm(2), Vmm(7), Vmm(12), Vmm(17), Vmm(22)}};
    static const std::array<Vmm, 5> vsum2 {
            {Vmm(3), Vmm(8), Vmm(13), Vmm(18), Vmm(23)}};
    static const std::array<Vmm, 5> vdst {
            {Vmm(4), Vmm(9), Vmm(14), Vmm(19), Vmm(24)}};
    static const std::array<Vmm, 5> vtmp {
            {Vmm(5), Vmm(10), Vmm(15), Vmm(20), Vmm(25)}};
    static const std::array<Vmm, 5> vscratch {
            {Vmm(6), Vmm(11), Vmm(16), Vmm(21), Vmm(26)}};
    static const std::array<Vmm, 2> vtmp2 {{Vmm(12), Vmm(13)}};
    static const Vmm vaux = Vmm(14);
    const bool has_ne_convert_xf16_support = isa == avx2_vnni_2
            && utils::one_of(d_type, data_type::bf16, data_type::f16);

    IRB_LOOP(this->uni_vxorps(vsum[irb], vsum[irb], vsum[irb]));
    for (int i = hoff; i <= Hoff; ++i) {
        for (int j = woff; j <= Woff; ++j) {
            const auto p_off = pixel_offset
                    + (i * stride + j) * this->single_pixel_offset_;
            const bool can_load_two_simdw = has_ne_convert_xf16_support
                    && Woff - j >= 2 && !(i == 0 && (j == 0 || j + 1 == 0));
            if (can_load_two_simdw) {
                IRB_LOOP(this->io_.at(d_type)->load_two_simdw_xf16(
                        this->ptr[src_ + p_off + irb_off], vtmp[irb],
                        vtmp2[irb]));
                IRB_LOOP(this->io_.at(d_type)->merge_interleaved_to_plain(
                        vtmp[irb], vtmp2[irb], vaux));
                IRB_LOOP(this->vfmadd231ps(vsum[irb], vtmp[irb], vtmp[irb]));
                IRB_LOOP(this->vfmadd231ps(vsum[irb], vtmp2[irb], vtmp2[irb]));
                ++j;
            } else {
                const auto vdata = (i == 0 && j == 0) ? vdst : vtmp;
                IRB_LOOP(this->io_.at(d_type)->load(
                        this->ptr[(src_ + p_off + irb_off)], vdata[irb],
                        false));
                IRB_LOOP(this->vfmadd231ps(vsum[irb], vdata[irb], vdata[irb]));
            }
        }
    }

    IRB_LOOP(this->vfmadd132ps(
            vsum[irb], vk_, valpha_)); // ysum <- ysum*valpha_+yk_
    IRB_LOOP(this->vmovaps(vscratch[irb], vsum[irb]));

    IRB_LOOP(this->vmulps(vsum2[irb], vsum[irb], vsum[irb]));
    IRB_LOOP(this->vmulps(
            vsum[irb], vsum[irb], vsum2[irb])); // ysum = (ysum*valpha_+yk_)^3;
    IRB_LOOP(this->vsqrtps(vsum[irb], vsum[irb]));
    IRB_LOOP(this->vsqrtps(
            vsum[irb], vsum[irb])); // ysum = (ysum*valpha_+yk_)^0.75
    IRB_LOOP(this->vdivps(
            vdst[irb], vdst[irb], vsum[irb])); // ydst <- ydst / ysum

    if (pk_ != prop_kind::forward_inference) {
        IRB_LOOP(this->io_.at(d_type)->store(vsum[irb],
                this->ptr[scratch_ + pixel_offset + irb_off], false));
        IRB_LOOP(this->vdivps(vscratch[irb], vdst[irb], vscratch[irb]));
        IRB_LOOP(this->io_.at(d_type)->store(vscratch[irb],
                this->ptr[bwd_intermediate_res_ + pixel_offset + irb_off],
                false));
    }

    IRB_LOOP(this->io_.at(d_type)->store(
            vdst[irb], this->ptr[dst_ + pixel_offset + irb_off], false));
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::move_data_pointers(
        int pixel_count, prop_kind_t pk) {

    const int pixel_offset = this->single_pixel_offset_ * pixel_count;
    this->add(src_, pixel_offset);
    this->add(dst_, pixel_offset);
    if (pk_ != prop_kind::forward_inference) {
        this->add(scratch_, pixel_offset);
        this->add(bwd_intermediate_res_, pixel_offset);
    }
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::jit_uni_lrn_fwd_kernel_t(
        const within_config_t &config, float A, float K, prop_kind_t pk)
    : Base(config, jit_name())
    , config_(lrn_config_t::within_config)
    , within_config_(config)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::generate(
        const within_config_t &config) {
    this->preamble();
    if (this->emulate_bfloat_) this->io_.init_bf16();

#define GET_OFF(field) offsetof(jit_args_fwd_t, field)
    this->mov(src_, this->ptr[this->param1 + GET_OFF(src)]);
    this->mov(dst_, this->ptr[this->param1 + GET_OFF(dst)]);
    if (pk_ != prop_kind::forward_inference) {
        this->mov(scratch_, this->ptr[this->param1 + GET_OFF(scratch)]);
        this->mov(bwd_intermediate_res_,
                this->ptr[this->param1 + GET_OFF(bwd_intermediate_res)]);
    }
#undef GET_OFF

    this->load_constant(alpha_, valpha_, xalpha_);
    this->load_constant(k_, vk_, xk_);

    static const int max_reg_blocks = is_superset(isa, avx512_core) ? 5
            : is_superset(isa, avx2)                                ? 2
                                                                    : 1;
    this->within_loop(config, max_reg_blocks, pk_);

    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::jit_uni_lrn_fwd_kernel_t(
        const struct nchw8c_across_t &J, float A, float K, prop_kind_t pk)
    : Base(jit_name())
    , config_(lrn_config_t::nchw8c_across)
    , nchw8c_across_(J)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::generate(const nchw8c_across_t &J) {
    const Xbyak::Reg64 &t = this->rsp;
    const Xbyak::Reg64 &hw = this->r9;
    const Xbyak::Xmm &xsrc_prev = this->xmm2;
    const Xbyak::Ymm &ysrc = this->ymm3;
    const Xbyak::Ymm &yc = this->ymm3;
    const Xbyak::Xmm &xsrc_next = this->xmm4;
    const Xbyak::Ymm &ya = this->ymm5;
    const Xbyak::Ymm &yb = this->ymm6;
    const Xbyak::Ymm &yd = this->ymm7;
    const Xbyak::Ymm &ye = this->ymm8;
    const Xbyak::Ymm &ysum = this->ymm9;
    const Xbyak::Ymm &ysum2 = this->ymm10;
    const Xbyak::Ymm &ydst = this->ymm11;
    const Xbyak::Ymm &ybase = this->ymm12;

    this->preamble();
    if (this->emulate_bfloat_) this->io_.init_bf16();

    this->mov(src_, this->ptr[this->param1 + 0]);
    this->mov(dst_, this->ptr[this->param1 + 8]);
    if (pk_ != prop_kind::forward_inference)
        this->mov(scratch_, this->ptr[this->param1 + 16]);
    this->sub(t, 64);
    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->vmovq(xalpha_, this->imm_addr64_);
    this->vbroadcastss(valpha_, xalpha_);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->vmovq(xk_, this->imm_addr64_);
    this->vbroadcastss(yk_, xk_);

    if (J.version == -1) {
        this->vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
        this->vmovups(this->ptr[t + 0], xsrc_prev);
    }
    if (J.version == +1) {
        this->vxorps(xsrc_next, xsrc_next, xsrc_next);
        this->vmovups(this->ptr[t + 48], xsrc_next);
    }

    this->mov(hw, J.H * J.W);

    Label lrn_loop;
    this->L(lrn_loop);

    if (J.version != -1)
        this->vmovups(xsrc_prev, this->ptr[src_ - J.H * J.W * 32 + 16]);
    this->vmovups(ysrc, this->ptr[src_]);
    if (J.version != +1)
        this->vmovups(xsrc_next, this->ptr[src_ + J.H * J.W * 32]);

    if (J.version != -1) this->vmovups(this->ptr[t + 0], xsrc_prev);
    this->vmovups(this->ptr[t + 16], ysrc);
    if (J.version != +1) this->vmovups(this->ptr[t + 48], xsrc_next);

    this->vmovups(ya, this->ptr[t + 16 - 8]);
    this->vmovups(yb, this->ptr[t + 16 - 4]);
    this->vmovups(yd, this->ptr[t + 16 + 4]);
    this->vmovups(ye, this->ptr[t + 16 + 8]);
    this->vmulps(ysum, yc, yc);
    this->vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya*ya
    this->vfmadd231ps(ysum, yb, yb);
    this->vfmadd231ps(ysum, yd, yd);
    this->vfmadd231ps(ysum, ye, ye);
    this->vfmadd132ps(ysum, yk_, valpha_); // ysum <- ysum*valpha_+yk_

    this->vmovaps(ybase, ysum);
    if (pk_ != prop_kind::forward_inference)
        this->vmovups(this->ptr[scratch_], ybase);
    this->vmulps(ysum2, ysum, ysum);
    this->vmulps(ysum, ysum, ysum2); // ysum = ybase^3;
    this->vsqrtps(ysum, ysum);
    this->vsqrtps(ysum, ysum); // ysum = ybase^0.75
    this->vdivps(ydst, ysrc, ysum); // ydst = ysrc / ysum
    this->vmovups(this->ptr[dst_], ydst);

    this->add(src_, 32);
    this->add(dst_, 32);
    if (pk_ != prop_kind::forward_inference) this->add(scratch_, 32);
    this->dec(hw);
    this->cmp(hw, 0);
    this->jne(lrn_loop, this->T_NEAR);

    this->add(t, 64);
    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::jit_uni_lrn_fwd_kernel_t(
        const struct nhwc_across_t &J, float A, float K, prop_kind_t pk)
    : Base(jit_name())
    , config_(lrn_config_t::nhwc_across)
    , nhwc_across_(J)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::generate(const nhwc_across_t &J) {
    static const uint32_t mask[] = {0, 0, 0x80000000, 0x80000000, 0x80000000,
            0x80000000, 0x80000000, 0x80000000, 0x80000000, 0, 0};

    const Xbyak::Reg64 &c = this->r9;
    const Xbyak::Ymm &ya = this->ymm2;
    const Xbyak::Ymm &yb = this->ymm3;
    const Xbyak::Ymm &yc = this->ymm4;
    const Xbyak::Ymm &yd = this->ymm5;
    const Xbyak::Ymm &ye = this->ymm6;
    const Xbyak::Ymm &ysum = this->ymm7;
    const Xbyak::Ymm &ydst = this->ymm8;
    const Xbyak::Ymm &ybase = this->ymm9;
    const Xbyak::Ymm &ymask = this->ymm10;

    this->preamble();
    if (this->emulate_bfloat_) this->io_.init_bf16();

    this->mov(src_, this->ptr[this->param1 + 0]);
    this->mov(dst_, this->ptr[this->param1 + 8]);
    if (pk_ != prop_kind::forward_inference)
        this->mov(scratch_, this->ptr[this->param1 + 16]);
    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->vmovq(xalpha_, this->imm_addr64_);
    this->vbroadcastss(valpha_, xalpha_);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->vmovq(xk_, this->imm_addr64_);
    this->vbroadcastss(yk_, xk_);

    this->vxorps(ysum, ysum, ysum);

    this->mov(this->imm_addr64_, reinterpret_cast<size_t>(&mask[0]));
    this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    this->vmaskmovps(ya, ymask, this->ptr[src_ - 8]);
    this->vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

    this->mov(this->imm_addr64_, reinterpret_cast<size_t>(&mask[1]));
    this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    this->vmaskmovps(yb, ymask, this->ptr[src_ - 4]);
    this->vfmadd231ps(ysum, yb, yb);

    this->mov(c, J.C / 8 - 1);
    Label lrn_loop;
    this->L(lrn_loop);

    this->vmovups(yc, this->ptr[src_]);
    this->vmovups(yd, this->ptr[src_ + 4]);
    this->vmovups(ye, this->ptr[src_ + 8]);
    this->vfmadd231ps(ysum, yc, yc);
    this->vfmadd231ps(ysum, yd, yd);
    this->vfmadd231ps(ysum, ye, ye);

    this->vmovups(ydst, ysum);
    this->vfmadd132ps(ydst, yk_, valpha_); // ydst <- ysum*valpha_+yk_

    this->vmovaps(ybase, ydst);
    if (pk_ != prop_kind::forward_inference)
        this->vmovups(this->ptr[scratch_], ybase);
    this->vmulps(ydst, ydst, ydst);
    this->vmulps(ydst, ydst, ybase); // ydst = (ysum*valpha_+yk_)^3;
    this->vsqrtps(ydst, ydst);
    this->vsqrtps(ydst, ydst); // ydst = (ysum*valpha_+yk_)^0.75

    this->vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*valpha_+yk_)^0.75
    this->vmovups(this->ptr[dst_], ydst);

    this->vxorps(ysum, ysum, ysum);

    this->add(src_, 32);
    this->add(dst_, 32);
    if (pk_ != prop_kind::forward_inference) this->add(scratch_, 32);

    this->vmovups(ya, this->ptr[src_ - 8]);
    this->vfmadd231ps(ysum, ya, ya);
    this->vmovups(yb, this->ptr[src_ - 4]);
    this->vfmadd231ps(ysum, yb, yb);

    this->dec(c);
    this->cmp(c, 0);
    this->jne(lrn_loop, jit_generator_t::T_NEAR);

    this->vmovups(yc, this->ptr[src_]);
    this->vfmadd231ps(ysum, yc, yc);

    this->mov(this->imm_addr64_, reinterpret_cast<size_t>(&mask[2]));
    this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    this->vmaskmovps(yd, ymask, this->ptr[src_ + 4]);
    this->vfmadd231ps(ysum, yd, yd); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

    this->mov(this->imm_addr64_, reinterpret_cast<size_t>(&mask[3]));
    this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    this->vmaskmovps(ye, ymask, this->ptr[src_ + 8]);
    this->vfmadd231ps(ysum, ye, ye);

    this->vmovups(ydst, ysum);
    this->vfmadd132ps(ydst, yk_, valpha_); // ydst <- ysum*valpha_+yk_

    this->vmovaps(ybase, ydst);
    if (pk_ != prop_kind::forward_inference)
        this->vmovups(this->ptr[scratch_], ybase);
    this->vmulps(ydst, ydst, ydst);
    this->vmulps(ydst, ydst, ybase); // ydst = (ysum*valpha_+yk_)^3;
    this->vsqrtps(ydst, ydst);
    this->vsqrtps(ydst, ydst); // ydst = (ysum*valpha_+yk_)^0.75
    this->vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*valpha_+yk_)^0.75

    this->vmovups(this->ptr[dst_], ydst);

    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::nchw_body(int tail, int HW,
        prop_kind_t pk, Xbyak::Ymm ymask, Xbyak::Ymm ya, Xbyak::Ymm yb,
        Xbyak::Ymm yc, Xbyak::Ymm yd, Xbyak::Ymm ye, Xbyak::Ymm ysum) {
    const Xbyak::Ymm &ydst = this->ymm14;
    const Xbyak::Ymm &ybase = this->ymm15;

    this->vfmadd231ps(ysum, ye, ye);

    this->vmovups(ydst, ysum);
    this->vfmadd132ps(ydst, yk_, valpha_); // ydst <- ysum*valpha_+yk_

    this->vmovaps(ybase, ydst);
    if (pk_ != prop_kind::forward_inference) {
        if (tail != 0)
            this->vmaskmovps(this->ptr[scratch_], ymask, ybase);
        else
            this->vmovups(this->ptr[scratch_], ybase);
    }
    this->vmulps(ydst, ydst, ydst);
    this->vmulps(ydst, ydst, ybase); // ydst = (ysum*valpha_+yk_)^3;
    this->vsqrtps(ydst, ydst);
    this->vsqrtps(ydst, ydst); // ydst = (ysum*valpha_+yk_)^0.75
    this->vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*valpha_+yk_)^0.75

    if (tail != 0)
        this->vmaskmovps(this->ptr[dst_], ymask, ydst);
    else
        this->vmovups(this->ptr[dst_], ydst);

    this->vfnmadd231ps(ysum, ya, ya);
    this->vmovups(ya, yb);
    this->vmovups(yb, yc);
    this->vmovups(yc, yd);
    this->vmovups(yd, ye);
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::nchw_tail_sse41(int tail,
        Xbyak::Reg64 reg_dst, Xbyak::Xmm xtail_lo, Xbyak::Xmm xtail_hi) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::nchw_body_sse41(int tail, int HW,
        prop_kind_t pk, Xbyak::Xmm xe_lo, Xbyak::Xmm xe_hi, Xbyak::Xmm xsum_lo,
        Xbyak::Xmm xsum_hi) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::jit_uni_lrn_fwd_kernel_t(
        const nchw_across_t &J, float A, float K, prop_kind_t pk)
    : Base(jit_name())
    , config_(lrn_config_t::nchw_across)
    , nchw_across_(J)
    , alpha_(A)
    , k_(K)
    , pk_(pk) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_fwd_kernel_t<isa, d_type>::generate(const nchw_across_t &J) {
    static const uint32_t mask[]
            = {0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
                    0x80000000, 0x80000000, 0, 0, 0, 0, 0, 0, 0};
    const Xbyak::Reg64 &c = this->r10;
    const Xbyak::Ymm &ymask = this->ymm2;
    const Xbyak::Ymm &ye = this->ymm3;
    const Xbyak::Ymm &ya = this->ymm4;
    const Xbyak::Ymm &yb = this->ymm5;
    const Xbyak::Ymm &yc = this->ymm6;
    const Xbyak::Ymm &yd = this->ymm7;
    const Xbyak::Ymm &ysum = this->ymm8;

    this->preamble();
    if (this->emulate_bfloat_) this->io_.init_bf16();

    if (J.tail != 0) {
        this->mov(
                this->imm_addr64_, reinterpret_cast<size_t>(&mask[7 - J.tail]));
        this->vmovups(ymask, this->ptr[this->imm_addr64_]);
    }
    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->vmovq(xalpha_, this->imm_addr64_);
    this->vbroadcastss(valpha_, xalpha_);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->vmovq(xk_, this->imm_addr64_);
    this->vbroadcastss(yk_, xk_);

    this->mov(src_, this->ptr[this->param1 + 0]);
    this->mov(dst_, this->ptr[this->param1 + 8]);
    if (pk_ != prop_kind::forward_inference)
        this->mov(scratch_, this->ptr[this->param1 + 16]);

    this->vxorps(ya, ya, ya);
    this->vxorps(yb, yb, yb);
    if (J.tail != 0)
        this->vmaskmovps(yc, ymask, this->ptr[src_ + J.HW * 0]);
    else
        this->vmovups(yc, this->ptr[src_ + J.HW * 0]);
    if (J.tail != 0)
        this->vmaskmovps(yd, ymask, this->ptr[src_ + J.HW * 4]);
    else
        this->vmovups(yd, this->ptr[src_ + J.HW * 4]);

    this->vxorps(ysum, ysum, ysum);
    this->vfmadd231ps(ysum, yc, yc); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2
    this->vfmadd231ps(ysum, yd, yd);

    this->mov(c, J.C - 2);
    Label lrn_loop;
    this->L(lrn_loop);

    if (J.tail != 0)
        this->vmaskmovps(ye, ymask, this->ptr[src_ + J.HW * 8]);
    else
        this->vmovups(ye, this->ptr[src_ + J.HW * 8]);

    nchw_body(J.tail, J.HW, pk_, ymask, ya, yb, yc, yd, ye, ysum);

    this->add(src_, J.HW * 4);
    this->add(dst_, J.HW * 4);
    if (pk_ != prop_kind::forward_inference) this->add(scratch_, J.HW * 4);
    this->dec(c);
    this->cmp(c, 0);
    this->jne(lrn_loop, jit_generator_t::T_NEAR);

    this->vxorps(ye, ye, ye);

    nchw_body(J.tail, J.HW, pk_, ymask, ya, yb, yc, yd, ye, ysum);
    this->add(src_, J.HW * 4);
    this->add(dst_, J.HW * 4);
    if (pk_ != prop_kind::forward_inference) this->add(scratch_, J.HW * 4);

    nchw_body(J.tail, J.HW, pk_, ymask, ya, yb, yc, yd, ye, ysum);

    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_fwd_kernel_t<isa, d_type>::~jit_uni_lrn_fwd_kernel_t() = default;

//////////////////////////////////////////////////////////////////////////////
// backward kernel
template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_bwd_kernel_t<isa, d_type>::jit_uni_lrn_bwd_kernel_t(
        const nchw8c_across_t &J, float A, float B, int use_h_parallel)
    : Base(jit_name())
    , config_(lrn_config_t::nchw8c_across)
    , nchw8c_across_(J)
    , nalphabeta_(-2 * A * B)
    , use_h_parallelizm_(use_h_parallel) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_bwd_kernel_t<isa, d_type>::generate(const nchw8c_across_t &J) {

    const Xbyak::Reg64 &t = this->rsp;
    const Xbyak::Reg64 &hw = this->r10;
    const Xbyak::Xmm &xsrc_prev = this->xmm1;
    const Xbyak::Xmm &xws_prev = this->xmm2;
    const Xbyak::Xmm &xdiffdst_prev = this->xmm3;
    const Xbyak::Ymm &ysrc = this->ymm4;
    const Xbyak::Ymm &yws = this->ymm5;
    const Xbyak::Ymm &ydiffdst = this->ymm6;
    const Xbyak::Xmm &xsrc_next = this->xmm7;
    const Xbyak::Xmm &xws_next = this->xmm8;
    const Xbyak::Xmm &xdiffdst_next = this->xmm9;
    const Xbyak::Ymm &ya = this->ymm10;
    const Xbyak::Xmm &xa = this->xmm10;
    const Xbyak::Ymm &yb = this->ymm11;
    const Xbyak::Ymm &yd = this->ymm12;
    const Xbyak::Ymm &ye = this->ymm13;
    const Xbyak::Ymm &ysum = this->ymm14;
    const Xbyak::Ymm &ydiffsrc = this->ymm15;

    this->preamble();
    if (this->bf16_emu_) this->bf16_emu_->init_vcvtneps2bf16();

#define GET_OFF(field) offsetof(jit_args_bwd_t, field)
    this->mov(src_, this->ptr[this->param1 + GET_OFF(src)]);
    this->mov(diffdst_, this->ptr[this->param1 + GET_OFF(diff_dst)]);
    this->mov(scratch_, this->ptr[this->param1 + GET_OFF(scratch)]);
    this->mov(bwd_intermediate_res_,
            this->ptr[this->param1 + GET_OFF(bwd_intermediate_res)]);
    this->mov(diffsrc_, this->ptr[this->param1 + GET_OFF(diff_src)]);
#undef GET_OFF

    this->sub(t, 64);
    this->mov(this->imm_addr64_, float2int(this->nalphabeta_));
    this->vmovq(xnalphabeta_, this->imm_addr64_);
    this->vbroadcastss(vnalphabeta_, xnalphabeta_);

    bool is_single = J.version == 3;
    bool is_first = J.version == -1 || J.version == -2;
    bool is_last = J.version == +1 || J.version == -2;

    if (is_first || is_single) {
        this->vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
        this->vmovups(this->ptr[t + 0], xsrc_prev);
    }
    if (is_last || is_single) {
        this->vxorps(xsrc_next, xsrc_next, xsrc_next);
        this->vmovups(this->ptr[t + 48], xsrc_next);
    }
    this->mov(hw, this->use_h_parallelizm_ ? J.W : J.H * J.W);
    Label lrn_loop;
    this->L(lrn_loop);
    {
        if (!is_first && !is_single) {
            this->vmovups(xws_prev, this->ptr[scratch_ - J.H * J.W * 32 + 16]);
            this->vmovups(xsrc_prev, this->ptr[src_ - J.H * J.W * 32 + 16]);
            this->vmovups(
                    xdiffdst_prev, this->ptr[diffdst_ - J.H * J.W * 32 + 16]);
            this->vmulps(xa, xws_prev, xws_prev);
            this->vmulps(xa, xa, xws_prev);
            this->vsqrtps(xa, xa);
            this->vsqrtps(xa, xa);
            this->vmulps(xa, xa, xws_prev);
            this->vdivps(xsrc_prev, xsrc_prev, xa);
            this->vmulps(xdiffdst_prev, xdiffdst_prev, xsrc_prev);
        }

        this->vmovups(ysrc, this->ptr[src_]);
        this->vmovups(yws, this->ptr[scratch_]);
        this->vmovups(ydiffdst, this->ptr[diffdst_]);
        this->vmulps(ya, yws, yws);
        this->vmulps(ya, ya, yws);
        this->vsqrtps(ya, ya);
        this->vsqrtps(ya, ya);
        this->vdivps(ydiffsrc, ydiffdst, ya);
        this->vdivps(ysum, ydiffsrc, yws);
        this->vmulps(ysum, ysum, ysrc);

        if (!is_last && !is_single) {
            this->vmovups(xws_next, this->ptr[scratch_ + J.H * J.W * 32]);
            this->vmovups(xsrc_next, this->ptr[src_ + J.H * J.W * 32]);
            this->vmovups(xdiffdst_next, this->ptr[diffdst_ + J.H * J.W * 32]);
            this->vmulps(xa, xws_next, xws_next);
            this->vmulps(xa, xa, xws_next);
            this->vsqrtps(xa, xa);
            this->vsqrtps(xa, xa);
            this->vmulps(xa, xa, xws_next);
            this->vdivps(xsrc_next, xsrc_next, xa);
            this->vmulps(xdiffdst_next, xdiffdst_next, xsrc_next);
        }

        if (!is_first && !is_single)
            this->vmovups(this->ptr[t + 0], xdiffdst_prev);
        this->vmovups(this->ptr[t + 16], ysum);
        if (!is_last && !is_single)
            this->vmovups(this->ptr[t + 48], xdiffdst_next);

        this->vmovups(ya, this->ptr[t + 16 - 8]);
        this->vmovups(yb, this->ptr[t + 16 - 4]);
        this->vaddps(ysum, ysum, ya);
        this->vmulps(ysrc, ysrc, vnalphabeta_);
        this->vaddps(ysum, ysum, yb);

        this->vmovups(yd, this->ptr[t + 16 + 4]);
        this->vmovups(ye, this->ptr[t + 16 + 8]);
        this->vaddps(ysum, ysum, yd);
        this->vaddps(ysum, ysum, ye);

        this->vfmadd231ps(ydiffsrc, ysum, ysrc);

        this->vmovups(this->ptr[diffsrc_], ydiffsrc);

        this->add(src_, 32);
        this->add(diffsrc_, 32);
        this->add(diffdst_, 32);
        this->add(scratch_, 32);

        this->dec(hw);
        this->cmp(hw, 0);
        this->jne(lrn_loop, this->T_NEAR);
    }

    this->add(t, 64);
    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_lrn_bwd_kernel_t<isa, d_type>::jit_uni_lrn_bwd_kernel_t(
        const within_config_t &config, float A, float B)
    : Base(config, jit_name())
    , config_(lrn_config_t::within_config)
    , within_config_(config)
    , nalphabeta_(-2.0f * A * B)
    , use_h_parallelizm_(0) {}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_bwd_kernel_t<isa, d_type>::generate(
        const within_config_t &config) {

    this->preamble();
    if (this->bf16_emu_) this->bf16_emu_->init_vcvtneps2bf16();

#define GET_OFF(field) offsetof(jit_args_bwd_t, field)
    this->mov(src_, this->ptr[this->param1 + GET_OFF(src)]);
    this->mov(diffdst_, this->ptr[this->param1 + GET_OFF(diff_dst)]);
    this->mov(scratch_, this->ptr[this->param1 + GET_OFF(scratch)]);
    this->mov(bwd_intermediate_res_,
            this->ptr[this->param1 + GET_OFF(bwd_intermediate_res)]);
    this->mov(diffsrc_, this->ptr[this->param1 + GET_OFF(diff_src)]);
#undef GET_OFF
    this->load_constant(nalphabeta_, vnalphabeta_, xnalphabeta_);

    static const int max_reg_blocks = is_superset(isa, avx512_core) ? 4 : 2;
    this->within_loop(config, max_reg_blocks, prop_kind::backward);

    this->postamble();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_bwd_kernel_t<isa, d_type>::within_body(int hoff, int Hoff,
        int woff, int Woff, int stride, prop_kind_t pk, const int reg_block,
        int pixel_offset) {

    static const std::array<Vmm, 4> vsum {{Vmm(1), Vmm(7), Vmm(13), Vmm(19)}};
    static const std::array<Vmm, 4> diff_dst {
            {Vmm(2), Vmm(8), Vmm(14), Vmm(20)}};
    static const std::array<Vmm, 4> ws1 {{Vmm(3), Vmm(9), Vmm(15), Vmm(21)}};
    static const std::array<Vmm, 4> ws0 {{Vmm(4), Vmm(10), Vmm(16), Vmm(22)}};
    static const std::array<Vmm, 4> src {{Vmm(5), Vmm(11), Vmm(17), Vmm(23)}};
    static const std::array<Vmm, 4> a {{Vmm(6), Vmm(12), Vmm(18), Vmm(24)}};

    IRB_LOOP(this->uni_vxorps(vsum[irb], vsum[irb], vsum[irb]));
    for (int i = hoff; i <= Hoff; ++i) {
        for (int j = woff; j <= Woff; ++j) {
            IRB_LOOP(this->io_.at(d_type)->load(
                    this->ptr[(diffdst_ + pixel_offset + irb_off)
                            + (i * stride + j) * this->single_pixel_offset_],
                    diff_dst[irb], false));
            IRB_LOOP(this->io_.at(d_type)->load(
                    this->ptr[(bwd_intermediate_res_ + pixel_offset + irb_off)
                            + (i * stride + j) * this->single_pixel_offset_],
                    ws1[irb], false));

            if (i == 0 && j == 0) {
                if (utils::one_of(d_type, data_type::bf16, data_type::f16)) {
                    IRB_LOOP(this->io_.at(d_type)->load(
                            this->ptr[(scratch_ + pixel_offset + irb_off)],
                            ws0[irb], false));
                    IRB_LOOP(this->vdivps(a[irb], diff_dst[irb], ws0[irb]));
                } else {
                    IRB_LOOP(this->vdivps(a[irb], diff_dst[irb],
                            this->ptr[(scratch_ + pixel_offset + irb_off)]));
                }
            }

            IRB_LOOP(this->vfmadd231ps(vsum[irb], ws1[irb], diff_dst[irb]));
        }
    }

    if (utils::one_of(d_type, data_type::bf16, data_type::f16)) {
        IRB_LOOP(this->io_.at(d_type)->load(
                this->ptr[(src_ + pixel_offset + irb_off)], src[irb], false));
        IRB_LOOP(this->vmulps(src[irb], this->vnalphabeta_, src[irb]));
    } else {
        IRB_LOOP(this->vmulps(src[irb], this->vnalphabeta_,
                this->ptr[(src_ + pixel_offset + irb_off)]));
    }

    IRB_LOOP(this->vfmadd231ps(a[irb], src[irb], vsum[irb]));

    IRB_LOOP(this->io_.at(d_type)->store(
            a[irb], this->ptr[diffsrc_ + pixel_offset + irb_off], false));
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_lrn_bwd_kernel_t<isa, d_type>::move_data_pointers(
        int pixel_count, prop_kind_t pk) {
    const int pixel_offset = this->single_pixel_offset_ * pixel_count;
    this->add(src_, pixel_offset);
    this->add(diffsrc_, pixel_offset);
    this->add(diffdst_, pixel_offset);
    this->add(scratch_, pixel_offset);
    this->add(bwd_intermediate_res_, pixel_offset);
}

template class jit_uni_lrn_fwd_kernel_t<avx2, data_type::f32>;
template class jit_uni_lrn_fwd_kernel_t<avx2_vnni_2, data_type::bf16>;
template class jit_uni_lrn_fwd_kernel_t<avx2_vnni_2, data_type::f16>;
template class jit_uni_lrn_fwd_kernel_t<avx512_core, data_type::f32>;
template class jit_uni_lrn_fwd_kernel_t<avx512_core, data_type::bf16>;
template class jit_uni_lrn_fwd_kernel_t<avx512_core_fp16, data_type::f16>;

template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<avx2, data_type::f32>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<avx2_vnni_2, data_type::bf16>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<avx2_vnni_2, data_type::f16>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<avx512_core, data_type::f32>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<avx512_core, data_type::bf16>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_fwd_kernel_t<avx512_core_fp16, data_type::f16>>;

template class jit_uni_lrn_bwd_kernel_t<avx512_core_fp16, data_type::f16>;
template class jit_uni_lrn_bwd_kernel_t<avx512_core, data_type::f32>;
template class jit_uni_lrn_bwd_kernel_t<avx512_core, data_type::bf16>;
template class jit_uni_lrn_bwd_kernel_t<avx2, data_type::f32>;

template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_bwd_kernel_t<avx2, data_type::f32>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_bwd_kernel_t<avx512_core, data_type::f32>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_bwd_kernel_t<avx512_core, data_type::bf16>>;
template class jit_uni_lrn_kernel_t<
        jit_uni_lrn_bwd_kernel_t<avx512_core_fp16, data_type::f16>>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
