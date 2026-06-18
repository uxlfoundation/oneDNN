/*******************************************************************************
* Copyright 2026 SpacemiT Corporation
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

#include <limits>

#include "common/bit_cast.hpp"

#include "cpu/rv64/jit_uni_reduction_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define GET_OFF(field) offsetof(jit_uni_reduction_args_t, field)

namespace {

const Reg reg_param = a0;
const Reg reg_src = a1;
const Reg reg_dst = a2;
const Reg reg_n = a3;
const Reg reg_tmp = t0;

const VReg v_data = VReg(0);
const VReg v_acc = VReg(8);
const VReg v_red = VReg(16);

const FReg f_init = fa0;
const FReg f_tmp = ft0;
const FReg f_scale = ft1;

constexpr uint16_t f16_lowest = 0xfbff;
constexpr uint16_t f16_max = 0x7bff;

} // namespace

jit_uni_reduction_kernel_t::jit_uni_reduction_kernel_t(
        const jit_reduction_conf_t &conf)
    : jit_generator_t("jit_uni_reduction_kernel"), conf_(conf) {}

void jit_uni_reduction_kernel_t::load_f32_const(
        const Xbyak_riscv::FReg &freg, float value) {
    li(reg_tmp, static_cast<int32_t>(utils::bit_cast<uint32_t>(value)));
    fmv_w_x(freg, reg_tmp);
}

void jit_uni_reduction_kernel_t::load_f16_const(
        const Xbyak_riscv::FReg &freg, uint16_t raw_value) {
    li(reg_tmp, raw_value);
    fmv_h_x(freg, reg_tmp);
}

void jit_uni_reduction_kernel_t::advance_src(int shift) {
    slli(reg_tmp, reg_tmp, shift);
    add(reg_src, reg_src, reg_tmp);
}

void jit_uni_reduction_kernel_t::load_params() {
    ld(reg_src, reg_param, GET_OFF(src));
    ld(reg_dst, reg_param, GET_OFF(dst));
    ld(reg_n, reg_param, GET_OFF(reduce_size));
}

jit_uni_reduction_kernel_t::src_kind_t
jit_uni_reduction_kernel_t::src_kind() const {
    using namespace data_type;
    if (conf_.src_type == f32) return src_kind_t::f32;
    if (conf_.src_type == f16) return src_kind_t::f16;
    assert(!"unsupported reduction source data type");
    return src_kind_t::f32;
}

jit_uni_reduction_kernel_t::scalar_kind_t
jit_uni_reduction_kernel_t::dst_kind() const {
    using namespace data_type;
    if (conf_.dst_type == f32) return scalar_kind_t::f32;
    if (conf_.dst_type == f16) return scalar_kind_t::f16;
    assert(!"unsupported reduction destination data type");
    return scalar_kind_t::f32;
}

jit_uni_reduction_kernel_t::reduce_op_t
jit_uni_reduction_kernel_t::reduce_op() const {
    using namespace alg_kind;
    switch (conf_.alg) {
        case reduction_max: return reduce_op_t::max;
        case reduction_min: return reduce_op_t::min;
        case reduction_sum: return reduce_op_t::sum;
        case reduction_mean: return reduce_op_t::mean;
        default: assert(!"unsupported reduction alg");
    }
    return reduce_op_t::sum;
}

jit_uni_reduction_kernel_t::acc_kind_t jit_uni_reduction_kernel_t::acc_kind(
        src_kind_t src_kind) const {
    return src_kind == src_kind_t::f16 && !is_f16_widen_acc() ? acc_kind_t::f16
                                                              : acc_kind_t::f32;
}

bool jit_uni_reduction_kernel_t::is_f16_widen_acc() const {
    const reduce_op_t op = reduce_op();
    return op == reduce_op_t::sum || op == reduce_op_t::mean;
}

void jit_uni_reduction_kernel_t::emit_store_scalar(scalar_kind_t scalar_kind) {
    const scalar_kind_t dst_kind = this->dst_kind();
    if (scalar_kind == scalar_kind_t::f32) {
        if (dst_kind == scalar_kind_t::f32) {
            fsw(f_tmp, reg_dst, 0);
        } else {
            fcvt_h_s(Reg(f_tmp.getIdx()), Reg(f_tmp.getIdx()));
            fsh(f_tmp, reg_dst, 0);
        }
    } else {
        if (dst_kind == scalar_kind_t::f32) {
            fcvt_s_h(Reg(f_tmp.getIdx()), Reg(f_tmp.getIdx()));
            fsw(f_tmp, reg_dst, 0);
        } else {
            fsh(f_tmp, reg_dst, 0);
        }
    }
}

void jit_uni_reduction_kernel_t::emit_init(src_kind_t src_kind) {
    const reduce_op_t op = reduce_op();

    if (src_kind == src_kind_t::f16 && !is_f16_widen_acc()) {
        if (op == reduce_op_t::max)
            load_f16_const(f_init, f16_lowest);
        else if (op == reduce_op_t::min)
            load_f16_const(f_init, f16_max);
        else
            assert(!"unsupported f16 non-widen reduction alg");
    } else {
        if (op == reduce_op_t::max)
            load_f32_const(f_init, -std::numeric_limits<float>::infinity());
        else if (op == reduce_op_t::min)
            load_f32_const(f_init, std::numeric_limits<float>::infinity());
        else
            load_f32_const(f_init, 0.0f);
    }

    if (op == reduce_op_t::mean)
        load_f32_const(f_scale, 1.0f / static_cast<float>(conf_.reduce_size));

    const acc_kind_t acc = acc_kind(src_kind);
    if (acc == acc_kind_t::f32)
        vsetvli(reg_tmp, x0, SEW::e32, LMUL::m8, VTA::tu, VMA::mu);
    else
        vsetvli(reg_tmp, x0, SEW::e16, LMUL::m8, VTA::tu, VMA::mu);

    if (op == reduce_op_t::sum
            || (src_kind == src_kind_t::f16 && is_f16_widen_acc())) {
        vfmv_v_f(v_acc, f_init);
        vfmv_v_f(v_red, f_init);
    } else {
        vfmv_v_f(v_red, f_init);
        vfmv_v_f(v_acc, f_init);
    }
}

void jit_uni_reduction_kernel_t::emit_update_acc(src_kind_t src_kind) {
    const reduce_op_t op = reduce_op();
    switch (op) {
        case reduce_op_t::max: vfmax_vv(v_acc, v_data, v_acc); break;
        case reduce_op_t::min: vfmin_vv(v_acc, v_data, v_acc); break;
        case reduce_op_t::sum:
        case reduce_op_t::mean:
            if (src_kind == src_kind_t::f16)
                vfwadd_wv(v_acc, v_acc, v_data);
            else
                vfadd_vv(v_acc, v_data, v_acc);
            break;
        default: assert(!"unsupported reduction alg");
    }
}

void jit_uni_reduction_kernel_t::emit_loop(src_kind_t src_kind) {
    Label loop;

    L(loop);
    if (src_kind == src_kind_t::f32) {
        vsetvli(reg_tmp, reg_n, SEW::e32, LMUL::m8, VTA::tu, VMA::mu);
        sub(reg_n, reg_n, reg_tmp);
        vle32_v(v_data, reg_src);
        advance_src(2);
    } else {
        if (is_f16_widen_acc())
            vsetvli(reg_tmp, reg_n, SEW::e16, LMUL::m4, VTA::tu, VMA::mu);
        else
            vsetvli(reg_tmp, reg_n, SEW::e16, LMUL::m8, VTA::tu, VMA::mu);
        sub(reg_n, reg_n, reg_tmp);
        vle16_v(v_data, reg_src);
        advance_src(1);
    }
    emit_update_acc(src_kind);
    bnez(reg_n, loop);
}

void jit_uni_reduction_kernel_t::emit_horizontal_reduce(src_kind_t src_kind) {
    const reduce_op_t op = reduce_op();
    const acc_kind_t acc = acc_kind(src_kind);

    if (acc == acc_kind_t::f32) {
        vsetvli(reg_tmp, x0, SEW::e32, LMUL::m8, VTA::tu, VMA::mu);
        if (op == reduce_op_t::max)
            vfredmax_vs(v_red, v_acc, v_red);
        else if (op == reduce_op_t::min)
            vfredmin_vs(v_red, v_acc, v_red);
        else if (op == reduce_op_t::mean && src_kind == src_kind_t::f32)
            vfredusum_vs(v_red, v_acc, v_red);
        else
            vfredosum_vs(v_red, v_acc, v_red);
    } else {
        vsetvli(reg_tmp, x0, SEW::e16, LMUL::m8, VTA::tu, VMA::mu);
        if (op == reduce_op_t::max)
            vfredmax_vs(v_red, v_acc, v_red);
        else if (op == reduce_op_t::min)
            vfredmin_vs(v_red, v_acc, v_red);
        else
            assert(!"unsupported f16 non-widen reduction alg");
    }
}

void jit_uni_reduction_kernel_t::emit_mean_scale_if_needed() {
    if (reduce_op() == reduce_op_t::mean) fmul_s(f_tmp, f_tmp, f_scale);
}

void jit_uni_reduction_kernel_t::emit_finalize_f16_minmax() {
    emit_horizontal_reduce(src_kind_t::f16);
    vfmv_f_s(f_tmp, v_red);
    emit_store_scalar(scalar_kind_t::f16);
}

void jit_uni_reduction_kernel_t::emit_finalize_f16_widen_sum_or_mean() {
    emit_horizontal_reduce(src_kind_t::f16);
    if (reduce_op() == reduce_op_t::sum) {
        if (dst_kind() == scalar_kind_t::f32) {
            vfmv_f_s(f_tmp, v_red);
            fsw(f_tmp, reg_dst, 0);
        } else {
            vsetvli(reg_tmp, x0, SEW::e16, LMUL::m4, VTA::tu, VMA::mu);
            vfncvt_f_f_w(v_data, v_red);
            vfmv_f_s(f_tmp, v_data);
            fsh(f_tmp, reg_dst, 0);
        }
    } else if (dst_kind() == scalar_kind_t::f32) {
        vfmv_f_s(f_tmp, v_red);
        emit_mean_scale_if_needed();
        fsw(f_tmp, reg_dst, 0);
    } else {
        vsetvli(reg_tmp, x0, SEW::e16, LMUL::m4, VTA::tu, VMA::mu);
        vfncvt_f_f_w(v_data, v_red);
        vfmv_f_s(f_tmp, v_data);
        fcvt_s_h(Reg(f_tmp.getIdx()), Reg(f_tmp.getIdx()));
        emit_mean_scale_if_needed();
        fcvt_h_s(Reg(f_tmp.getIdx()), Reg(f_tmp.getIdx()));
        fsh(f_tmp, reg_dst, 0);
    }
}

void jit_uni_reduction_kernel_t::emit_finalize(src_kind_t src_kind) {
    if (src_kind == src_kind_t::f16) {
        if (is_f16_widen_acc())
            emit_finalize_f16_widen_sum_or_mean();
        else
            emit_finalize_f16_minmax();
        return;
    }

    emit_horizontal_reduce(src_kind_t::f32);
    vfmv_f_s(f_tmp, v_red);
    emit_mean_scale_if_needed();
    emit_store_scalar(scalar_kind_t::f32);
}

void jit_uni_reduction_kernel_t::emit_reduce(src_kind_t src_kind) {
    emit_init(src_kind);
    emit_loop(src_kind);
    emit_finalize(src_kind);
}

void jit_uni_reduction_kernel_t::generate() {
    load_params();

    emit_reduce(src_kind());

    ret();
}

#undef GET_OFF

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
