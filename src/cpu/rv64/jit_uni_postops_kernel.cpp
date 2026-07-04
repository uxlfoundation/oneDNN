/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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
#include <cstddef>

#include "common/memory_desc_wrapper.hpp"

#include "cpu/rv64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/rv64/jit_uni_postops_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define GET_OFF(field) \
    static_cast<int32_t>( \
            offsetof(jit_uni_postops_kernel_t::call_params_t, field))

// The actual code generator, hidden behind the pimpl so the public header stays
// free of the injector/xbyak includes.
struct jit_uni_postops_kernel_t::impl_t : public jit_generator_t {
    impl_t(const post_ops_t &po, bool with_bias, bool bias_per_element,
            bool has_per_elem_binary)
        : jit_generator_t("jit_uni_postops_apply")
        , po_(po)
        , with_bias_(with_bias)
        , bias_per_element_(bias_per_element)
        , has_per_elem_binary_(has_per_elem_binary) {
        create_kernel();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_postops_kernel_t::impl_t)

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

    void generate() override {
        const Reg reg_param = a0;
        const Reg reg_dst = a1;
        const Reg reg_bias = a2;
        const Reg reg_rhs = a3; // base of the per-binary rhs pointer array
        const Reg reg_len = a4;
        const Reg reg_vl = t0;
        const Reg reg_bytes = t1;
        const Reg reg_off = t3; // byte offset of the chunk within the unit
        const Reg reg_gpr = t4; // binary injector scratch

        ld(reg_dst, reg_param, GET_OFF(dst));
        ld(reg_bias, reg_param, GET_OFF(bias));
        ld(reg_rhs, reg_param, GET_OFF(rhs));
        ld(reg_len, reg_param, GET_OFF(len));

        // Accumulator group v24 (m4). The injectors may use v0 as a mask
        // register, so the data register group must not overlap it.
        const VReg v_data(24);
        const VReg v_bias(20);
        const FReg f_bias = fa3;
        eltwise_injector::static_params_t esp(
                VReg(4), VReg(8), VReg(12), fa0, fa1, t2);
        // Indirect binary rhs: reg_rhs is the pointer array, reg_off the shared
        // per-chunk byte offset; each binary loads its own base + reg_off.
        binary_injector::static_params_t bsp(
                VReg(16), fa2, reg_rhs, reg_off, reg_gpr);
        injector::jit_uni_postops_injector_t<v> inj(this, po_, esp, &bsp);

        if (has_per_elem_binary_) ld(reg_off, reg_param, GET_OFF(off0));

        Label loop, done;
        L(loop);
        beqz(reg_len, done);

        vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        vle32_v(v_data, reg_dst);

        // bias add (before the post-op chain), scalar or per-element.
        if (with_bias_) {
            if (bias_per_element_) {
                vle32_v(v_bias, reg_bias);
                vfadd_vv(v_data, v_data, v_bias);
            } else {
                flw(f_bias, reg_bias, 0);
                vfadd_vf(v_data, v_data, f_bias);
            }
        }

        inj.compute_vector(v_data.getIdx());
        vse32_v(v_data, reg_dst);

        slli(reg_bytes, reg_vl, 2);
        add(reg_dst, reg_dst, reg_bytes);
        // Per-element bias advances 1:1 with the unit; the binary rhs offset
        // advances likewise (each binary adds it to its own base). Scalars stay
        // fixed and ignore reg_off.
        if (with_bias_ && bias_per_element_) add(reg_bias, reg_bias, reg_bytes);
        if (has_per_elem_binary_) add(reg_off, reg_off, reg_bytes);
        sub(reg_len, reg_len, reg_vl);
        j_(loop);

        L(done);
        ret();
    }

    post_ops_t po_;
    bool with_bias_;
    bool bias_per_element_;
    bool has_per_elem_binary_;
};

jit_uni_postops_kernel_t::jit_uni_postops_kernel_t() = default;
jit_uni_postops_kernel_t::~jit_uni_postops_kernel_t() = default;

void jit_uni_postops_kernel_t::operator()(const call_params_t *p) const {
    (*impl_)(p);
}

bool jit_uni_postops_kernel_t::binary_broadcast_ok(
        const post_ops_t &po, dim_t unit_nelems) {
    for (int i = 0; i < po.len(); i++) {
        const auto &e = po.entry_[i];
        if (!e.is_binary()) continue;
        const memory_desc_wrapper src1_d(e.binary.src1_desc);
        const bool scalar = src1_d.nelems(true) == 1;
        const bool per_elem
                = src1_d.nelems(true) == unit_nelems && src1_d.is_dense(true);
        if (!scalar && !per_elem) return false;
    }
    return true;
}

bool jit_uni_postops_kernel_t::binary_rhs_dt_ok(const post_ops_t &po) {
    for (int i = 0; i < po.len(); i++) {
        const auto &e = po.entry_[i];
        if (e.is_binary() && e.binary.src1_desc.data_type != data_type::f32)
            return false;
    }
    return true;
}

bool jit_uni_postops_kernel_t::binary_per_last_dim_ok(
        const post_ops_t &po, dim_t last_dim) {
    for (int i = 0; i < po.len(); i++) {
        const auto &e = po.entry_[i];
        if (!e.is_binary()) continue;
        const memory_desc_wrapper s1(e.binary.src1_desc);
        if (s1.nelems(true) == 1) continue; // scalar broadcasts to everything
        const int nd = s1.ndims();
        if (s1.dims()[nd - 1] != last_dim) return false; // last dim must be N
        for (int k = 0; k < nd - 1; k++)
            if (s1.dims()[k] != 1) return false; // leading dims must broadcast
    }
    return true;
}

bool jit_uni_postops_kernel_t::post_ops_supported(
        const post_ops_t &po, dim_t unit_nelems) {
    return injector::jit_uni_postops_injector_t<v>::post_ops_ok(po)
            && binary_broadcast_ok(po, unit_nelems) && binary_rhs_dt_ok(po);
}

status_t jit_uni_postops_kernel_t::create(
        std::shared_ptr<jit_uni_postops_kernel_t> &kernel, const post_ops_t &po,
        const conf_t &conf) {
    if (conf.dst_dt != data_type::f32) return status::unimplemented;
    if (!injector::jit_uni_postops_injector_t<v>::post_ops_ok(po))
        return status::unimplemented;
    // The injector loads the binary rhs as f32 only (flw/vle32/vlse32).
    if (!binary_rhs_dt_ok(po)) return status::unimplemented;

    // A per-element (non-scalar) binary rhs advances in lockstep with dst.
    bool has_per_elem_binary = false;
    for (int i = 0; i < po.len(); i++) {
        const auto &e = po.entry_[i];
        if (e.is_binary()
                && memory_desc_wrapper(e.binary.src1_desc).nelems(true) != 1)
            has_per_elem_binary = true;
    }

    std::shared_ptr<jit_uni_postops_kernel_t> k(new jit_uni_postops_kernel_t());
    k->impl_.reset(new impl_t(
            po, conf.with_bias, conf.bias_per_element, has_per_elem_binary));
    kernel = std::move(k);
    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
