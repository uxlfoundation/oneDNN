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
        , has_per_elem_binary_(has_per_elem_binary) {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_postops_kernel_t::impl_t)

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

    void generate() override {
        const Reg reg_param = a0;
        const Reg reg_dst = a1;
        const Reg reg_bias = a2;
        const Reg reg_len = a4;
        const Reg reg_vl = t0;
        const Reg reg_bytes = t1;
        const Reg reg_off = t3; // element offset of the chunk within the unit
        const Reg reg_gpr = a5; // binary injector address scratch

        ld(reg_dst, reg_param, GET_OFF(dst));
        ld(reg_bias, reg_param, GET_OFF(bias));
        ld(reg_len, reg_param, GET_OFF(len));

        // Accumulator group v24 (m4). The injectors may use v0 as a mask
        // register, so the data register group must not overlap it.
        const VReg v_data(24);
        const VReg v_bias(20);
        const FReg f_bias = fa3;
        // 4 aux (v4/v8/v12 + v28) so log/soft_relu/gelu_erf post-ops fit; v28
        // is the one free m4 group here (v16=binary rhs, v20=bias, v24=data).
        eltwise_injector::static_params_t esp(VReg(4), VReg(8), VReg(12),
                VReg(28), VReg(28), fa0, fa1, t2, /*is_fwd=*/true);
        // Indirect binary rhs via the dynamic-params interface: reg_rhs is the
        // per-binary pointer array; the injector computes each rhs address from
        // the per-chunk output element offset (reg_off) it is handed at call
        // time. reg_bytes is the injector's address scratch.
        // Host-positioned injector mode (null dst_d): the epilogue works on
        // flat units, so the rhs classifies as scalar / no_broadcast from its
        // element count and the unit offset is handed in at call time.
        // The rhs pointer array is read through the args pointer (x64 model).
        // The hosted scalar/no_broadcast paths clobber only the injector's t4
        // scratch, which is free here -> no GPR preservation.
        binary_injector::rhs_arg_static_params_t rhs_arg_bsp(16, reg_gpr,
                reg_bytes, reg_bytes, false /*preserve gpr*/,
                false /*preserve vmm*/, GET_OFF(rhs),
                memory_desc_wrapper(nullptr));
        rhs_arg_bsp.rhs_dt_helper_freg = fa2;
        const binary_injector::static_params_t bsp(reg_param, rhs_arg_bsp);
        injector::jit_uni_postops_injector_t<v> inj(this, po_, bsp, esp);
        binary_injector::rhs_arg_dynamic_params_t rhs_dyn;

        if (has_per_elem_binary_) {
            ld(reg_off, reg_param, GET_OFF(off0)); // byte offset
            srli(reg_off, reg_off, 2); // -> element offset (rhs is f32)
            rhs_dyn.vmm_idx_to_out_reg.emplace(v_data.getIdx(), reg_off);
        }

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

        // Epilogue computes at e32/m4 (group_stride 4).
        inj.compute_vector(v_data.getIdx(), rhs_dyn, 4 /*group_stride*/);
        vse32_v(v_data, reg_dst);

        slli(reg_bytes, reg_vl, 2);
        add(reg_dst, reg_dst, reg_bytes);
        // Per-element bias advances 1:1 with the unit; the binary rhs offset
        // advances likewise (each binary adds it to its own base). Scalars stay
        // fixed and ignore reg_off.
        if (with_bias_ && bias_per_element_) add(reg_bias, reg_bias, reg_bytes);
        if (has_per_elem_binary_)
            add(reg_off, reg_off, reg_vl); // element offset
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

// Chain-level injector acceptance. The epilogue works on flat units without a
// dst descriptor, so it cannot use injector::post_ops_ok (which classifies the
// binary rhs against dst); the chain entry kinds are checked directly: eltwise
// with a 4-aux budget (heavy algs included), any non-ternary binary alg
// (select needs a condition scratch group this host does not reserve), no
// other kinds.
static bool injector_chain_ok(const post_ops_t &po) {
    for (int i = 0; i < po.len(); i++) {
        const auto &e = po.entry_[i];
        if (e.is_eltwise()) {
            if (!eltwise_injector::is_supported(e.eltwise.alg)) return false;
        } else if (e.is_binary()) {
            if (e.is_binary_with_ternary_op()) return false;
        } else
            return false;
    }
    return true;
}

bool jit_uni_postops_kernel_t::post_ops_supported(
        const post_ops_t &po, dim_t unit_nelems) {
    return injector_chain_ok(po) && binary_broadcast_ok(po, unit_nelems)
            && binary_rhs_dt_ok(po);
}

status_t jit_uni_postops_kernel_t::create(
        std::shared_ptr<jit_uni_postops_kernel_t> &kernel, const post_ops_t &po,
        const conf_t &conf) {
    if (conf.dst_dt != data_type::f32) return status::unimplemented;
    if (!injector_chain_ok(po)) return status::unimplemented;
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
    CHECK(k->impl_->create_kernel());
    kernel = std::move(k);
    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
