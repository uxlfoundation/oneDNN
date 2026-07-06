/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#include "cpu/rv64/brgemm/jit_brgemm_kernel.hpp"
#include "cpu/rv64/rvjit/rvjit.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;
using namespace rvjit;

// Single generator for uniform and widening operations
struct jit_brgemm_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_t)

    jit_brgemm_kernel_t(const brgemm_desc_t &brg)
        : jit_generator_t("rv64_brgemm_kernel_jit"), brg_(brg) {}

    void operator()(brgemm_kernel_params_t *p) const {
        jit_generator_t::operator()(p);
    }

    const brgemm_desc_t &get_brg() const { return brg_; }

protected:
    void generate() override;

private:
    brgemm_desc_t brg_;
};

void jit_brgemm_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1

    const data_type_t dt_c = brg_.dt_c;
    const bool c_is_int = is_int_optype(to_rvjit_optype(dt_c));

    // int8 A is pre-widened to s32 during packing (see rvv_brgemm_matmul.cpp's
    // jit_pack_a_tile_t), so LDA must stride by the widened 4-byte element,
    // not brg_.typesize_A (which stays 1, matching the descriptor's logical
    // s8 dtype).
    const int sewbb = brg_.typesize_B;
    const dim_t LDA
            = brg_.LDA * (brg_.is_int8 ? brg_.typesize_C : brg_.typesize_A);
    const dim_t LDB = brg_.LDB * sewbb;
    const dim_t LDC = brg_.LDC * brg_.typesize_C;

    // Live registers
    const Reg args = a0; // Arguments
    const Reg ptra = a1; // A pointer
    const Reg ptrb = a2; // B pointer
    const Reg ptrc = a3; // C pointer

    // rvjit component system setup

    rvjit_t m(*this);
    m.set_model(rv64_rvjit_model());
    auto &ca = m.const_folding();
    auto &cf = m.control_flow();
    auto &pool = m.register_pool();
    auto &mem = m.memory_move();
    auto &eng = m.matmul_engine();

    pool.int_register_file_excluding({args, ptra, ptrb, ptrc});

    matmul_plan_t plan;
    // int8 A is pre-widened to s32 during packing; B stays raw s8. Every
    // other dtype combination has A and B share one on-disk dtype.
    plan.dti = brg_.is_int8 ? optype_t::s32 : to_rvjit_optype(brg_.dt_a);
    plan.dto = to_rvjit_optype(brg_.dt_c);
    plan.dtb = brg_.is_int8 ? optype_t::s8 : plan.dti;
    plan.ptra = ptra;
    plan.ptrb = ptrb;
    plan.ptrc = ptrc;
    plan.strides = matmul_strides_t::from_bytes(LDA, LDB, LDC);

    plan.max_n_ur = brg_.n_step;
    plan.k_ur = brg_.rd_block;
    plan.k_ur_tail = 4;

    plan.n_loop = loop_t::unroll_and_switch().limit([&](const Reg &lim) {
        ld(lim, args, offsetof(brgemm_kernel_params_t, N));
    });
    plan.k_loop = loop_t::unroll().limit([&](const Reg &lim) {
        ld(lim, args, offsetof(brgemm_kernel_params_t, K));
    });
    plan.avl = [&](const Reg &avl) {
        ld(avl, args, offsetof(brgemm_kernel_params_t, M));
    };

    eng.configure(plan);

    // Post-ops: k_loop's iter is dead after dense_loop, reuse it
    const Reg bias = eng.k_loop().iter();
    const Reg beta = bias;
    const Reg Boff = bias;

    // Code start

    pool.preserve();

    ld(ptra, args, offsetof(brgemm_kernel_params_t, ptr_A));
    ld(ptrb, args, offsetof(brgemm_kernel_params_t, ptr_B));
    ld(ptrc, args, offsetof(brgemm_kernel_params_t, ptr_C));

    eng.generate([&](v_block_t c, v_block_t vtmp) {
        const VReg t = vtmp(0);

        ld(bias, args, offsetof(brgemm_kernel_params_t, ptr_bias));
        cf.if_(branch_t::nez(bias), [&] {
            mem.vle(t, bias, to_rvjit_sew(dt_c));
            for (int i = 0; i < c.size(); ++i)
                c_is_int ? vadd_vv(c[i], c[i], t) : vfadd_vv(c[i], c[i], t);
        });

        lw(beta, args, offsetof(brgemm_kernel_params_t, beta));
        cf.if_(branch_t::nez(beta), [&](bool nez) {
            if (nez) {
                // beta != 0: C[col] = C[col] + accum
                for (int i = 0; i < c.size(); ++i) {
                    mem.vle(t, ptrc, to_rvjit_sew(dt_c));
                    c_is_int ? vadd_vv(c[i], c[i], t) : vfadd_vv(c[i], c[i], t);
                    mem.vse(c[i], ptrc, to_rvjit_sew(dt_c));
                    ca.add_const(ptrc, ptrc, eng.ldc());
                }
            } else {
                // beta == 0: C[col] = accum (overwrite)
                for (int i = 0; i < c.size(); ++i) {
                    mem.vse(c[i], ptrc, to_rvjit_sew(dt_c));
                    ca.add_const(ptrc, ptrc, eng.ldc());
                }
            }
        });

        // Advance B pointer for next N group
        const const_t B_off = ca.init_constant(const_t(c.size() * LDB, Boff));
        ca.add_const(ptrb, ptrb, B_off);
    });

    pool.restore();
    ret();
#else
    // RVV JIT is disabled at build time.
    ret();
#endif
}

brgemm_kernel_common_t::brgemm_kernel_common_t(const brgemm_desc_t &brg)
    : brg_(brg), jit_kernel_(new jit_brgemm_kernel_t(brg)) {}

brgemm_kernel_common_t::~brgemm_kernel_common_t() {
    delete jit_kernel_;
}

status_t brgemm_kernel_common_t::create_kernel() {
    return jit_kernel_->create_kernel();
}

void brgemm_kernel_common_t::operator()(brgemm_kernel_params_t *p) const {
    (*jit_kernel_)(p);
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
