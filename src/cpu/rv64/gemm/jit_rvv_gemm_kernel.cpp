/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#include "cpu/rv64/gemm/jit_rvv_gemm_kernel.hpp"
#include "common/verbose.hpp"
#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"
#include "cpu/rv64/rvjit/rvjit.hpp"

#include <memory>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

using namespace Xbyak_riscv;
using namespace rvjit;

jit_rvv_gemm_kernel_t::jit_rvv_gemm_kernel_t(
        bool isTransA, bool isTransB, bool has_bias)
    : jit_generator_t("rv64_gemm_kernel_f32_jit")
    , isTransA_(isTransA)
    , isTransB_(isTransB)
    , has_bias_(has_bias) {
    create_kernel();
}

void jit_rvv_gemm_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1

    // Operand data types — uniform f32
    const data_type_t dt_a = data_type::f32;
    const data_type_t dt_c = data_type::f32;
    const int sewba = sizeof(float);
    const int sewbb = sizeof(float);
    const int sewbc = sizeof(float);

    // rvjit component system
    rvjit_t m(*this);
    m.set_model(rv64_rvjit_model());
    auto &cf = m.control_flow();
    auto &pool = m.register_pool();
    auto &mem = m.memory_move();
    auto &eng = m.matmul_engine();

    // Live registers
    const Reg args = a0;
    const Reg ptra = a1;
    const Reg ptrb = a2;
    const Reg ptrc = a3;
    const Reg lda = a4;
    const Reg ldb = a5;
    const Reg ldc = a6;
    const FReg alpha = fa0;
    const FReg beta = fa1;

    pool.int_register_file_excluding({args, ptra, ptrb, ptrc, lda, ldb, ldc});

    matmul_plan_t plan;
    plan.dti = to_rvjit_optype(dt_a);
    plan.dto = to_rvjit_optype(dt_c);
    plan.dtb = plan.dti;
    plan.ptra = ptra;
    plan.ptrb = ptrb;
    plan.ptrc = ptrc;
    plan.is_transa = isTransA_;
    plan.is_transb = isTransB_;
    plan.strides = matmul_strides_t::from_regs(lda, ldb, ldc);
    plan.max_n_ur
            = static_cast<int>(gemm_utils_traits<float>::get_n_unroll_factor());

    plan.n_loop = loop_t::switch_().id(
            [&](const Reg &r) { ld(r, args, offsetof(call_params_t, n)); });
    plan.k_loop = loop_t::unroll().limit(
            [&](const Reg &lim) { ld(lim, args, offsetof(call_params_t, K)); });
    plan.avl = [&](const Reg &avl) {
        ld(avl, args, offsetof(call_params_t, m));
    };

    if (!eng.configure(plan)) {
        VERROR(primitive, create,
                "rv64: gemm: failed to configure matmul "
                "component (N_UR too large for the "
                "available register file)");
        return;
    }

    // Post-ops: the switch dispatch id is dead once inside a case body, reuse it
    const Reg beta_bits = eng.n_loop().id();
    const Reg bias_ptr = eng.n_loop().id();

    // Code start

    pool.preserve();

    // Prepare pointers
    ld(ptra, args, offsetof(call_params_t, A));
    ld(ptrb, args, offsetof(call_params_t, B));
    ld(ptrc, args, offsetof(call_params_t, C));

    // Prepare strides
    ld(lda, args, offsetof(call_params_t, lda));
    ld(ldb, args, offsetof(call_params_t, ldb));
    ld(ldc, args, offsetof(call_params_t, ldc));
    slli(lda, lda, math::ilog2q(sewba));
    slli(ldb, ldb, math::ilog2q(sewbb));
    slli(ldc, ldc, math::ilog2q(sewbc));

    eng.generate([&](v_block_t c, v_block_t vtmp) {
        const VReg t = vtmp(0);

        // Post-ops
        lw(beta_bits, args, offsetof(call_params_t, beta));
        flw(alpha, args, offsetof(call_params_t, alpha));
        flw(beta, args, offsetof(call_params_t, beta));
        cf.if_(branch_t::nez(beta_bits), [&](bool nonzero) {
            for (int n = 0; n < c.size(); n++) {
                if (nonzero) {
                    // beta != 0: result = alpha*acc + beta*C [+ bias]
                    mem.vle(t, ptrc, to_rvjit_sew(dt_c));
                    vfmul_vf(c[n], c[n], alpha);
                    vfmacc_vf(c[n], beta, t);
                    if (has_bias_) {
                        ld(bias_ptr, args, offsetof(call_params_t, bias));
                        cf.if_(branch_t::nez(bias_ptr), [&] {
                            mem.vle(t, bias_ptr, to_rvjit_sew(dt_c));
                            vfadd_vv(c[n], c[n], t);
                        });
                    }
                    mem.vse(c[n], ptrc, to_rvjit_sew(dt_c));
                } else {
                    // beta == 0: result = alpha*acc [+ bias]
                    vfmul_vf(c[n], c[n], alpha);
                    if (has_bias_) {
                        ld(bias_ptr, args, offsetof(call_params_t, bias));
                        cf.if_(branch_t::nez(bias_ptr), [&] {
                            mem.vle(t, bias_ptr, to_rvjit_sew(dt_c));
                            vfadd_vv(c[n], c[n], t);
                        });
                    }
                    mem.vse(c[n], ptrc, to_rvjit_sew(dt_c));
                }
                add(ptrc, ptrc, ldc);
            }
        });
    });

    pool.restore();
    ret();
#else
    ret();
#endif
}

namespace {

template <bool isTransA, bool isTransB>
struct jit_rvv_gemm_kernel_storage_t {
    jit_rvv_gemm_kernel_t nb {isTransA, isTransB, false};
    jit_rvv_gemm_kernel_t b {isTransA, isTransB, true};
};

template <bool isTransA, bool isTransB>
void jit_rvv_gemm_kernel_dispatch(const float *A, const float *B, float *C,
        dim_t lda, dim_t ldb, dim_t ldc, dim_t K, float alpha, float beta,
        dim_t m, dim_t n, const float *bias) {
    static jit_rvv_gemm_kernel_storage_t<isTransA, isTransB> storage;

    static bool verbose_printed = false;
    if (!verbose_printed) {
        VINFO(primitive, create, dispatch, rvv_gemm_jit,
                "JIT gemm kernel taking over: m=%d, n=%d", (int)m, (int)n);
        verbose_printed = true;
    }

    jit_rvv_gemm_kernel_t::call_params_t p;
    p.A = A;
    p.B = B;
    p.C = C;
    p.lda = lda;
    p.ldb = ldb;
    p.ldc = ldc;
    p.K = K;
    p.m = m;
    p.n = n;
    p.alpha = alpha;
    p.beta = beta;
    p.bias = bias;

    auto *kernel = bias ? &storage.b : &storage.nb;
    (*kernel)(&p);
}

} // namespace

void jit_rvv_gemm_kernel(const float *A, const float *B, float *C, dim_t lda,
        dim_t ldb, dim_t ldc, dim_t K, float alpha, float beta, dim_t m,
        dim_t n, bool isTransA, bool isTransB, const float *bias) {
    if (!isTransA && !isTransB) {
        jit_rvv_gemm_kernel_dispatch<false, false>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n, bias);
    } else if (isTransA && !isTransB) {
        jit_rvv_gemm_kernel_dispatch<true, false>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n, bias);
    } else if (!isTransA && isTransB) {
        jit_rvv_gemm_kernel_dispatch<false, true>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n, bias);
    } else {
        jit_rvv_gemm_kernel_dispatch<true, true>(
                A, B, C, lda, ldb, ldc, K, alpha, beta, m, n, bias);
    }
}

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
