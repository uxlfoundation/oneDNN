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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/platform.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/rvv_brgemm_matmul.hpp"
#include "cpu/rv64/rvv_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;

// ---------------------------------------------------------------------------
// JIT kernel: pack_a_tile
// Copies valid_rows floats per column from col-major A (stride LDA_orig)
// into contiguous workspace (stride bd). Vectorized with LMUL=m4.
// ---------------------------------------------------------------------------
struct jit_pack_a_tile_t : public jit_generator_t {
    struct call_params_t {
        float *ws; // offset 0
        const float *A; // offset 8
        dim_t LDA_orig; // offset 16
        dim_t bd; // offset 24
        dim_t valid_rows; // offset 32
        dim_t K_inner; // offset 40
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_pack_a_tile_t)

    jit_pack_a_tile_t() : jit_generator_t("jit_pack_a_tile") {
        create_kernel();
    }

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
        using namespace Xbyak_riscv;

        const Reg reg_param = a0;
        const Reg reg_ws = a1;
        const Reg reg_A = a2;
        const Reg reg_LDA = t0;
        const Reg reg_bd = t1;
        const Reg reg_K = t2;
        const Reg reg_k = t3; // outer loop counter
        const Reg reg_src = t4;
        const Reg reg_dst = t5;
        const Reg reg_rows_remaining = t6;
        const Reg reg_vl = a3;
        const Reg reg_bytes = a4;
        const Reg reg_tmp = a5;

        const VReg v_tmp(0);

        // Load parameters
        ld(reg_ws, reg_param, 0);
        ld(reg_A, reg_param, 8);
        ld(reg_LDA, reg_param, 16);
        ld(reg_bd, reg_param, 24);
        ld(reg_rows_remaining, reg_param, 32); // reuse as valid_rows temp
        ld(reg_K, reg_param, 40);

        // Compute LDA_orig * sizeof(float) and bd * sizeof(float)
        slli(reg_LDA, reg_LDA, 2);
        slli(reg_bd, reg_bd, 2);

        // Save valid_rows for reuse in each k iteration
        const Reg reg_valid_rows = a6;
        ld(reg_valid_rows, reg_param, 32);

        xor_(reg_k, reg_k, reg_k); // k = 0

        Label k_loop, k_done;
        L(k_loop);
        beq(reg_k, reg_K, k_done);

        // src = A + k * LDA_bytes
        mul(reg_tmp, reg_k, reg_LDA);
        add(reg_src, reg_A, reg_tmp);

        // dst = ws + k * bd_bytes
        mul(reg_tmp, reg_k, reg_bd);
        add(reg_dst, reg_ws, reg_tmp);

        // Inner copy loop: copy valid_rows floats
        mv(reg_rows_remaining, reg_valid_rows);
        Label copy_loop, copy_done;
        L(copy_loop);
        beqz(reg_rows_remaining, copy_done);

        vsetvli(reg_vl, reg_rows_remaining, SEW::e32, LMUL::m4);
        vle32_v(v_tmp, reg_src);
        vse32_v(v_tmp, reg_dst);

        slli(reg_bytes, reg_vl, 2);
        add(reg_src, reg_src, reg_bytes);
        add(reg_dst, reg_dst, reg_bytes);
        sub(reg_rows_remaining, reg_rows_remaining, reg_vl);
        j_(copy_loop);

        L(copy_done);

        addi(reg_k, reg_k, 1);
        j_(k_loop);

        L(k_done);
        ret();
#else
        ret();
#endif
    }
};

// ---------------------------------------------------------------------------
// JIT kernel: bias + post-ops for one row
// Processes N elements: load dst, add bias (vector/scalar/none),
// apply post-op (none/relu), store dst. Vectorized with LMUL=m1.
// ---------------------------------------------------------------------------
struct jit_bias_postops_row_t : public jit_generator_t {
    // bias_type: 0=none, 1=scalar broadcast, 2=per-element vector
    // postop: 0=none, 1=relu
    struct call_params_t {
        float *row_dst; // offset 0
        dim_t N; // offset 8
        const float *bias_ptr; // offset 16
        int32_t bias_type; // offset 24
        int32_t postop; // offset 28
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bias_postops_row_t)

    jit_bias_postops_row_t() : jit_generator_t("jit_bias_postops_row") {
        create_kernel();
    }

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
        using namespace Xbyak_riscv;

        const Reg reg_param = a0;
        const Reg reg_dst = a1;
        const Reg reg_N = a2;
        const Reg reg_bias = a3;
        const Reg reg_vl = t0;
        const Reg reg_bytes = t1;
        const Reg reg_tmp = t2;
        const Reg reg_bias_type = t3;
        const Reg reg_postop = t4;

        const FReg f_bias_scalar = fa0;
        const FReg f_zero = fa1;
        const VReg v_acc(0);
        const VReg v_bias(1);

        // Load parameters
        ld(reg_dst, reg_param, 0);
        ld(reg_N, reg_param, 8);
        ld(reg_bias, reg_param, 16);
        lw(reg_bias_type, reg_param, 24);
        lw(reg_postop, reg_param, 28);

        // Prepare f_zero = 0.0 for ReLU
        xor_(reg_tmp, reg_tmp, reg_tmp);
        fmv_w_x(f_zero, reg_tmp);

        Label loop, done;
        L(loop);
        beqz(reg_N, done);

        vsetvli(reg_vl, reg_N, SEW::e32, LMUL::m1);
        vle32_v(v_acc, reg_dst);

        // Bias: 0=none, 1=scalar, 2=vector
        Label bias_scalar, bias_vector, after_bias;
        bnez(reg_bias_type, bias_scalar);

        // bias_type == 0: no bias
        j_(after_bias);

        L(bias_scalar);
        // Load bias[0] into float register
        flw(f_bias_scalar, reg_bias, 0);
        // Check if bias_type == 1 (scalar) or 2 (vector)
        addi(reg_tmp, reg_bias_type, -1);
        bnez(reg_tmp, bias_vector);

        // Scalar bias: acc += broadcast
        vfadd_vf(v_acc, v_acc, f_bias_scalar);
        j_(after_bias);

        L(bias_vector);
        // Vector bias: acc += bias[n0..n0+vl)
        vle32_v(v_bias, reg_bias);
        vfadd_vv(v_acc, v_acc, v_bias);
        // Advance bias pointer for next iteration
        slli(reg_bytes, reg_vl, 2);
        add(reg_bias, reg_bias, reg_bytes);

        L(after_bias);

        // Post-op: ReLU (max(acc, 0))
        Label postop_done;
        beqz(reg_postop, postop_done);
        vfmax_vf(v_acc, v_acc, f_zero);
        L(postop_done);

        // Store result
        vse32_v(v_acc, reg_dst);

        // Advance dst pointer
        slli(reg_bytes, reg_vl, 2);
        add(reg_dst, reg_dst, reg_bytes);
        sub(reg_N, reg_N, reg_vl);
        j_(loop);

        L(done);
        ret();
#else
        ret();
#endif
    }
};

status_t rvv_brgemm_matmul_t::pd_t::init(engine_t *engine) {
    using smask_t = primitive_attr_t::skip_mask_t;

    VDISPATCH_MATMUL(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);

    VDISPATCH_MATMUL(dnnl_get_max_threads() <= 1, VERBOSE_IMPL_HEURISTIC_FAIL,
            "brgemm matmul is single-thread only");

    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    const memory_desc_wrapper src_mdw(src_md(0));
    const memory_desc_wrapper wei_mdw(weights_md(0));
    const memory_desc_wrapper dst_mdw(dst_md(0));
    const memory_desc_wrapper bias_mdw = bias_md_;

    VDISPATCH_MATMUL(!src_mdw.has_runtime_dims_or_strides()
                    && !wei_mdw.has_runtime_dims_or_strides()
                    && !dst_mdw.has_runtime_dims_or_strides()
                    && !bias_mdw.has_runtime_dims_or_strides(),
            VERBOSE_UNSUPPORTED_TAG);

    const bool types_ok = src_mdw.data_type() == f32
            && wei_mdw.data_type() == f32 && dst_mdw.data_type() == f32
            && IMPLICATION(!bias_mdw.is_zero(), bias_mdw.data_type() == f32)
            && desc()->accum_data_type == f32;
    VDISPATCH_MATMUL(types_ok, VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_MATMUL(attr()->has_default_values(smask_t::post_ops, f32),
            VERBOSE_UNSUPPORTED_ATTR);

    VDISPATCH_MATMUL(rvv_postops_t::post_ops_ok(attr()->post_ops_),
            VERBOSE_UNSUPPORTED_POSTOP);

    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    const int ndims = src_mdw.ndims();
    const int wei_ndims = wei_mdw.ndims();

    // Plain dense tensors
    VDISPATCH_MATMUL(src_mdw.blocking_desc().inner_nblks == 0
                    && wei_mdw.blocking_desc().inner_nblks == 0
                    && dst_mdw.blocking_desc().inner_nblks == 0,
            VERBOSE_UNSUPPORTED_TAG);

    // All tensors must be dense row-major (full stride chain verified).
    // Weights must be row-major; col-major would require transpose which
    // is not yet supported with copy_A packing.
    VDISPATCH_MATMUL(is_row_major(src_mdw), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(is_row_major(dst_mdw), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(is_row_major(wei_mdw), VERBOSE_UNSUPPORTED_TAG);

    // Check bias
    if (!bias_mdw.is_zero()) {
        VDISPATCH_MATMUL(bias_mdw.data_type() == f32, VERBOSE_UNSUPPORTED_DT);
        const int bias_ndims = bias_mdw.ndims();
        const auto *bias_dims = bias_mdw.dims();
        const auto *dst_dims = dst_mdw.dims();
        const int dst_ndims = dst_mdw.ndims();
        VDISPATCH_MATMUL(bias_ndims <= dst_ndims, VERBOSE_UNSUPPORTED_BIAS_CFG);
        for (int d = 1; d <= bias_ndims; ++d) {
            dim_t bias_dim = bias_dims[bias_ndims - d];
            dim_t dst_dim = dst_dims[dst_ndims - d];
            VDISPATCH_MATMUL(bias_dim == 1 || bias_dim == dst_dim,
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
        }
    }

    // Check weight broadcast
    {
        bool bc_ok = true;
        for (int i = 0; i < wei_ndims - 2; ++i) {
            if (src_mdw.dims()[i] != wei_mdw.dims()[i]
                    && wei_mdw.dims()[i] != 1) {
                bc_ok = false;
                break;
            }
        }
        VDISPATCH_MATMUL(bc_ok, VERBOSE_UNSUPPORTED_TAG);
    }

    // Extract dimensions (same as rvv_matmul.cpp)
    const dim_t *src_dims = src_mdw.dims();
    const dim_t *wei_dims = wei_mdw.dims();

    batch_ = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch_ *= src_dims[i];

    M_ = src_dims[ndims - 2];
    K_ = src_dims[ndims - 1];
    N_ = wei_dims[wei_ndims - 1];

    dim_t weights_batch_size = 1;
    for (int i = 0; i < wei_ndims - 2; ++i)
        weights_batch_size *= wei_dims[i];
    weights_are_broadcast_ = (weights_batch_size == 1 && batch_ > 1);

    // Shape guards
    VDISPATCH_MATMUL(weights_are_broadcast_, VERBOSE_IMPL_HEURISTIC_FAIL,
            "weights are not broadcast across batch");
    VDISPATCH_MATMUL(
            N_ >= 16, VERBOSE_IMPL_HEURISTIC_FAIL, "N too small for brgemm");

    // BRGEMM's copy_A packing reads A sequentially (stride bd_block) while
    // GEMM reads A with stride N. When A exceeds L2 cache, sequential reads
    // from DRAM are ~2x faster than strided reads. Two conditions ensure
    // BRGEMM is beneficial:
    // 1. K >= 4096: K-blocking amortizes packing overhead
    // 2. A exceeds L2: N * K * 4 > L2_size
    // 3. batch * M >= 128: BRGEMM kernel's N-loop needs enough iterations
    //    to pipeline efficiently; fewer causes regression (benchmarked).
    const dim_t A_bytes = N_ * K_ * (dim_t)sizeof(float);
    const auto L2_bytes = platform::get_per_core_cache_size(3);
    VDISPATCH_MATMUL(K_ >= 4096 && A_bytes > L2_bytes && batch_ * M_ >= 128,
            VERBOSE_IMPL_HEURISTIC_FAIL,
            "shape not beneficial for brgemm matmul");

    // Compute blocking parameters
    const int vlen_f32 = get_platform_vlen() / 32;
    const int bd_block = vlen_f32 * 4; // LMUL=m4
    const dim_t M_brg = N_;
    const dim_t K_brg = K_;
    const dim_t LDA = bd_block; // packed stride (not N)
    const dim_t LDB = K_;
    const dim_t LDC = N_;
    const dim_t N_brg = batch_ * M_;

    brgemm_desc_t brg_desc;
    CHECK(brgemm_desc_init(&brg_desc, v, brgemm_strd, f32, f32,
            brgemm_col_major, 1.0f, 0.0f, LDA, LDB, LDC, M_brg, N_brg, K_brg));

    brgemm_kernel_t *kernel = nullptr;
    CHECK(brgemm_kernel_create(&kernel, brg_desc));
    brg_kernel_.reset(kernel);

    // Create JIT kernels for pack_a_tile and bias+postops
    pack_kernel_.reset(new jit_pack_a_tile_t());
    bias_postops_kernel_.reset(new jit_bias_postops_row_t());

    init_scratchpad();

    return status::success;
}

void rvv_brgemm_matmul_t::pd_t::init_scratchpad() {
    using namespace memory_tracking::names;
    const auto &brg = brg_kernel_->get_brg();
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.template book<float>(
            key_brgemm_primitive_buffer_a, brg.bd_block * K_);
}

status_t rvv_brgemm_matmul_t::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const dim_t M = pd()->M_;
    const dim_t N = pd()->N_;
    const dim_t K = pd()->K_;
    const dim_t batch = pd()->batch_;

    const auto &brg = pd()->brg_kernel_->get_brg();
    const int bd = brg.bd_block;
    const int bdb = brg.bdb;
    const int bdb_tail = brg.bdb_tail;
    const dim_t total_N = batch * M;
    const dim_t LDA_orig = N; // row-major weights: original col-major LDA = N

    const auto *brg_kernel = pd()->brg_kernel_.get();
    const auto *pack_kernel = pd()->pack_kernel_.get();
    const auto *bias_postops_kernel = pd()->bias_postops_kernel_.get();

    // Packing workspace from scratchpad.
    // Size: bd × K × sizeof(float). For bd=32, K=4096: 512KB.
    // Pack once per M-tile, then call kernel per K-block with offset.
    const dim_t BK = BRGEMM_BK;
    auto &grantor = ctx.get_scratchpad_grantor();
    float *ws = grantor.template get<float>(
            memory_tracking::names::key_brgemm_primitive_buffer_a);

    // Manual M-tile × K-block loop with copy_A packing.
    //
    // For each M-tile:
    // 1. Pack the tile's full A data from col-major (LDA=N) into contiguous
    //    layout (LDA=bd).
    // 2. For each K-block, call the JIT kernel with ptr_A pointing into the
    //    packed buffer at the correct K-block offset.
    const int num_tiles = bdb + (bdb_tail > 0 ? 1 : 0);
    for (int t = 0; t < num_tiles; t++) {
        const bool is_tail = (t == bdb);
        const int rows = is_tail ? bdb_tail : bd;
        const float *A_tile = weights + t * bd;

        // JIT pack_a_tile
        jit_pack_a_tile_t::call_params_t pack_p;
        pack_p.ws = ws;
        pack_p.A = A_tile;
        pack_p.LDA_orig = LDA_orig;
        pack_p.bd = bd;
        pack_p.valid_rows = rows;
        pack_p.K_inner = K;
        (*pack_kernel)(&pack_p);

        for (dim_t kb = 0; kb < K; kb += BK) {
            const dim_t K_inner = nstl::min(BK, K - kb);
            const float beta_kb = (kb == 0) ? 0.0f : 1.0f;

            brgemm_kernel_params_t p;
            p.ptr_A = ws + kb * bd;
            p.ptr_B = src + kb;
            p.ptr_C = dst + t * bd;
            p.N = total_N;
            p.M = rows;
            p.K = K_inner;
            p.beta = beta_kb;
            p.ptr_bias = nullptr;
            (*brg_kernel)(&p);
        }
    }

    // Apply bias + post-ops using JIT kernel
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);
    const float *bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    const post_ops_t &post_ops = pd()->attr()->post_ops_;

    if (!bias && post_ops.len() == 0) return status::success;

    const int dst_ndims = dst_d.ndims();
    const int bias_ndims = bias_d.ndims();
    const dim_t *bias_dims = bias_d.dims();
    const memory_desc_wrapper src_d(pd()->src_md());
    const dim_t *src_dims_ptr = src_d.dims();
    const dim_t dst_batch_stride = M * N;

    // Determine postop type for JIT kernel
    int32_t postop = 0;
    if (post_ops.len() > 0 && post_ops.entry_[0].is_eltwise()
            && post_ops.entry_[0].eltwise.alg == alg_kind::eltwise_relu)
        postop = 1;

    parallel_nd(batch, [&](dim_t b) {
        float *dst_base = dst + b * dst_batch_stride;

        dim_t dst_idx_prefix[DNNL_MAX_NDIMS] = {};
        size_t bias_strides[DNNL_MAX_NDIMS] = {};

        if (bias && bias_ndims > 1) {
            bias_strides[bias_ndims - 1] = 1;
            for (int d = bias_ndims - 2; d >= 0; --d)
                bias_strides[d]
                        = bias_strides[d + 1] * (size_t)bias_dims[d + 1];
        }

        for (dim_t m = 0; m < M; ++m) {
            if (dst_ndims > 2) {
                utils::l_dims_by_l_offset(
                        dst_idx_prefix, b, src_dims_ptr, dst_ndims - 2);
            }
            dst_idx_prefix[dst_ndims - 2] = m;

            float *row_dst = dst_base + m * N;

            int32_t bias_type = 0; // 0=none
            const float *bias_ptr = nullptr;
            if (bias) {
                if (bias_d.nelems() == 1) {
                    bias_type = 1; // scalar broadcast
                    bias_ptr = bias;
                } else {
                    size_t base_bias_off = 0;
                    if (bias_ndims > 1) {
                        for (int d = 0; d < bias_ndims - 1; ++d) {
                            int dst_dim_idx = d + (dst_ndims - bias_ndims);
                            dim_t idx = (bias_dims[d] == 1)
                                    ? 0
                                    : dst_idx_prefix[dst_dim_idx];
                            base_bias_off += idx * bias_strides[d];
                        }
                    }
                    bias_ptr = bias + base_bias_off;
                    if (bias_dims[bias_ndims - 1] == 1)
                        bias_type = 1; // scalar
                    else
                        bias_type = 2; // vector
                }
            }

            jit_bias_postops_row_t::call_params_t bp;
            bp.row_dst = row_dst;
            bp.N = N;
            bp.bias_ptr = bias_ptr;
            bp.bias_type = bias_type;
            bp.postop = postop;
            (*bias_postops_kernel)(&bp);
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
