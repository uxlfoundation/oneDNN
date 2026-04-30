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

#include <cstring>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/rvv_winograd_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::memory_tracking::names;

namespace {
// Winograd F(2x2, 3x3) filter transform matrix
// Filter transform matrix G (4x3)
constexpr float G[4][3] = {{1.0f, 0.0f, 0.0f}, {0.5f, 0.5f, 0.5f},
        {0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}};

// Pre-compute filter transform with GEMM-layout: 3x3 -> 4x4
// Transform = G * filter * G^T
// Output layout: [16][ic_rounded][oc_rounded] for brgemm col-major access
// Computes directly without intermediate allocation.
void compute_filter_transform_3x3_to_4x4_gemm_layout(const float *filter,
        float *transformed, int oc, int ic, int ic_rounded, int oc_rounded) {

    std::memset(transformed, 0, 16 * ic_rounded * oc_rounded * sizeof(float));

    for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
        for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
            const float *f = &filter[(oc_idx * ic + ic_idx) * 9];

            // Step 1: temp = G * filter (4x3)
            float temp[4][3];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        sum += G[i][k] * f[k * 3 + j];
                    }
                    temp[i][j] = sum;
                }
            }

            // Step 2: result = temp * G^T, store directly to GEMM layout
            // Layout: [elem][ic_idx * oc_rounded + oc_idx]
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        sum += temp[i][k] * G[j][k];
                    }
                    int elem = i * 4 + j;
                    transformed[elem * oc_rounded * ic_rounded
                            + ic_idx * oc_rounded + oc_idx]
                            = sum;
                }
            }
        }
    }
}
} // namespace

// -----------------------------------------------------------------------
// jit_wino_input_transform_t implementation
// -----------------------------------------------------------------------
jit_wino_input_transform_t::jit_wino_input_transform_t(
        const rvv_winograd_conf_t &conf)
    : jit_generator_t("wino_input_xform")
    , ic_(conf.ic)
    , ih_(conf.ih)
    , iw_(conf.iw)
    , pad_t_(conf.pad_t)
    , pad_l_(conf.pad_l)
    , input_ld_row_(conf.wspec.input_ld_row)
    , V_elem_stride_(conf.wspec.input_ld_batch) {}

void jit_wino_input_transform_t::generate() {
    using namespace Xbyak_riscv;

    // --- Scalar register allocation ---
    const Reg reg_src = a1; // src_batch base
    const Reg reg_V = a2; // V buffer base
    const Reg reg_ic_base = a3; // IC loop counter
    const Reg reg_vl = a4; // current VL
    const Reg reg_ic_total = a5; // ic_ (baked)
    const Reg reg_ic_spb = a6; // ic_spatial_stride * 4 (bytes)
    const Reg reg_oh_s = a7; // oh_s (computed per tile)
    const Reg reg_ow_s = t5; // ow_s (computed per tile)
    const Reg reg_tile_voff = t6; // tile offset in bytes (per tile)
    // Saved: baked constants + tile loop state
    const Reg reg_ih = s0, reg_iw = s1, reg_pad_t = s2, reg_pad_l = s3;
    const Reg reg_nb_oh = Xbyak_riscv::s4, reg_nb_ow = Xbyak_riscv::s5;
    const Reg reg_oh_tile = Xbyak_riscv::s6, reg_ow_tile = Xbyak_riscv::s7;

    // --- Vector registers (LMUL=1) ---
    const VReg vr0(0), vr1(1), vr2(2), vr3(3);
    const VReg vt0(4), vt1(5), vt2(6), vt3(7);
    const VReg vm0(8), vm1(9), vm2(10), vm3(11);

    // --- Prologue: save s0-s7 ---
    addi(sp, sp, -64);
    sd(reg_ih, sp, 0);
    sd(reg_iw, sp, 8);
    sd(reg_pad_t, sp, 16);
    sd(reg_pad_l, sp, 24);
    sd(reg_nb_oh, sp, 32);
    sd(reg_nb_ow, sp, 40);
    sd(reg_oh_tile, sp, 48);
    sd(reg_ow_tile, sp, 56);

    // Load params
    ld(reg_src, a0, 0); // src_batch
    ld(reg_V, a0, 8); // V
    ld(t0, a0, 16); // ic_spatial_stride
    ld(reg_nb_oh, a0, 24); // nb_oh
    ld(reg_nb_ow, a0, 32); // nb_ow

    slli(reg_ic_spb, t0, 2); // ic_spat_bytes

    // Bake constants
    li(reg_ic_total, ic_);
    li(reg_ih, ih_);
    li(reg_iw, iw_);
    li(reg_pad_t, pad_t_);
    li(reg_pad_l, pad_l_);

    // ---- Outer tile loop: oh_tile ----
    mv(reg_oh_tile, x0);
    Label lbl_oh_loop, lbl_oh_done;
    L(lbl_oh_loop);
    bge(reg_oh_tile, reg_nb_oh, lbl_oh_done);

    // oh_s = oh_tile * 2
    slli(reg_oh_s, reg_oh_tile, 1);

    // ---- Inner tile loop: ow_tile ----
    mv(reg_ow_tile, x0);
    Label lbl_ow_loop, lbl_ow_done;
    L(lbl_ow_loop);
    bge(reg_ow_tile, reg_nb_ow, lbl_ow_done);

    // ow_s = ow_tile * 2
    slli(reg_ow_s, reg_ow_tile, 1);

    // tile_voff = (oh_tile * nb_ow + ow_tile) * input_ld_row * 4
    mul(reg_tile_voff, reg_oh_tile, reg_nb_ow);
    add(reg_tile_voff, reg_tile_voff, reg_ow_tile);
    li(t0, input_ld_row_);
    mul(reg_tile_voff, reg_tile_voff, t0);
    slli(reg_tile_voff, reg_tile_voff, 2);

    // ---- IC loop ----
    mv(reg_ic_base, x0);
    Label lbl_ic_loop, lbl_ic_end;
    L(lbl_ic_loop);
    bge(reg_ic_base, reg_ic_total, lbl_ic_end);

    sub(t0, reg_ic_total, reg_ic_base);
    vsetvli(reg_vl, t0, SEW::e32, LMUL::m1);

    // src_row_base = src + ic_base * ic_spb
    mul(a0, reg_ic_base, reg_ic_spb);
    add(a0, a0, reg_src);

    // Lambda: load one k row, B-sparse, accumulate
    auto emit_load_k = [&](int k, bool positive) {
        addi(t1, reg_oh_s, k);
        sub(t1, t1, reg_pad_t);

        Label lbl_zero_row, lbl_row_done;
        blt(t1, x0, lbl_zero_row);
        bge(t1, reg_ih, lbl_zero_row);

        mul(t2, t1, reg_iw);
        slli(t2, t2, 2);
        add(t2, t2, a0);

        auto emit_col_load = [&](int j, const VReg &vt) {
            addi(t3, reg_ow_s, j);
            sub(t3, t3, reg_pad_l);
            Label lbl_skip_col, lbl_col_done;
            blt(t3, x0, lbl_skip_col);
            bge(t3, reg_iw, lbl_skip_col);
            slli(t4, t3, 2);
            add(t4, t4, t2);
            vlse32_v(vt, t4, reg_ic_spb);
            j_(lbl_col_done);
            L(lbl_skip_col);
            vmv_v_i(vt, 0);
            L(lbl_col_done);
        };

        emit_col_load(0, vt0);
        emit_col_load(1, vt1);
        emit_col_load(2, vt2);
        emit_col_load(3, vt3);

        j_(lbl_row_done);
        L(lbl_zero_row);
        vmv_v_i(vt0, 0);
        vmv_v_i(vt1, 0);
        vmv_v_i(vt2, 0);
        vmv_v_i(vt3, 0);
        L(lbl_row_done);

        vfsub_vv(vm0, vt0, vt2);
        vfadd_vv(vm1, vt1, vt2);
        vfsub_vv(vm2, vt2, vt1);
        vfsub_vv(vm3, vt3, vt1);

        if (positive) {
            vfadd_vv(vr0, vr0, vm0);
            vfadd_vv(vr1, vr1, vm1);
            vfadd_vv(vr2, vr2, vm2);
            vfadd_vv(vr3, vr3, vm3);
        } else {
            vfsub_vv(vr0, vr0, vm0);
            vfsub_vv(vr1, vr1, vm1);
            vfsub_vv(vr2, vr2, vm2);
            vfsub_vv(vr3, vr3, vm3);
        }
    };

    // Lambda: zero accum, 2 k values, store 4 results
    auto emit_out_i
            = [&](int out_i, int k_a, bool k_a_pos, int k_b, bool k_b_pos) {
        vmv_v_i(vr0, 0);
        vmv_v_i(vr1, 0);
        vmv_v_i(vr2, 0);
        vmv_v_i(vr3, 0);

        emit_load_k(k_a, k_a_pos);
        emit_load_k(k_b, k_b_pos);

        // Store to V buffer
        slli(t0, reg_ic_base, 2);
        add(t0, t0, reg_tile_voff);
        add(t0, t0, reg_V);
        li(t1, static_cast<uint64_t>(out_i) * 4 * V_elem_stride_ * 4);
        add(t0, t0, t1);
        li(t1, V_elem_stride_ * 4);

        vse32_v(vr0, t0);
        add(t0, t0, t1);
        vse32_v(vr1, t0);
        add(t0, t0, t1);
        vse32_v(vr2, t0);
        add(t0, t0, t1);
        vse32_v(vr3, t0);
    };

    emit_out_i(0, 0, true, 2, false);
    emit_out_i(1, 1, true, 2, true);
    emit_out_i(2, 1, false, 2, true);
    emit_out_i(3, 1, false, 3, true);

    add(reg_ic_base, reg_ic_base, reg_vl);
    j_(lbl_ic_loop);
    L(lbl_ic_end);

    // Advance ow_tile
    addi(reg_ow_tile, reg_ow_tile, 1);
    j_(lbl_ow_loop);
    L(lbl_ow_done);

    // Advance oh_tile
    addi(reg_oh_tile, reg_oh_tile, 1);
    j_(lbl_oh_loop);
    L(lbl_oh_done);

    // --- Epilogue: restore s0-s7 ---
    ld(reg_ih, sp, 0);
    ld(reg_iw, sp, 8);
    ld(reg_pad_t, sp, 16);
    ld(reg_pad_l, sp, 24);
    ld(reg_nb_oh, sp, 32);
    ld(reg_nb_ow, sp, 40);
    ld(reg_oh_tile, sp, 48);
    ld(reg_ow_tile, sp, 56);
    addi(sp, sp, 64);
    ret();
}

// -----------------------------------------------------------------------
// jit_wino_output_transform_t implementation
// -----------------------------------------------------------------------
jit_wino_output_transform_t::jit_wino_output_transform_t(
        const rvv_winograd_conf_t &conf)
    : jit_generator_t("wino_output_xform")
    , oc_(conf.oc)
    , oh_(conf.oh)
    , ow_(conf.ow)
    , N_(conf.wspec.N)
    , M_elem_stride_(conf.wspec.output_ld_batch)
    , oc_spatial_stride_(conf.oh * conf.ow)
    , with_bias_(conf.with_bias) {}

void jit_wino_output_transform_t::generate() {
    using namespace Xbyak_riscv;

    const Reg reg_M = a1;
    const Reg reg_dst = a2;
    const Reg reg_oc_base = a3;
    const Reg reg_vl = a4;
    const Reg reg_oc_total = a5;
    const Reg reg_bias = a6;
    const Reg reg_oc_spb = a7;
    const Reg reg_oh_s = t4;
    const Reg reg_ow_s = t5;
    const Reg reg_tile_moff = t6;
    // Saved
    const Reg reg_oh_lim = s0, reg_ow_lim = s1;
    const Reg reg_M_eb = s2, reg_N = s3;
    const Reg reg_nb_oh = Xbyak_riscv::s4, reg_nb_ow = Xbyak_riscv::s5;
    const Reg reg_oh_tile = Xbyak_riscv::s6, reg_ow_tile = Xbyak_riscv::s7;

    // --- Vector registers (LMUL=1) ---
    const VReg vr0(0), vr1(1);
    const VReg vw0(4), vw1(5), vw2(6), vw3(7);
    const VReg vtmp0(8), vtmp1(9);
    const VReg v_bias(10);

    // --- Prologue: save s0-s7 ---
    addi(sp, sp, -64);
    sd(reg_oh_lim, sp, 0);
    sd(reg_ow_lim, sp, 8);
    sd(reg_M_eb, sp, 16);
    sd(reg_N, sp, 24);
    sd(reg_nb_oh, sp, 32);
    sd(reg_nb_ow, sp, 40);
    sd(reg_oh_tile, sp, 48);
    sd(reg_ow_tile, sp, 56);

    // Load params
    ld(reg_M, a0, 0);
    ld(reg_bias, a0, 8);
    ld(reg_dst, a0, 16);
    ld(reg_nb_oh, a0, 24);
    ld(reg_nb_ow, a0, 32);

    // Bake constants
    li(reg_oc_total, oc_);
    li(reg_oh_lim, oh_);
    li(reg_ow_lim, ow_);
    li(reg_N, N_);
    li(reg_M_eb, M_elem_stride_ * 4);
    li(reg_oc_spb, oc_spatial_stride_ * 4);

    // Lambda: load w0-w3 and accumulate for one k
    auto emit_load_k = [&](int k, bool positive) {
        slli(t0, reg_oc_base, 2);
        add(t0, t0, reg_tile_moff);
        add(t0, t0, reg_M);
        if (k > 0) {
            li(t1, static_cast<uint64_t>(k) * 4);
            mul(t1, t1, reg_M_eb);
            add(t0, t0, t1);
        }

        vle32_v(vw0, t0);
        add(t0, t0, reg_M_eb);
        vle32_v(vw1, t0);
        add(t0, t0, reg_M_eb);
        vle32_v(vw2, t0);
        add(t0, t0, reg_M_eb);
        vle32_v(vw3, t0);

        vfadd_vv(vtmp0, vw0, vw1);
        vfadd_vv(vtmp0, vtmp0, vw2);
        vfsub_vv(vtmp1, vw1, vw2);
        vfadd_vv(vtmp1, vtmp1, vw3);

        if (positive) {
            vfadd_vv(vr0, vr0, vtmp0);
            vfadd_vv(vr1, vr1, vtmp1);
        } else {
            vfsub_vv(vr0, vr0, vtmp0);
            vfsub_vv(vr1, vr1, vtmp1);
        }
    };

    // Lambda: zero accum, 3 k values, bias, store
    auto emit_out_i = [&](int out_i, int k0, bool pos0, int k1, bool pos1,
                              int k2, bool pos2) {
        vmv_v_i(vr0, 0);
        vmv_v_i(vr1, 0);

        emit_load_k(k0, pos0);
        emit_load_k(k1, pos1);
        emit_load_k(k2, pos2);

        if (with_bias_) {
            vfadd_vv(vr0, vr0, v_bias);
            vfadd_vv(vr1, vr1, v_bias);
        }

        // Store with boundary checks
        addi(t0, reg_oh_s, out_i);
        Label lbl_store_done;
        bge(t0, reg_oh_lim, lbl_store_done);

        mul(t1, reg_oc_base, reg_oc_spb);
        add(t1, t1, reg_dst);
        mul(t2, t0, reg_ow_lim);
        slli(t2, t2, 2);
        add(t2, t2, t1);
        slli(t3, reg_ow_s, 2);
        add(t2, t2, t3);

        vsse32_v(vr0, t2, reg_oc_spb);

        addi(t3, reg_ow_s, 1);
        bge(t3, reg_ow_lim, lbl_store_done);
        addi(t2, t2, 4);
        vsse32_v(vr1, t2, reg_oc_spb);

        L(lbl_store_done);
    };

    // ---- Outer tile loop: oh_tile ----
    mv(reg_oh_tile, x0);
    Label lbl_oh_loop, lbl_oh_done;
    L(lbl_oh_loop);
    bge(reg_oh_tile, reg_nb_oh, lbl_oh_done);

    slli(reg_oh_s, reg_oh_tile, 1);

    // ---- Inner tile loop: ow_tile ----
    mv(reg_ow_tile, x0);
    Label lbl_ow_loop, lbl_ow_done;
    L(lbl_ow_loop);
    bge(reg_ow_tile, reg_nb_ow, lbl_ow_done);

    slli(reg_ow_s, reg_ow_tile, 1);

    // tile_moff = (oh_tile * nb_ow + ow_tile) * N * 4
    mul(reg_tile_moff, reg_oh_tile, reg_nb_ow);
    add(reg_tile_moff, reg_tile_moff, reg_ow_tile);
    mul(reg_tile_moff, reg_tile_moff, reg_N);
    slli(reg_tile_moff, reg_tile_moff, 2);

    // ---- OC loop ----
    mv(reg_oc_base, x0);
    Label lbl_oc_loop, lbl_oc_end;
    L(lbl_oc_loop);
    bge(reg_oc_base, reg_oc_total, lbl_oc_end);

    sub(t0, reg_oc_total, reg_oc_base);
    vsetvli(reg_vl, t0, SEW::e32, LMUL::m1);

    // Load bias once per OC chunk
    if (with_bias_) {
        slli(t0, reg_oc_base, 2);
        add(t0, reg_bias, t0);
        vle32_v(v_bias, t0);
    }

    emit_out_i(0, 0, true, 1, true, 2, true);
    emit_out_i(1, 1, true, 2, false, 3, true);

    add(reg_oc_base, reg_oc_base, reg_vl);
    j_(lbl_oc_loop);
    L(lbl_oc_end);

    addi(reg_ow_tile, reg_ow_tile, 1);
    j_(lbl_ow_loop);
    L(lbl_ow_done);

    addi(reg_oh_tile, reg_oh_tile, 1);
    j_(lbl_oh_loop);
    L(lbl_oh_done);

    // --- Epilogue: restore s0-s7 ---
    ld(reg_oh_lim, sp, 0);
    ld(reg_ow_lim, sp, 8);
    ld(reg_M_eb, sp, 16);
    ld(reg_N, sp, 24);
    ld(reg_nb_oh, sp, 32);
    ld(reg_nb_ow, sp, 40);
    ld(reg_oh_tile, sp, 48);
    ld(reg_ow_tile, sp, 56);
    addi(sp, sp, 64);
    ret();
}

// -----------------------------------------------------------------------
// rvv_wino_convolution_fwd_t implementation
// -----------------------------------------------------------------------
status_t rvv_wino_convolution_fwd_t::init(engine_t *engine) {
    auto *input = new jit_wino_input_transform_t(pd()->conf_);
    input_xform_.reset(input);
    CHECK(input->create_kernel());

    auto *output = new jit_wino_output_transform_t(pd()->conf_);
    output_xform_.reset(output);
    CHECK(output->create_kernel());

    return status::success;
}

status_t rvv_winograd_init_conf(rvv_winograd_conf_t &conf,
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr) {
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);

    const dim_t mb = src_d.dims()[0];
    conf.mb = mb;

    conf.ih = src_d.dims()[2];
    conf.iw = src_d.dims()[3];
    conf.oh = dst_d.dims()[2];
    conf.ow = dst_d.dims()[3];

    conf.ic = src_d.dims()[1];
    conf.oc = dst_d.dims()[1];

    conf.kh = weights_d.dims()[2];
    conf.kw = weights_d.dims()[3];

    conf.stride_h = cd.strides[0];
    conf.stride_w = cd.strides[1];

    conf.pad_t = cd.padding[0][0];
    conf.pad_l = cd.padding[0][1];
    conf.pad_b = cd.padding[1][0];
    conf.pad_r = cd.padding[1][1];

    conf.with_bias = cd.bias_desc.data_type != data_type::undef;

    // Compute Winograd domain specification for GEMM-based execution
    constexpr dim_t CACHE_LINE_SIZE = platform::get_cache_line_size();
    constexpr dim_t CACHE_LINE_FLOATS = CACHE_LINE_SIZE / sizeof(float); // 16

    // Matrix dimensions: C[M][N] = A[M][K] * B[K][N]
    conf.wspec.M = ((conf.oh + 1) / 2) * ((conf.ow + 1) / 2); // Total 2x2 tiles
    conf.wspec.K = conf.ic; // Input channels
    conf.wspec.N = conf.oc; // Output channels
    conf.wspec.n_gemms = 16; // Winograd F(2x2, 3x3)
    conf.wspec.n_batches = conf.mb;

    // 64-byte aligned leading dimensions (for efficient vectorization)
    // Weight matrix: B[N][K] where N=OC, K=IC (row-major for GEMM)
    // For weight transform, we need separate rounded dimensions
    conf.wspec.weight_oc_rounded = rnd_up(conf.wspec.N, CACHE_LINE_FLOATS);
    conf.wspec.weight_ic_rounded = rnd_up(conf.wspec.K, CACHE_LINE_FLOATS);

    // weight_ld_row is the leading dimension for column-major OC x IC matrix
    // A[oc_idx + ic_idx * lda], so lda = oc_rounded
    conf.wspec.weight_ld_row = conf.wspec.weight_oc_rounded;
    conf.wspec.weight_ld_matrix
            = conf.wspec.weight_oc_rounded * conf.wspec.weight_ic_rounded;

    // Input matrix: A[K][M] column-major where K=IC, M=tiles
    // Input buffer per thread: [16][tile_chunk x IC_rounded] per element
    conf.wspec.input_ld_row
            = rnd_up(conf.wspec.K, CACHE_LINE_FLOATS); // LDB = IC_rounded
    conf.wspec.input_ld_batch
            = conf.wspec.input_ld_row * conf.wspec.M; // per-elem stride

    // Output buffer per thread: [16][tiles x OC] per element
    conf.wspec.output_ld_row = conf.wspec.N; // LDC = OC
    conf.wspec.output_ld_batch
            = conf.wspec.output_ld_row * conf.wspec.M; // per-elem stride

    // Buffer sizes in floats
    conf.wspec.weight_matrix_size
            = conf.wspec.n_gemms * conf.wspec.weight_ld_matrix;
    conf.wspec.V_buffer_size = conf.wspec.n_gemms * conf.wspec.input_ld_batch;
    conf.wspec.M_buffer_size = conf.wspec.n_gemms * conf.wspec.output_ld_batch;

    // Scratchpad: U (transformed weights), V (transformed input), M (GEMM output)
    using namespace memory_tracking::names;

    scratchpad.book<float>(key_wino_U, conf.wspec.weight_matrix_size);
    scratchpad.book<float>(key_wino_V, conf.wspec.V_buffer_size);
    scratchpad.book<float>(key_wino_M, conf.wspec.M_buffer_size);

    return status::success;
}

status_t rvv_wino_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const auto &conf = pd()->conf_;
    const auto scratchpad = ctx.get_scratchpad_grantor();
    const auto *brg_kernel = pd()->brg_kernel_.get();

    using namespace memory_tracking::names;

    // Transform weights into scratchpad buffer every execute (like x64 brgemm)
    float *transformed_weights = scratchpad.template get<float>(key_wino_U);
    compute_filter_transform_3x3_to_4x4_gemm_layout(weights,
            transformed_weights, conf.wspec.N, conf.wspec.K,
            conf.wspec.weight_ic_rounded, conf.wspec.weight_oc_rounded);

    float *V = scratchpad.template get<float>(key_wino_V);
    float *M = scratchpad.template get<float>(key_wino_M);

    auto *input_xform = input_xform_.get();
    auto *output_xform = output_xform_.get();

    // Sequential batch processing: input transform -> GEMM -> output transform
    // per batch, keeping working set in L2 cache.
    const dim_t nb_oh = (conf.oh + 1) / 2;
    const dim_t nb_ow = (conf.ow + 1) / 2;
    const dim_t total_tiles = nb_oh * nb_ow;
    const dim_t V_elem_stride = conf.wspec.input_ld_batch;
    const dim_t M_elem_stride = conf.wspec.output_ld_batch;
    const dim_t ic_spatial_stride = conf.ih * conf.iw;
    const dim_t oc_spatial_stride = conf.oh * conf.ow;

    for (dim_t mb_idx = 0; mb_idx < conf.mb; mb_idx++) {
        const float *src_batch = src + mb_idx * conf.ic * ic_spatial_stride;

        // Step 1: Input transform via JIT kernel (tile loop is inside JIT)
        {
            jit_wino_input_transform_t::call_params_t params;
            params.src_batch = src_batch;
            params.V = V;
            params.ic_spatial_stride = ic_spatial_stride;
            params.nb_oh = nb_oh;
            params.nb_ow = nb_ow;
            (*input_xform)(&params);
        }

        // Step 2: GEMM using brgemm kernel (16 elements)
        for (int elem = 0; elem < 16; elem++) {
            const float *A_weights
                    = transformed_weights + elem * conf.wspec.weight_ld_matrix;
            const float *B_input = V + elem * V_elem_stride;
            float *C_output = M + elem * M_elem_stride;

            brgemm_kernel_execute(brg_kernel, A_weights, B_input, C_output,
                    total_tiles, 0.0f);
        }

        // Step 3: Output transform via JIT kernel (tile loop is inside JIT)
        {
            data_t *dst_batch = dst + mb_idx * conf.oc * oc_spatial_stride;
            jit_wino_output_transform_t::call_params_t params;
            params.M = M;
            params.bias = bias;
            params.dst_batch = dst_batch;
            params.nb_oh = nb_oh;
            params.nb_ow = nb_ow;
            (*output_xform)(&params);
        }
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
