/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <algorithm>
#include <utility>
#include <initializer_list>

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/ref_io_helper.hpp"
#include "cpu/x64/matmul/brgemm_matmul_per_mn_comp.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::memory_tracking::names;
using namespace data_type;
using namespace Xbyak;

// V is prepared in chunks of this many columns.
static constexpr dim_t max_n_chunk = 512;

// Column sums of B over K for one block of columns. The values are summed as
// unsigned (s4 / s8 are flipped to u4 / u8), the caller subtracts the bias.
// Plain B: 128 (int4) or 64 (int8) columns of one row per K step. Blocked
// int4 ([n / n_blk][k / 4][n_blk][4]): 32 columns, 4 K values each, per step.
struct jit_col_sum_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src; // B at the first (k, n) of the block
        float *dst; // n_cols sums
        dim_t k_iters; // K steps (rows, or k quads of the blocked layout)
        dim_t src_stride; // bytes between K steps
        dim_t n_cols; // <= block_cols()
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_col_sum_kernel_t)

    jit_col_sum_kernel_t(data_type_t dt, bool blocked)
        : jit_generator_t("per_mn_comp_col_sum", avx512_core)
        , is_int4_(utils::one_of(dt, u4, s4))
        , is_signed_(utils::one_of(dt, s4, s8))
        , blocked_(blocked) {}

    int block_cols() const { return blocked_ ? 32 : is_int4_ ? 128 : 64; }

private:
    const bool is_int4_, is_signed_, blocked_;
    // One 64-byte load per K step feeds this many word / dword accumulators.
    int num_w() const { return blocked_ ? 1 : is_int4_ ? 4 : 2; }

    const Reg64 reg_src = r8, reg_dst = r9, reg_k = r10, reg_stride = r11,
                reg_n = r12, reg_cnt = r13, reg_tmp = rax, reg_tmp2 = rcx;
    const Opmask k_load = k1, k_store = k2;
    Opmask k_cols(int i) const {
        return Opmask(3 + i);
    } // columns [64i, 64i+64)
    const Zmm zmm_nib = zmm0, zmm_flip = zmm1, zmm_ones = zmm2, zmm_ld = zmm3,
              zmm_lo = zmm4, zmm_hi = zmm5, zmm_tmp = zmm6, zmm_idx_lo = zmm7,
              zmm_idx_hi = zmm8;
    Zmm W(int i) const { return Zmm(9 + i); }
    Zmm D(int i) const { return Zmm(13 + i); }

    void widen_add(const Zmm &lo, const Zmm &hi, const Zmm &src, bool words) {
        auto ext = [&](const Zmm &dst, const Ymm &s) {
            words ? vpmovzxwd(dst, s) : vpmovzxbw(dst, s);
        };
        auto add = [&](const Zmm &acc) {
            words ? vpaddd(acc, acc, zmm_tmp) : vpaddw(acc, acc, zmm_tmp);
        };
        ext(zmm_tmp, Ymm(src.getIdx()));
        add(lo);
        vextracti64x4(Ymm(zmm_tmp.getIdx()), src, 1);
        ext(zmm_tmp, Ymm(zmm_tmp.getIdx()));
        add(hi);
    }

    // k = low max(0, n_cols - base) bits (all 64 when >= 64).
    void cols_mask(const Opmask &k, int base) {
        mov(reg_tmp, reg_n);
        sub(reg_tmp, base);
        xor_(reg_tmp2, reg_tmp2);
        cmp(reg_tmp, 0);
        cmovl(reg_tmp, reg_tmp2);
        mov(reg_tmp2, -1);
        bzhi(reg_tmp2, reg_tmp2, reg_tmp);
        kmovq(k, reg_tmp2);
    }

    void generate() override {
        // Word accumulators take at most 255 per step, flush every 128 steps.
        constexpr int k_chunk = 128;
        preamble();
        mov(reg_src, ptr[param1 + offsetof(call_params_t, src)]);
        mov(reg_dst, ptr[param1 + offsetof(call_params_t, dst)]);
        mov(reg_k, ptr[param1 + offsetof(call_params_t, k_iters)]);
        mov(reg_stride, ptr[param1 + offsetof(call_params_t, src_stride)]);
        mov(reg_n, ptr[param1 + offsetof(call_params_t, n_cols)]);

        // Load mask: bytes = n_cols / 2 (int4), n_cols (int8), 2 n_cols
        // (blocked).
        mov(reg_tmp, reg_n);
        if (blocked_)
            add(reg_tmp, reg_tmp);
        else if (is_int4_) {
            inc(reg_tmp);
            shr(reg_tmp, 1);
        }
        mov(reg_tmp2, -1);
        bzhi(reg_tmp2, reg_tmp2, reg_tmp);
        kmovq(k_load, reg_tmp2);
        for (int i = 0; i < utils::div_up(block_cols(), 64); ++i)
            cols_mask(k_cols(i), 64 * i);

        mov(reg_tmp.cvt32(), 0x0F0F0F0F);
        vpbroadcastd(zmm_nib, reg_tmp.cvt32());
        mov(reg_tmp.cvt32(), is_int4_ ? 0x88888888 : 0x80808080);
        vpbroadcastd(zmm_flip, reg_tmp.cvt32());
        mov(reg_tmp.cvt32(), 0x01010101);
        vpbroadcastd(zmm_ones, reg_tmp.cvt32());
        for (int i = 0; i < 2 * num_w(); ++i)
            vpxord(D(i), D(i), D(i));

        Label l_chunk, l_step, l_done;
        L(l_chunk);
        test(reg_k, reg_k);
        jz(l_done, T_NEAR);
        for (int i = 0; i < num_w(); ++i)
            vpxord(W(i), W(i), W(i));
        mov(reg_cnt, k_chunk);
        cmp(reg_k, k_chunk);
        cmovb(reg_cnt, reg_k);
        sub(reg_k, reg_cnt);

        L(l_step);
        vmovdqu8(zmm_ld | k_load | T_z, ptr[reg_src]);
        if (is_signed_) vpxord(zmm_ld, zmm_ld, zmm_flip);
        if (is_int4_) {
            vpandd(zmm_lo, zmm_ld, zmm_nib);
            vpsrlw(zmm_hi, zmm_ld, 4);
            vpandd(zmm_hi, zmm_hi, zmm_nib);
        }
        if (blocked_) {
            // The 4 nibbles of a column form one word.
            vpaddb(zmm_lo, zmm_lo, zmm_hi);
            vpmaddubsw(zmm_lo, zmm_lo, zmm_ones);
            vpaddw(W(0), W(0), zmm_lo);
        } else if (is_int4_) {
            // Even columns are the low nibbles, odd ones the high nibbles.
            widen_add(W(0), W(1), zmm_lo, false);
            widen_add(W(2), W(3), zmm_hi, false);
        } else
            widen_add(W(0), W(1), zmm_ld, false);
        add(reg_src, reg_stride);
        dec(reg_cnt);
        jnz(l_step, T_NEAR);

        for (int i = 0; i < num_w(); ++i)
            widen_add(D(2 * i), D(2 * i + 1), W(i), true);
        jmp(l_chunk, T_NEAR);
        L(l_done);

        Label l_idx;
        if (!blocked_ && is_int4_) {
            // Interleave the even (D0..3) and odd (D4..7) column sums.
            mov(reg_tmp, l_idx);
            vmovdqu32(zmm_idx_lo, ptr[reg_tmp]);
            vmovdqu32(zmm_idx_hi, ptr[reg_tmp + 64]);
            for (int j = 0; j < 4; ++j) {
                vmovdqa32(W(j), D(j));
                vpermt2d(D(j), zmm_idx_lo, D(4 + j)); // out[2j]
                vpermt2d(W(j), zmm_idx_hi, D(4 + j)); // out[2j + 1]
                vmovdqa32(D(4 + j), W(j));
            }
        }
        const int num_out = utils::div_up(block_cols(), 16);
        for (int v = 0; v < num_out; ++v) {
            const int d
                    = !blocked_ && is_int4_ ? (v % 2 ? 4 + v / 2 : v / 2) : v;
            kshiftrq(k_store, k_cols(v / 4), 16 * (v % 4));
            vcvtdq2ps(D(d), D(d));
            vmovups(ptr[reg_dst + 64 * v] | k_store, D(d));
        }
        postamble();

        if (!blocked_ && is_int4_) {
            align(64);
            L(l_idx);
            for (int i = 0; i < 16; ++i) {
                dd(i);
                dd(16 + i);
            }
        }
    }
};

// One U/V row per brgemm K call is needed when the raw sum over A is offset
// by a src zero point or by the s8 -> u8 shift of the kernel.
static bool has_src_rows(const brgemm_matmul_conf_t &bgmmc) {
    return bgmmc.has_zero_point_a || bgmmc.s8s8_compensation_required;
}

// The B shift of the kernel acts as an extra weights zero point.
static bool has_wei_rows(const brgemm_matmul_conf_t &bgmmc) {
    return bgmmc.has_zero_point_b || bgmmc.int8_b_shift() != 0;
}

// gcd of the active per-K group sizes, or K when there is none.
static dim_t gcd_gsize(const brgemm_matmul_conf_t &bgmmc,
        std::initializer_list<std::pair<bool, dim_t>> groups) {
    dim_t gsize = 0;
    for (const auto &g : groups)
        if (g.first && g.second > 0)
            gsize = gsize ? math::gcd(gsize, g.second) : g.second;
    return gsize ? nstl::min(gsize, bgmmc.K) : bgmmc.K;
}

bool brgemm_matmul_per_mn_comp_t::init_conf(brgemm_matmul_conf_t &bgmmc) {
    const bool wei_scales_in_kernel
            = bgmmc.is_wei_scale_per_k && !bgmmc.apply_scales_in_buffer_b;
    // The S term reads column sums of B: plain N-contiguous, or VNNI-blocked
    // int4 ([n / wei_n_blk][k / 4][wei_n_blk][4]).
    const bool B_plain = !bgmmc.blocked_B && !bgmmc.transposed_B
            && bgmmc.B_strides[0] == bgmmc.b_dt_sz;
    const bool B_blocked_int4 = bgmmc.blocked_B && bgmmc.is_int4_weights
            && bgmmc.B_strides[1]
                    == bgmmc.b_dt_sz * bgmmc.wei_k_blk * bgmmc.wei_n_blk;
    const bool ok = bgmmc.use_buffer_c && !bgmmc.is_c_buf_dst_dt
            && !bgmmc.is_gemv && !bgmmc.is_runtime_M && !bgmmc.is_runtime_N
            && !bgmmc.is_runtime_K && !is_runtime_value(bgmmc.batch)
            && is_superset(bgmmc.isa, avx512_core)
            && (bgmmc.is_amx || bgmmc.N_blk <= bgmmc.LDC)
            // Row sums of A are read from plain, K-contiguous src.
            && !bgmmc.transposed_A
            && bgmmc.A_strides[0] == bgmmc.a_dt_sz
            // Blocked int8 weights carry their compensation in the md.
            && IMPLICATION(bgmmc.blocked_B, bgmmc.is_int4_weights)
            && IMPLICATION(has_src_rows(bgmmc), B_plain || B_blocked_int4);
    if (!ok) return false;

    bgmmc.per_mn_comp_wei_gsize = gcd_gsize(bgmmc,
            {{wei_scales_in_kernel, bgmmc.wei_scales_k_gsize},
                    {bgmmc.is_wei_zp_per_k, bgmmc.wei_zp_k_gsize}});
    bgmmc.per_mn_comp_wei_rows = has_wei_rows(bgmmc)
            ? utils::div_up(bgmmc.K, bgmmc.per_mn_comp_wei_gsize)
            : 0;
    bgmmc.per_mn_comp_src_gsize = gcd_gsize(bgmmc,
            {{wei_scales_in_kernel, bgmmc.wei_scales_k_gsize},
                    {bgmmc.is_src_scale_per_k, bgmmc.src_scales_k_gsize},
                    {bgmmc.is_src_zp_per_k, bgmmc.src_zp_k_gsize}});
    bgmmc.per_mn_comp_src_rows = has_src_rows(bgmmc)
            ? utils::div_up(bgmmc.K, bgmmc.per_mn_comp_src_gsize)
            : 0;
    bgmmc.per_mn_comp_K
            = bgmmc.per_mn_comp_wei_rows + bgmmc.per_mn_comp_src_rows;
    // V is read in LDC-wide stripes starting anywhere in [0, N).
    bgmmc.per_mn_comp_ldv = bgmmc.N + bgmmc.LDC;
    return true;
}

void brgemm_matmul_per_mn_comp_t::book_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const brgemm_matmul_conf_t &bgmmc) {
    scratchpad.book(key_brgemm_primitive_per_mn_comp_u,
            static_cast<size_t>(bgmmc.batch * bgmmc.M * bgmmc.per_mn_comp_K),
            types::data_type_size(f32));
    scratchpad.book(key_brgemm_primitive_per_mn_comp_v,
            static_cast<size_t>(
                    bgmmc.batch * bgmmc.per_mn_comp_K * bgmmc.per_mn_comp_ldv),
            types::data_type_size(f32));
}

status_t brgemm_matmul_per_mn_comp_t::init_brgemm_descs(
        brgemm_desc_t (&descs)[2], const brgemm_matmul_conf_t &bgmmc) {
    for (int i_M = 0; i_M < 2; i_M++) {
        const dim_t vM = i_M ? bgmmc.M_tail : bgmmc.M_blk;
        if (vM == 0) continue;
        brgemm_desc_t &brg = descs[i_M];
        CHECK(brgemm_desc_init(&brg, avx512_core, brgemm_addr, f32, f32, false,
                false, brgemm_row_major, -1.0f, 0.0f, bgmmc.per_mn_comp_K,
                bgmmc.per_mn_comp_ldv, bgmmc.LDC, vM,
                nstl::min(bgmmc.LDC, bgmmc.N_blk), bgmmc.per_mn_comp_K,
                nullptr));
        brgemm_attr_t brgattr;
        brgattr.max_bs = 1;
        CHECK(brgemm_desc_set_attr(&brg, brgattr));
        CHECK(brgemm_desc_finalize(&brg));
    }
    return status::success;
}

status_t brgemm_matmul_per_mn_comp_t::create_kernels(
        const brgemm_desc_t (&descs)[2], const brgemm_matmul_conf_t &bgmmc) {
    for (int i_M = 0; i_M < 2; i_M++) {
        if ((i_M ? bgmmc.M_tail : bgmmc.M_blk) == 0) continue;
        brgemm_kernel_t *ker = nullptr;
        CHECK(brgemm_kernel_create(&ker, descs[i_M]));
        CHECK(safe_ptr_assign(kernels_[i_M], ker));
    }
    if (has_src_rows(bgmmc)) {
        CHECK(safe_ptr_assign(col_sum_kernel_,
                new jit_col_sum_kernel_t(bgmmc.orig_wei_dt, bgmmc.blocked_B)));
        CHECK(col_sum_kernel_->create_kernel());
    }
    return status::success;
}

void brgemm_matmul_per_mn_comp_t::init_buffer_c(
        const brgemm_matmul_conf_t &bgmmc, const float *U_row, const float *V,
        int i_M, dim_t n, dim_t n_size, char *buf_C) const {
    brgemm_batch_element_t be;
    be.ptr.A = U_row;
    for (dim_t s = 0; s * bgmmc.LDC < n_size; ++s) {
        be.ptr.B = V + n + s * bgmmc.LDC;
        brgemm_kernel_execute(kernels_[i_M].get(), 1, &be,
                buf_C + s * bgmmc.M_blk * bgmmc.LDC * bgmmc.acc_dt_sz, nullptr);
    }
}

// Converts len values of a per-N (or broadcast) row to f32.
static void load_row(data_type_t dt, const void *base, dim_t idx0, bool per_n,
        dim_t len, float *dst) {
    if (!per_n) {
        std::fill_n(dst, len, io::load_float_value(dt, base, idx0));
    } else if (dt == f32) {
        const float *p = static_cast<const float *>(base) + idx0;
        std::copy(p, p + len, dst);
    } else if (utils::one_of(dt, u4, s4)) {
        const auto *p = static_cast<const uint8_t *>(base);
        for (dim_t i = 0; i < len; ++i) {
            const dim_t idx = idx0 + i;
            const int v = (p[idx >> 1] >> ((idx & 1) * 4)) & 0xF;
            dst[i] = static_cast<float>(dt == s4 ? (v ^ 8) - 8 : v);
        }
    } else if (utils::one_of(dt, u8, s8)) {
        const auto *p = static_cast<const uint8_t *>(base) + idx0;
        for (dim_t i = 0; i < len; ++i)
            dst[i] = static_cast<float>(
                    dt == s8 ? static_cast<int8_t>(p[i]) : p[i]);
    } else {
        for (dim_t i = 0; i < len; ++i)
            dst[i] = io::load_float_value(dt, base, idx0 + i);
    }
}

// S[n0 .. n0 + nv) over k in [k0, k0 + len) of plain B.
static void column_sums_plain(const brgemm_matmul_conf_t &bgmmc,
        const jit_col_sum_kernel_t *ker, const char *B, dim_t k0, dim_t len,
        dim_t n0, dim_t nv, float *S) {
    const auto dt = bgmmc.orig_wei_dt;
    const bool is_int4 = utils::one_of(dt, u4, s4);
    // Int4 rows must start on a byte boundary for the kernel.
    if (ker
            && IMPLICATION(
                    is_int4, bgmmc.B_strides[1] % 2 == 0 && n0 % 2 == 0)) {
        const int bits = is_int4 ? 4 : 8;
        const int bias = dt == s4 ? 8 : dt == s8 ? 128 : 0;
        for (dim_t n = 0; n < nv; n += ker->block_cols()) {
            jit_col_sum_kernel_t::call_params_t p;
            p.src = B + ((k0 * bgmmc.B_strides[1] + n0 + n) * bits >> 3);
            p.dst = S + n;
            p.k_iters = len;
            p.src_stride = bgmmc.B_strides[1] * bits >> 3;
            p.n_cols = nstl::min<dim_t>(ker->block_cols(), nv - n);
            (*ker)(&p);
        }
        for (dim_t i = 0; i < nv; ++i)
            S[i] -= static_cast<float>(bias * len);
        return;
    }
    float row[max_n_chunk];
    for (dim_t k = k0; k < k0 + len; ++k) {
        load_row(dt, B, k * bgmmc.B_strides[1] + n0, true, nv, row);
        for (dim_t i = 0; i < nv; ++i)
            S[i] += row[i];
    }
}

// Same for VNNI-blocked int4 B ([n / wei_n_blk][k / 4][wei_n_blk][4]); the
// kernel takes the full k quads, the remaining k values are summed here.
static void column_sums_blocked_int4(const brgemm_matmul_conf_t &bgmmc,
        const jit_col_sum_kernel_t *ker, const char *B, dim_t k0, dim_t len,
        dim_t n0, dim_t nv, float *S) {
    const auto *p0 = reinterpret_cast<const uint8_t *>(B);
    const bool is_s4 = bgmmc.orig_wei_dt == s4;
    const dim_t n_blk = bgmmc.wei_n_blk;
    const dim_t k_quads = len / 4;
    for (dim_t n = n0; n < n0 + nv;) {
        const dim_t nb = n / n_blk;
        const dim_t n_end = nstl::min(n0 + nv, (nb + 1) * n_blk);
        const dim_t cnt = nstl::min<dim_t>(n_end - n, ker->block_cols());
        float *s = S + (n - n0);
        const auto at = [&](dim_t k) {
            return p0
                    + ((nb * bgmmc.B_strides[0] + k * n_blk + (n % n_blk) * 4)
                            >> 1);
        };
        jit_col_sum_kernel_t::call_params_t p;
        p.src = at(k0);
        p.dst = s;
        p.k_iters = k_quads;
        p.src_stride = n_blk * 2;
        p.n_cols = cnt;
        (*ker)(&p);
        if (is_s4)
            for (dim_t i = 0; i < cnt; ++i)
                s[i] -= static_cast<float>(32 * k_quads);
        const uint8_t *tail = at(k0 + 4 * k_quads);
        for (dim_t i = 0; i < cnt; ++i, tail += 2)
            for (dim_t j = 0; j < len % 4; ++j) {
                const int v = (tail[j >> 1] >> ((j & 1) * 4)) & 0xF;
                s[i] += static_cast<float>(is_s4 ? (v ^ 8) - 8 : v);
            }
        n += cnt;
    }
}

void brgemm_matmul_per_mn_comp_t::prepare(
        const brgemm_matmul_conf_t &bgmmc, const ctx_t &ctx) const {
    const dim_t wei_rows = bgmmc.per_mn_comp_wei_rows;
    const dim_t wei_gsize = bgmmc.per_mn_comp_wei_gsize;
    const dim_t src_gsize = bgmmc.per_mn_comp_src_gsize;
    const bool src_rows = has_src_rows(bgmmc);
    const bool wei_rows_used = has_wei_rows(bgmmc);
    const float b_shift = static_cast<float>(bgmmc.int8_b_shift());
    const bool src_signed = bgmmc.orig_src_dt == s8;
    const float shift = bgmmc.s8s8_compensation_required ? 128.f : 0.f;
    const bool sa_per_k = bgmmc.is_src_scale_per_k;
    const bool sb_per_k
            = bgmmc.is_wei_scale_per_k && !bgmmc.apply_scales_in_buffer_b;

    // sa and za are constant over a K range of this size, which lies within
    // one wei row and one src row.
    const dim_t a_gsize
            = wei_rows_used ? math::gcd(wei_gsize, src_gsize) : src_gsize;
    parallel_nd(bgmmc.batch, bgmmc.M, [&](dim_t b, dim_t m) {
        const char *row = ctx.A_batch(b) + m * bgmmc.A_strides[1];
        float *u = U_ptr(bgmmc, ctx.U, b, m);
        std::fill_n(u, wei_rows, 0.f);
        for (dim_t k0 = 0; k0 < bgmmc.K; k0 += a_gsize) {
            const dim_t len = nstl::min(a_gsize, bgmmc.K - k0);
            int32_t t = 0;
            for (dim_t k = k0; k < k0 + len; ++k)
                t += src_signed ? static_cast<int8_t>(row[k])
                                : static_cast<uint8_t>(row[k]);
            float za = 0.f;
            if (bgmmc.has_zero_point_a) {
                int nibble = 0;
                const void *zp = ctx.src_zp(b, k0, m, &nibble);
                za = io::load_float_value(bgmmc.src_zp_dt, zp, nibble);
            }
            const float sa = sa_per_k
                    ? io::load_float_value(
                              bgmmc.src_scales_dt, ctx.src_scales(b, k0, m), 0)
                    : 1.f;
            if (wei_rows_used)
                u[k0 / wei_gsize] += sa
                        * (static_cast<float>(t)
                                - static_cast<float>(len) * za);
            if (src_rows) u[wei_rows + k0 / src_gsize] = sa * (za + shift);
        }
    });

    // The AMX ukernel applies a common src scale in every per-K call, the
    // other kernels apply it once with the post-ops.
    const float src_common_scale = sb_per_k && bgmmc.with_src_scales
                    && !bgmmc.is_src_scale_per_k
                    && bgmmc.per_mn_comp_src_scale_per_call
            ? io::load_float_value(
                      bgmmc.src_scales_dt, ctx.src_scales(0, 0, 0), 0)
            : 1.f;
    const dim_t ldv = bgmmc.per_mn_comp_ldv;
    constexpr dim_t n_chunk = max_n_chunk;
    const dim_t num_n_chunks = utils::div_up(ldv, n_chunk);
    parallel_nd(bgmmc.batch, bgmmc.per_mn_comp_K, num_n_chunks,
            [&](dim_t b, dim_t r, dim_t nc) {
        const bool is_wei_row = r < wei_rows;
        const dim_t k0
                = is_wei_row ? r * wei_gsize : (r - wei_rows) * src_gsize;
        const dim_t len
                = nstl::min(is_wei_row ? wei_gsize : src_gsize, bgmmc.K - k0);
        const dim_t n0 = nc * n_chunk;
        const dim_t n_len = nstl::min(n_chunk, ldv - n0);
        float *v = V_ptr(bgmmc, ctx.V, b, r) + n0;
        std::fill_n(v, n_len, 0.f);
        const dim_t nv = nstl::min(n_len, bgmmc.N - n0);
        if (nv <= 0) return;

        if (is_wei_row) {
            if (bgmmc.has_zero_point_b)
                load_row(bgmmc.wei_zp_dt, ctx.wei_zp, ctx.wei_zp_idx(b, k0, n0),
                        !bgmmc.is_wei_zp_common && bgmmc.is_wei_zp_per_n, nv,
                        v);
            for (dim_t i = 0; i < nv; ++i)
                v[i] += b_shift;
        } else {
            const auto *ker = static_cast<const jit_col_sum_kernel_t *>(
                    col_sum_kernel_.get());
            if (bgmmc.blocked_B)
                column_sums_blocked_int4(
                        bgmmc, ker, ctx.B_batch(b), k0, len, n0, nv, v);
            else
                column_sums_plain(
                        bgmmc, ker, ctx.B_batch(b), k0, len, n0, nv, v);
            // The kernel sums the shifted B.
            for (dim_t i = 0; i < nv; ++i)
                v[i] += static_cast<float>(len) * b_shift;
        }
        if (sb_per_k) {
            float sb[max_n_chunk];
            load_row(bgmmc.wei_scales_dt, ctx.wei_scales(b, k0),
                    bgmmc.is_wei_scale_per_n ? n0 : 0, bgmmc.is_wei_scale_per_n,
                    nv, sb);
            for (dim_t i = 0; i < nv; ++i)
                v[i] *= sb[i] * src_common_scale;
        }
    });
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
