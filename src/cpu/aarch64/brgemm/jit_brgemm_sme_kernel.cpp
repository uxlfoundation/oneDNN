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
#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#include "cpu/aarch64/brgemm/jit_brdgmm_kernel.hpp"
#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"

using namespace Xbyak_aarch64;
#define GET_OFF(field) (uint32_t) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) \
    (uint32_t) offsetof(brgemm_batch_element_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
struct jit_brgemm_sme_kernel_base_t : public jit_generator_t {
    jit_brgemm_sme_kernel_base_t(const brgemm_desc_t &abrd) : brg(abrd) {}

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_sme_kernel_base_t)

    brgemm_desc_t brg;

private:
    void generate() override;
};

void jit_brgemm_sme_kernel_base_t::generate() {
    const int sme_capacity
            = get_sme_length() >> 2u; // float == 4 bytes per sample

    /** Fill A into ZA tile 0
     * @param zaRegOffset Base offset of ZA tile
     * @param pg Predicate of ZA tile which controls how much samples to load from A along K (reduced) dim
     * @param A_ptr (temporary) X register which contains pointer to A
     * @param bd Number of rows left to load from A along M (broadcast) dim
     *
     * @return Number of rows loaded into ZA tile (can differ from m because Tile size is limited)
     */
    auto ld1w_A_to_za0
            = [=](const WReg &zaRegOffset, const _PReg &pA /*div T_z*/,
                      const XReg &A_ptr, int bd) {
        int step_offset
                = 0; // used to increase base offset of ZA tile to access lines > 15 (for ld1w instruction)
        int loads = (bd > sme_capacity) ? sme_capacity : bd;
        for (int i = 0; i < loads; i++) {
            if (0 == (i & 0x3)) {
                mov(zaRegOffset, step_offset);
                step_offset += 4;
            }
            ld1w(za0h.s(zaRegOffset, i & 0x3), pA, ptr(A_ptr));
            add(A_ptr, A_ptr, x27);
        }
        return loads;
    };

    /** FMOPA operation applied:
     *     1) ld1w (cont load of 32-bit words) of B lines (number of lines is based on reduce dim but limited by size of SME tile)
     *     2) mova vertical lines from tile 0 which contains A (extracting A.T)
     *     3) as soon as Z registers are full => FMOPA operations are applied for up to 3 tiles (Tile 0 is used to contain A)
     *
     * @param A_zregs Z registers which are used to extract A.T from Tile 0
     * @param B_zregs Z registers which are used to load B vectors (up to 3 vectors per B line)
     * @param pA_T Predicate for active elements of A.T
     * @param pB Predicate for active elements of B (always all true)
     * @param pB_tail Predicate for active elements of B (if tail along load (N) dim)
     * @param zaRegOffset Register to use as base offset of ZA tile
     * @param B_ptr (temporary) X register which contains pointer to B
     * @param rd Number of rows to load from B (reduce (K) dim)
     * @param ld Number of elements to load from B (load (N) dim)
     */
    auto fmopa_A_T_by_B =
            [=](const std::vector<ZReg> &A_zregs,
                    const std::vector<ZReg> &B_zregs,
                    const _PReg &pA_T /*div T_m*/, const _PReg &pB /*div T_m*/,
                    const _PReg &pB_tail /*div T_m*/, const WReg &zaRegOffset,
                    const XReg &B_ptr, int rd, int ld) {
        int step_offset
                = 0; // used to increase base offset of ZA tile to access lines > 3 (for mova instruction)
        int cycles = (rd > sme_capacity) ? sme_capacity : rd;
        int used_tiles = std::min((ld / sme_capacity),
                3); // Up to 3 tiles can be used (4th is used for A)
        int ld_tail = ld % sme_capacity; // Load dim tail (along N of B)
        auto A_itr = A_zregs.begin();
        auto B_itr = B_zregs.begin();
        bool isTail = (used_tiles < 3) && ld_tail;
        auto pB_last = isTail ? pB_tail : pB;
        if (isTail) used_tiles += 1;

        if (3 == used_tiles) {
            assert(0 == (B_zregs.size() % 3));
            assert(A_zregs.size() <= (B_zregs.size() / 3));
            for (int i = 0; i < cycles; ++i) {
                if (0 == (i & 0x3)) {
                    mov(zaRegOffset, step_offset);
                    step_offset += 4;
                }
                ld1w((B_itr++)->s, pB, ptr(B_ptr));
                ld1w((B_itr++)->s, pB, ptr(B_ptr, 1 /*mul vl*/));
                ld1w((B_itr++)->s, pB_last, ptr(B_ptr, 2 /*mul vl*/));
                add(B_ptr, B_ptr, x28);
                mova((A_itr++)->s, pA_T, za0v.s(zaRegOffset, i & 0x3));
                if (A_itr == A_zregs.end()) {
                    A_itr = A_zregs.begin();
                    B_itr = B_zregs.begin();
                    do {
                        fmopa(za1.s, pA_T, pB, A_itr->s, (B_itr++)->s);
                        fmopa(za2.s, pA_T, pB, A_itr->s, (B_itr++)->s);
                        fmopa(za3.s, pA_T, pB_last, A_itr->s, (B_itr++)->s);
                        ++A_itr;
                    } while (A_itr != A_zregs.end());
                    A_itr = A_zregs.begin();
                    B_itr = B_zregs.begin();
                }
            }

            // To finish left fmopa operations in case A_zregs vector is not fully filled:
            if (A_itr != A_zregs.begin()) {
                B_itr = B_zregs.begin();
                for (auto itr = A_zregs.begin(); itr != A_itr; ++itr) {
                    fmopa(za1.s, pA_T, pB, itr->s, (B_itr++)->s);
                    fmopa(za2.s, pA_T, pB, itr->s, (B_itr++)->s);
                    fmopa(za3.s, pA_T, pB_last, itr->s, (B_itr++)->s);
                }
            }
        } else if (2 == used_tiles) {
            assert(0 == (B_zregs.size() % 2));
            assert(A_zregs.size() <= (B_zregs.size() / 2));
            for (int i = 0; i < cycles; ++i) {
                if (0 == (i & 0x3)) {
                    mov(zaRegOffset, step_offset);
                    step_offset += 4;
                }
                ld1w((B_itr++)->s, pB, ptr(B_ptr));
                ld1w((B_itr++)->s, pB_last, ptr(B_ptr, 1 /*mul vl*/));
                add(B_ptr, B_ptr, x28);
                mova((A_itr++)->s, pA_T, za0v.s(zaRegOffset, i & 0x3));
                if (A_itr == A_zregs.end()) {
                    A_itr = A_zregs.begin();
                    B_itr = B_zregs.begin();
                    do {
                        fmopa(za1.s, pA_T, pB, A_itr->s, (B_itr++)->s);
                        fmopa(za2.s, pA_T, pB, A_itr->s, (B_itr++)->s);
                        ++A_itr;
                    } while (A_itr != A_zregs.end());
                    A_itr = A_zregs.begin();
                    B_itr = B_zregs.begin();
                }
            }

            // To finish left fmopa operations in case A_zregs vector is not fully filled:
            if (A_itr != A_zregs.begin()) {
                B_itr = B_zregs.begin();
                for (auto itr = A_zregs.begin(); itr != A_itr; ++itr) {
                    fmopa(za1.s, pA_T, pB, itr->s, (B_itr++)->s);
                    fmopa(za2.s, pA_T, pB, itr->s, (B_itr++)->s);
                }
            }
        } else {
            assert(A_zregs.size() <= B_zregs.size());
            for (int i = 0; i < cycles; ++i) {
                if (0 == (i & 0x3)) {
                    mov(zaRegOffset, step_offset);
                    step_offset += 4;
                }
                ld1w((B_itr++)->s, pB_last, ptr(B_ptr));
                add(B_ptr, B_ptr, x28);
                mova((A_itr++)->s, pA_T, za0v.s(zaRegOffset, i & 0x3));
                if (A_itr == A_zregs.end()) {
                    A_itr = A_zregs.begin();
                    B_itr = B_zregs.begin();
                    do {
                        fmopa(za1.s, pA_T, pB_last, A_itr->s, (B_itr++)->s);
                        ++A_itr;
                    } while (A_itr != A_zregs.end());
                    A_itr = A_zregs.begin();
                    B_itr = B_zregs.begin();
                }
            }
            // To finish left fmopa operations in case A_zregs vector is not fully filled:
            if (A_itr != A_zregs.begin()) {
                B_itr = B_zregs.begin();
                for (auto itr = A_zregs.begin(); itr != A_itr; ++itr) {
                    fmopa(za1.s, pA_T, pB, itr->s, (B_itr++)->s);
                }
            }
        }
    };

    /** Store ZA tile(s) 1(,2,3) to D pointer
     *
     * @param D_zregs Vector of Z registers which are allow
     * @param pD Predicate for active elements of D to store (always all true)
     * @param pD_tail Predicate for active elements of D to store (if tail along load (N) dim)
     * @param zaRegOffset (temporary) Register to use as base offset of ZA tile
     * @param D_ptr (temporary) X register which contains pointer to D
     * @param bd Number of rows to store along M (broadcast) dim
     * @param ld Number of elements to store per row (load (N) dim)
     */
    auto st1w_tiles_to_D =
            [=](const std::vector<ZReg> &D_zregs, const PReg &pD /*==pB*/,
                    const PReg &pD_tail /*==pB_tail*/, const WReg &zaRegOffset,
                    const XReg &D_ptr, int stride, int bd, int ld) {
        int step_offset
                = 0; // used to increase base offset of ZA tile to access lines > 3 (for mova instruction)
        int cycles = (bd > sme_capacity) ? sme_capacity : bd;
        int used_tiles = std::min((ld / sme_capacity),
                3); // Up to 3 tiles can be used (4th is used for A)
        int ld_tail = ld % sme_capacity; // Load dim tail (along N of B)
        bool isTail = (used_tiles < 3) && ld_tail;
        auto pD_last = isTail ? pD_tail : pD;
        if (isTail) used_tiles += 1;

        if (3 == used_tiles) {
            assert(D_zregs.size() >= 3);
            for (int i = 0; i < cycles; ++i) {
                if (0 == (i & 0x3)) {
                    mov(zaRegOffset, step_offset);
                    step_offset += 4;
                }
                mova(D_zregs[0].s, pD / T_m, za1h.s(zaRegOffset, i & 0x3));
                mova(D_zregs[1].s, pD / T_m, za2h.s(zaRegOffset, i & 0x3));
                mova(D_zregs[2].s, pD_last / T_m, za3h.s(zaRegOffset, i & 0x3));
                st1w(D_zregs[0].s, pD, ptr(D_ptr));
                st1w(D_zregs[1].s, pD, ptr(D_ptr, 1 /*mul vl*/));
                st1w(D_zregs[2].s, pD_last, ptr(D_ptr, 2 /*mul vl*/));
                add(D_ptr, D_ptr, stride);
            }
        } else if (2 == used_tiles) {
            assert(D_zregs.size() >= 2);
            for (int i = 0; i < cycles; ++i) {
                if (0 == (i & 0x3)) {
                    mov(zaRegOffset, step_offset);
                    step_offset += 4;
                }
                mova(D_zregs[0].s, pD / T_m, za1h.s(zaRegOffset, i & 0x3));
                mova(D_zregs[1].s, pD_last / T_m, za2h.s(zaRegOffset, i & 0x3));
                st1w(D_zregs[0].s, pD, ptr(D_ptr));
                st1w(D_zregs[1].s, pD_last, ptr(D_ptr, 1 /*mul vl*/));
                add(D_ptr, D_ptr, stride);
            }
        } else {
            assert(!D_zregs.empty());
            for (int i = 0; i < cycles; ++i) {
                if (0 == (i & 0x3)) {
                    mov(zaRegOffset, step_offset);
                    step_offset += 4;
                }
                mova(D_zregs[0].s, pD_last / T_m, za1h.s(zaRegOffset, i & 0x3));
                st1w(D_zregs[0].s, pD_last, ptr(D_ptr));
                add(D_ptr, D_ptr, stride);
            }
        }
    };

    int bd_block = sme_capacity;
    int bdb = brg.bcast_dim
            / bd_block; // Number of blocks along broadcast dim of A
    int bd_tail = brg.bcast_dim % bd_block;

    int ld_block = 3 * sme_capacity; // use up to 3 tiles in one go
    int ldb = brg.load_dim / ld_block;
    int ld_tail = brg.load_dim % ld_block;

    int rd_block = sme_capacity;
    int rdb = brg.reduce_dim / rd_block;
    int rd_tail = brg.reduce_dim % rd_block;

    int A_stride = brg.LDA * brg.typesize_A;
    int B_stride = brg.LDB * brg.typesize_B;
    int D_stride = brg.LDD * brg.typesize_D;
    std::vector<ZReg> A_zregs = {z24, z25, z26, z27, z28, z29, z30, z31};
    std::vector<ZReg> B_zregs = {z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
            z11, z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, z23};
    std::vector<ZReg> D_zregs = {z0, z1, z2};

    struct p_block_desc_t {
        const PReg &pReg;
        int size;
    };
    auto p_bdb = p_block_desc_t {p0, bd_block};
    auto p_bdt = p_block_desc_t {p1, bd_tail};
    auto p_ldb = p_block_desc_t {p0, ld_block};
    auto p_ldt = p_block_desc_t {p2, ld_tail}; // pReg of leading
    /**
     * Implementation of loop along reduce dim (main processing here)
     *
     * @param bd Descriptor of block along broadcast (M) dim (predicate and number of elements to process in block along broadcast dim)
     * @param bd Descriptor of block along load (N) dim (predicate and number of elements to process in block along load dim)
     */
    auto rd_loop = [=](p_block_desc_t &bd, p_block_desc_t &ld) {
        Label bs_loop, rdb_loop;
        add(x23, x0, GET_OFF(batch));
        zero(za);
        ldr(x9, ptr(x23)); // restore Batch pointer
        mov(x16, x19); // Batch size
        L(bs_loop);
        add(x23, x9, GET_OFF_BATCH_ELEMENT(ptr.A));
        add(x24, x9, GET_OFF_BATCH_ELEMENT(ptr.B));
        ldr(x1, ptr(x23));
        ldr(x2, ptr(x24));
        add(x9, x9, sizeof(brgemm_batch_element_t));
        add(x11, x1, x7); // A pointer to move
        add(x12, x2, x8); // B pointer to move
        if (rdb > 0) {
            if (rdb > 1) {
                mov(w22, rdb);
                L(rdb_loop);
            }
            // Inner loop:
            // Load A to ZA Tile 0 to transpose it further
            mov(x23, x11);
            ld1w_A_to_za0(w13, p0 / T_z, x23, bd.size);
            add(x11, x11, get_sme_length());
            fmopa_A_T_by_B(A_zregs, B_zregs, bd.pReg / T_m, p0 / T_m, p2 / T_m,
                    w13, x12, rd_block, ld.size);
            if (rdb > 1) {
                subs(w22, w22, 1);
                b(GT, rdb_loop);
            }
        }
        if (rd_tail) {
            mov(x23, x11);
            ld1w_A_to_za0(w13, p3 / T_z, x23, bd.size);
            fmopa_A_T_by_B(A_zregs, B_zregs, bd.pReg / T_m, p0 / T_m, p2 / T_m,
                    w13, x12, rd_tail, ld.size);
        }
        subs(w16, w16, 1);
        b(GT, bs_loop);
        // Store Tile(s)
        mov(x23, x10);
        st1w_tiles_to_D(D_zregs, p0, p2, w13, x23, D_stride, bd.size, ld.size);
        add(x10, x10, ld.size * brg.typesize_D);
        add(x8, x8, ld.size * brg.typesize_B);
    };

    preamble();
    smstart();
    add(x20, x0, GET_OFF(batch));
    add(x21, x0, GET_OFF(BS));
    ldr(x9, ptr(x20));
    add(x20, x0, GET_OFF(ptr_D));
    ldr(x19, ptr(x21));
    ldr(x3, ptr(x20));
    mov(x27, A_stride);
    mov(x28, B_stride);
    mov(x30, D_stride);
    mov(x4, brg.LDA);
    mov(x5, brg.LDD);
    mov(x20, brg.typesize_A * sme_capacity);
    mov(x21, brg.typesize_D * sme_capacity);
    mul(x4, x4, x20); // stride between A blocks
    mul(x5, x5, x21); // stride between D blocks
    mov(x7, 0);
    ptrue(p0.s);
    mov(x20, bd_tail);
    mov(x21,
            ld_tail % sme_capacity); // used up to 3 Tiles => tail predicate for last Tile only
    mov(x22, rd_tail);
    whilelo(p1.s, xzr, x20); // Broadcast dim tail predicate (M tail)
    whilelo(p2.s, xzr, x21); // Load dim tail predicate (N tail)
    whilelo(p3.s, xzr, x22); // Reduce dim tail predicate (K tail)

    if (bdb > 0) { // Full SME tile is used for loading along broadcast (M) dim
        Label bdb_loop, ldb_loop;
        if (bdb > 1) {
            mov(w20, bdb);
            L(bdb_loop);
        }
        mov(x8, 0); // reset B offset
        mov(x10, x3); // move D pointer into tmp register
        if (ldb > 0) { // Full SME tile is used for loading along load (N) dim
            if (ldb > 1) {
                mov(w21, ldb);
                L(ldb_loop);
            }
            rd_loop(p_bdb, p_ldb);
            if (ldb > 1) {
                subs(w21, w21, 1);
                b(GT, ldb_loop);
            }
        }
        if (ld_tail) rd_loop(p_bdb, p_ldt);
        add(x7, x7, x4); // move A offset forward
        add(x3, x3, x5); // move D forward
        if (bdb > 1) {
            subs(w20, w20, 1);
            b(GT, bdb_loop);
        }
    }
    if (bd_tail) {
        Label ldb_loop_bd_tail;
        mov(x8, 0); // reset B offset
        mov(x10, x3); // D pointer to move
        if (ldb > 0) {
            if (ldb > 1) {
                mov(w21, ldb);
                L(ldb_loop_bd_tail);
            }
            rd_loop(p_bdt, p_ldb);
            if (ldb > 1) {
                subs(w21, w21, 1);
                b(GT, ldb_loop_bd_tail);
            }
        }
        if (ld_tail) rd_loop(p_bdt, p_ldt);
    }
    smstop();
    postamble();
}

brgemm_sme_kernel_t::brgemm_sme_kernel_t(const brgemm_desc_t abrd)
    : brgemm_kernel_(new jit_brgemm_sme_kernel_base_t(abrd)) {}

status_t brgemm_sme_kernel_t::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

void brgemm_sme_kernel_t::operator()(brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

const jit_generator_t *brgemm_sme_kernel_t::get_jit_generator() const {
    return brgemm_kernel_;
}

brgemm_sme_kernel_t::~brgemm_sme_kernel_t() {
    delete brgemm_kernel_;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
