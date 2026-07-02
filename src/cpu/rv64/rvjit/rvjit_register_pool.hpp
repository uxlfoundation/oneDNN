/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef CPU_RV64_RVJIT_RVJIT_REGISTER_POOL_HPP
#define CPU_RV64_RVJIT_RVJIT_REGISTER_POOL_HPP

#include <array>
#include <cstdint>
#include <initializer_list>

#include "cpu/rv64/rvjit/rvjit_rvv.hpp"
#include "cpu/rv64/rvjit/rvjit_types.hpp"
#include "cpu/rv64/rvjit/rvjit_utils.hpp"

#if defined(RVJIT_DEBUG)
#include "common/verbose.hpp"
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) > 1) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

/// Fixed-capacity register array with a forward-advancing allocation cursor
template <typename T, int N = 32>
struct partition_t {
    partition_t() = default;

    template <typename It>
    partition_t(It s, It e) : count_(e - s) {
        int i = 0;
        for (auto it = s; it != e; ++it)
            ids_[i++] = *it;
    }

    partition_t(std::initializer_list<T> items)
        : partition_t(items.begin(), items.end()) {}

    int size() const { return count_; }
    int allocated() const { return alloc_; }
    int available() const { return count_ - alloc_; }

    /// Enumerates the full register list regardless of allocation state.
    const T *begin() const { return ids_.data(); }
    const T *end() const { return ids_.data() + count_; }

    /// Allocates a single register
    ///
    /// @pre Caller must ensure the partition has enough registers
    T allocate() {
        if (alloc_ >= count_) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed partition_t::allocate() due to "
                        "insufficient registers\n");
            });
            return T();
        }
        return ids_[alloc_++];
    }

    /// Allocates a 1D block of registers
    ///
    /// @pre Caller must ensure the partition has enough registers
    block_t<T> allocate(int cols) { return allocate(1, cols); }

    /// Allocates a 2D block of registers
    ///
    /// @pre Caller must ensure the partition has enough registers
    block_t<T> allocate(int rows, int cols) {
        if (alloc_ + rows * cols > count_) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed partition_t::allocate() due to "
                        "insufficient registers\n");
            });
            return {};
        }
        const T *start = ids_.data() + alloc_;
        alloc_ += rows * cols;
        return block_t<T>(start, rows, cols);
    }

private:
    std::array<T, N> ids_;
    int count_ = 0;
    int alloc_ = 0;
};

#if XBYAK_RISCV_V

/// Vector register allocation context with physical-overlap cross-invalidation
///
/// @note Accumulator and vector candidate lists share one occupancy bitmask,
///     so allocating from either blocks overlapping registers in the other;
///     needed for mixed-precision kernels where inputs/accumulators differ
///     in EMUL
struct vector_ctx_t {
    static constexpr int NVPR = 32;

    // Whether or not this object manages acc separately from vec registers
    bool with_acc = false;

    // Number of unique registers names per group on each class of registers
    int a_gsize = 0;
    int v_gsize = 0;

    // Determines which registers are allocated in the context
    uint32_t used_mask = 0;

    // Candidates for accumulators and vector registers
    VReg a_cand[NVPR];
    VReg v_cand[NVPR];
    int a_ncand = 0;
    int v_ncand = 0;

    // Contiguous stable storage to construct pointers for block_t allocations
    VReg acc[NVPR];
    VReg vec[NVPR];
    int nacc = 0;
    int nvec = 0;

    /// Allocates a single vector register
    VReg allocate_vector() { return allocate_vector(1, 1)[0]; }

    /// Allocates a single vector accumulator register
    VReg allocate_accumulator() { return allocate_accumulator(1, 1)[0]; }

    /// Allocates a vector register block
    v_block_t allocate_vector(int rows, int cols) {
        return alloc_block(v_cand, v_ncand, v_gsize, vec, nvec, rows, cols,
                "new_vector()");
    }

    /// Allocates a vector accumulator register block
    v_block_t allocate_accumulator(int rows, int cols) {
        if (!with_acc) return allocate_vector(rows, cols);
        return alloc_block(a_cand, a_ncand, a_gsize, acc, nacc, rows, cols,
                "new_vector_accumulator()");
    }

private:
    v_block_t alloc_block(VReg *cands, int n, int phys, VReg *out, int &out_n,
            int rows, int cols, const char *caller) {
        const int start = out_n;
        const uint32_t phys_save = used_mask;
        const uint32_t bits = (1u << phys) - 1u;
        for (int k = 0; k < rows * cols; ++k) {
            bool found = false;
            for (int i = 0; i < n; ++i) {
                const uint32_t m = bits << cands[i].getIdx();
                if (!(used_mask & m)) {
                    used_mask |= m;
                    out[out_n++] = cands[i];
                    found = true;
                    break;
                }
            }
            if (!found) {
                used_mask = phys_save;
                out_n = start;
                DEBUG({
                    verbose_printf(verbose_t::debuginfo,
                            "rvjit: Failed vector_ctx_t::%s block allocation\n",
                            caller);
                });
                return {};
            }
        }
        return v_block_t(out + start, rows, cols);
    }
};

#endif // XBYAK_RISCV_V

/// Component for managing architectural state allocation
struct register_pool_t {

    register_pool_t(emitter_t e) : e_(e) {}

    // Integer register file

    static constexpr int NGPR = 27; // integer registers available (t+a+s)
    using x_partition_t = partition_t<Reg, NGPR>;

    /// Sets the integer register file, optionally excluding the listed registers
    void int_register_file(std::initializer_list<Reg> excl = {}) {
        const Reg *regs = integer_registers();
        x_ctx_ = filter<Reg, NGPR>(regs, NGPR, excl.begin(), excl.end());
        callee_saved_boundary_ = find_callee_saved_boundary(x_ctx_);
    }

    template <typename It>
    void int_register_file_excluding(It s, It e) {
        const Reg *regs = integer_registers();
        x_ctx_ = filter<Reg, NGPR>(regs, NGPR, s, e);
        callee_saved_boundary_ = find_callee_saved_boundary(x_ctx_);
    }

    void int_register_file_excluding(std::initializer_list<Reg> excl) {
        int_register_file_excluding(excl.begin(), excl.end());
    }

    int x_available() const { return x_ctx_.available(); }

    Reg new_int() { return x_ctx_.allocate(); }
    x_block_t new_int(int cols) { return x_ctx_.allocate(cols); }
    x_block_t new_int(int r, int c) { return x_ctx_.allocate(r, c); }

    /// Conditionally allocates a register for a value that !is_simm12()
    const_t new_const(int value) {
        const_t c(value);
        if (!c.needs_reg()) return c;
        const Reg r = new_int();
        if (!is_not_zero(r)) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed register_pool_t::new_const() due to "
                        "insufficient registers\n");
            });
            return c;
        }
        return c.reg(r);
    }

    /// Materializes `rv.reg` via `new_int()` if it isn't already set,
    /// preserving any existing lazy loader
    runtime_value_t new_runtime_value(runtime_value_t rv = {}) {
        if (!rv.is_valid()) rv.reg = new_int();
        return rv;
    }

    /// Resolves any registers `loop` still needs
    ///
    /// @details Supplies registers for `iter`, `tmp`, and `branch.rhs` (the
    ///     latter via `new_runtime_value`, which preserves its lazy loader
    ///     while materializing the register)
    void new_loop(loop_t &loop) {
        using m = loop_t::mode_t;
        switch (loop.mode) {
            case m::literal: return;
            case m::switch_:
                if (loop.needs_tmp()) loop.tmp(new_int());
                loop.value(new_runtime_value(loop.value()));
                break;
            case m::unroll:
            case m::unroll_and_switch:
                if (loop.needs_iter()) loop.iter(new_int());
                loop.value(new_runtime_value(loop.value()));
                if (loop.needs_tmp()) {
                    const bool distinct = loop.is_eager()
                            || needs_distinct_tmp(loop.unrolling());
                    loop.tmp(distinct ? new_int() : loop.limit());
                }
                break;
        }
        if (!loop.is_valid()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed register_pool_t::new_loop() due to "
                        "insufficient registers\n");
            });
        }
    }

    // Float register file

    static constexpr int NFPR = 20; // caller-saved fp registers available
    using f_partition_t = partition_t<FReg, NFPR>;

    /// Sets the float register file, optionally excluding the listed registers
    void float_register_file(std::initializer_list<FReg> excl = {}) {
        const FReg *regs = float_registers();
        f_ctx_ = filter<FReg, NFPR>(regs, NFPR, excl.begin(), excl.end());
    }

    void float_register_file_excluding(std::initializer_list<FReg> excl) {
        float_register_file(excl);
    }

    int f_available() const { return f_ctx_.available(); }

    FReg new_float() { return f_ctx_.allocate(); }
    f_block_t new_float(int cols) { return f_ctx_.allocate(cols); }
    f_block_t new_float(int r, int c) { return f_ctx_.allocate(r, c); }

    // Callee-saved save / restore

    /// Preserves all callee-saved gpr allocated using this component
    ///
    /// @pre Caller must not further allocate registers after calling preserve
    void preserve() {
        const int hi = x_ctx_.allocated();
        if (hi <= callee_saved_boundary_) return;
        const int lo = callee_saved_boundary_;
        const int n = hi - lo;
        const Reg *ids = x_ctx_.begin();
        e_->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -(n * 8));
        for (int i = 0; i < n; ++i)
            e_->sd(ids[lo + i], Xbyak_riscv::sp, i * 8);
    }

    /// Restores all callee-saved gpr allocated using this component
    ///
    /// @pre Caller must not further allocate registers after calling preserve
    void restore() {
        const int hi = x_ctx_.allocated();
        if (hi <= callee_saved_boundary_) return;
        const int lo = callee_saved_boundary_;
        const int n = hi - lo;
        const Reg *ids = x_ctx_.begin();
        for (int i = 0; i < n; ++i)
            e_->ld(ids[lo + i], Xbyak_riscv::sp, i * 8);
        e_->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, n * 8);
    }

    /// Preserves a list of gpr in the same order as they are given
    void preserve(std::initializer_list<Reg> list) {
        const int n = list.size();
        if (n == 0) return;
        e_->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -(n * 8));
        for (int i = 0; i < n; ++i)
            e_->sd(*(list.begin() + i), Xbyak_riscv::sp, i * 8);
    }

    /// Restores a list of gpr in the same order as they are given
    void restore(std::initializer_list<Reg> list) {
        const int n = list.size();
        if (n == 0) return;
        for (int i = 0; i < n; ++i)
            e_->ld(*(list.begin() + i), Xbyak_riscv::sp, i * 8);
        e_->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, n * 8);
    }

#if XBYAK_RISCV_V

    /// Sets a single vector register file (non-widening kernels)
    void vector_register_file(
            const LMUL &m, std::initializer_list<VReg> excl = {}) {
        v_ctx_ = {};
        int count = 0;
        const uint32_t em = excl_vmask(excl.begin(), excl.end());
        const VReg *regs = vector_registers(m, count);

        for (int i = 0; i < count; ++i)
            if (!(em & (1u << regs[i].getIdx())))
                v_ctx_.v_cand[v_ctx_.v_ncand++] = regs[i];

        v_ctx_.v_gsize = vgroup_size(m);
    }

    /// Sets two vector register files for widening kernels
    ///
    /// @param m_vec  LMUL for input (narrow) operands
    /// @param m_acc  LMUL for accumulator (wide) operands; must be >= inp
    void vector_register_file(const LMUL &m_vec, const LMUL &m_acc,
            std::initializer_list<VReg> excl = {}) {
        // Uniform precision
        if (m_vec == m_acc) {
            vector_register_file(m_vec, excl);
            return;
        }

        // Setup context group sizes and allocation type
        v_ctx_ = {};
        v_ctx_.with_acc = true;
        v_ctx_.v_gsize = vgroup_size(m_vec);
        v_ctx_.a_gsize = vgroup_size(m_acc);

        // Setup register pools
        int nvec = 0;
        int nacc = 0;
        const VReg *vec = vector_registers(m_vec, nvec);
        const VReg *acc = vector_registers(m_acc, nacc);

        // Exclude elements signaled to be removed
        const uint32_t em = excl_vmask(excl.begin(), excl.end());

        for (int i = 0; i < nvec; ++i)
            if (!(em & (1u << vec[i].getIdx())))
                v_ctx_.v_cand[v_ctx_.v_ncand++] = vec[i];

        for (int i = 0; i < nacc; ++i)
            if (!(em & (1u << acc[i].getIdx())))
                v_ctx_.a_cand[v_ctx_.a_ncand++] = acc[i];
    }

    /// Get the number of available v registers
    int v_available() const {
        const uint32_t bits = (1u << v_ctx_.v_gsize) - 1u;
        int n = 0;
        for (int i = 0; i < v_ctx_.v_ncand; ++i) {
            const uint32_t m = bits << v_ctx_.v_cand[i].getIdx();
            if (!(v_ctx_.used_mask & m)) ++n;
        }
        return n;
    }

    /// Allocates a vector register from the context
    VReg new_vector() { return v_ctx_.allocate_vector(); }
    v_block_t new_vector(int cols) { return v_ctx_.allocate_vector(1, cols); }
    v_block_t new_vector(int rows, int cols) {
        return v_ctx_.allocate_vector(rows, cols);
    }

    /// Allocates accumulator registers from the accumulator register pool
    ///
    /// @note On uniform precision micro-kernels, the input and accumulator
    ///     pools are the same and calling new_vector() yields the same reg.
    VReg new_vector_accumulator() { return v_ctx_.allocate_accumulator(); }
    v_block_t new_vector_accumulator(int cols) {
        return v_ctx_.allocate_accumulator(1, cols);
    }
    v_block_t new_vector_accumulator(int rows, int cols) {
        return v_ctx_.allocate_accumulator(rows, cols);
    }

#endif

private:
    emitter_t e_;
    x_partition_t x_ctx_;
    f_partition_t f_ctx_;
    int callee_saved_boundary_ = 0;
#if XBYAK_RISCV_V
    vector_ctx_t v_ctx_;
#endif

    // Scans `p` forward to find the first index containing an s-register
    static int find_callee_saved_boundary(const x_partition_t &p) {
        // s0=x8, s1=x9, s2..s11=x18..x27
        static constexpr uint32_t s_mask = (1u << 8) | (1u << 9) | (1u << 18)
                | (1u << 19) | (1u << 20) | (1u << 21) | (1u << 22) | (1u << 23)
                | (1u << 24) | (1u << 25) | (1u << 26) | (1u << 27);
        const Reg *ids = p.begin();
        for (int i = 0; i < p.size(); ++i)
            if (s_mask & (1u << ids[i].getIdx())) return i;
        return p.size();
    }

    static const Reg *integer_registers() {
        using namespace Xbyak_riscv;
        static const Reg x[NGPR]
                = {t0, t1, t2, t3, t4, t5, t6, a0, a1, a2, a3, a4, a5, a6, a7,
                        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11};
        return x;
    }

    static const FReg *float_registers() {
        using namespace Xbyak_riscv;
        static const FReg f[NFPR] = {ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7,
                ft8, ft9, ft10, ft11, fa0, fa1, fa2, fa3, fa4, fa5, fa6, fa7};
        return f;
    }

    template <typename T, int N, typename It>
    static partition_t<T, N> filter(const T *list, int list_size, It s, It e) {
        uint32_t excl_mask = 0;
        for (auto it = s; it != e; ++it)
            excl_mask |= 1u << (*it).getIdx();
        T buf[N];
        int n = 0;
        for (int i = 0; i < list_size; ++i)
            if (!(excl_mask & (1u << list[i].getIdx()))) buf[n++] = list[i];
        return partition_t<T, N>(buf, buf + n);
    }

#if XBYAK_RISCV_V
    template <typename It>
    static uint32_t excl_vmask(It s, It e) {
        uint32_t m = 0;
        for (auto it = s; it != e; ++it)
            m |= 1u << (*it).getIdx();
        return m;
    }

    static const VReg *vector_registers(const LMUL &m, int &count) {
        using namespace Xbyak_riscv;
        switch (m) {
            case LMUL::m8: {
                static const VReg regs[] = {v0, v8, v16, v24};
                count = 4;
                return regs;
            }
            case LMUL::m4: {
                static const VReg regs[]
                        = {v0, v4, v8, v12, v16, v20, v24, v28};
                count = 8;
                return regs;
            }
            case LMUL::m2: {
                static const VReg regs[] = {v0, v2, v4, v6, v8, v10, v12, v14,
                        v16, v18, v20, v22, v24, v26, v28, v30};
                count = 16;
                return regs;
            }
            default: {
                static const VReg regs[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8,
                        v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30,
                        v31};
                count = 32;
                return regs;
            }
        }
    }
#endif
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#undef DEBUg
#undef DEBUG

#endif
