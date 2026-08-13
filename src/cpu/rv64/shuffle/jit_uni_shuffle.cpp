/*******************************************************************************
* Copyright 2026 Léandre Le Duc
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
#include <cstdint>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "common/z_magic.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/shuffle/jit_uni_shuffle.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

// Gathers `cb_loop_size` elements per spatial position and repeats over
// `sp_loop_size` consecutive positions. The channel permutation itself never
// reaches the kernel: it was folded into a table of byte offsets at init, so
// this is a plain gather-and-pack loop that knows nothing about layouts,
// dimensionality, or propagation direction.
struct jit_uni_shuffle_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *src = nullptr;
        void *dst = nullptr;
        const void *input_off = nullptr;
        size_t sp_stride = 0;
        size_t sp_loop_size = 0;
        size_t cb_loop_size = 0;
    };
#define GET_OFF(field) offsetof(call_params_t, field)

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_shuffle_kernel_t);
    jit_uni_shuffle_kernel_t(size_t dt_size)
        : jit_generator_t("jit_uni_shuffle"), dt_size(dt_size) {}

    // Register layout. a0 holds call_params_t*, per the LP64D convention; every
    // register below is caller-saved, so there is no prologue or stack frame.
    //
    //   a1 in      gather base ; fixed inside a run, += stride between runs
    //   a2 out     destination cursor ;  never reset, one contiguous stream
    //   a3 off     table base           a4 stride   bytes between runs
    //   a5 runs    positions left       a6          elements per run
    //   t0 vl      t1 bytes (scratch)   t2 cb       elements left in the run
    //   t3 cursor  table cursor ; reset to a3 at the start of every run
    //   v8-v11     offsets, always e32/m4
    //   v16+       gathered data, m4 for 32-bit and m2 for 16-bit
    void generate() override {
        ld(a1, a0, GET_OFF(src));
        ld(a2, a0, GET_OFF(dst));
        ld(a3, a0, GET_OFF(input_off));
        ld(a4, a0, GET_OFF(sp_stride));
        ld(a5, a0, GET_OFF(sp_loop_size));
        ld(a6, a0, GET_OFF(cb_loop_size));

        const Reg in = a1, out = a2, off = a3, stride = a4, runs = a5,
                  cb_loop_size = a6, vl = t0, bytes = t1, cb = t2, cursor = t3;
        const VReg voff(8), vout(16);
        const bool is16 = dt_size == 2;

        Label sploop, spdone;
        Label cloop, cdone;

        L(sploop);
        beqz(runs, spdone);

        // A run re-reads the table from its start and gathers relative to a
        // fixed base, so both cursors reset here; only `in` walks between runs.
        mv(cb, cb_loop_size);
        mv(cursor, off);

        L(cloop);
        beqz(cb, cdone);

        // Index vtype first, so vl is sized for the offsets we are about to
        // load; the data vtype then reuses that same vl (x0 destination).
        vsetvli(vl, cb, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
        vle32_v(voff, cursor);
        if (is16) vsetvli(x0, vl, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);

        // `in` stays put: the table holds absolute byte offsets from it.
        vluxei32_v(vout, in, voff);
        if (is16)
            vse16_v(vout, out);
        else
            vse32_v(vout, out);

        // The destination advances by the element width, the table cursor
        // always by 4. Indices are uint32_t whatever the data type is.
        slli(bytes, vl, is16 ? 1 : 2);
        add(out, out, bytes);
        slli(bytes, vl, 2);
        add(cursor, cursor, bytes);
        sub(cb, cb, vl);
        j_(cloop);
        L(cdone);

        add(in, in, stride);
        addi(runs, runs, -1);
        j_(sploop);
        L(spdone);
        ret();
    }
    void operator()(const call_params_t *args) const {
        jit_generator_t::operator()(args);
    }

    const size_t dt_size;
};

status_t jit_uni_shuffle_t::precompute_offsets() {
    const auto conf = pd()->get_conf();
    const dim_t axis_size = pd()->axis_size();
    const dim_t group_size = pd()->group_size();
    const dim_t transpose_row
            = pd()->is_fwd() ? group_size : axis_size / group_size;
    const dim_t transpose_col
            = pd()->is_fwd() ? axis_size / group_size : group_size;
    std::vector<dim_t> rev_transposed(axis_size);

    // Precompute transposed axis helper array
    parallel_nd(transpose_col, transpose_row, [&](dim_t i, dim_t j) {
        rev_transposed[j * transpose_col + i] = i * transpose_row + j;
    });
    const dim_t C = conf.c;
    input_off_ = (uint32_t *)malloc(
            C * sizeof(uint32_t), platform::get_cache_line_size());
    if (input_off_ == nullptr) return status::out_of_memory;

    const dim_t SP = conf.sp, es = conf.dt_size, blk = conf.blk_size;
    parallel_nd(C, [&](dim_t c) {
        const dim_t ic = rev_transposed[c];
        // We get the proper lane index in the corresponding block.
        // Then if we need a specific spatial point, we just need to add sp * blk.
        input_off_[c] = (uint32_t)((ic / blk * SP * blk + ic % blk) * es);
    });

    return status::success;
}

status_t jit_uni_shuffle_t::init(engine_t *engine) {
    UNUSED(engine);
    CHECK(precompute_offsets());
    CHECK(safe_ptr_assign(
            kernel_, new jit_uni_shuffle_kernel_t(pd()->get_conf().dt_size)));
    CHECK(kernel_->create_kernel());
    return status::success;
}

status_t jit_uni_shuffle_t::execute(const exec_ctx_t &ctx) const {
    using namespace utils;
    using kparams_t = jit_uni_shuffle_kernel_t::call_params_t;

    const auto i_arg = pd()->is_fwd() ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
    const auto o_arg = pd()->is_fwd() ? DNNL_ARG_DST : DNNL_ARG_DIFF_SRC;

    auto input = CTX_IN_MEM(const char *, i_arg);
    auto output = CTX_OUT_MEM(char *, o_arg);
    const auto conf = pd()->get_conf();

    const dim_t MB = conf.mb;
    const dim_t nb_c = conf.nb_blk;

    // If we already use all the threads, no tiling on spatial dim.
    // Otherwise we balance the remaining tasks between all the threads.
    const int nthr = dnnl_get_current_num_threads();
    const dim_t tasks = MB * nb_c;
    const dim_t sp_split_size = (tasks >= nthr)
            ? conf.sp
            : nstl::max<dim_t>(1, div_up(conf.sp, div_up((dim_t)nthr, tasks)));
    const dim_t nb_sp = div_up(conf.sp, sp_split_size);

    parallel_nd(MB, nb_c, nb_sp,
            [= COMPAT_THIS_CAPTURE](dim_t mb, dim_t cb, dim_t spb) {
        const dim_t c_curr = cb * conf.blk_size;
        const dim_t sp0 = spb * sp_split_size;
        const dim_t sp_blk_size = nstl::min(sp_split_size, conf.sp - sp0);
        const dim_t base = mb * conf.stride_mb + sp0 * conf.blk_size;

        kparams_t args;
        args.src = input + base * conf.dt_size;
        args.dst = output + (base + conf.sp * c_curr) * conf.dt_size;
        args.input_off = input_off_ + c_curr;
        args.sp_stride = conf.blk_size * conf.dt_size;
        args.sp_loop_size = sp_blk_size;
        args.cb_loop_size = conf.blk_size;

        (*kernel_)(&args);
    });
    return status::success;
}
jit_uni_shuffle_t::jit_uni_shuffle_t(const pd_t *apd) : primitive_t(apd) {}
jit_uni_shuffle_t::~jit_uni_shuffle_t() {
    free(input_off_);
}

#undef GET_OFF
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
