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

#ifndef CPU_X64_IR_POSTOPS_INJECTOR_HPP
#define CPU_X64_IR_POSTOPS_INJECTOR_HPP

// Lowers the `inject_postops` IR operation onto the JIT post-ops injector,
// which is not IR-based (see `emitter.hpp`). It owns the injector and turns the
// operation's operands into the arguments the injector takes, so it is where
// the IR and Xbyak worlds meet.
//
// The IR operation is builder-independent. `generate_kernel()` sets this object
// up:
//   1. Create it from the post-ops chain and the destination descriptor.
//   2. Pass it to the emitter, which calls `init()` once and then `inject()`
//      per `inject_postops` operation.
//   3. Call `maybe_prepare_table()` once after the postamble.
//
// ISA handling:
// - The vector width (Ymm or Zmm) is chosen from the `isa`. The width-specific
//   injector is stored type-erased, so IR-based kernels do not need to be
//   templated.
//
// Register handling:
// - The injector saves and restores the gpr and vec registers it borrows, so
//   those need not be reserved in the IR allocator. The save/restore cost is
//   paid once per call.
// - It does not do the same for opmasks. On AVX-512 the eltwise injector and
//   the binary injector each take one as a fixed register and restore neither,
//   so `generate_kernel()` reserves both and keeps them out of the allocator's
//   mask file (see `reserved_masks` in `make_reg_config()`):
//     eltwise_opmask     - scratch the eltwise injector overwrites. It is
//                          written before it is read, so it needs no setup.
//     binary_tail_opmask - active-element pattern the binary injector reads for
//                          a partial right-hand-side load. Binary, prelu, and
//                          sum post-ops all route through it. `init()` writes
//                          the pattern.
//   The two must be different registers. An eltwise post-op ahead of a binary
//   one would otherwise destroy the tail pattern.
//
//   Both are unused on AVX2*, where a mask is a vector register and the tail
//   size is an integer inside the injector.
// - An fp8 binary right-hand side is converted by emulation on an ISA without
//   native fp8 conversions. The converters write fixed scratch registers
//   without restoring them, so `generate_kernel()` reserves those too (see
//   `fp8_cvt_regs_t`).

#include <memory>
#include <vector>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct fp8_conversion_e5m2_t;
struct fp8_conversion_e4m3_t;

namespace ir {

// Interfaces marked with DNNL_API are exported for testing purposes only.
struct postops_injector_t {
    // Fixed registers of the fp8 emulation of a binary right-hand side (see
    // `needs_fp8_cvt()`). The converters write them without restoring them,
    // so the kernel keeps them out of the allocator's pools (see
    // `reserved_regs_t`). Empty when the chain needs no emulation.
    //   vecs   - the converters' scratch vectors, and last the vector the
    //            binary injector converts a right-hand side into
    //   opmask - the converters' scratch opmask
    //   gpr    - the converters' table pointer
    struct fp8_cvt_regs_t {
        std::vector<int> vecs;
        int opmask = -1;
        int gpr = -1;
    };

    // Whether a binary post-op of `post_ops` reads an fp8 right-hand side, which
    // the injector converts through the fp8 converters.
    static bool DNNL_API needs_fp8_cvt(const post_ops_t &post_ops);

    // The fixed registers the fp8 converters take on `isa`, chosen clear of
    // the registers the injector itself borrows.
    static fp8_cvt_regs_t DNNL_API fp8_cvt_regs(cpu_isa_t isa);

    // gen                - generator the injected post-op code is emitted into
    // isa                - kernel ISA
    // post_ops           - attribute post-ops chain to apply
    // dst_md             - destination memory descriptor (used by binary
    //                      post-ops)
    // bcast_set          - broadcasting strategies the binary injector may
    //                      pick for a right-hand-side argument
    // param_reg          - gpr holding the kernel-parameter-struct pointer
    //                      (used by binary post-ops to reach their
    //                      right-hand-side arguments)
    // rhs_arg_offset     - byte offset in the parameter struct of the binary
    //                      right-hand-side argument pointer array
    // dst_orig_off       - byte offset in the parameter struct of the
    //                      destination origin pointer, used to turn an
    //                      accumulator address into its position in the
    //                      destination tensor
    // tail_elems         - right-hand-side elements a partial (tail) load
    //                      reads. 0 means every accumulator holds a full vector
    // eltwise_opmask     - AVX-512 opmask the eltwise injector uses as scratch
    // binary_tail_opmask - AVX-512 opmask holding the `tail_elems` pattern for
    //                      the binary injector
    // fp8_regs           - registers of the fp8 converters. Required when
    //                      `needs_fp8_cvt(post_ops)`, ignored otherwise
    DNNL_API postops_injector_t(jit_generator_t &gen, cpu_isa_t isa,
            const post_ops_t &post_ops, const memory_desc_t &dst_md,
            const bcast_set_t &bcast_set, const Xbyak::Reg64 &param_reg,
            int rhs_arg_offset, dim_t dst_orig_off, int tail_elems,
            int eltwise_opmask, int binary_tail_opmask,
            const fp8_cvt_regs_t &fp8_regs);
    DNNL_API ~postops_injector_t();

    postops_injector_t(const postops_injector_t &) = delete;
    postops_injector_t &operator=(const postops_injector_t &) = delete;

    // Set up the fixed registers `inject()` reads. Call once, ahead of the
    // first `inject()`, on the generator the object was created for. A no-op
    // unless the ISA and the chain need the tail opmask.
    void DNNL_API init(jit_generator_t &gen) const;

    // Apply the post-ops to the accumulators in `acc_phys` (physical vec
    // register indices). For binary and sum post-ops, `base_phys` and
    // `out_byte_off` give each accumulator's output address: binary reaches its
    // right-hand-side argument through it, sum reads the previous destination
    // value from it. `base_phys` is -1 for a chain that needs neither.
    // `n_elems` is the number of valid leading elements in each accumulator:
    // `tail_elems` for a partially filled accumulator, the vector length
    // otherwise.
    void DNNL_API inject(const std::vector<int> &acc_phys, int base_phys,
            const std::vector<dim_t> &out_byte_off, int n_elems);

    // Emit the post-ops constant tables. Call once, after the postamble. Only
    // eltwise and sum post-ops and the fp8 converters have a table, so this is
    // a no-op without one.
    void DNNL_API maybe_prepare_table();

private:
    // Type-erased `jit_uni_postops_injector_t<Vmm>`. Cast to the target
    // type when used, based on `is_zmm_`.
    std::shared_ptr<void> injector_;
    // The fp8 converters of a binary right-hand side, one per fp8 type the
    // chain reads. Null otherwise.
    std::unique_ptr<fp8_conversion_e5m2_t> f8_e5m2_cvt_;
    std::unique_ptr<fp8_conversion_e4m3_t> f8_e4m3_cvt_;
    bool is_zmm_ = false;
    // Whether the chain has an eltwise post-op.
    bool with_eltwise_ = false;
    // Whether the chain has a sum post-op.
    bool with_sum_ = false;
    // Whether any post-op needs the per-accumulator destination address: binary
    // and prelu to reach their right-hand-side argument, sum to read the
    // previous destination value.
    bool needs_rhs_args_ = false;
    // Number of right-hand-side elements a tail load reads.
    int tail_elems_ = 0;
    // AVX-512 opmask holding the `tail_elems_` pattern, written by `init()`.
    int binary_tail_opmask_ = -1;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
