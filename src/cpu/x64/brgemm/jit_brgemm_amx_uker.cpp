/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include <functional>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/brgemm/jit_brgemm_amx_uker.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace injector_utils;
using namespace Xbyak;

struct jit_brgemm_amx_uker_t : public brgemm_kernel_t {
    jit_brgemm_amx_uker_t(const brgemm_desc_t &abrg)
        : brgemm_kernel_t(jit_name(), abrg.isa_impl)
        , brg(abrg)
        , postops_injector_(nullptr) {

        bool has_f8_e5m2_binary_postops = false;
        bool has_f8_e4m3_binary_postops = false;
        bool has_f8_dt = one_of(data_type::f8_e5m2, brg.dt_d, brg.dt_bias)
                || one_of(data_type::f8_e4m3, brg.dt_d, brg.dt_bias);
        if (brg.with_binary) {
            const auto &post_ops = brg.attr()->post_ops_;
            for (int i = 0; i < post_ops.len(); i++) {
                const auto &entry = post_ops.entry_[i];
                if (!entry.is_binary()) continue;
                has_f8_e5m2_binary_postops
                        = entry.binary.src1_desc.data_type == data_type::f8_e5m2
                        || has_f8_e5m2_binary_postops;
                has_f8_e4m3_binary_postops
                        = entry.binary.src1_desc.data_type == data_type::f8_e4m3
                        || has_f8_e4m3_binary_postops;
            }
        }

        if (brg.is_fp8 || has_f8_dt || has_f8_e5m2_binary_postops
                || has_f8_e4m3_binary_postops) {
            if (one_of(data_type::f8_e5m2, brg.dt_a, brg.dt_b, brg.dt_d,
                        brg.dt_bias)
                    || has_f8_e5m2_binary_postops)
                f8_e5m2_cvt_ = utils::make_unique<fp8_conversion_e5m2_t>(this,
                        fp8_emu_xmm_1(), fp8_emu_xmm_2(), fp8_emu_xmm_3(),
                        fp8_tmp_mask, fp8_tmp_reg);
            if (one_of(data_type::f8_e4m3, brg.dt_a, brg.dt_b, brg.dt_d,
                        brg.dt_bias)
                    || has_f8_e4m3_binary_postops)
                f8_e4m3_cvt_ = utils::make_unique<fp8_conversion_e4m3_t>(this,
                        fp8_emu_xmm_1(), fp8_emu_xmm_2(), fp8_emu_xmm_3(),
                        fp8_emu_xmm_4(), fp8_emu_xmm_5(), fp8_tmp_reg);
        }

        if (brg.with_eltwise || brg.with_binary || brg.with_sum) {

            static constexpr bool preserve_gpr = true;
            // we don't use zmm1 for storing vectors
            // so we don't need to preserve vmm
            static constexpr bool preserve_vmm = false;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            const auto dst_md_wrapper = memory_desc_wrapper(brg.dst_md());

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    Xbyak::Zmm(1).getIdx(), this->r14, this->r15, this->r13,
                    preserve_gpr, preserve_vmm,
                    GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                    dst_md_wrapper, brg.ldb_tail, ld_tail_mask,
                    use_exact_tail_scalar_bcast};

            const binary_injector::static_params_t bsp(this->param1,
                    binary_injector::get_all_strategies_supported_by_injector(),
                    rhs_sp, f8_e5m2_cvt_.get(), f8_e4m3_cvt_.get());

            eltwise_injector::static_params_t esp;
            esp.preserve_vmm = preserve_vmm;
            esp.preserve_p_table = false;

            postops_injector_ = utils::make_unique<po_injector_t>(this,
                    brg.attr()->post_ops_, bsp, esp,
                    /* inject_sum = */ brg.with_sum);

            using namespace dnnl::impl::cpu::binary_injector_utils;
            std::tie(with_binary_per_oc_bcast_, with_binary_per_oc_sp_bcast_,
                    with_binary_per_oc_d_bcast_, with_binary_per_mb_bcast_,
                    with_binary_channel_bcast_, with_binary_per_mb_w_bcast_,
                    with_binary_per_w_bcast_, with_binary_per_hw_bcast_,
                    with_binary_batch_bcast_, with_binary_spatial_bcast_,
                    with_binary_no_bcast_)
                    = bcast_strategies_present_tup(brg.attr()->post_ops_.entry_,
                            dst_md_wrapper, broadcasting_strategy_t::per_oc,
                            broadcasting_strategy_t::per_oc_spatial,
                            broadcasting_strategy_t::per_oc_d,
                            broadcasting_strategy_t::per_mb,
                            broadcasting_strategy_t::per_mb_spatial,
                            broadcasting_strategy_t::per_mb_w,
                            broadcasting_strategy_t::per_w,
                            broadcasting_strategy_t::per_hw,
                            broadcasting_strategy_t::batch,
                            broadcasting_strategy_t::spatial,
                            broadcasting_strategy_t::no_broadcast);
            handle_binary_po_offset_ = with_binary_per_oc_bcast_
                    || with_binary_per_oc_sp_bcast_
                    || with_binary_per_oc_d_bcast_ || with_binary_per_mb_bcast_
                    || with_binary_channel_bcast_ || with_binary_per_mb_w_bcast_
                    || with_binary_per_w_bcast_ || with_binary_per_hw_bcast_
                    || with_binary_batch_bcast_ || with_binary_spatial_bcast_
                    || with_binary_no_bcast_;
        }
        use_ils_ = brg.brgattr.use_interleave_stores && !brg.is_ace();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_amx_uker_t)

    brgemm_desc_t brg;

private:
    using po_injector_t = injector::jit_uni_postops_injector_t<Zmm>;
    std::unique_ptr<po_injector_t> postops_injector_;

    std::unique_ptr<fp8_conversion_e5m2_t> f8_e5m2_cvt_;
    std::unique_ptr<fp8_conversion_e4m3_t> f8_e4m3_cvt_;

    enum {
        simd_w = 16,
        zmm_width_in_bytes = cpu_isa_traits_t<avx512_core>::vlen,
    };

    // Register decomposition
    using reg64_t = const Xbyak::Reg64;

    registry_scratchpad_t regscratchpad_ {*this, brg.isa_impl};

    const reg64_t param1 = abi_param1;
    const reg64_t reg_iter_label = r9;
    const reg64_savable_t reg_iter_labels_list {regscratchpad_, rax, r16};

    const reg64_t reg_addr_batch = r13;
    const reg64_savable_t reg_aux1_batch {
            regscratchpad_, rbx, rbp, may_use_rbp()};
    const reg64_savable_t reg_A {regscratchpad_, r11};
    const reg64_savable_t reg_B {regscratchpad_, r10};
    const reg64_t reg_stride_lda = r14;
    const reg64_t reg_stride_ldb = abi_not_param1;
    const reg64_savable_t reg_C {regscratchpad_, r15};
    const reg64_savable_t reg_D {regscratchpad_, r12};

    const reg64_savable_t reg_buf {regscratchpad_, r8};
    const reg64_t reg_BS = rbx;
    const reg64_t reg_BS_loop = r9;
    const reg64_savable_t reg_bias {regscratchpad_, rbx};
    const reg64_savable_t reg_bias_backup {regscratchpad_, rbx};
    const reg64_savable_t reg_src_scales {regscratchpad_, rbx};
    // Copy of reg_src_scales taken before the bd loop and restored after it,
    // so the per-bd-iteration advance (bd_iteration_t::A_scales_shift) does
    // not leak into the next ld iteration when the bd loop is the inner one.
    const reg64_savable_t reg_src_scales_bd_loop {regscratchpad_, rbx};
    const reg64_savable_t reg_wei_scales {regscratchpad_, rbx};
    const reg64_savable_t reg_wei_scales_backup {regscratchpad_, rbx};
    const reg64_savable_t reg_dst_scales {regscratchpad_, rbx};
    // Copy of reg_dst_scales taken before the bd loop and restored after it,
    // Booked only for MXFP8 dst quantization.
    const reg64_savable_t reg_dst_scales_bd_loop {
            regscratchpad_, rbx, brg.quantize_dst_to_mxfp8};

    const reg64_t reg_stride_ld_block = rdx;
    const reg64_t reg_do_post_ops = rbx;
    const reg64_t reg_do_skip_accum = reg_do_post_ops;
    const reg64_t reg_tmp_gpr = rbx;

    const reg64_savable_t reg_zp_comp_a {regscratchpad_, rbx, r17};
    const reg64_savable_t reg_zp_a_values {regscratchpad_, rbx, r18};
    const reg64_savable_t reg_zp_comp_b {regscratchpad_, rbx, r19};
    const reg64_savable_t reg_zp_c_values {regscratchpad_, rbx, r20};
    const reg64_savable_t reg_src_scales_per_k {regscratchpad_, rbx, r21};
    const reg64_savable_t reg_per_mn_comp {regscratchpad_, rbx, r22};
    const reg64_t reg_converted_stride = rsi;
    const reg64_t reg_zp_comp_pad_a = rsi;

    const reg64_savable_t reg_long_offt = {regscratchpad_, r11};

    bool are_post_ops_applicable_ = false;
    bool need_to_apply_alpha_beta_ = false;
    bool may_load_accumulators_ = false;

    bool handle_binary_po_offset_ = false;
    bool with_binary_per_oc_bcast_ = false;
    bool with_binary_per_oc_sp_bcast_ = false;
    bool with_binary_per_oc_d_bcast_ = false;
    bool with_binary_channel_bcast_ = false;
    bool with_binary_per_mb_bcast_ = false;
    bool with_binary_per_mb_w_bcast_ = false;
    bool with_binary_per_w_bcast_ = false;
    bool with_binary_batch_bcast_ = false;
    bool with_binary_spatial_bcast_ = false;
    bool with_binary_per_hw_bcast_ = false;
    bool with_binary_no_bcast_ = false;
    bool prepare_post_ops_registers_once_ = false;

    const char *bd_mask_buffer_ptr_ = nullptr;
    std::vector<dim_t> adj_bd_mask_buffer_;
    std::vector<dim_t> skipped_bd_mask_buffer_;
    palette_config_t palette_ {};
    // used to store offsets within wsp buffer where the data is
    // transformed(downconverted), to reuse when needed.
    std::unordered_map<std::string, dim_t> transform_buf_map_A_;
    std::unordered_map<std::string, dim_t> transform_buf_map_B_;

    // Deferred micro-kernel body.
    // rdb_loop_call_based() emits `call(*entry)` into the hot path and
    // registers here an emitter that will later emit
    // `L(*entry); rdb_loop_body(bi); ret();` out of the hot path.
    // The label and the emitter are one object with one lifetime: top_loop()
    // owns them, emits every registered body exactly once and then destroys
    // both. No label outlives the code generation of the loop nest that
    // referenced it.
    // The label is held by shared_ptr because the emitter is created before
    // the `call` that references it: a copy of a not-yet-used Xbyak::Label is
    // an independent label, so the very same object must be shared by the
    // call site and by the deferred `L()`.
    struct deferred_uk_body_t {
        std::shared_ptr<Xbyak::Label> entry;
        std::function<void()> emit;
    };
    std::vector<deferred_uk_body_t> deferred_uk_bodies_;

    dim_t LDA_size_ = 0, LDA2_size_ = 0;
    dim_t LDB_size_ = 0, LDB2_size_ = 0;
    dim_t LDC_size_ = 0, LDC2_size_M_ = 0, LDC2_size_N_ = 0;
    dim_t LDD_size_ = 0;
    dim_t ld_block_B_size_ = 0;
    dim_t ld_block_C_size_ = 0;
    dim_t ld_block_D_size_ = 0;
    dim_t ld_block_bias_size_ = 0;
    dim_t ld_block_scales_size_ = 0;
    dim_t ld_block_zp_size_ = 0;

    dim_t ldb_tail_B_size_ = 0;
    dim_t ldb_tail_C_size_ = 0;
    dim_t ldb_tail_D_size_ = 0;
    dim_t ldb_tail_zp_size_ = 0;

    enum matrix_kind_t { matrix_A, matrix_B, matrix_C, matrix_D };

    // Loops in brgemm kernel are (two outermost loops depend on loop order):
    // by bd block2
    //     by ld block2
    //          by batch_size
    //              by rd block
    //                  gemm_microkernel
    // Structures below (iteration_block_t, dim_iteration_t, bs_iteration_t and
    // iteration_map_t) describe the structure of cycles and are used for
    // JIT code generation
    struct iteration_block_t {
        int block = 0;
        dim_t pos = 0;
        bool is_tail = false;
        iteration_block_t(dim_t pos_, int block_, bool is_tail_ = false)
            : block(block_), pos(pos_), is_tail(is_tail_) {}
        bool operator==(const iteration_block_t &rhs) const {
            return block == rhs.block && is_tail == rhs.is_tail;
        }
    };

    struct dim_iteration_t {
        size_t idx = 0;
        std::vector<iteration_block_t> blocks;
        bool operator==(const dim_iteration_t &rhs) const {
            return blocks == rhs.blocks;
        }
        bool operator!=(const dim_iteration_t &rhs) const {
            return !operator==(rhs);
        }

        dim_t pos(size_t b) const {
            assert(b < blocks.size());
            return blocks[b].pos;
        }

        dim_t rel_pos(size_t b) const {
            assert(b < blocks.size());
            return (blocks[b].pos - blocks[0].pos);
        }

        int block(size_t b) const {
            assert(b < blocks.size());
            return blocks[b].block;
        }

        bool is_tail(size_t b) const {
            assert(b < blocks.size());
            return blocks[b].is_tail;
        }

        int block2() const { return static_cast<int>(blocks.size()); }

        int length() const {
            if (blocks.empty()) return 0;
            const int n = static_cast<int>(blocks.size());
            // only last block may be different
            return ((n - 1) * blocks[0].block + blocks[n - 1].block);
        }
    };

    struct bd_iteration_t : public dim_iteration_t {
        dim_t A_shift {0};
        // MXFP8: advance of reg_src_scales between this and the previous bd
        // iteration under ununroll_bd_loop (A_offset_scales() is then
        // relative to the current iteration).
        dim_t A_scales_shift {0};
        dim_t D_scales_shift {0};
        dim_t C_shift {0};
        dim_t D_shift {0};
        dim_t zp_comp_pad_a_shift {0};
        std::vector<char> bd_mask;
        std::vector<dim_t> adj_bd_mask;
        bd_iteration_t *similar {nullptr};
        Label lstart;

        bool operator==(const bd_iteration_t &rhs) const {
            return dim_iteration_t::operator==(rhs) && A_shift == rhs.A_shift
                    && A_scales_shift == rhs.A_scales_shift
                    && C_shift == rhs.C_shift && D_shift == rhs.D_shift
                    && bd_mask == rhs.bd_mask
                    && zp_comp_pad_a_shift == rhs.zp_comp_pad_a_shift
                    && D_scales_shift == rhs.D_scales_shift;
        }
        bool operator!=(const bd_iteration_t &_rhs) const {
            return !operator==(_rhs);
        }
    };

    struct bs_iteration_t {
        dim_t idx = 0;
        dim_t pos = 0;
        bool is_first = false;
        bool is_last = false;
        bs_iteration_t() = default;
        bs_iteration_t(dim_t pos_, bool is_first_ = true, bool is_last_ = false)
            : pos(pos_), is_first(is_first_), is_last(is_last_) {}
    };

    class iteration_map_t {
    public:
        struct top_loop_t {
            std::vector<dim_iteration_t> ldis;
            std::vector<bd_iteration_t> bdis;
            std::vector<bs_iteration_t> bsis;
            std::vector<dim_iteration_t> rdis;
            int duplicated {0};
            bool is_last_rdi(const dim_iteration_t *rdi) const {
                return (rdi->idx == rdis.size() - 1);
            }
        };

        iteration_map_t() : tloops(2) {}

        inline top_loop_t &operator[](bool bidx) {
            return tloops[static_cast<int>(bidx)];
        }
        inline const top_loop_t &operator[](bool bidx) const {
            return tloops[static_cast<int>(bidx)];
        }

    private:
        std::vector<top_loop_t> tloops;
    };

    struct brgemm_iteration_t {
        const bd_iteration_t *bdi {nullptr};
        const dim_iteration_t *ldi {nullptr};
        const bs_iteration_t *bsi {nullptr};
        const dim_iteration_t *rdi {nullptr};
        bool apply_postops {false};
        bool skip_accumulation {false};
        bool first_bsi {false};
        bool last_bsi {false};
        // Registers holding scalar binary RHS values loaded before the loop.
        binary_injector::preloaded_rhs_t preloaded_po_rhs;
        brgemm_iteration_t() = default;
    };

    struct prf_t {
        brgemm_kernel_prefetching_t pft = brgemm_prf_default;
        int dist = -1;
        int vec = 0;
        void set(brgemm_kernel_prefetching_t pft_, int dist_) {
            pft = pft_;
            dist = dist_;
            vec = 0;
        }
        void reset() { vec = 0; }
    };

    struct prf_sprinkled_t {
        std::vector<dim_t> prefetch_offsets;
        size_t current_prefetch_idx;
        void reset() {
            prefetch_offsets.clear();
            current_prefetch_idx = 0;
        }
    };

    // iteration map
    iteration_map_t imap_;

    prf_sprinkled_t prf_sprinkled_a, prf_sprinkled_b;
    size_t num_amx_ops;
    size_t current_num_amx_ops;
    // interleave stores
    bool use_ils_ = false;
    bool was_prev_bi_ = false;
    // saved parameters for storing
    brgemm_iteration_t prev_bi_;
    // current storing coordinates
    int ils_vec_ = 0, ils_bdb_ = 0, ils_ldb_ = 0, ils_bd_start_ = 0;
    prf_t prf0A, prf1A, prf2A, prfntaA, prf0B, prf1B, prf2B, prfntaB, prf0C,
            prf1C;

    bool dt_requires_saturation_ = false;
    bool use_sat_cvt_ = false;

    bool ununroll_bd_loop = false;
    // If set, the rd loop is emitted as a sequence of calls to a few shared
    // micro-kernel bodies instead of being fully unrolled. It keeps the code
    // size small while still allowing rd iterations to use different
    // (statically known) offsets: the shared bodies address A, B and the
    // transform buffer relative to registers that the caller advances once
    // per group of call_based_rd_unroll iterations.
    bool call_based_rd_loop = false;
    // Number of rd iterations sharing one group of micro-kernel bodies. All
    // matrix pointers are advanced once per group.
    static constexpr int call_based_rd_unroll = 4;
    // Number of ZMM registers per bd block for ACE microkernel.
    static constexpr int ace_zmms_per_bd_block
            = brgemm_desc_t::ace_zmms_per_bd_block;

    Xbyak::Opmask ld_full_mask = Xbyak::Opmask(0);
    // Post-ops may use and clobber Opmask(1), so it is not allocated here.
    // TODO: check whether post-ops can be made to preserve it.
    Xbyak::Opmask ld_tail_mask = Xbyak::Opmask(7);
    Xbyak::Opmask fp_col_mask = Xbyak::Opmask(2);
    Xbyak::Opmask rd_tail_mask = Xbyak::Opmask(3);
    // There are 3 uses of k4: ld_scale_tail_mask, fp8_tmp_mask and ace_load_A_mask.
    // All 3 usage sites have a mask load before the use.
    Xbyak::Opmask ld_scale_tail_mask = Xbyak::Opmask(4);
    Xbyak::Opmask fp8_tmp_mask = Xbyak::Opmask(4);

    // The four constant masks below are set up once in generate(), so they
    // must avoid k4: the fp8 converters clobber it and are instantiated for
    // f8 binary post-ops even when ACE itself computes bf16/int8. k2 and k3
    // are only written by fp8 convert and AMX k-tail paths ACE never takes.
    // ace_load_A_mask is rewritten before every use, so k4 is safe for it.
    Xbyak::Opmask ace_load_A_mask = Xbyak::Opmask(4);
    Xbyak::Opmask ace_load_A_mask_f = Xbyak::Opmask(2);
    Xbyak::Opmask ace_load_A_mask_f0 = Xbyak::Opmask(3);
    Xbyak::Opmask ace_load_A_mask_f00 = Xbyak::Opmask(5);
    Xbyak::Opmask ace_load_A_mask_f000 = Xbyak::Opmask(6);

    // Zmm map below
    const Xbyak::Zmm &zmm_tmp_1() const noexcept { return this->zmm0; }
    const Xbyak::Zmm &zmm_tmp_2() const noexcept { return this->zmm1; }
    const Xbyak::Zmm &zmm_tmp_3() const noexcept { return this->zmm2; }

    /* for fp8 emulation only */
    Xmm fp8_emu_xmm_1() const noexcept { return Xmm(1); }
    Xmm fp8_emu_xmm_2() const noexcept { return Xmm(2); }
    Xmm fp8_emu_xmm_3() const noexcept { return Xmm(3); }
    Xmm fp8_emu_xmm_4() const noexcept { return Xmm(6); }
    Xmm fp8_emu_xmm_5() const noexcept { return Xmm(7); }
    const reg64_t fp8_tmp_reg = rax;

    const Xbyak::Zmm zmm_bf32_permute = zmm6;
    const Xbyak::Zmm zmm_zp_comp_a = zmm6;
    const Xbyak::Zmm zmm_zp_c = zmm7;
    const Xbyak::Zmm zmm_lbound = zmm8;
    const Xbyak::Zmm zmm_ubound = zmm9;

    // MXFP8 only. The B scales are stored flat along the ld dimension in
    // memory, while the Block Scale Register expects `b_scales_perm_groups`
    // interleaved groups of `b_scales_perm_group_size` scales, so a `vpermb`
    // is needed before moving them into the BSR. The permutation maps the
    // memory layout
    //     [ 0, 1, 2, ..., 63 ]
    // to the BSR layout
    //     [ 0, 16, 32, 48, 1, 17, 33, 49, ..., 15, 31, 47, 63 ]
    // i.e. destination byte `b_scales_perm_groups * i + j` is taken from
    // source byte `i + b_scales_perm_group_size * j`.
    //
    // zmm22 is free on this path: postop scale vectors (zmm18-23) are not
    // used by MXFP8 (the scales are consumed by the outer product, see
    // prepare_post_ops_registers()), and the A/B operand registers stay well
    // below it -- asserted in init().
    const Xbyak::Zmm zmm_wei_scale_permute = zmm22;
    static constexpr int b_scales_perm_group_size = 16;
    static constexpr int b_scales_perm_groups = 4;
    Xbyak::Label b_scales_perm_index_table;
    // MX block scale group size, in elements of the reduction dimension.
    static constexpr int mx_group_size = 32;

    // Number of levels of the max-reduction tree used to compute the MXFP8
    // dst scales: 2^4 = 16 rows of a tile are reduced into a single vector of
    // per-group exponents.
    static constexpr int mxfp8_reduce_levels = 4;

    // Tiles of the C accumulator, indexed as [bdb * mxfp8_max_ld_blocks + ldb].
    // The blocking heuristics cap the ACE MXFP8 output at 2 x 4 tiles, see
    // brgemm_blocking() in brgemm_utils.cpp.
    static constexpr int mxfp8_max_bd_blocks = 2;
    static constexpr int mxfp8_max_ld_blocks = 4;
    static constexpr int mxfp8_max_tiles
            = mxfp8_max_bd_blocks * mxfp8_max_ld_blocks;
    // Number of rows of a single tile.
    static constexpr int mxfp8_tile_rows = 16;
    // Two tiles (2 x 16 elements of f32) are packed into one vector, so a pair
    // of tiles covers one 32-element MX scale group.
    static constexpr int mxfp8_tiles_per_group = 2;

    // Upper bound of the node range of quantize_to_mxfp8(), i.e. "emit all the
    // nodes":
    //   1                                                          constant tables
    // + (2 * mxfp8_tile_rows - 1)                  max-reduction tree (16 leaves and 15 inner nodes)
    // + 1                                                        e8m0 scale computation and store
    // + bd_blocks * ld_pairs *  2                  prep inv scales per tile pair
    // + bd_blocks * ld_pairs *  rows             number of vector pairs to quantize
    static constexpr int mxfp8_all_nodes = 1 + (2 * mxfp8_tile_rows - 1) + 1
            + mxfp8_max_bd_blocks
                    * (mxfp8_max_ld_blocks / mxfp8_tiles_per_group)
                    * (2 + mxfp8_tile_rows);

    // Constant tables used by quantize_to_mxfp8().
    // `mxfp8_permute_table` holds one 64-byte permutation per reduction level
    // and `mxfp8_mask_table` the matching 64-bit blend mask
    Xbyak::Label mxfp8_permute_table;
    Xbyak::Label mxfp8_mask_table;
    // Gathers the per-group exponents produced by the reduction tree into
    // consecutive bytes.
    Xbyak::Label mxfp8_final_permute_table;
    // Reorders the e8m0 scales into the layout expected by the dst scales
    // copy kernel
    Xbyak::Label mxfp8_final_permute_store_table;
    // Selects the high halves of two f32 vectors, i.e. performs a truncating
    // (round-to-zero) f32 -> bf16 conversion of both of them at once. This is
    // the rounding required by the e8m0 scale definition.
    Xbyak::Label mxfp8_bf16_truncation_table;

    int store_bd_step() const {
        return brg.is_ace() ? 8 : 3; /*heuristic values*/
    }

    // zmm_bias, zmm_bias and accm shouldn't be overlapped
    Xbyak::Zmm accm(int bd) const {
        assert(bd < 16);
        return Xbyak::Zmm(31 - (bd % store_bd_step()));
    }

    int ace_reserved_zmms() { return 5; }

    Xbyak::Zmm zmm_bias(int ldb) const {
        if (brg.is_ace()) {
            // ACE: zmm10 - zmm17 (avoids zmm6=zmm_zp_comp_a,
            // zmm7=zmm_zp_c, zmm8=zmm_lbound, zmm9=zmm_ubound)
            assert(ldb < 8); // ld_block2 capped at 8 (no-scales) in blocking
            return Xbyak::Zmm(10 + ldb);
        } else {
            assert(ldb < 5);
            // zmm10 - zmm14
            return Xbyak::Zmm(10 + ldb);
        }
    }

    Xbyak::Zmm zmm_scales(int ldb) const {
        if (brg.is_ace()) {
            // ACE: zmm18 - zmm23 (safe below accm zmm24-31)
            // ld_block2 capped at 6 when scales active in blocking
            assert(ldb < 6); // max safe: zmm18+5=zmm23 < zmm24(accm)
            assert(store_bd_step() < 10);
            return Xbyak::Zmm(18 + ldb);
        } else {
            assert(ldb < 5);
            assert(store_bd_step() < 10);
            // zmm15 - zmm19
            return Xbyak::Zmm(15 + ldb);
        }
    }

    // Scalar binary post-op RHS values, loaded once per kernel call.
    // Non-ACE: zmm20 - zmm28, between zmm_scales and accm. ACE uses all zmms
    // for A and B, so the RHS is read from memory there.
    int n_zmm_po_rhs() const {
        return brg.is_ace() ? 0 : accm(store_bd_step() - 1).getIdx() - 20;
    }
    Xbyak::Zmm zmm_po_rhs(int i) const {
        assert(0 <= i && i < n_zmm_po_rhs());
        return Xbyak::Zmm(20 + i);
    }

    // ACE rd steps = ZMM count used per A/B block in the micro-kernel.
    // This is the number of rd steps for each block.
    int ace_rd_steps(int rd_block) const noexcept {
        return div_up(rd_block, brg.rd_step);
    }

    Xbyak::Zmm ace_zmm_tmp(int i) {
        assert(1 <= i && i <= 4);
        return Xbyak::Zmm(i);
    }

    // ACE A register: bdb block index, rds register index.
    // Returns the ZMM for the ACE micro-kernel A matrix.
    Xbyak::Zmm ace_zmm_A(int bdb, int rds) {
        const int base_idx
                = ace_rd_steps(brg.rd_block) * (brg.n_bcast_1_load ? bdb : 0);
        const int idx = ace_reserved_zmms() + base_idx + rds;
        assert(idx < 32 && "ZMM register index overflow in ace_zmm_A");
        return Xbyak::Zmm(idx);
    }

    // ACE B register: ldb block index, rds register index.
    // Returns the ZMM for the ACE micro-kernel B matrix.
    Xbyak::Zmm ace_zmm_B(int ldb, int rds) {
        // n_bcast_1_load reuses one B register; only rds 0 is valid.
        assert(IMPLICATION(brg.n_bcast_1_load, rds == 0));
        const int base_idx = ace_rd_steps(brg.rd_block)
                * (brg.n_bcast_1_load ? brg.bd_block2 : (1 + ldb));
        const int idx = ace_reserved_zmms() + base_idx + rds;
        assert(idx < 32 && "ZMM register index overflow in ace_zmm_B");
        return Xbyak::Zmm(idx);
    }

    template <typename U>
    U vmm_mask(const U &vmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask) const;

    void cvt2ps(data_type_t type_in, const Xbyak::Zmm &zmm_in,
            const Xbyak::Operand &op, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);

    void read_params();
    void load_accumulators(brgemm_iteration_t &bi);

    void maybe_saturation(Xbyak::Zmm &zmm);
    void apply_alpha_beta_to_vector(
            int idx, const Address &addr, bool is_ld_tail);
    void apply_post_ops_to_range(brgemm_iteration_t &bi, int bd_start,
            int bd_finish, int bdb, int ldb);
    void store_vector_with_post_ops(
            int idx, const Address &addr, bool is_ld_tail);
    void prepare_post_ops_registers_ldb(brgemm_iteration_t &bi, int ldb);
    void prepare_post_ops_registers(brgemm_iteration_t &bi);

    bool bi_shift_output(
            brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi);
    bool bi_shift_A(
            brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi);
    bool bi_shift_B(
            brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi);

    void uni_prefetch(const Address &addr, brgemm_kernel_prefetching_t pft,
            bool for_write);
    void prefetch_CD_range(brgemm_iteration_t &bi,
            brgemm_kernel_prefetching_t pft, int bd_start, int bd_finish,
            int bdb, int ldb);
    dim_t calc_ops_CD(brgemm_iteration_t &bi) const noexcept;
    void prefetch_CD(brgemm_iteration_t &bi, brgemm_iteration_t &pfo_bi,
            prf_t &prf, bool prefetch_all);

    void prefetch_A(brgemm_iteration_t &bi, brgemm_iteration_t &pfo_bi,
            prf_t &prf, bool prefetch_all);
    void prefetch_B(brgemm_iteration_t &bi, brgemm_iteration_t &pfo_bi,
            prf_t &prf, bool prefetch_all);
    void prefetching(brgemm_iteration_t &bi, bool prefetch_all);

    void process_output_range(brgemm_iteration_t &bi, int bd_start,
            int bd_finish, int bdb, int ldb);
    void store_vector_without_post_ops(int idx, const Address &addr,
            bool is_ld_tail, data_type_t dst_dt = data_type::f32);
    void store_vector(brgemm_iteration_t &bi, int bdb, int bd, int ldb);
    void apply_comp_pad_to_vector(brgemm_iteration_t &bi, int bdb, int inp_bd,
            int ldb, const int idx);

    void interleave_store(brgemm_iteration_t &bi, bool store_all);

    void store_accumulators(brgemm_iteration_t &bi);
    void store_accumulators_default(brgemm_iteration_t &bi);

    bool need_to_output_mxfp8() const;
    bool is_mxfp8_quantization_after_postops(
            const brgemm_iteration_t &bi) const;

    void set_A_B_matrices(dim_t bs);
    void set_A_B_matrices();

    void bf32_downconvert(brgemm_iteration_t &bi, int num_rows,
            int tile_num_col_bytes, reg64_t reg_data, dim_t offset,
            reg64_t reg_data_stride, reg64_t reg_buf);
    void fp8_to_f16_upconvert(brgemm_iteration_t &bi, int num_rows,
            int tile_num_col_bytes, reg64_t reg_data, dim_t offset,
            reg64_t reg_data_stride, reg64_t reg_buf, data_type_t dt);

    void fp8_to_f16_upconvert_to_vnni(brgemm_iteration_t &bi, int num_rows,
            int tile_num_col_bytes, reg64_t reg_data, dim_t offset,
            reg64_t reg_data_stride, reg64_t reg_buf, data_type_t dt);

    void bf32_downconvert_to_vnni(brgemm_iteration_t &bi, int num_rows,
            int tile_num_col_bytes, reg64_t reg_data, dim_t offset,
            reg64_t reg_data_stride, reg64_t reg_buf);

    void maybe_pre_process_data(brgemm_iteration_t &bi, const Tmm &t1,
            reg64_t reg_base, dim_t offset, reg64_t reg_stride,
            matrix_kind_t mk);

    bool maybe_pre_process_k_tail(brgemm_iteration_t &bi, const Tmm &t1,
            reg64_t reg_base, dim_t offset, reg64_t reg_stride,
            matrix_kind_t mk, bool use_memadvice);

    void pre_process_k_tail_fused_copy_a(brgemm_iteration_t &bi, int bdb,
            const Tmm &t1, reg64_t reg_base, dim_t offset_src, dim_t offset_dst,
            bool mem_advice_A);

    void maybe_tileloadd_nt(
            brgemm_iteration_t &bi, matrix_kind_t mk, int xdb, dim_t offset);

    void maybe_fused_copy_A_nt_load(brgemm_iteration_t &bi, int bdb);

    void maybe_sprinkle_prefetches();
    void ace_load_A_4x16bytes(
            const Zmm &zmm, size_t mask, const Reg64 &reg_A, dim_t offset);

    void ace_load_A(brgemm_iteration_t &bi, int bdb, dim_t offset);
    void ace_load_B(brgemm_iteration_t &bi, int ldb, dim_t offset, int rdstep);
    // `imm8` is the BSR selector: bits [2:0] pick the A scale sub-block and
    // bits [5:3] the B one. It is meaningful on the MXFP8 path only; the
    // unscaled fp8 path passes 0 against an all-ones BSR.
    void outer_product(
            const Zmm &zmm_a, const Zmm &zmm_b, const Tmm &accm, int imm8 = 0);

    // True when the block scales are applied by the outer product itself,
    // i.e. this is an MXFP8 ACE descriptor.
    bool is_mxfp8_compute() const noexcept { return brg.is_mxfp8_ace; }

    // Loads the B block scales of the current (ldi, rdi) into zmm_tmp_2 and
    // permutes them into the BSR order. Reads ld_scale_tail_mask, which
    // set_b_scale_tail_mask() must have set for the same bi.
    void load_b_scale(const brgemm_iteration_t &bi);
    bool is_b_scale_tail(const brgemm_iteration_t &bi) const;
    // Sets ld_scale_tail_mask. It depends on bi.ldi only, so it is emitted
    // once above the rd loop rather than on every rd iteration.
    void set_b_scale_tail_mask(const brgemm_iteration_t &bi);
    // Refreshes the BSR for the rd window `bi.rdi` if that window starts a
    // new BSR field; a no-op otherwise. See the implementation for the
    // window-to-field mapping.
    void maybe_load_mxfp8_scales(const brgemm_iteration_t &bi);

    void tdpbxxd(brgemm_iteration_t &bi, int bdb_idx, int ldb_idx,
            bool do_pre_tilestore, bool do_post_tilestore);

    void gemm_microkernel_amx(brgemm_iteration_t &bi);
    void gemm_microkernel_ace(brgemm_iteration_t &bi);

    void rdb_loop_body(brgemm_iteration_t &bi);
    // Emits the rd loop as a sequence of calls to a small set of shared
    // micro-kernel bodies (see call_based_rd_loop).
    void rdb_loop_call_based(brgemm_iteration_t &bi);
    // Emits, out of the hot path, the bodies registered by
    // rdb_loop_call_based() and destroys them together with their labels.
    void emit_deferred_uk_bodies();
    void rdb_loop(brgemm_iteration_t &bi);

    void bs_loop_body(brgemm_iteration_t &bi);
    void bs_loop(brgemm_iteration_t &bi);

    void ldb_loop_body(brgemm_iteration_t &bi);
    void ldb_loop(brgemm_iteration_t &bi);

    void bdb_loop_body(brgemm_iteration_t &bi);
    void bdb_loop(brgemm_iteration_t &bi);

    void init(brgemm_iteration_t &bi);
    void generate() override;

    void prepare_bd_mask() noexcept;
    dim_t skipped_bd_mask(dim_t inp_bd) noexcept;

    bool get_store_by_vectors(bool apply_post_ops) const {
        const bool need_to_apply_post_ops
                = are_post_ops_applicable_ && apply_post_ops;
        // Tile-direct-store fast-path dumps raw accumulator dtype; for
        // narrow dt_c (bf16/f16) we must go through the per-vector store
        // path that downconverts via store_vector_without_post_ops.
        const bool c_dt_needs_cvt
                = !utils::one_of(brg.dt_c, data_type::f32, data_type::s32);
        const auto store_by_vectors = need_to_apply_alpha_beta_
                || need_to_apply_post_ops || brg.brgattr.bd_mask_level
                || brg.has_per_k_scales() || brg.with_per_mn_compensation
                || c_dt_needs_cvt;
        return store_by_vectors;
    }
    bool actual_ils(bool apply_post_ops, bool skip_accumulation = false) const {
        return (use_ils_ && get_store_by_vectors(apply_post_ops)
                && !skip_accumulation);
    }

    // Quantizes the accumulators of the current iteration to MXFP8: computes
    // the e8m0 scale of every group of 32 elements along the ld dimension,
    // stores the scales through `reg_dst_scales` and down-converts the values
    // to `brg.dt_d`. Values are read from the C tiles if `tile_src` is set and
    // from the tile workspace otherwise.
    // The emitted code is split into a sequence of numbered nodes; only the
    // nodes in the [node_start, node_end) range are emitted. This is used to
    // interleave quantization with the rest of the kernel.
    void quantize_to_mxfp8(brgemm_iteration_t &bi, bool tile_src,
            int node_start, int node_end);

    // Emits the constant tables used by quantize_to_mxfp8() into the generated
    // code.
    void emit_mxfp8_tables();

    // Sets the constant row masks of the ACE A load. quantize_to_mxfp8()
    // clobbers them, so they are re-emitted before the rd loop.
    void emit_ace_load_A_masks();

    dim_t A_offset(
            const brgemm_iteration_t &bi, int bdb, int rdb = 0) const noexcept;

    dim_t A_offset_wsp(
            const brgemm_iteration_t &bi, int bdb, int rdb = 0) const noexcept;

    dim_t A_offset_line(const brgemm_iteration_t &bi, int bdb, int rdb = 0,
            int bd_elem_idx = 0) const noexcept;

    dim_t B_offset(
            const brgemm_iteration_t &bi, int ldb, int rdb = 0) const noexcept;

    dim_t B_offset_line(const brgemm_iteration_t &bi, int ldb, int rdb = 0,
            int rd_elem_idx = 0) const noexcept;

    dim_t C_offset(const brgemm_iteration_t &bi, int bdb, int inp_bd,
            dim_t ldb) const noexcept;

    dim_t C_offset_row(const brgemm_iteration_t &bi, int bdb, int inp_bd,
            dim_t ldb) const noexcept;

    dim_t C_offset_wsp(const brgemm_iteration_t &bi, int bdb, int ldb,
            int inp_bd) const noexcept;

    dim_t D_offset(const brgemm_iteration_t &bi, int bdb, int inp_bd,
            dim_t ldb) const noexcept;

    dim_t D_scales_offset(const bd_iteration_t *bdi, int bdb, int inp_bd,
            dim_t ldb, bool force_global_pos = false) const noexcept;

    dim_t D_scales_offset(const brgemm_iteration_t &bi, int bdb, int inp_bd,
            dim_t ldb) const noexcept;

    dim_t lda() const noexcept;
    dim_t ldb() const noexcept;

    dim_t bias_offset(dim_t ldb) const noexcept;

    dim_t scales_offset(dim_t ldb) const noexcept;
    // MXFP8 only. Offset, in bytes, of the A block scales of the given
    // iteration inside the repacked A-scales slab of one (M_blk, K_blk)
    // block. The layout is documented on A_offset_scales(m, k).
    dim_t A_offset_scales(const brgemm_iteration_t &bi, int bdb, int rdb = 0,
            int bd_elem_idx = 0) const noexcept;
    dim_t A_offset_scales(dim_t m, dim_t k) const noexcept;
    // MXFP8 only. Offset, in bytes, of the B block scales of the given
    // iteration in the user tensor, whose layout is the plain mathematical
    // [K / mx_group_size][N] with a row stride of brg.LDB_scales.
    dim_t B_offset_scales(
            const brgemm_iteration_t &bi, int ldb, int rdb = 0) const noexcept;
    dim_t zp_comp_a_offset(dim_t ldb) const noexcept;
    dim_t zp_comp_pad_a_offset(const brgemm_iteration_t &bi, int bdb,
            int inp_bd, dim_t ldb) const noexcept;
    dim_t zp_comp_b_offset(dim_t bd) const noexcept;
    dim_t zp_c_values_offset(brgemm_iteration_t &bi, int ldb) const noexcept;
    dim_t per_mn_comp_offset(const brgemm_iteration_t &bi, int bdb, int inp_bd,
            dim_t ldb) const noexcept;
    bool is_out_bd(const bd_iteration_t *bdi, int bdb, int inp_bd) const;
    dim_t get_out_bd(const bd_iteration_t *bdi, int bdb, int inp_bd) const;

    void maybe_tilestore(brgemm_iteration_t &bi, int bdb_idx, int ldb_idx,
            bool do_pre_tilestore, bool do_post_tilestore);
    int get_C_tensor(brgemm_iteration_t &bi, int m, int n) const noexcept;
    void top_loop(brgemm_iteration_t &bi);
    bd_iteration_t *find_similar(const bd_iteration_t *bdi, bool apply_postops);

    void fill_imap();
    void copy_k_tail_to_wsp(const Tmm &t1,
            jit_brgemm_amx_uker_t::reg64_t &reg_base, dim_t src_offset,
            jit_brgemm_amx_uker_t::reg64_t &reg_src_stride, bool use_memadvice);
};

bool jit_brgemm_amx_uker_t::bi_shift_output(
        brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi) {
    res_bi = bi;
    if (shift == 0) return true;

    const auto &tloop = imap_[bi.apply_postops];
    const auto nldis = tloop.ldis.size();
    const auto nbdis = tloop.bdis.size();
    size_t lidx = 0;
    size_t bd_idx = 0;
    size_t ld_idx = 0;
    if (brg.innermost_loop == brgemm_ld_loop_innermost) {
        lidx = bi.bdi->idx * nldis + bi.ldi->idx;
        lidx += shift;
        bd_idx = lidx / nldis;
        ld_idx = lidx % nldis;
    } else if (brg.innermost_loop == brgemm_bd_loop_innermost) {
        lidx = bi.ldi->idx * nbdis + bi.bdi->idx;
        lidx += shift;
        ld_idx = lidx / nbdis;
        bd_idx = lidx % nbdis;
    } else
        assert(!"Unknown loop order!");
    if (lidx >= nldis * nbdis) return false;
    res_bi.bdi = &(tloop.bdis[bd_idx]);
    res_bi.ldi = &(tloop.ldis[ld_idx]);

    return true;
}

bool jit_brgemm_amx_uker_t::bi_shift_A(
        brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi) {
    res_bi = bi;
    const auto &tloop = imap_[bi.apply_postops];
    const auto nbdis = tloop.bdis.size();
    const auto nrdis = tloop.rdis.size();

    auto lidx = bi.bdi->idx * nrdis + bi.rdi->idx;
    lidx += shift;
    if (lidx >= nrdis * nbdis) return false;

    const auto bd_idx = lidx / nrdis;
    const auto rd_idx = lidx % nrdis;

    res_bi.bdi = &(tloop.bdis[bd_idx]);
    res_bi.rdi = &(tloop.rdis[rd_idx]);

    return true;
}

bool jit_brgemm_amx_uker_t::bi_shift_B(
        brgemm_iteration_t &bi, int shift, brgemm_iteration_t &res_bi) {
    res_bi = bi;
    const auto &tloop = imap_[bi.apply_postops];
    const auto nldis = tloop.ldis.size();
    const auto nrdis = tloop.rdis.size();

    auto lidx = bi.ldi->idx * nrdis + bi.rdi->idx;
    lidx += shift;
    if (lidx >= nrdis * nldis) return false;

    const auto ld_idx = lidx / nrdis;
    const auto rd_idx = lidx % nrdis;

    res_bi.ldi = &(tloop.ldis[ld_idx]);
    res_bi.rdi = &(tloop.rdis[rd_idx]);

    return true;
}

int jit_brgemm_amx_uker_t::get_C_tensor(
        brgemm_iteration_t &bi, int m, int n) const noexcept {
    return brg.get_C_tensor(m, n, bi.bdi->is_tail(m), bi.ldi->is_tail(n));
}

void jit_brgemm_amx_uker_t::prepare_bd_mask() noexcept {
    if (!brg.brgattr.bd_mask_level) return;
    bd_mask_buffer_ptr_ = brg.brgattr.bd_mask;
    const auto bd_mask_size = brg.bcast_dim;
    adj_bd_mask_buffer_.resize(bd_mask_size);
    skipped_bd_mask_buffer_.resize(bd_mask_size);
    if (bd_mask_buffer_ptr_ != nullptr) {
        dim_t out_ibd = 0;
        for (dim_t i = 0; i < bd_mask_size; i++) {
            adj_bd_mask_buffer_[i] = out_ibd;
            out_ibd += bd_mask_buffer_ptr_[i];
            skipped_bd_mask_buffer_[i] = i;
            for (auto ii = i; ii < bd_mask_size; ii++) {
                if (bd_mask_buffer_ptr_[ii]) {
                    skipped_bd_mask_buffer_[i] = ii;
                    break;
                }
            }
        }
    } else
        assert(!"struct nullptr error");
}

dim_t jit_brgemm_amx_uker_t::skipped_bd_mask(dim_t inp_bd) noexcept {
    if (brg.brgattr.bd_mask_level != 2)
        return inp_bd;
    else
        return skipped_bd_mask_buffer_[inp_bd];
}

dim_t jit_brgemm_amx_uker_t::A_offset_wsp(
        const brgemm_iteration_t &bi, int bdb, int rdb) const noexcept {
    // Fused copy A layout: [bs][k / rd_block][m][k = rd_block].
    const auto transform_offset
            = brg.get_wsp_base_offset(brgemm_desc_t::wsp_fused_copy_a);

    const auto bs_offs = bi.bsi->pos * brg.bcast_dim
            * rnd_up(brg.reduce_dim, brg.max_rd_block()) * brg.typesize_A;

    const auto bdb_offs = bi.bdi->pos(bdb) * brg.rd_block_A_size();
    const auto rdb_offs
            = bi.rdi->pos(rdb) * brg.bcast_dim * brg.rd_block_A_size();

    return transform_offset + bs_offs + bdb_offs + rdb_offs;
}

dim_t jit_brgemm_amx_uker_t::A_offset(
        const brgemm_iteration_t &bi, int bdb, int rdb) const noexcept {
    const auto bs_offs = (brg.type == brgemm_static_offs)
            ? brg.brgattr.static_offsets[bi.bsi->idx].offset.A
            : 0;
    const auto bdb_offs
            = ununroll_bd_loop ? bi.bdi->rel_pos(bdb) : bi.bdi->pos(bdb);
    return bdb_offs * LDA2_size_ + bs_offs
            + bi.rdi->pos(rdb) * brg.rd_block_A_size();
}

dim_t jit_brgemm_amx_uker_t::A_offset_line(const brgemm_iteration_t &bi,
        int bdb, int rdb, int bd_elem_idx) const noexcept {
    return A_offset(bi, bdb, rdb) + bd_elem_idx * LDA2_size_;
}

dim_t jit_brgemm_amx_uker_t::B_offset(
        const brgemm_iteration_t &bi, int ldb, int rdb) const noexcept {
    const auto bs_offs = (brg.type == brgemm_static_offs)
            ? brg.brgattr.static_offsets[bi.bsi->idx].offset.B
            : 0;

    const auto rdb_B_offset = bi.rdi->pos(rdb) * brg.rd_block * LDB_size_;

    const auto ldb_offs = bi.ldi->pos(ldb) * brg.ld_block;
    const auto ldb_B_offset = brg.typesize_B
            * ((ldb_offs / brg.LDB) * brg.brgattr.LDB2
                    + (ldb_offs % brg.LDB) * brg.rd_step);

    return rdb_B_offset + ldb_B_offset + bs_offs;
}

dim_t jit_brgemm_amx_uker_t::B_offset_line(const brgemm_iteration_t &bi,
        int ldb, int rdb, int rd_elem_idx) const noexcept {
    return B_offset(bi, ldb, rdb) + rd_elem_idx * LDB_size_;
}

dim_t jit_brgemm_amx_uker_t::C_offset(const brgemm_iteration_t &bi, int bdb,
        int inp_bd, dim_t ldb) const noexcept {
    const auto bi_bd_start = get_out_bd(bi.bdi, 0, 0);
    const auto bd = get_out_bd(bi.bdi, bdb, inp_bd);
    const auto bd_shift = bd - (ununroll_bd_loop ? bi_bd_start : 0);
    dim_t ldc_elem = (dim_t)ldb * brg.ld_block;
    dim_t bloc_idx = ldc_elem / brg.LDC;
    dim_t in_block = ldc_elem % brg.LDC;

    return (dim_t)bd_shift * LDC2_size_M_ + (dim_t)bloc_idx * LDC2_size_N_
            + in_block * brg.typesize_C;
}

// ACE stores accumulators one row at a time, so the bd stride within a block is
// LDC_size_ and not LDC2_size_M_, which matters when LDC2_M != LDC. The offset
// between bdb blocks still uses LDC2_size_M_, as legacy AMX tilestored does.
dim_t jit_brgemm_amx_uker_t::C_offset_row(const brgemm_iteration_t &bi, int bdb,
        int inp_bd, dim_t ldb) const noexcept {
    const auto bi_bd_start = get_out_bd(bi.bdi, 0, 0);
    const auto bdb_start = get_out_bd(bi.bdi, bdb, 0);
    const auto bdb_start_shift
            = bdb_start - (ununroll_bd_loop ? bi_bd_start : 0);
    dim_t ldc_elem = (dim_t)ldb * brg.ld_block;
    dim_t bloc_idx = ldc_elem / brg.LDC;
    dim_t in_block = ldc_elem % brg.LDC;

    return (dim_t)bdb_start_shift * LDC2_size_M_ + (dim_t)inp_bd * LDC_size_
            + (dim_t)bloc_idx * LDC2_size_N_ + in_block * brg.typesize_C;
}

dim_t jit_brgemm_amx_uker_t::D_offset(const brgemm_iteration_t &bi, int bdb,
        int inp_bd, dim_t ldb) const noexcept {
    const auto bi_bd_start = get_out_bd(bi.bdi, 0, 0);
    const auto bd = get_out_bd(bi.bdi, bdb, inp_bd);
    const auto bd_shift = bd - (ununroll_bd_loop ? bi_bd_start : 0);
    return (dim_t)bd_shift * LDD_size_ + (dim_t)ldb * ld_block_D_size_;
}

dim_t jit_brgemm_amx_uker_t::D_scales_offset(const bd_iteration_t *bdi, int bdb,
        int inp_bd, dim_t ldb, bool force_global_pos) const noexcept {
    // `top_loop` computes shifts between iterations and hence needs absolute
    // positions, while in-loop uses are relative to the current bd iteration.
    const bool global_pos = force_global_pos || !ununroll_bd_loop;
    const auto bi_bd_start = get_out_bd(bdi, 0, 0);
    const auto bd = get_out_bd(bdi, bdb, inp_bd);
    const auto bd_shift = bd - (global_pos ? 0 : bi_bd_start);
    // The scales are staged as [n1 = N/64][m = rnd_up(bcast_dim, 32)][n0 = 2]:
    // each group of 4 ld blocks (4 * 16 = 64 elements) holds 2 e8m0 scale
    // groups of 32 elements each.
    constexpr dim_t scales_per_ld_group = 2;
    constexpr dim_t ld_blocks_per_ld_group = 4;
    const dim_t total_m = rnd_up(brg.bcast_dim, mx_group_size);
    return bd_shift * scales_per_ld_group
            + (ldb / ld_blocks_per_ld_group) * total_m * scales_per_ld_group;
}

dim_t jit_brgemm_amx_uker_t::D_scales_offset(const brgemm_iteration_t &bi,
        int bdb, int inp_bd, dim_t ldb) const noexcept {
    return D_scales_offset(bi.bdi, bdb, inp_bd, ldb);
}

dim_t jit_brgemm_amx_uker_t::C_offset_wsp(const brgemm_iteration_t &bi, int bdb,
        int ldb, int inp_bd) const noexcept {
    // The workspace holds the C tiles of the iteration one after another, each
    // of them stored as `bd_block` rows of `ld_block_C_size_` bytes. Passing
    // bdb = ldb = 0 gives the offset of the single slot shared by all the
    // tiles when they are not staged per tile.
    return brg.get_wsp_base_offset(brgemm_desc_t::wsp_c_tiles)
            + (bdb * bi.ldi->block2() + ldb) * bi.bdi->block(0)
            * ld_block_C_size_
            + inp_bd * ld_block_C_size_;
}
dim_t jit_brgemm_amx_uker_t::lda() const noexcept {
    return LDA_size_;
}

dim_t jit_brgemm_amx_uker_t::ldb() const noexcept {
    return LDB_size_ * brg.rd_step;
}

dim_t jit_brgemm_amx_uker_t::bias_offset(dim_t ldb) const noexcept {
    return ldb * ld_block_bias_size_;
}

dim_t jit_brgemm_amx_uker_t::scales_offset(dim_t ldb) const noexcept {
    return brg.is_per_n_wei_scales * ldb * ld_block_scales_size_;
}
dim_t jit_brgemm_amx_uker_t::A_offset_scales(const brgemm_iteration_t &bi,
        int bdb, int rdb, int bd_elem_idx) const noexcept {
    // One BSR A field covers 32 rows x 2 scale groups, and the selector
    // encoded in the outer product picks the (16-row, 1-group) quarter of it,
    // so the address is rounded down to that granularity here.
    const auto bdb_pos
            = ununroll_bd_loop ? bi.bdi->rel_pos(bdb) : bi.bdi->pos(bdb);
    const dim_t m = rnd_dn(bdb_pos + bd_elem_idx, 32);
    const dim_t k = rnd_dn(bi.rdi->pos(rdb) * brg.rd_block, 2 * mx_group_size);
    return A_offset_scales(m, k);
}

dim_t jit_brgemm_amx_uker_t::A_offset_scales(dim_t m, dim_t k) const noexcept {
    // Repacked A scales of a single (M_blk, K_blk) block, one byte per e8m0
    // scale:
    //     [m2 = rnd_up(M_blk, 32) / 32][k1 = rnd_up(K_blk, 64) / 64]
    //             [m0 = 16][m1 = 2][k0 = 2]
    // The innermost [m0][m1][k0] tile is exactly one 64-byte BSR A field.
    // The same layout is produced by jit_brgemm_matmul_copy_a_scales_t and
    // addressed by brgemm_matmul; all three are block-local, the driver owns
    // the outer [M_chunk][K_chunk] indexing.
    constexpr dim_t m0_size = 16;
    constexpr dim_t m1_size = 2;
    constexpr dim_t k0_size = 2;

    const dim_t k_scales = div_up(brg.reduce_dim, mx_group_size);
    const dim_t k1_count = div_up(k_scales, k0_size);
    const dim_t k_scale_idx = k / mx_group_size;

    const dim_t m2 = m / (m0_size * m1_size);
    const dim_t m1 = (m % (m0_size * m1_size)) / m0_size;
    const dim_t m0 = m % m0_size;
    const dim_t k1 = k_scale_idx / k0_size;
    const dim_t k0 = k_scale_idx % k0_size;

    const dim_t m1_stride = k0_size;
    const dim_t m0_stride = m1_stride * m1_size;
    const dim_t k1_stride = m0_size * m0_stride;
    const dim_t m2_stride = k1_count * k1_stride;

    return m2 * m2_stride + k1 * k1_stride + m0 * m0_stride + m1 * m1_stride
            + k0;
}

dim_t jit_brgemm_amx_uker_t::B_offset_scales(
        const brgemm_iteration_t &bi, int ldb, int rdb) const noexcept {
    // B scales in the plain mathematical layout [K / mx_group_size][N] with a
    // row stride of brg.LDB_scales (== N), one byte per e8m0 scale. One BSR B
    // field holds the 64 N values of a single scale group; the selector picks
    // the 16-wide ldb slice of it, so the address is rounded down to 64.
    const dim_t n = rnd_dn(bi.ldi->pos(ldb) * brg.ld_block, 64);
    const dim_t k = rnd_dn(bi.rdi->pos(rdb) * brg.rd_block, mx_group_size);
    return (k / mx_group_size) * brg.LDB_scales + n;
}

dim_t jit_brgemm_amx_uker_t::zp_comp_a_offset(dim_t ldb) const noexcept {
    return ldb * ld_block_zp_size_;
}

dim_t jit_brgemm_amx_uker_t::zp_comp_pad_a_offset(const brgemm_iteration_t &bi,
        int bdb, int inp_bd, dim_t ldb) const noexcept {
    const auto bi_bd_start = get_out_bd(bi.bdi, 0, 0);
    const auto bd = get_out_bd(bi.bdi, bdb, inp_bd);
    const auto bd_shift = bd - (ununroll_bd_loop ? bi_bd_start : 0);
    return (dim_t)bd_shift * brg.LDB * sizeof(int32_t)
            + (dim_t)ldb * ld_block_zp_size_;
}

dim_t jit_brgemm_amx_uker_t::zp_comp_b_offset(dim_t bd) const noexcept {
    return sizeof(int32_t) * bd;
}

dim_t jit_brgemm_amx_uker_t::zp_c_values_offset(
        brgemm_iteration_t &bi, int ldb) const noexcept {
    if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
        return (bi.ldi->is_tail(ldb)) ? ldb_tail_zp_size_
                                      : bi.ldi->pos(ldb) * ld_block_zp_size_;
    }

    return 0;
}

dim_t jit_brgemm_amx_uker_t::per_mn_comp_offset(const brgemm_iteration_t &bi,
        int bdb, int inp_bd, dim_t ldb) const noexcept {
    const auto bi_bd_start = get_out_bd(bi.bdi, 0, 0);
    const auto bd = get_out_bd(bi.bdi, bdb, inp_bd);
    assert(bd >= 0);
    const auto bd_shift = bd - (ununroll_bd_loop ? bi_bd_start : 0);
    const dim_t ldc_elem = (dim_t)ldb * brg.ld_block;
    const dim_t bloc_idx = ldc_elem / brg.LDC;
    const dim_t in_block = ldc_elem % brg.LDC;
    return (dim_t)sizeof(float)
            * ((dim_t)bd_shift * brg.LDC2_M + (dim_t)bloc_idx * brg.LDC2_N
                    + in_block);
}

bool jit_brgemm_amx_uker_t::is_out_bd(
        const bd_iteration_t *bdi, int bdb, int inp_bd) const {
    const auto bd = bdi->pos(bdb) + inp_bd;
    return IMPLICATION(
            brg.brgattr.bd_mask_level, bdi->bd_mask[bd - bdi->pos(0)] != 0);
}

dim_t jit_brgemm_amx_uker_t::get_out_bd(
        const bd_iteration_t *bdi, int bdb, int inp_bd) const {
    if (!is_out_bd(bdi, bdb, inp_bd)) return -1;
    const auto bd = bdi->pos(bdb) + inp_bd;
    if (brg.brgattr.bd_mask_level) {
        assert(bdi->adj_bd_mask[bd - bdi->pos(0)] == adj_bd_mask_buffer_[bd]);
        return bdi->adj_bd_mask[bd - bdi->pos(0)];
    } else
        return bd;
}

template <typename U>
U jit_brgemm_amx_uker_t::vmm_mask(const U &vmm_in, bool mask_flag, bool store,
        Xbyak::Opmask ktail_mask) const {
    return mask_flag ? (store ? vmm_in | ktail_mask : vmm_in | ktail_mask | T_z)
                     : vmm_in;
}

void jit_brgemm_amx_uker_t::cvt2ps(data_type_t type_in,
        const Xbyak::Zmm &zmm_in, const Xbyak::Operand &op, bool mask_flag,
        bool store, Xbyak::Opmask ktail_mask) {
    const Xbyak::Zmm zmm = vmm_mask(zmm_in, mask_flag, store, ktail_mask);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: vmovups(zmm, op); break;
        case data_type::bf16:
            vpmovzxwd(zmm, op);
            vpslld(zmm, zmm, 16);
            break;
        case data_type::f16: vcvtph2ps(zmm, op); break;
        case data_type::f8_e5m2: f8_e5m2_cvt_->vcvt_f8_to_f32(zmm, op); break;
        case data_type::f8_e4m3: f8_e4m3_cvt_->vcvt_f8_to_f32(zmm, op); break;
        case data_type::s8: vpmovsxbd(zmm, op); break;
        case data_type::u8: vpmovzxbd(zmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (types::is_integral_dt(type_in)) vcvtdq2ps(zmm_in, zmm_in);
}

void jit_brgemm_amx_uker_t::read_params() {
    Label label_done;

    mov(reg_BS, ptr[param1 + GET_OFF(BS)]);

    mov(reg_addr_batch, ptr[param1 + GET_OFF(batch)]);

    mov(reg_buf, ptr[param1 + GET_OFF(ptr_buf)]);

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_a, ptr[param1 + GET_OFF(a_zp_compensations)]);
        reg_zp_comp_a.save();
        mov(reg_zp_a_values, ptr[param1 + GET_OFF(zp_a_val)]);
        reg_zp_a_values.save();

        if (brg.req_comp_pads_with_bcast)
            mov(reg_zp_comp_pad_a, ptr[param1 + GET_OFF(a_zp_compensations)]);
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        mov(reg_zp_comp_b, ptr[param1 + GET_OFF(b_zp_compensations)]);
        reg_zp_comp_b.save();
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        mov(reg_zp_c_values, ptr[param1 + GET_OFF(c_zp_values)]);
        reg_zp_c_values.save();
    }

    if (brg.is_per_k_src_scales) {
        mov(reg_src_scales_per_k, ptr[param1 + GET_OFF(ptr_src_scales)]);
        reg_src_scales_per_k.save();
    }

    if (brg.with_per_mn_compensation) {
        mov(reg_per_mn_comp, ptr[param1 + GET_OFF(ptr_per_mn_compensation)]);
        reg_per_mn_comp.save();
    }

    if (brg.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
        reg_bias.save();
    }
    if (brg.with_src_scales) {
        mov(reg_src_scales, ptr[param1 + GET_OFF(ptr_src_scales)]);
        reg_src_scales.save();
    }
    if (brg.with_wei_scales) {
        mov(reg_wei_scales, ptr[param1 + GET_OFF(ptr_wei_scales)]);
        reg_wei_scales.save();
    }
    if (brg.with_dst_scales) {
        mov(reg_dst_scales, ptr[param1 + GET_OFF(ptr_dst_scales)]);
        reg_dst_scales.save();
    }

    if (brg.type == brgemm_offs || brg.type == brgemm_static_offs) {
        if (brg.layout == brgemm_row_major) {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
        } else {
            mov(reg_A, ptr[param1 + GET_OFF(ptr_B)]);
            mov(reg_B, ptr[param1 + GET_OFF(ptr_A)]);
        }
        reg_A.save();
        reg_B.save();
    }

    mov(reg_C, ptr[param1 + GET_OFF(ptr_C)]);
    reg_C.save();
    mov(reg_D, ptr[param1 + GET_OFF(ptr_D)]);
    reg_D.save();
}

void jit_brgemm_amx_uker_t::load_accumulators(brgemm_iteration_t &bi) {
    dim_t ils_shift = 0;
    if (may_load_accumulators_) {
        mov(reg_stride_ld_block, LDC_size_);
        const auto need_ils_shift
                = (actual_ils(bi.apply_postops, bi.skip_accumulation)
                        && ununroll_bd_loop && bi.ldi->idx == 0);
        // if need_ils_shift then we have to add shift to C because reg_C points
        // to previous iteration in this case
        ils_shift = need_ils_shift ? bi.bdi->C_shift : 0;
    }

    for_(int bdb = 0; bdb < bi.bdi->block2(); bdb++)
    for (int ldb = 0; ldb < bi.ldi->block2(); ldb++) {
        if (may_load_accumulators_) {
            // TILELOADD is not supported under the ACE palette and raises
            // #UD. may_load_accumulators_ is therefore never set for ACE;
            // see init().
            assert(!brg.is_ace()
                    && "TILELOADD is illegal under the ACE palette");
            auto c_offset = C_offset(bi, bdb, 0, bi.ldi->pos(ldb)) + ils_shift;
            tileloadd(Tmm(get_C_tensor(bi, bdb, ldb)),
                    ptr[reg_C + c_offset + reg_stride_ld_block]);
        } else {
            // call tilezero on very first iteration
            if (!brg.interleave_tilestores_
                    || everyone_is(0u, bi.bdi->idx, bi.ldi->idx))
                tilezero(Tmm(get_C_tensor(bi, bdb, ldb)));
        }
    }
}

void jit_brgemm_amx_uker_t::apply_alpha_beta_to_vector(
        int idx, const Address &addr, bool is_ld_tail) {
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    auto zmm = Zmm(idx);
    auto zmm_beta = zmm_tmp_1();
    auto zmm_alpha = zmm_tmp_2();
    auto zmm_prev_dst = zmm_tmp_3();

    const bool apply_alpha = brg.alpha != 1.f;
    const bool apply_beta = brg.beta != 0.f;
    if (!apply_alpha && !apply_beta) return;

    // vaddps/vpaddd beta=1 fast-path reads the C buffer as raw accumulator
    // dtype; only valid when dt_c matches it. For narrow dt_c (bf16/f16/f8)
    // fall through to cvt2ps + vfmadd231ps so previous-dst is upconverted.
    const bool c_dt_matches_acc
            = utils::one_of(brg.dt_c, data_type::f32, data_type::s32);
    // When K-scales (wei or src) are applied, accumulator is already
    // converted to float (dq2ps + scales before alpha_beta), C stores floats.
    const bool use_vadd_for_beta
            = brg.beta == 1.f && !brg.do_dq2ps_cvt() && c_dt_matches_acc;

    if (apply_beta && !use_vadd_for_beta) {
        mov(reg_tmp_gpr, float2int(static_cast<float>(brg.beta)));
        vmovq(Xmm(zmm_beta.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_beta, Xmm(zmm_beta.getIdx()));
    }
    if (apply_alpha) {
        mov(reg_tmp_gpr, float2int(static_cast<float>(brg.alpha)));
        vmovq(Xmm(zmm_alpha.getIdx()), reg_tmp_gpr);
        vbroadcastss(zmm_alpha, Xmm(zmm_alpha.getIdx()));
    }
    if (brg.do_dq2ps_cvt()) vcvtdq2ps(zmm, zmm);
    if (apply_alpha) vmulps(zmm, zmm, zmm_alpha);
    if (apply_beta) {
        if (use_vadd_for_beta) {
            auto zmm_masked = zmm | k_mask | T_z;
            if (brg.is_integer_acc())
                vpaddd(zmm_masked, zmm, addr);
            else
                vaddps(zmm_masked, zmm, addr);
        } else {
            cvt2ps(brg.dt_c, zmm_prev_dst, addr, true, false, k_mask);
            vfmadd231ps(zmm, zmm_prev_dst, zmm_beta);
        }
    }
}

void jit_brgemm_amx_uker_t::apply_post_ops_to_range(
        brgemm_iteration_t &bi, int bd_start, int bd_finish, int bdb, int ldb) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    rhs_arg_params.preloaded_rhs = bi.preloaded_po_rhs;
    const auto ldb_pos = bi.ldi->pos(ldb);
    const auto is_ld_tail = bi.ldi->is_tail(ldb);

    // Sum post-op requires binary parameters to be set.
    const bool set_for_binary = brg.with_binary && handle_binary_po_offset_;
    if (set_for_binary || brg.with_sum) {
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            // We have no way to tell the injector to skip some vectors.
            // Therefore, we must set parameters correctly for all registers.
            // TODO: Make it possible to specify "skipped" vectors to injector
            const auto idx = accm(bd).getIdx();
            if (is_ld_tail) rhs_arg_params.vmm_tail_idx_.emplace(idx);
            rhs_arg_params.vmm_idx_to_out_reg.emplace(idx, reg_D);

            // Rows that are not stored keep the default zero offset. Sum then
            // reads the head of `reg_D`, which stays in bounds, and the result
            // is dropped along with the accumulator.
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            const auto d_offset = D_offset(bi, bdb, bd, ldb_pos);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(idx, d_offset);
        }
    }

    // Using knowledge how "accm" assign zmm registers.
    // TODO: make this code more clear
    const auto finish_idx = accm(bd_start).getIdx() + 1;
    const auto start_idx = accm(bd_finish - 1).getIdx();
    postops_injector_->compute_vector_range(
            start_idx, finish_idx, rhs_arg_params);
}

void jit_brgemm_amx_uker_t::maybe_saturation(Xbyak::Zmm &zmm) {
    if (!dt_requires_saturation_) return;
    // For ACE, zmm_lbound/zmm_ubound (zmm8/zmm9) are reused as ace_zmm_B
    // registers and clobbered during computation, so restore the saturation
    // state before converting.
    if (brg.is_ace())
        init_saturate_f32(zmm_lbound, zmm_ubound, reg_tmp_gpr, data_type::f32,
                brg.dt_d, false, use_sat_cvt_);
    saturate_cvt_f32(
            zmm, zmm_lbound, zmm_ubound, brg.dt_d, false, use_sat_cvt_);
}

void jit_brgemm_amx_uker_t::prepare_post_ops_registers_ldb(
        brgemm_iteration_t &bi, int ldb) {
    if (!bi.apply_postops) return;
    auto k_mask = (!bi.ldi->is_tail(ldb)) ? ld_full_mask : ld_tail_mask;

    if (brg.zp_type_a != brgemm_broadcast_t::none) {
        const auto zmm_zp_a_val = zmm_tmp_1();
        reg_zp_a_values.restore();
        vpbroadcastd(zmm_zp_a_val, reg_zp_a_values.cvt32());
        vcvtdq2ps(zmm_zp_a_val, zmm_zp_a_val);
        reg_zp_comp_a.restore();

        const auto zp_comp_a_off = zp_comp_a_offset(bi.ldi->pos(ldb));
        const auto zp_comp_a_addr
                = EVEX_compress_addr(reg_zp_comp_a, zp_comp_a_off);
        cvt2ps(data_type::s32, zmm_zp_comp_a, zp_comp_a_addr, true, false,
                k_mask);
        vmulps(zmm_zp_comp_a, zmm_zp_comp_a, zmm_zp_a_val);
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        reg_zp_c_values.restore();
        if (brg.zp_type_c == brgemm_broadcast_t::per_tensor) {
            vcvtdq2ps(zmm_zp_c, EVEX_compress_addr(reg_zp_c_values, 0, true));
        }
        if (brg.zp_type_c == brgemm_broadcast_t::per_n) {
            const auto zp_c_off = zp_c_values_offset(bi, ldb);
            const auto zp_c_addr
                    = EVEX_compress_addr(reg_zp_c_values, zp_c_off);
            cvt2ps(data_type::s32, zmm_zp_c, zp_c_addr, true, false, k_mask);
        }
    }
}

void jit_brgemm_amx_uker_t::prepare_post_ops_registers(brgemm_iteration_t &bi) {
    const auto ldi = bi.ldi;

    // MXFP8 consumes both scale tensors inside the outer product (they are
    // staged in the BSR, see maybe_load_mxfp8_scales()), so none of the
    // epilogue scale vectors below are loaded. They are also not merely
    // redundant: the scales are e8m0 bytes, and the loaders here would read
    // them as f32/bf16/f16.
    const bool scales_in_epilogue = !is_mxfp8_compute();

    // Load wei_scales for per K-block application (is_per_k_wei_scales).
    // This must happen for both apply_postops and non-apply_postops paths.
    if (scales_in_epilogue && brg.with_wei_scales && brg.is_per_k_wei_scales) {
        reg_wei_scales.restore();
        for (int ldb = 0; ldb < ldi->block2(); ldb++) {
            auto scales_ptr = EVEX_compress_addr(
                    reg_wei_scales, scales_offset(ldi->pos(ldb)));
            auto k_mask = ldi->is_tail(ldb) ? ld_tail_mask : ld_full_mask;
            if (brg.is_single_wei_scale) {
                // Broadcast a single scale value — handle non-f32 types.
                switch (brg.dt_wei_scales) {
                    case data_type::f32:
                        vbroadcastss(zmm_scales(ldb), scales_ptr);
                        break;
                    case data_type::bf16:
                        vpbroadcastw(zmm_scales(ldb), scales_ptr);
                        vpslld(zmm_scales(ldb), zmm_scales(ldb), 16);
                        break;
                    case data_type::f16:
                        vpbroadcastw(zmm_scales(ldb), scales_ptr);
                        vcvtph2ps(zmm_scales(ldb),
                                Xbyak::Ymm(zmm_scales(ldb).getIdx()));
                        break;
                    default: vbroadcastss(zmm_scales(ldb), scales_ptr); break;
                }
            } else {
                // Load a vector of scale values with proper type conversion.
                cvt2ps(brg.dt_wei_scales, zmm_scales(ldb), scales_ptr, true,
                        false, k_mask);
            }
        }

        if (brg.with_src_scales && !brg.is_per_k_src_scales) {
            reg_src_scales.restore();
            auto zmm_src_sc = zmm_tmp_1();
            auto src_sc_addr = EVEX_compress_addr(reg_src_scales, 0);
            switch (brg.dt_src_scales) {
                case data_type::bf16:
                    vpbroadcastw(zmm_src_sc, src_sc_addr);
                    vpslld(zmm_src_sc, zmm_src_sc, 16);
                    break;
                case data_type::f16:
                    vpbroadcastw(zmm_src_sc, src_sc_addr);
                    vcvtph2ps(zmm_src_sc, Xbyak::Ymm(zmm_src_sc.getIdx()));
                    break;
                case data_type::f32:
                default: vbroadcastss(zmm_src_sc, src_sc_addr); break;
            }
            for (int ldb = 0; ldb < ldi->block2(); ldb++)
                vmulps(zmm_scales(ldb), zmm_scales(ldb), zmm_src_sc);
        }
    }

    if (!bi.apply_postops) return;

    if (brg.with_bias) {
        reg_bias.restore();

        for (int ldb = 0; ldb < ldi->block2(); ldb++) {
            auto ptr_bias
                    = EVEX_compress_addr(reg_bias, bias_offset(ldi->pos(ldb)));
            auto k_mask = ldi->is_tail(ldb) ? ld_tail_mask : ld_full_mask;
            cvt2ps(brg.dt_bias, zmm_bias(ldb), ptr_bias, true, false, k_mask);
        }
    }

    if (scales_in_epilogue && brg.with_src_scales && !brg.is_per_k_src_scales
            && !brg.is_per_k_wei_scales) {
        reg_src_scales.restore();
        for (int ldb = 0; ldb < ldi->block2(); ldb++) {
            // Hard-coded assumption for a single src scale value being
            // supported, thus, offset is 0.
            auto scales_ptr
                    = EVEX_compress_addr(reg_src_scales, /* offset = */ 0);
            auto k_mask = ldi->is_tail(ldb) ? ld_tail_mask : ld_full_mask;
            const auto zmm_scale = zmm_scales(ldb);
            const auto zmm_scale_masked = zmm_scales(ldb) | k_mask | T_z;
            switch (brg.dt_src_scales) {
                case data_type::f32:
                    vbroadcastss(zmm_scale_masked, scales_ptr);
                    break;
                case data_type::bf16:
                    vpbroadcastw(zmm_scale, scales_ptr);
                    uni_vpslld(zmm_scale_masked, zmm_scale, 16);
                    break;
                case data_type::f16:
                    vpbroadcastw(zmm_scale, scales_ptr);
                    vcvtph2psx(
                            Xmm(zmm_scale.getIdx()), Xmm(zmm_scale.getIdx()));
                    vbroadcastss(zmm_scale_masked, Xmm(zmm_scale.getIdx()));
                    break;
                default: assert(!"unsupported src_scales data type");
            }
        }
    }

    if (scales_in_epilogue && brg.with_wei_scales && !brg.is_per_k_wei_scales) {
        reg_wei_scales.restore();
        for (int ldb = 0; ldb < ldi->block2(); ldb++) {
            auto scales_ptr = EVEX_compress_addr(
                    reg_wei_scales, scales_offset(ldi->pos(ldb)));
            auto k_mask = ldi->is_tail(ldb) ? ld_tail_mask : ld_full_mask;

            const auto zmm_scale = zmm_scales(ldb);
            const auto zmm_scale_masked = zmm_scales(ldb) | k_mask | T_z;

            if (brg.is_single_wei_scale) {
                if (brg.with_src_scales && !brg.is_per_k_src_scales) {
                    // Single value is not anticipated to be of any other type
                    // when both scales are defined.
                    assert(brg.dt_wei_scales == data_type::f32);
                    // Src scales are set, need to multiply by their value.
                    auto scales_bcast_ptr = EVEX_compress_addr(reg_wei_scales,
                            scales_offset(ldi->pos(ldb)),
                            /* bcast = */ true);
                    vmulps(zmm_scale_masked, zmm_scale, scales_bcast_ptr);
                } else {
                    switch (brg.dt_wei_scales) {
                        case data_type::f32:
                            vbroadcastss(zmm_scale, scales_ptr);
                            break;
                        case data_type::bf16:
                            vpbroadcastw(zmm_scale, scales_ptr);
                            uni_vpslld(zmm_scale, zmm_scale, 16);
                            break;
                        case data_type::f16:
                            vpbroadcastw(zmm_scale, scales_ptr);
                            vcvtph2psx(Xmm(zmm_scale.getIdx()),
                                    Xmm(zmm_scale.getIdx()));
                            vbroadcastss(zmm_scale, Xmm(zmm_scale.getIdx()));
                            break;
                        default: assert(!"unsupported wei_scales data type");
                    }
                }
                continue;
            }

            const auto zmm_wei_scale = zmm_tmp_1();
            const auto zmm_wei_scale_masked = zmm_wei_scale | k_mask | T_z;
            switch (brg.dt_wei_scales) {
                case data_type::f32:
                    uni_vmovups(zmm_wei_scale_masked, scales_ptr);
                    break;
                case data_type::bf16:
                    uni_vpmovzxwd(zmm_wei_scale_masked, scales_ptr);
                    uni_vpslld(zmm_wei_scale, zmm_wei_scale, 16);
                    break;
                case data_type::f16:
                    vcvtph2ps(zmm_wei_scale_masked, scales_ptr);
                    break;
                default: assert(!"unsupported wei_scales data type");
            }

            if (brg.with_src_scales && !brg.is_per_k_src_scales) {
                // Src scales are set, need to multiply by their value.
                vmulps(zmm_scale_masked, zmm_scale, zmm_wei_scale);
            } else {
                // No src scales, just load a vector of values.
                vmovups(zmm_scale, zmm_wei_scale);
            }
        }
    }
}

void jit_brgemm_amx_uker_t::uni_prefetch(
        const Address &addr, brgemm_kernel_prefetching_t pft, bool for_write) {
    if (for_write) {
        switch (pft) {
            case brgemm_prf0: prefetchw(addr); break;
            default: break;
        }
    } else {
        switch (pft) {
            case brgemm_prf0: prefetcht0(addr); break;
            case brgemm_prf1: prefetcht1(addr); break;
            case brgemm_prf2: prefetcht2(addr); break;
            case brgemm_prfNTA: prefetchnta(addr); break;
            default: break;
        }
    }
}

void jit_brgemm_amx_uker_t::prefetch_CD_range(brgemm_iteration_t &bi,
        brgemm_kernel_prefetching_t pft, int bd_start, int bd_finish, int bdb,
        int ldb) {
    const auto ldb_pos = bi.ldi->pos(ldb);
    for (int bd = bd_start; bd < bd_finish; bd++) {
        if (!is_out_bd(bi.bdi, bdb, bd)) continue;
        if (bi.apply_postops) {
            const auto d_offset = D_offset(bi, bdb, bd, ldb_pos);
            auto ptr_D = EVEX_compress_addr_safe(reg_D, d_offset, reg_tmp_gpr);
            uni_prefetch(ptr_D, pft, true);
        } else if (are_post_ops_applicable_) {
            //            TODO: split hints C and D hints
            //              Using prefetchw for the C matrix is generally harmful
            //              because the C matrix is frequently reused and remains in the cache.
            //              However, it is very necessary for the D matrix

            //            const auto c_offset = C_offset(bi, bdb, bd, ldb_pos);
            //            auto ptr_C = EVEX_compress_addr(reg_C, c_offset);
            //            uni_prefetch(ptr_C, pft, true);
        } else {
            const auto d_offset = D_offset(bi, bdb, bd, ldb_pos);
            auto ptr_D = EVEX_compress_addr(reg_D, d_offset);
            uni_prefetch(ptr_D, pft, true);
        }
    }
}

dim_t jit_brgemm_amx_uker_t::calc_ops_CD(
        brgemm_iteration_t &bi) const noexcept {
    const auto &tloop = imap_[bi.apply_postops];
    return static_cast<dim_t>(tloop.rdis.size()) * bi.ldi->block2()
            * bi.bdi->block2() * (brg.brgattr.var_bs ? 1 : brg.brgattr.max_bs);
}

void jit_brgemm_amx_uker_t::prefetch_CD(brgemm_iteration_t &bi,
        brgemm_iteration_t &pfo_bi, prf_t &prf, bool prefetch_all) {

    const auto calc_ops = calc_ops_CD(bi);
    const auto bdb_row = pfo_bi.bdi->block(0) * pfo_bi.ldi->block2();
    const auto tot_vecs = pfo_bi.bdi->length() * pfo_bi.ldi->block2();
    const auto pfo_vecs_per_store = (calc_ops) ? div_up(tot_vecs, calc_ops) : 0;

    const auto nvecs = prefetch_all
            ? tot_vecs
            : nstl::min(pfo_vecs_per_store, tot_vecs - prf.vec);

    const auto out_typesize
            = (are_post_ops_applicable_ && !prev_bi_.apply_postops)
            ? brg.typesize_C
            : brg.typesize_D;
    for (int iv = 0; iv < nvecs && prf.vec < tot_vecs; iv++) {
        const auto bdb = prf.vec / bdb_row;
        const auto vec_in_bdb_row = prf.vec - bdb * bdb_row;
        const auto ldb = vec_in_bdb_row / pfo_bi.bdi->block(bdb);
        const auto bd = vec_in_bdb_row % pfo_bi.bdi->block(bdb);
        // prefetch output cache lines only once
        if (pfo_bi.ldi->pos(ldb) % (4 / out_typesize) == 0) {
            prefetch_CD_range(pfo_bi, prf.pft, bd, bd + 1, bdb, ldb);
        }
        prf.vec++;
    }
}

void jit_brgemm_amx_uker_t::prefetch_A(brgemm_iteration_t &bi,
        brgemm_iteration_t &pfo_bi, prf_t &prf, bool prefetch_all) {

    const auto calc_ops = bi.ldi->block2() * bi.bdi->block2();
    const auto tot_vecs = pfo_bi.bdi->length();
    const auto pfo_vecs_per_store = (calc_ops) ? div_up(tot_vecs, calc_ops) : 0;

    const auto nvecs = prefetch_all
            ? tot_vecs
            : nstl::min(pfo_vecs_per_store, tot_vecs - prf.vec);

    for (int iv = 0; iv < nvecs && prf.vec < tot_vecs; iv++) {
        const auto bdb = prf.vec / pfo_bi.bdi->block(0);
        const auto bd = prf.vec % pfo_bi.bdi->block(0);

        //TODO: looks like we have to prefetch in each bs separately
        const auto ptr_A = EVEX_compress_addr(
                reg_A, A_offset(pfo_bi, bdb) + bd * LDA_size_);
        uni_prefetch(ptr_A, prf.pft, false);
        prf.vec++;
    }
}

void jit_brgemm_amx_uker_t::prefetch_B(brgemm_iteration_t &bi,
        brgemm_iteration_t &pfo_bi, prf_t &prf, bool prefetch_all) {

    const auto calc_ops = bi.ldi->block2() * bi.bdi->block2();
    const auto tot_vecs = pfo_bi.ldi->length();
    const auto pfo_vecs_per_store = (calc_ops) ? div_up(tot_vecs, calc_ops) : 0;

    const auto nvecs = prefetch_all
            ? tot_vecs
            : nstl::min(pfo_vecs_per_store, tot_vecs - prf.vec);

    // TODO: check these addressing for correctness
    for (int iv = 0; iv < nvecs && prf.vec < tot_vecs; iv++) {

        const auto ldb = prf.vec / pfo_bi.rdi->block(0);
        const auto rb = prf.vec % pfo_bi.rdi->block(0);
        //TODO: looks like we have to prefetch in each bs separately
        const auto ptr_B = EVEX_compress_addr(
                reg_B, B_offset(pfo_bi, ldb) + rb * LDB_size_);

        uni_prefetch(ptr_B, prf.pft, false);
        prf.vec++;
    }
}

void jit_brgemm_amx_uker_t::prefetching(
        brgemm_iteration_t &bi, bool prefetch_all) {
    // for var_bs we do prefetch on last iteration by bs only
    if (brg.brgattr.var_bs && !bi.last_bsi) return;
    brgemm_iteration_t pfo_bi;
    auto maybe_prefetch_C = [&](prf_t &prf) {
        if (prf.dist < 0) return;
        bool is_pfo_bi = false;
        brgemm_iteration_t pfo_bi;
        if (actual_ils(bi.apply_postops, bi.skip_accumulation)) {
            if (was_prev_bi_ && prf.dist == 0) {
                is_pfo_bi = true;
                pfo_bi = prev_bi_;
            } else if (prf.dist > 0) {
                is_pfo_bi = bi_shift_output(bi, prf.dist - 1, pfo_bi);
            }
        } else {
            is_pfo_bi = bi_shift_output(bi, prf.dist, pfo_bi);
        }
        if (is_pfo_bi) prefetch_CD(bi, pfo_bi, prf, prefetch_all);
    };

    auto maybe_prefetch_A = [&](prf_t &prf) {
        if (prf.dist < 0) return;
        if (bi_shift_A(bi, prf.dist, pfo_bi))
            prefetch_A(bi, pfo_bi, prf, prefetch_all);
    };

    auto maybe_prefetch_B = [&](prf_t &prf) {
        if (prf.dist < 0) return;
        if (bi_shift_B(bi, prf.dist, pfo_bi))
            prefetch_B(bi, pfo_bi, prf, prefetch_all);
    };

    maybe_prefetch_C(prf0C);
    maybe_prefetch_C(prf1C);

    maybe_prefetch_A(prf0A);
    maybe_prefetch_A(prf1A);
    maybe_prefetch_A(prf2A);
    maybe_prefetch_A(prfntaA);

    maybe_prefetch_B(prf0B);
    maybe_prefetch_B(prf1B);
    maybe_prefetch_B(prf2B);
    maybe_prefetch_B(prfntaB);
    if (!prefetch_all) maybe_sprinkle_prefetches();
}

void jit_brgemm_amx_uker_t::apply_comp_pad_to_vector(
        brgemm_iteration_t &bi, int bdb, int inp_bd, int ldb, const int idx) {
    const auto is_ld_tail = bi.ldi->is_tail(ldb);
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    auto zmm = Zmm(idx);
    auto zmm_masked = zmm | k_mask | T_z;
    const auto zmm_zp_a_val = zmm_tmp_1();

    reg_zp_a_values.restore();
    vpbroadcastd(zmm_zp_a_val, reg_zp_a_values.cvt32());
    vcvtdq2ps(zmm_zp_a_val, zmm_zp_a_val);
    reg_zp_comp_a.restore();
    const auto comp_pad_offset
            = zp_comp_pad_a_offset(bi, bdb, inp_bd, bi.ldi->pos(ldb));
    const auto zp_comp_pad_a_addr
            = EVEX_compress_addr(reg_zp_comp_pad_a, comp_pad_offset);
    cvt2ps(data_type::s32, zmm_zp_comp_a, zp_comp_pad_a_addr, true, false,
            k_mask);
    vmulps(zmm_zp_comp_a, zmm_zp_comp_a, zmm_zp_a_val);
    vaddps(zmm_masked, zmm, zmm_zp_comp_a);
}

void jit_brgemm_amx_uker_t::process_output_range(
        brgemm_iteration_t &bi, int bd_start, int bd_finish, int bdb, int ldb) {

    const auto k_mask = bi.ldi->is_tail(ldb) ? ld_tail_mask : ld_full_mask;

    // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
    // accumulated values are already converted to ps in apply_alpha_beta()
    const bool alpha_or_beta_applicable = brg.alpha != 1.0f || brg.beta != 0.f;
    const bool beta_uses_vadd
            = brg.beta == 1.f && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
    const bool dq2ps_required = brg.is_int8
            && IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);

    bool some_bd_mask = false;
    for (auto bd = bd_start; bd < bd_finish; bd++) {
        auto zmm = accm(bd);
        if (!is_out_bd(bi.bdi, bdb, bd)) continue;

        some_bd_mask = true;

        auto vreg_acc = accm(bd);
        // fill accumulator vector by data
        if (bi.skip_accumulation) {
            vpxord(vreg_acc, vreg_acc, vreg_acc);
        } else if (brg.is_ace()) {
            tilemovrow(vreg_acc, Tmm(get_C_tensor(bi, bdb, ldb)), bd);
        } else {
            vreg_acc = bi.ldi->is_tail(ldb) ? vreg_acc | ld_tail_mask | T_z
                                            : vreg_acc;
            const bool per_tile_wsp = use_ils_ || brg.interleave_tilestores_;
            const auto wsp_offset = per_tile_wsp
                    ? C_offset_wsp(prev_bi_, bdb, ldb, bd)
                    : C_offset_wsp(prev_bi_, 0, 0, bd);
            vmovups(vreg_acc, ptr[reg_buf + wsp_offset]);
        }

        // Per-(M,N) compensation: convert int32->float and subtract the
        // unscaled delta BEFORE per-K scale multiply.
        if (brg.with_per_mn_compensation && !bi.skip_accumulation) {
            vcvtdq2ps(zmm, zmm);

            reg_per_mn_comp.restore();
            const auto global_ldb = bi.ldi->pos(ldb);
            const auto delta_off = per_mn_comp_offset(bi, bdb, bd, global_ldb);
            const auto delta_ptr = EVEX_compress_addr_safe(
                    reg_per_mn_comp, delta_off, reg_long_offt);
            const auto zmm_delta = zmm_tmp_1();
            const Xbyak::Zmm zmm_delta_masked
                    = vmm_mask(zmm_delta, true, false, k_mask);
            vmovups(zmm_delta_masked, delta_ptr);
            vsubps(zmm, zmm, zmm_delta);
        }

        // For K-scales (wei and/or src), convert int32->float (if not
        // already done by per-MN compensation) and apply the current
        // K-group's scales BEFORE alpha_beta accumulation. Non-integer
        // accumulators (e.g. fp8/bf16 TMUL) are already f32.
        // MXFP8 is excluded: its per-K flags describe block scales that the
        // outer product has already applied, so the accumulator is final.
        if (brg.has_per_k_scales() && !is_mxfp8_compute()
                && !bi.skip_accumulation) {
            if (brg.is_int8 && !brg.with_per_mn_compensation)
                vcvtdq2ps(zmm, zmm);

            const Xbyak::Zmm scaled_zmm = vmm_mask(zmm, true, false, k_mask);
            // Apply K-wei_scales if present (pre-loaded per-ldb vector).
            if (brg.is_per_k_wei_scales) {
                vmulps(scaled_zmm, scaled_zmm, zmm_scales(ldb));
            }
            // Apply K-src_scales per-bd: each M-row has its own scalar.
            if (brg.is_per_k_src_scales) {
                reg_src_scales_per_k.restore();
                const auto out_bd = get_out_bd(bi.bdi, bdb, bd);
                const auto src_sc_offset = out_bd * brg.src_scale_m_stride;
                const auto src_sc_ptr = EVEX_compress_addr(
                        reg_src_scales_per_k, src_sc_offset);
                const auto zmm_src_sc = zmm_tmp_1();
                switch (brg.dt_src_scales) {
                    case data_type::f32:
                        vbroadcastss(zmm_src_sc, src_sc_ptr);
                        break;
                    case data_type::bf16:
                        vpbroadcastw(zmm_src_sc, src_sc_ptr);
                        vpslld(zmm_src_sc, zmm_src_sc, 16);
                        break;
                    case data_type::f16:
                        vpbroadcastw(zmm_src_sc, src_sc_ptr);
                        vcvtph2ps(zmm_src_sc, Xbyak::Ymm(zmm_src_sc.getIdx()));
                        break;
                    default: assert(!"unsupported src_scales data type");
                }
                vmulps(scaled_zmm, scaled_zmm, zmm_src_sc);
            }
        }

        // For ACE with LDC2 layout: use LDC_size_ per row (not LDC2_size_M_
        // which is the tile-block stride). For other cases, use C_offset as-is.
        const bool ace_ldc2 = brg.is_ace() && brg.brgattr.LDC2_M > 0;
        const auto c_offset = ace_ldc2
                ? C_offset_row(bi, bdb, bd, bi.ldi->pos(ldb))
                : C_offset(bi, bdb, bd, bi.ldi->pos(ldb));
        if (ace_ldc2) lea(reg_long_offt, ptr[reg_C + c_offset]);
        const auto ptr_C = ace_ldc2
                ? zword[reg_long_offt]
                : EVEX_compress_addr_safe(reg_C, c_offset, reg_tmp_gpr);

        if (need_to_apply_alpha_beta_ || bi.skip_accumulation) {
            apply_alpha_beta_to_vector(
                    zmm.getIdx(), ptr_C, bi.ldi->is_tail(ldb));
        }

        if (!bi.apply_postops) continue;

        // When K-scales or per-MN comp are used, dq2ps already applied above.
        if (dq2ps_required && !brg.has_per_k_scales()
                && !brg.with_per_mn_compensation)
            vcvtdq2ps(zmm, zmm);

        if (brg.req_comp_pads_with_bcast)
            apply_comp_pad_to_vector(bi, bdb, bd, ldb, zmm.getIdx());
    }

    if (!some_bd_mask) return;

    if (!bi.apply_postops) return;

    if (brg.zp_type_a != brgemm_broadcast_t::none
            && !brg.req_comp_pads_with_bcast) {
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            vaddps(zmm, zmm, zmm_zp_comp_a);
        }
    }

    if (brg.zp_type_b != brgemm_broadcast_t::none) {
        reg_zp_comp_b.restore();

        auto zmm_zp_comp_b = zmm_tmp_1();
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);

            const auto zp_comp_b_off
                    = zp_comp_b_offset(get_out_bd(bi.bdi, bdb, bd));
            vcvtdq2ps(zmm_zp_comp_b,
                    EVEX_compress_addr(reg_zp_comp_b, zp_comp_b_off, true));

            vaddps(zmm, zmm, zmm_zp_comp_b);
        }
    }

    // When K-scales (wei) are used, they were already applied
    // per K-block above. Only apply the remaining scales (non-K) in postops.
    // For per-K wei + common src, the common src scalar was folded into the
    // per-K wei load in `prepare_post_ops_registers` so it has already been
    // applied per K-block; skip the postop apply to avoid double-multiply.
    const bool src_scales_in_postops = brg.with_src_scales
            && !brg.is_per_k_src_scales && !brg.is_per_k_wei_scales;
    const bool wei_scales_in_postops
            = brg.with_wei_scales && !brg.is_per_k_wei_scales;
    const bool apply_scales_in_postops
            = (src_scales_in_postops || wei_scales_in_postops)
            // MXFP8 applies both scale tensors in the outer product.
            && !is_mxfp8_compute();
    if (apply_scales_in_postops) {
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            const Xbyak::Zmm scaled_zmm = vmm_mask(zmm, true, false, k_mask);
            vmulps(scaled_zmm, scaled_zmm, zmm_scales(ldb));
        }
    }

    if (brg.with_bias) {
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            vaddps(zmm, zmm, zmm_bias(ldb));
        }
    }

    if (postops_injector_) {
        apply_post_ops_to_range(bi, bd_start, bd_finish, bdb, ldb);
    }

    if (brg.with_dst_scales && !need_to_output_mxfp8()) {
        reg_dst_scales.restore();
        auto zmm_dst_scales = zmm_tmp_1();
        vbroadcastss(zmm_dst_scales, ptr[reg_dst_scales]);
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            vmulps(zmm, zmm, zmm_dst_scales);
        }
    }

    if (brg.zp_type_c != brgemm_broadcast_t::none) {
        for (auto bd = bd_start; bd < bd_finish; bd++) {
            if (!is_out_bd(bi.bdi, bdb, bd)) continue;

            auto zmm = accm(bd);
            vaddps(zmm, zmm, zmm_zp_c);
        }
    }
}

void jit_brgemm_amx_uker_t::store_vector_with_post_ops(
        int idx, const Address &addr, bool is_ld_tail) {
    auto zmm = Zmm(idx);

    maybe_saturation(zmm);

    auto ymm = Xbyak::Ymm(idx);
    auto xmm = Xbyak::Xmm(idx);
    auto k_mask = (!is_ld_tail) ? ld_full_mask : ld_tail_mask;
    const Xbyak::Zmm r_zmm = vmm_mask(zmm, true, true, k_mask);
    const Xbyak::Ymm r_ymm = vmm_mask(ymm, true, true, k_mask);
    const Xbyak::Xmm r_xmm = vmm_mask(xmm, true, true, k_mask);
    if (use_sat_cvt_) {
        assert(one_of(brg.dt_d, data_type::s8, data_type::u8));
        auto zmm_perm = zmm_ubound;
        vpermb(zmm, zmm_perm, zmm);
        vmovdqu8(addr, r_xmm);
        return;
    }

    switch (brg.dt_d) {
        case data_type::f32:
        case data_type::s32: vmovups(addr, r_zmm); break;
        case data_type::bf16:
            vcvtneps2bf16(ymm, zmm);
            vmovdqu16(addr, r_ymm);
            break;
        case data_type::f16:
            vcvtps2ph(ymm, zmm, _op_mxcsr);
            vmovdqu16(addr, r_ymm);
            break;
        case data_type::f8_e5m2:
            f8_e5m2_cvt_->vcvt_f32_to_f8(xmm, zmm);
            vmovdqu8(addr, r_xmm);
            break;
        case data_type::f8_e4m3:
            f8_e4m3_cvt_->vcvt_f32_to_f8(xmm, zmm);
            vmovdqu8(addr, r_xmm);
            break;
        case data_type::s8: vpmovsdb(addr, r_zmm); break;
        case data_type::u8: vpmovusdb(addr, r_zmm); break;
        default: assert(!"unknown dst_dt");
    }
}

void jit_brgemm_amx_uker_t::store_vector_without_post_ops(
        int idx, const Address &addr, bool is_ld_tail, data_type_t dst_dt) {
    auto zmm = Zmm(idx);

    maybe_saturation(zmm);

    // Downconvert f32 accumulator before the masked store for bf16/f16 dst.
    if (dst_dt == data_type::bf16 || dst_dt == data_type::f16) {
        auto k_mask = is_ld_tail ? ld_tail_mask : ld_full_mask;
        auto ymm = Xbyak::Ymm(idx);
        const Xbyak::Ymm r_ymm = vmm_mask(ymm, true, true, k_mask);
        if (dst_dt == data_type::bf16)
            vcvtneps2bf16(ymm, zmm);
        else
            vcvtps2ph(ymm, zmm, _op_mxcsr);
        vmovdqu16(addr, r_ymm);
        return;
    }

    if (is_ld_tail)
        vmovups(addr | ld_tail_mask | T_z, zmm);
    else
        vmovups(addr, zmm);
}

void jit_brgemm_amx_uker_t::store_vector(
        brgemm_iteration_t &bi, int bdb, int inp_bd, int ldb) {

    if (!is_out_bd(bi.bdi, bdb, inp_bd)) return;

    auto acc_idx = accm(inp_bd).getIdx();

    auto ldb_pos = bi.ldi->pos(ldb);
    auto is_ld_tail = bi.ldi->is_tail(ldb);
    const auto c_offset = C_offset(bi, bdb, inp_bd, ldb_pos);
    const auto d_offset = D_offset(bi, bdb, inp_bd, ldb_pos);

    // When post-ops have to be applied before quantization, the result is
    // staged in the tile workspace and quantized once the whole bd x ld block
    // is final.
    if (is_mxfp8_quantization_after_postops(bi)) {
        vmovups(ptr[reg_buf + C_offset_wsp(bi, bdb, ldb, inp_bd)],
                Zmm(acc_idx));
    } else if (bi.apply_postops) {
        auto ptr_D = EVEX_compress_addr_safe(reg_D, d_offset, reg_tmp_gpr);
        store_vector_with_post_ops(acc_idx, ptr_D, is_ld_tail);
    } else if (are_post_ops_applicable_) {
        // Intermediate C-buffer store; dtype may be narrower than the
        // accumulator (e.g. bf16/f16) under dst-dtype C-buffer mode.
        auto ptr_C = EVEX_compress_addr_safe(reg_C, c_offset, reg_tmp_gpr);
        store_vector_without_post_ops(acc_idx, ptr_C, is_ld_tail, brg.dt_c);
    } else if (brg.is_ace() && brg.brgattr.LDC2_M > 0) {
        const auto row_offset = C_offset_row(bi, bdb, inp_bd, ldb_pos);
        lea(reg_long_offt, ptr[reg_C + row_offset]);
        store_vector_without_post_ops(
                acc_idx, zword[reg_long_offt], is_ld_tail, brg.dt_c);
    } else {
        auto ptr_D = EVEX_compress_addr_safe(reg_D, d_offset, reg_tmp_gpr);
        store_vector_without_post_ops(acc_idx, ptr_D, is_ld_tail, brg.dt_d);
    }
}

void jit_brgemm_amx_uker_t::interleave_store(
        brgemm_iteration_t &bi, bool store_all) {

    if (store_all) { prev_bi_ = bi; }
    if (!was_prev_bi_) return;
    if (!actual_ils(prev_bi_.apply_postops, bi.skip_accumulation)) return;

    if (store_all) prefetching(prev_bi_, true);

    auto cur_bdb = ils_bdb_;
    auto cur_ldb = ils_ldb_;

    // if first block
    if (ils_vec_ == 0) {
        if (!prepare_post_ops_registers_once_) {
            prepare_post_ops_registers(prev_bi_);
        }
        prepare_post_ops_registers_ldb(prev_bi_, 0);
        ils_bd_start_ = 0;
        auto bd_finish = nstl::min(store_bd_step(), prev_bi_.bdi->block(0));
        process_output_range(prev_bi_, 0, bd_finish, cur_bdb, cur_ldb);
    }

    const auto calc_ops = calc_ops_CD(bi);
    // we use maximum estimation (prev_bi_.bdi.block2() * prev_bi_.bdi.block())
    // to calculate ils_store_ops to avoid error when we didn't store all
    // vectors from tile buffer but it is already overwritten in a new iteration
    const auto ils_store_ops = prev_bi_.ldi->block2() * prev_bi_.bdi->block2()
            * prev_bi_.bdi->block(0);
    const auto ils_vecs_per_store
            = (calc_ops) ? div_up(ils_store_ops, calc_ops) : 0;

    // last bd_block may be bd_tail
    const auto bdb_row = prev_bi_.bdi->block(0) * prev_bi_.ldi->block2();
    const auto total_vectors = prev_bi_.bdi->length() * prev_bi_.ldi->block2();
    const auto nvecs = store_all ? total_vectors : ils_vecs_per_store;
    for (int vec = 0; vec < nvecs && ils_vec_ < total_vectors; vec++) {
        const auto bdb = ils_vec_ / bdb_row;
        const auto vec_in_bdb_row = ils_vec_ - bdb * bdb_row;
        const auto ldb = vec_in_bdb_row / prev_bi_.bdi->block(bdb);
        const auto bd = vec_in_bdb_row % prev_bi_.bdi->block(bdb);

        if (ldb != cur_ldb) prepare_post_ops_registers_ldb(prev_bi_, ldb);

        if (bdb != cur_bdb || ldb != cur_ldb
                || rnd_dn(bd, store_bd_step()) != ils_bd_start_) {
            ils_bd_start_ = rnd_dn(bd, store_bd_step());
            auto bd_finish = nstl::min(
                    ils_bd_start_ + store_bd_step(), prev_bi_.bdi->block(bdb));
            process_output_range(prev_bi_, ils_bd_start_, bd_finish, bdb, ldb);
        }

        store_vector(prev_bi_, bdb, bd, ldb);
        cur_bdb = bdb;
        cur_ldb = ldb;
        ils_vec_++;
    }
    ils_ldb_ = cur_ldb;
    ils_bdb_ = cur_bdb;
}

void jit_brgemm_amx_uker_t::store_accumulators(brgemm_iteration_t &bi) {

    const auto store_by_vectors = get_store_by_vectors(bi.apply_postops);

    if (store_by_vectors) {
        if (!brg.interleave_tilestores_)
            mov(reg_stride_ld_block, ld_block_C_size_);
    } else
        mov(reg_stride_ld_block, LDC_size_);

    prev_bi_ = bi;
    was_prev_bi_ = true;

    ils_vec_ = 0;
    ils_bdb_ = 0;
    ils_ldb_ = 0;

    prf0C.reset();
    prf1C.reset();

    // Nothing has to be applied to the accumulators before quantization, so
    // they are quantized directly from the C tiles and the regular store path
    // is not needed at all.
    if (need_to_output_mxfp8() && bi.apply_postops
            && !is_mxfp8_quantization_after_postops(bi)) {
        quantize_to_mxfp8(bi, /*tile_src=*/true, 0, mxfp8_all_nodes);
        return;
    }

    store_accumulators_default(bi);

    // Post-ops were applied and their result staged in the tile workspace by
    // store_vector(); quantize it now that the whole block is final.
    if (is_mxfp8_quantization_after_postops(bi))
        quantize_to_mxfp8(bi, /*tile_src=*/false, 0, mxfp8_all_nodes);
}

void jit_brgemm_amx_uker_t::store_accumulators_default(brgemm_iteration_t &bi) {

    const auto store_by_vectors = get_store_by_vectors(bi.apply_postops);
    const bool real_ils = actual_ils(bi.apply_postops, bi.skip_accumulation);
    if (store_by_vectors && !real_ils && !prepare_post_ops_registers_once_)
        prepare_post_ops_registers(bi);

    for_(int bdb = 0; bdb < bi.bdi->block2(); bdb++)
    for (int ldb = 0; ldb < bi.ldi->block2(); ldb++) {

        // ACE palette does not support tilestored, so always use
        // vector store path (tilemovrow + vmovups) for ACE.
        const bool tile_store_by_vectors = store_by_vectors || brg.is_ace();
        if (tile_store_by_vectors) {
            if (!brg.interleave_tilestores_ && !bi.skip_accumulation
                    && !brg.is_ace()) {
                const dim_t wsp_offset = use_ils_
                        ? C_offset_wsp(bi, bdb, ldb, 0)
                        : C_offset_wsp(bi, 0, 0, 0);
                tilestored(ptr[reg_buf + reg_stride_ld_block + wsp_offset],
                        Tmm(get_C_tensor(bi, bdb, ldb)));
            }
            if (real_ils) continue;

            prepare_post_ops_registers_ldb(bi, ldb);

            for (int bd_step = 0; bd_step < bi.bdi->block(bdb);
                    bd_step += store_bd_step()) {
                auto bd_finish = nstl::min(
                        bd_step + store_bd_step(), bi.bdi->block(bdb));
                process_output_range(bi, bd_step, bd_finish, bdb, ldb);

                for (auto bd = bd_step; bd < bd_finish; bd++)
                    store_vector(bi, bdb, bd, ldb);
            }
        } else if (!brg.interleave_tilestores_) {
            const auto c_offset = C_offset(bi, bdb, 0, bi.ldi->pos(ldb));
            tilestored(ptr[reg_C + reg_stride_ld_block + c_offset],
                    Tmm(get_C_tensor(bi, bdb, ldb)));
        }
    }
}
bool jit_brgemm_amx_uker_t::need_to_output_mxfp8() const {
    return brg.quantize_dst_to_mxfp8;
}

// Quantization has to happen after the post-ops whenever anything modifies the
// accumulators between the tiles and the dst, since the scale of a group can
// only be computed once every contribution to that group is final.
bool jit_brgemm_amx_uker_t::is_mxfp8_quantization_after_postops(
        const brgemm_iteration_t &bi) const {
    if (!need_to_output_mxfp8() || !bi.apply_postops) return false;

    const bool with_zp_ab = brg.zp_type_a != brgemm_broadcast_t::none
            || brg.zp_type_b != brgemm_broadcast_t::none
            || brg.req_s8s8_compensation;
    const bool with_epilogue_scales
            = (brg.with_src_scales || brg.with_wei_scales)
            && !is_mxfp8_compute();
    return need_to_apply_alpha_beta_ || bi.skip_accumulation
            || brg.req_comp_pads_with_bcast || brg.with_bias
            || postops_injector_ != nullptr || with_zp_ab
            || brg.with_per_mn_compensation || with_epilogue_scales;
}

void jit_brgemm_amx_uker_t::set_A_B_matrices(dim_t bs) {
    if (one_of(brg.type, brgemm_static_offs)) return;
    assert(one_of(brg.type, brgemm_addr, brgemm_offs));
    if (brg.brgattr.max_bs == 1) return;
    const auto batch_offset = (dim_t)bs * sizeof(brgemm_batch_element_t);
    if (brg.type == brgemm_addr) {
        if (brg.layout == brgemm_row_major) {
            mov(reg_A,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(ptr.A)));
            mov(reg_B,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(ptr.B)));
        } else {
            mov(reg_A,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(ptr.B)));
            mov(reg_B,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(ptr.A)));
        }
    } else if (brg.type == brgemm_offs) {
        reg_A.restore();
        reg_B.restore();
        if (brg.layout == brgemm_row_major) {
            add(reg_A,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(offset.A)));
            add(reg_B,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(offset.B)));
        } else {
            add(reg_A,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(offset.B)));
            add(reg_B,
                    EVEX_compress_addr(reg_addr_batch,
                            batch_offset + GET_OFF_BATCH_ELEMENT(offset.A)));
        }
    }
}

void jit_brgemm_amx_uker_t::set_A_B_matrices() {
    if (one_of(brg.type, brgemm_static_offs)) return;
    assert(one_of(brg.type, brgemm_addr, brgemm_offs));
    assert(brg.brgattr.var_bs);
    if (brg.brgattr.max_bs == 1) return;

    if (brg.type == brgemm_addr) {
        reg_aux1_batch.restore();
        if (brg.layout == brgemm_row_major) {
            mov(reg_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
            mov(reg_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
        } else {
            mov(reg_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.B)]);
            mov(reg_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(ptr.A)]);
        }
    } else if (brg.type == brgemm_offs) {
        reg_aux1_batch.restore();
        // The offsets are relative to the original matrix pointers, so the
        // base pointers have to be reloaded before every batch element.
        reg_A.restore();
        reg_B.restore();
        if (brg.layout == brgemm_row_major) {
            add(reg_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(offset.A)]);
            add(reg_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(offset.B)]);
        } else {
            add(reg_A, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(offset.B)]);
            add(reg_B, ptr[reg_aux1_batch + GET_OFF_BATCH_ELEMENT(offset.A)]);
        }
    }
}

void jit_brgemm_amx_uker_t::maybe_sprinkle_prefetches() {
    auto jit_prefetches
            = [&](prf_sprinkled_t &prf_sprinkled, Xbyak::Reg64 base) {
        // Calculate the number of cache lines to jit
        float total_cache_lines_to_prefetch
                = (float)prf_sprinkled.prefetch_offsets.size();
        float cache_lines_per_amx_op = total_cache_lines_to_prefetch
                / static_cast<float>(num_amx_ops);
        int num_prefetches_to_jit
                = (int)(static_cast<float>(current_num_amx_ops + 1)
                          * cache_lines_per_amx_op)
                - (int)(static_cast<float>(current_num_amx_ops)
                        * cache_lines_per_amx_op);

        // Jit the prefetches
        for (size_t i = prf_sprinkled.current_prefetch_idx;
                i < num_prefetches_to_jit + prf_sprinkled.current_prefetch_idx;
                i++) {
            const auto ptr = EVEX_compress_addr(
                    base, prf_sprinkled.prefetch_offsets[i]);
            uni_prefetch(ptr, brgemm_prf1, false);
        }

        // Update idx of last prefetched line
        prf_sprinkled.current_prefetch_idx += num_prefetches_to_jit;
    };

    if (brg.prfA.sprinkled) jit_prefetches(prf_sprinkled_a, reg_A);
    if (brg.prfB.sprinkled) jit_prefetches(prf_sprinkled_b, reg_B);

    current_num_amx_ops++;
}
void jit_brgemm_amx_uker_t::maybe_tileloadd_nt(
        brgemm_iteration_t &bi, matrix_kind_t mk, int xdb, dim_t offset) {

    const bool is_A = mk == matrix_kind_t::matrix_A;
    bool load_nt = is_A ? brg.load_nt_A : brg.load_nt_B;

    auto t1 = Tmm(is_A ? brg.get_A_tensor(xdb, bi.bdi->is_tail(xdb))
                       : brg.get_B_tensor(xdb, bi.ldi->is_tail(xdb)));
    auto &reg_base = is_A ? reg_A : reg_B;
    auto reg_stride = is_A ? reg_stride_lda : reg_stride_ldb;

    const bool mem_advice_A = utils::one_of(brg.brgattr.mem_advice,
            brgemm_hint_mem_advice_A, brgemm_hint_mem_advice_A_B);
    const bool mem_advice_B = utils::one_of(brg.brgattr.mem_advice,
            brgemm_hint_mem_advice_B, brgemm_hint_mem_advice_A_B);
    bool has_mem_advice = is_A ? mem_advice_A : mem_advice_B;

    if (brg.is_input_convert()) {
        // try_load_nt is not supported in maybe_pre_process_data as there is
        // no guarantee that the data is cache line aligned.
        maybe_pre_process_data(bi, t1, reg_base, offset, reg_stride, mk);
        return;
    }

    if (maybe_pre_process_k_tail(
                bi, t1, reg_base, offset, reg_stride, mk, has_mem_advice))
        return;

    if (load_nt) {
        if (has_mem_advice)
            tileloaddrst1(t1, ptr[reg_base + offset + reg_stride]);
        else
            tileloaddt1(t1, ptr[reg_base + offset + reg_stride]);
    } else {
        if (has_mem_advice)
            tileloaddrs(t1, ptr[reg_base + offset + reg_stride]);
        else
            tileloadd(t1, ptr[reg_base + offset + reg_stride]);
    }
}

void jit_brgemm_amx_uker_t::maybe_fused_copy_A_nt_load(
        brgemm_iteration_t &bi, int bdb) {
    auto t1 = Tmm(brg.get_A_tensor(bdb, bi.bdi->is_tail(bdb)));

    auto load_a_tile_from_wsp = [&]() {
        if (brg.load_nt_A) {
            tileloaddt1(
                    t1, ptr[reg_buf + A_offset_wsp(bi, bdb) + reg_stride_lda]);
        } else {
            tileloadd(
                    t1, ptr[reg_buf + A_offset_wsp(bi, bdb) + reg_stride_lda]);
        }
    };

    if (bi.ldi->pos(0) == 0) {
        mov(reg_stride_lda, lda());
        const bool has_mem_advice = utils::one_of(brg.brgattr.mem_advice,
                brgemm_hint_mem_advice_A, brgemm_hint_mem_advice_A_B);

        auto src_offset = A_offset(bi, bdb);
        // fused copy A: load from A orig and store it aligned in the wsp buffer
        if (bi.rdi->is_tail(0)) {
            pre_process_k_tail_fused_copy_a(bi, bdb, t1, reg_A, src_offset,
                    A_offset_wsp(bi, bdb), has_mem_advice);
            mov(reg_stride_lda, 64);
            load_a_tile_from_wsp();
        } else {
            if (has_mem_advice)
                tileloaddrst1(t1, ptr[reg_A + src_offset + reg_stride_lda]);
            else
                tileloaddt1(t1, ptr[reg_A + src_offset + reg_stride_lda]);
            mov(reg_stride_lda, 64);
            tilestored(
                    ptr[reg_buf + A_offset_wsp(bi, bdb) + reg_stride_lda], t1);
        }

    } else {
        load_a_tile_from_wsp();
    };
}
void jit_brgemm_amx_uker_t::maybe_tilestore(brgemm_iteration_t &bi, int bdb_idx,
        int ldb_idx, bool do_pre_tilestore, bool do_post_tilestore) {
    if (bi.skip_accumulation) return;
    auto current_tensor_idx = get_C_tensor(bi, bdb_idx, ldb_idx);

    if (!brg.interleave_tilestores_) return;
    const auto current_tensor_number
            = current_tensor_idx - get_C_tensor(bi, 0, 0);
    const auto store_tensor_shift
            = do_pre_tilestore ? (bi.bdi->block2() == 1 ? 2 : 1) : 0;
    const auto store_tensor_idx = current_tensor_idx + store_tensor_shift;
    const auto store_tensor_number = current_tensor_number + store_tensor_shift;

    const auto &store_bi = do_pre_tilestore ? prev_bi_ : bi;
    const int max_store_tensor_number
            = store_bi.bdi->block2() * store_bi.ldi->block2();
    bool perform_store
            = (do_pre_tilestore
                      && (store_tensor_number >= 2
                              && store_tensor_number < max_store_tensor_number))
            || (do_post_tilestore && (store_tensor_number < 2));

    if (!perform_store) return;
    if (do_pre_tilestore) {
        bdb_idx = store_tensor_idx / bi.ldi->block2();
        ldb_idx = store_tensor_idx % bi.ldi->block2();
    }
    const bool store_by_vectors = get_store_by_vectors(bi.apply_postops);
    Tmm acc = Tmm(store_tensor_idx);
    if (store_by_vectors) {
        const bool per_tile_wsp = use_ils_ || brg.interleave_tilestores_;
        const auto wsp_offset = per_tile_wsp
                ? C_offset_wsp(bi, bdb_idx, ldb_idx, 0)
                : C_offset_wsp(bi, 0, 0, 0);
        tilestored(ptr[reg_buf + reg_stride_ld_block + wsp_offset], acc);
    } else {
        const auto store_ldb_ind
                = do_pre_tilestore ? prev_bi_.ldi->pos(0) : bi.ldi->pos(0);
        const auto c_offset
                = C_offset(store_bi, bdb_idx, 0, store_ldb_ind + ldb_idx);
        tilestored(ptr[reg_C + reg_stride_ld_block + c_offset], acc);
    }
    tilezero(acc);
}

void jit_brgemm_amx_uker_t::tdpbxxd(brgemm_iteration_t &bi, int bdb_idx,
        int ldb_idx, bool do_pre_tilestore, bool do_post_tilestore) {
    prefetching(bi, false);
    maybe_tilestore(bi, bdb_idx, ldb_idx, do_pre_tilestore, false);

    const Tmm &x1 = Tmm(get_C_tensor(bi, bdb_idx, ldb_idx));
    const Tmm &x2 = Tmm(brg.get_A_tensor(bdb_idx, bi.bdi->is_tail(bdb_idx)));
    const Tmm &x3 = Tmm(brg.get_B_tensor(ldb_idx, bi.ldi->is_tail(ldb_idx)));

    using namespace data_type;
    if (brg.is_bf32 || (brg.dt_a == bf16 && brg.dt_b == bf16)) {
        tdpbf16ps(x1, x2, x3);
    } else if (brg.dt_a == f16 && brg.dt_b == f16) {
        tdpfp16ps(x1, x2, x3);
    } else if (brg.is_fp8 && brg.is_fp8_via_convert()) {
        tdpfp16ps(x1, x2, x3);
    } else if (brg.dt_a == f8_e5m2 && brg.dt_b == f8_e5m2) {
        tdpbf8ps(x1, x2, x3);
    } else if (brg.dt_a == f8_e5m2 && brg.dt_b == f8_e4m3) {
        tdpbhf8ps(x1, x2, x3);
    } else if (brg.dt_a == f8_e4m3 && brg.dt_b == f8_e4m3) {
        tdphf8ps(x1, x2, x3);
    } else if (brg.dt_a == f8_e4m3 && brg.dt_b == f8_e5m2) {
        tdphbf8ps(x1, x2, x3);
    } else if (brg.dt_a == u8 && brg.dt_b == u8) {
        tdpbuud(x1, x2, x3);
    } else if (brg.dt_a == u8 && brg.dt_b == s8) {
        tdpbusd(x1, x2, x3);
    } else if (brg.dt_a == s8 && brg.dt_b == u8) {
        tdpbsud(x1, x2, x3);
    } else if (brg.dt_a == s8 && brg.dt_b == s8) {
        tdpbssd(x1, x2, x3);
    } else {
        assert(!"unsupported combination");
    }
    interleave_store(bi, false);
    maybe_tilestore(bi, bdb_idx, ldb_idx, false, do_post_tilestore);
}

// This method up-converts the data from bf8 to f16 and saves at reg_buf.
// Generally used by matrix_A, where no vnni transformation of data is needed.
void jit_brgemm_amx_uker_t::fp8_to_f16_upconvert(brgemm_iteration_t &bi,
        int num_rows, int tile_num_col_bytes, reg64_t reg_data, dim_t offset,
        reg64_t reg_data_stride, reg64_t reg_buf, data_type_t dt) {
    const auto rd_block = bi.rdi->block(0);
    const dim_t max_num_cols = nstl::min<dim_t>(
            tile_num_col_bytes / sizeof(float16_t), rd_block);
    const dim_t col_tail = max_num_cols % 32;
    auto zmm_1 = zmm_tmp_1();
    auto zmm_1_masked = col_tail ? zmm_1 | fp_col_mask | T_z : zmm_1;

    assert(max_num_cols > 0);

    if (col_tail) {
        const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
        mov(reg_tmp_gpr, tail_mask);
        kmovq(fp_col_mask, reg_tmp_gpr);
    }

    // Note: using the same register used in col_tail, so order is important
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_data + offset]);

    for (int r = 0; r < num_rows; ++r) {
        if (dt == data_type::f8_e5m2)
            f8_e5m2_cvt_->vcvt_f8_to_f16(zmm_1_masked, ptr[reg_data_aux]);
        else if (dt == data_type::f8_e4m3)
            f8_e4m3_cvt_->vcvt_f8_to_f16(zmm_1_masked, ptr[reg_data_aux]);
        else
            assert(!"unsupported data type");

        vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_1);
        add(reg_data_aux, reg_data_stride);
    }
}

// This method down-converts the data from f32 to bf16 and saves at reg_buf.
// Generally used by matrix_A, where no vnni transformation of data is needed.
void jit_brgemm_amx_uker_t::bf32_downconvert(brgemm_iteration_t &bi,
        int num_rows, int tile_num_col_bytes, reg64_t reg_data, dim_t offset,
        reg64_t reg_data_stride, reg64_t reg_buf) {
    const auto rd_block = bi.rdi->block(0);
    const int max_num_cols
            = nstl::min<int>(tile_num_col_bytes / sizeof(bfloat16_t), rd_block);
    const auto col_tail = max_num_cols % simd_w;
    auto zmm_1 = zmm_tmp_1();
    auto zmm_2 = zmm_tmp_2();
    auto zmm_2_masked = col_tail ? zmm_2 | fp_col_mask | T_z : zmm_2;

    assert(max_num_cols > 0);

    if (col_tail) {
        const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
        mov(reg_tmp_gpr, tail_mask);
        kmovq(fp_col_mask, reg_tmp_gpr);
    }

    // Note: using the same register used in col_tail, so order is important
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_data + offset]);

    for (int r = 0; r < num_rows; ++r) {
        if (max_num_cols > 16) {
            vmovups(zmm_1, ptr[reg_data_aux]);
            vmovups(zmm_2_masked, ptr[reg_data_aux + zmm_width_in_bytes]);
            vcvtne2ps2bf16(zmm_1, zmm_2, zmm_1);
            // we assume enough padding space is available.
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_1);
        } else {
            auto ymm_1 = Ymm(zmm_1.getIdx());
            auto ymm_1_masked
                    = max_num_cols == 16 ? ymm_1 : ymm_1 | fp_col_mask | T_z;
            vcvtneps2bf16(ymm_1_masked, ptr[reg_data_aux]);
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], ymm_1);
        }
        add(reg_data_aux, reg_data_stride);
    }
}

// This method up-converts and transforms the data from fp8_vnni to f16_vnni
// format. Generally used by matrix_B.
void jit_brgemm_amx_uker_t::fp8_to_f16_upconvert_to_vnni(brgemm_iteration_t &bi,
        int num_rows, int tile_num_col_bytes, reg64_t reg_data, dim_t offset,
        reg64_t reg_data_stride, reg64_t reg_buf, data_type_t dt) {
    const int num_cols_ele = tile_num_col_bytes / 2; // 32 for full tile
    const int num_N = num_cols_ele / 2; // 16 for full tile
    const auto zmm_2 = zmm_tmp_2();

    assert(num_N > 0 && "bad tile parameters");
    MAYBE_UNUSED(num_N);

    const auto rd_block = bi.rdi->block(0);
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_data + offset]);

    const int vnni_granularity = 2;
    const dim_t r_end = utils::div_up(rd_block, vnni_granularity);
    assert(r_end <= num_rows && "bad tile parameters");

    if (dt == data_type::f8_e5m2)
        f8_e5m2_cvt_->vcvt_f8_to_f16_vnni_block(static_cast<int>(r_end),
                reg_data_aux, reg_data_stride, reg_buf);
    else if (dt == data_type::f8_e4m3)
        f8_e4m3_cvt_->vcvt_f8_to_f16_vnni_block(static_cast<int>(r_end),
                reg_data_aux, reg_data_stride, reg_buf);
    else
        assert(!"unsupported data type");

    // zero rest of the tile data
    if (r_end < num_rows) {
        vpxord(zmm_2, zmm_2, zmm_2);
        for (dim_t r = r_end; r < num_rows; ++r)
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_2);
    }
}

// This method down-converts and transforms the data from f32 to bf16_vnni
// format. Generally used by matrix_B.
void jit_brgemm_amx_uker_t::bf32_downconvert_to_vnni(brgemm_iteration_t &bi,
        int num_rows, int tile_num_col_bytes, reg64_t reg_data, dim_t offset,
        reg64_t reg_data_stride, reg64_t reg_buf) {
    const auto num_cols_ele = tile_num_col_bytes / sizeof(bfloat16_t);
    const auto num_N = num_cols_ele / sizeof(bfloat16_t);
    const auto col_tail = num_N % simd_w;
    const auto zmm_1 = zmm_tmp_1();
    const auto zmm_2 = zmm_tmp_2();

    assert(num_N > 0);

    auto load = [&](Zmm zmm, Address addr) {
        if (col_tail)
            vmovups(zmm | fp_col_mask | T_z, addr);
        else
            vmovups(zmm, addr);
    };

    if (col_tail) {
        const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
        mov(reg_tmp_gpr, tail_mask);
        kmovq(fp_col_mask, reg_tmp_gpr);
    }

    // Note: using the same register used in col_tail, so order is important
    const auto reg_data_aux = reg_tmp_gpr;
    lea(reg_data_aux, ptr[reg_data + offset]);

    const auto rd_block = bi.rdi->block(0);
    // data_type_vnni_granularity() returns size_t but is always a tiny,
    // fixed ISA constant (1, 2 or 4), so narrow it at this boundary.
    const int vnni_granularity = static_cast<int>(
            data_type_vnni_granularity(data_type_t::dnnl_bf16));
    const int r_end
            = nstl::min(utils::div_up(rd_block, vnni_granularity), num_rows);

    for (dim_t r = 0; r < r_end; ++r) {
        load(zmm_1, ptr[reg_data_aux]);

        if (r * vnni_granularity + 1 >= rd_block) {
            vpxord(zmm_2, zmm_2, zmm_2);
        } else {
            load(zmm_2, ptr[reg_data_aux + reg_data_stride]);
        }

        vcvtne2ps2bf16(zmm_1, zmm_2, zmm_1);
        vpermw(zmm_1, zmm_bf32_permute, zmm_1);
        vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_1);
        lea(reg_data_aux,
                ptr[reg_data_aux + reg_data_stride * vnni_granularity]);
    }

    // zero rest of the tile data
    if (r_end < num_rows) {
        vpxord(zmm_2, zmm_2, zmm_2);
        for (dim_t r = r_end; r < num_rows; ++r)
            vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_2);
    }
}

void jit_brgemm_amx_uker_t::maybe_pre_process_data(brgemm_iteration_t &bi,
        const Tmm &t1, reg64_t reg_base, dim_t offset, reg64_t reg_stride,
        matrix_kind_t mk) {

    const auto &tloop = imap_[bi.apply_postops];
    auto should_save_transform = [&](matrix_kind_t mk) {
        if (mk == matrix_A)
            return brg.save_transform_A();
        else
            return brg.save_transform_B();
    };

    const auto dt = mk == matrix_A ? brg.dt_a : brg.dt_b;

    const bool is_A = mk == matrix_A;
    auto &transform_buf = is_A ? transform_buf_map_A_ : transform_buf_map_B_;

    const auto convert_base
            = brg.get_wsp_base_offset(brgemm_desc_t::wsp_convert);
    const auto max_bdb2 = tloop.bdis[0].block2();
    const auto max_rdb = static_cast<dim_t>(tloop.rdis.size());
    dim_t matrix_a_tiles;
    if (should_save_transform(matrix_A))
        matrix_a_tiles = brg.brgattr.max_bs * max_bdb2 * max_rdb;
    else if (should_save_transform(matrix_B))
        matrix_a_tiles = 1;
    else
        matrix_a_tiles = 0;

    const auto matrix_a_offset = static_cast<dim_t>(convert_base);
    const auto matrix_b_offset
            = matrix_a_offset + brgemm_desc_t::tilesize * matrix_a_tiles;
    const auto matrix_offset = is_A ? matrix_a_offset : matrix_b_offset;
    const std::string key
            = std::to_string(bi.bsi->pos) + "_" + std::to_string(offset);

    if (transform_buf.find(key) != transform_buf.end()) {
        auto buf_idx = transform_buf[key];
        auto offt = matrix_offset + buf_idx * brgemm_desc_t::tilesize;
        tileloadd(t1, ptr[reg_buf + reg_converted_stride + offt]);
        return;
    }

    dim_t buf_offt = matrix_offset;
    // save offset of the transformation if required.
    if (should_save_transform(mk)) {
        auto buf_idx = transform_buf.size();
        buf_offt = matrix_offset + buf_idx * brgemm_desc_t::tilesize;
        transform_buf[key] = buf_idx;
    }

    if (buf_offt) add(reg_buf, buf_offt);
    mov(reg_converted_stride, zmm_width_in_bytes);

    const int max_tiles = amx::get_max_palette_size();
    JIT_ASSERT(t1.getIdx() >= 0 && t1.getIdx() < max_tiles);
    const auto num_rows = palette_.rows[t1.getIdx()];
    const auto num_col_bytes = palette_.cols[t1.getIdx()];
    if (is_A) {
        if (brg.is_bf32)
            bf32_downconvert(bi, num_rows, num_col_bytes, reg_base, offset,
                    reg_stride, reg_buf);
        else
            fp8_to_f16_upconvert(bi, num_rows, num_col_bytes, reg_base, offset,
                    reg_stride, reg_buf, dt);
    } else {
        if (brg.is_bf32)
            bf32_downconvert_to_vnni(bi, num_rows, num_col_bytes, reg_base,
                    offset, reg_stride, reg_buf);
        else
            fp8_to_f16_upconvert_to_vnni(bi, num_rows, num_col_bytes, reg_base,
                    offset, reg_stride, reg_buf, dt);
    }

    // load into tmm from the transformed data.
    tileloadd(t1, ptr[reg_buf + reg_converted_stride]);

    // reset buf pointer.
    if (buf_offt) sub(reg_buf, buf_offt);
}

void jit_brgemm_amx_uker_t::pre_process_k_tail_fused_copy_a(
        brgemm_iteration_t &bi, int bdb, const Tmm &t1, reg64_t reg_base,
        dim_t offset_src, dim_t offset_dst, bool mem_advice_A) {
    if (offset_dst) add(reg_buf, offset_dst);
    copy_k_tail_to_wsp(t1, reg_base, offset_src, reg_stride_lda, mem_advice_A);
    if (offset_dst) sub(reg_buf, offset_dst);
}

bool jit_brgemm_amx_uker_t::maybe_pre_process_k_tail(brgemm_iteration_t &bi,
        const Tmm &t1, reg64_t reg_base, dim_t offset, reg64_t reg_stride,
        matrix_kind_t mk, bool use_memadvice) {
    const auto &tloop = imap_[bi.apply_postops];

    // With the extendable_k strategy matrix B is zero-padded along the K
    // dimension, so brgemm reads the full K-tail tile from matrix A and relies
    // on B's zeros to cancel the columns beyond the valid K range. Reading past
    // the valid K columns of A, however, pulls in bytes that belong to adjacent
    // rows of A. If those bytes contain NaN/Inf the identity 'NaN * 0 == 0' no
    // longer holds and the garbage leaks into otherwise clean output rows. To
    // avoid this the K-tail of A is copied into a zero-padded scratch buffer
    // for every M tile (not only the last one) before it is loaded into the
    // AMX tile.
    const bool need_k_tail_processing = mk == matrix_A && brg.amx_wary_k_tail()
            && brg.rdb_tail != 0 && tloop.is_last_rdi(bi.rdi);

    if (!need_k_tail_processing) return false;

    const auto transform_offset
            = brg.get_wsp_base_offset(brgemm_desc_t::wsp_wary_k_tail);

    if (transform_offset) add(reg_buf, transform_offset);
    mov(reg_converted_stride, zmm_width_in_bytes);

    copy_k_tail_to_wsp(t1, reg_base, offset, reg_stride, use_memadvice);
    // load into tmm from the transformed data.
    tileloadd(t1, ptr[reg_buf + reg_converted_stride]);

    // reset buf pointer
    if (transform_offset) sub(reg_buf, transform_offset);
    return true;
}
void jit_brgemm_amx_uker_t::copy_k_tail_to_wsp(const Tmm &t1,
        jit_brgemm_amx_uker_t::reg64_t &reg_base, dim_t src_offset,
        jit_brgemm_amx_uker_t::reg64_t &reg_src_stride, bool use_memadvice) {
    const int max_tiles = amx::get_max_palette_size();
    JIT_ASSERT(t1.getIdx() >= 0 && t1.getIdx() < max_tiles);
    const auto num_rows = palette_.rows[t1.getIdx()];
    const auto num_col_bytes = palette_.cols[t1.getIdx()];

    const auto max_num_cols
            = nstl::min<dim_t>(num_col_bytes / brg.typesize_A, brg.rdb_tail);
    const size_t col_tail
            = max_num_cols % (zmm_width_in_bytes / brg.typesize_A);
    if (col_tail) {
        const auto tail_mask = (static_cast<size_t>(1) << col_tail) - 1;
        mov(reg_tmp_gpr, tail_mask);
        kmovq(rd_tail_mask, reg_tmp_gpr);
    }
    auto zmm_1 = zmm_tmp_1();
    auto zmm_1_masked = col_tail ? zmm_1 | rd_tail_mask | T_z : zmm_1;

    assert(max_num_cols > 0);

    const auto reg_src = reg_tmp_gpr;
    lea(reg_src, ptr[reg_base + src_offset]);

    for (int r = 0; r < num_rows; ++r) {
        switch (brg.dt_a) {
            case data_type::bf16:
            case data_type::f16:
                if (use_memadvice)
                    vmovrsw(zmm_1_masked, ptr[reg_src]);
                else
                    vmovdqu16(zmm_1_masked, ptr[reg_src]);
                break;
            case data_type::f8_e5m2:
            case data_type::f8_e4m3:
            case data_type::s8:
            case data_type::u8:
                if (use_memadvice)
                    vmovrsb(zmm_1_masked, ptr[reg_src]);
                else
                    vmovdqu8(zmm_1_masked, ptr[reg_src]);
                break;
            default: assert(!"unsupported data type");
        }
        vmovups(ptr[reg_buf + r * zmm_width_in_bytes], zmm_1);
        add(reg_src, reg_src_stride);
    }
}

void jit_brgemm_amx_uker_t::ace_load_A_4x16bytes(
        const Zmm &zmm, size_t mask, const Reg64 &reg_A, dim_t offset) {
    constexpr size_t full_mask = 0xFFFF;
    constexpr size_t all_full_mask = 0xFFFFFFFFFFFFFFFF;

    const auto zmm_tmp1 = ace_zmm_tmp(1);
    const auto xmm_tmp = Xmm(zmm_tmp1.getIdx());
    if (mask != all_full_mask) vpxord(zmm, zmm, zmm);

    for (int i = 0; i < 4; ++i) {
        const auto cur_offset = offset + i * lda() * ace_zmms_per_bd_block;
        const size_t cur_mask = (mask >> (16 * i)) & full_mask;
        if (cur_mask == 0) continue;

        if (cur_mask != full_mask) {
            // Partial mask: load bytes and insert into ZMM.
            // vbroadcasti32x4 is unsafe because it reads the full 128-bit block first.
            mov(reg_tmp_gpr, cur_mask);
            kmovq(ace_load_A_mask, reg_tmp_gpr);
            vmovdqu8(xmm_tmp | ace_load_A_mask | T_z, ptr[reg_A + cur_offset]);
            vinserti64x2(zmm, zmm, xmm_tmp, i);
        } else {
            // Note: bind by value; assigning through a reference to
            // ace_load_A_mask would clobber the member used above.
            const Xbyak::Opmask bcst_mask
                    = utils::pick(i, ace_load_A_mask_f, ace_load_A_mask_f0,
                            ace_load_A_mask_f00, ace_load_A_mask_f000);
            vbroadcasti32x4(zmm | bcst_mask, ptr[reg_A + cur_offset]);
        }
    }
}

void jit_brgemm_amx_uker_t::ace_load_A(
        brgemm_iteration_t &bi, int bdb, dim_t offset) {

    // ACE inputs are fetched through ZMMs rather than TILELOADD/TILELOADDT1.
    // ACE also does not imply MOVRS, so NT and memory-advice hints do not
    // affect the ACE A/B load paths.

    const auto bd_block = bi.bdi->block(bdb);
    const auto rd_block = bi.rdi->block(0);
    const auto base_zmm_idx = ace_zmm_A(bdb, 0).getIdx();
    const auto a_zmm1 = Zmm(base_zmm_idx + 0);
    const auto a_zmm2 = Zmm(base_zmm_idx + 1);
    const auto a_zmm3 = Zmm(base_zmm_idx + 2);
    const auto a_zmm4 = Zmm(base_zmm_idx + 3);

    const auto transformed_data_base
            = brg.get_wsp_base_offset(brgemm_desc_t::wsp_a_transform)
            + (bi.bsi->idx * brg.all_rdb() + bi.rdi->pos(0))
                    * brg.ace_transformed_A_bd_block2_size()
            + bdb * brg.ace_transformed_A_bd_block_size();
    const auto transf_addr = [&](int i) {
        const auto transformed_data_offset
                = transformed_data_base + i * zmm_width_in_bytes;
        return ptr[reg_buf + transformed_data_offset];
    };

    if (brg.ace_save_transform_A() && bi.ldi->idx > 0) {

        vmovups(a_zmm1, transf_addr(0));
        vmovups(a_zmm2, transf_addr(1));
        vmovups(a_zmm3, transf_addr(2));
        vmovups(a_zmm4, transf_addr(3));
        return; // done
    }

    // Load 4 zmm registers with 16 rows, 4 bytes per each row: each register
    // holds 16 x rd_step elements (2 for bf16, 4 for int8).

    const int mask_stride = 16; // 16 bytes per row
    const auto rd_block2 = rd_block;
    for (int rb = 0; rb < ace_zmms_per_bd_block; ++rb) {
        auto zmm = ace_zmm_A(bdb, rb);
        if (rb >= bd_block) {
            vpxord(zmm, zmm, zmm);
            continue;
        }
        size_t mask = 0;
        int mask_row_offs = 0;
        for (int row = rb; row < bd_block; row += ace_zmms_per_bd_block) {
            for (int col = 0; col < rd_block2; ++col) {
                if (brg.typesize_A == 2) {
                    // bf16, f16
                    mask |= (size_t)0b11 << (mask_row_offs + col * 2);
                } else if (brg.typesize_A == 1) {
                    // f8_e5m2, f8_e4m3, s8, u8
                    mask |= (size_t)0b1 << (mask_row_offs + col);
                } else {
                    assert(!"Unsupported data type for ace_load_A");
                }
            }
            mask_row_offs += mask_stride;
        }
        ace_load_A_4x16bytes(zmm, mask, reg_A, offset + rb * lda());
    }
    // transpose four 4x4 blocks
    const auto tmp_zmm1 = ace_zmm_tmp(1);
    const auto tmp_zmm2 = ace_zmm_tmp(2);
    const auto tmp_zmm3 = ace_zmm_tmp(3);
    const auto tmp_zmm4 = ace_zmm_tmp(4);
    vpunpckldq(tmp_zmm1, a_zmm1, a_zmm2);
    vpunpckhdq(tmp_zmm2, a_zmm1, a_zmm2);
    vpunpckldq(tmp_zmm3, a_zmm3, a_zmm4);
    vpunpckhdq(tmp_zmm4, a_zmm3, a_zmm4);
    vpunpcklqdq(a_zmm1, tmp_zmm1, tmp_zmm3);
    vpunpckhqdq(a_zmm2, tmp_zmm1, tmp_zmm3);
    vpunpcklqdq(a_zmm3, tmp_zmm2, tmp_zmm4);
    vpunpckhqdq(a_zmm4, tmp_zmm2, tmp_zmm4);

    if (brg.ace_save_transform_A() && bi.ldi->idx == 0) {

        vmovups(transf_addr(0), a_zmm1);
        vmovups(transf_addr(1), a_zmm2);
        vmovups(transf_addr(2), a_zmm3);
        vmovups(transf_addr(3), a_zmm4);
    }
}

void jit_brgemm_amx_uker_t::ace_load_B(
        brgemm_iteration_t &bi, int ldb, dim_t offset, int rdstep) {
    // if rdstep is -1, then we load all registers for this ldb
    const auto start_rdstep = (rdstep == -1) ? 0 : rdstep;
    const auto finish_rdstep
            = (rdstep == -1) ? ace_rd_steps(bi.rdi->block(0)) : rdstep + 1;
    // EVEX_compress_addr uses rbp (= reg_aux1_batch) as a scale multiplier
    // for offsets >= 0x200. Since rbp holds a live batch pointer during the
    // microkernel, we must not let it be used as an address component.
    // Pre-add the base offset to reg_tmp_gpr and use small per-rds offsets.
    const dim_t rds_stride = brg.rd_step * LDB_size_;
    // When called with a specific rdstep (n_bcast_1_load=true path), the
    // caller passes B_offset for the ldb block but does not advance for the
    // K-step. We must add rdstep * rds_stride so that each K-step reads from
    // the correct row of B (not always row 0).
    const dim_t effective_offset
            = offset + (rdstep == -1 ? 0 : rdstep * rds_stride);
    const bool needs_base_fixup = (effective_offset >= EVEX_max_8b_offt);
    if (needs_base_fixup) { lea(reg_tmp_gpr, ptr[reg_B + effective_offset]); }
    for (int rds = start_rdstep; rds < finish_rdstep; rds++) {
        // if rdstep != -1 , then (rds - start_rdstep) should be 0
        auto zmm = ace_zmm_B(ldb, rds - start_rdstep);
        auto k_mask = (!bi.ldi->is_tail(ldb)) ? ld_full_mask : ld_tail_mask;
        const dim_t rds_off = (rds - start_rdstep) * rds_stride;
        // Use dword-granular vmovups: each ld element occupies one dword of
        // the B register, and ld_tail_mask has one bit per element. A
        // qword-granular load would fetch twice the tail bytes and can read
        // past the end of B.
        if (needs_base_fixup) {
            vmovups(zmm | k_mask | T_z,
                    EVEX_compress_addr(reg_tmp_gpr, rds_off));
        } else {
            vmovups(zmm | k_mask | T_z,
                    EVEX_compress_addr(reg_B, effective_offset + rds_off));
        }
    }
}

bool jit_brgemm_amx_uker_t::is_b_scale_tail(
        const brgemm_iteration_t &bi) const {
    // One BSR B field is 64 scales wide, so a tail exists whenever fewer than
    // 64 N columns remain from the start of the field.
    const dim_t n = rnd_dn(bi.ldi->pos(0) * brg.ld_block, 64);
    return (brg.load_dim - n) < 64;
}

void jit_brgemm_amx_uker_t::set_b_scale_tail_mask(
        const brgemm_iteration_t &bi) {
    if (!is_b_scale_tail(bi)) return;
    const dim_t n = rnd_dn(bi.ldi->pos(0) * brg.ld_block, 64);
    const size_t remaining_n = static_cast<size_t>(brg.load_dim - n);
    assert(remaining_n > 0 && remaining_n < 64);
    const size_t tail_mask = (static_cast<size_t>(1) << remaining_n) - 1;
    mov(reg_tmp_gpr, tail_mask);
    kmovq(ld_scale_tail_mask, reg_tmp_gpr);
}

void jit_brgemm_amx_uker_t::load_b_scale(const brgemm_iteration_t &bi) {
    set_b_scale_tail_mask(bi);
    const auto zmm_b_scales = zmm_tmp_2();
    // Out-of-range N columns are zeroed rather than left undefined: an e8m0
    // of 0 is a denormal scale, and the corresponding B elements are zero as
    // well, so the products of the tail lanes stay zero.
    const auto zmm_masked = is_b_scale_tail(bi)
            ? (zmm_b_scales | ld_scale_tail_mask | T_z)
            : zmm_b_scales;
    reg_wei_scales.restore();
    vmovdqu8(zmm_masked, ptr[reg_wei_scales + B_offset_scales(bi, 0, 0)]);
    vpermb(zmm_b_scales, zmm_wei_scale_permute, zmm_b_scales);
}

void jit_brgemm_amx_uker_t::maybe_load_mxfp8_scales(
        const brgemm_iteration_t &bi) {
    if (!is_mxfp8_compute()) return;
    // Four consecutive rd windows share one BSR state:
    //   window 0: the A field (32 rows x 2 scale groups) and the high B field
    //             (64 N columns of the first group) are written together by
    //             bsrmovf;
    //   window 2: only the low B field (the second group) changes, bsrmovl;
    //   windows 1 and 3 reuse what the previous window wrote -- the outer
    //   product selector, not the BSR contents, distinguishes them.
    const int window = static_cast<int>(bi.rdi->pos(0) % 4);
    if (window != 0 && window != 2) return;

    const auto zmm_a_scales = zmm_tmp_1();
    const auto zmm_b_scales = zmm_tmp_2();

    load_b_scale(bi);
    if (window == 0) {
        reg_src_scales.restore();
        vmovups(zmm_a_scales, ptr[reg_src_scales + A_offset_scales(bi, 0, 0)]);
        bsrmovf(bsr0, zmm_a_scales, zmm_b_scales);
    } else {
        bsrmovl(bsr0, zmm_b_scales);
    }
}

void jit_brgemm_amx_uker_t::outer_product(
        const Zmm &zmm_a, const Zmm &zmm_b, const Tmm &accm, int imm8) {
    using namespace data_type;
    // ACE has no unscaled fp8 outer product: the TOP4MX*PS forms below are
    // always the MX-scaled ones. On the MXFP8 path `imm8` selects the block
    // scales staged in the BSR register by load_mxfp8_scales(). On the plain
    // fp8 path the caller passes 0 and generate() has primed the BSR with
    // bsrinit, so every scale read is 1.0 and the selector is immaterial.
    assert(IMPLICATION(brg.is_fp8, brg.is_ace())
            && "fp8 outer products require the ACE BSR to be initialized");
    assert(IMPLICATION(!is_mxfp8_compute(), imm8 == 0));
    if (brg.dt_a == bf16 && brg.dt_b == bf16) {
        top2bf16ps(accm, zmm_a, zmm_b);
    } else if (brg.dt_a == u8 && brg.dt_b == u8) {
        top4buud(accm, zmm_a, zmm_b);
    } else if (brg.dt_a == u8 && brg.dt_b == s8) {
        top4busd(accm, zmm_a, zmm_b);
    } else if (brg.dt_a == s8 && brg.dt_b == u8) {
        top4bsud(accm, zmm_a, zmm_b);
    } else if (brg.dt_a == s8 && brg.dt_b == s8) {
        top4bssd(accm, zmm_a, zmm_b);
    } else if (brg.dt_a == f8_e5m2 && brg.dt_b == f8_e5m2) {
        top4mxbf8ps(accm, zmm_a, zmm_b, imm8);
    } else if (brg.dt_a == f8_e5m2 && brg.dt_b == f8_e4m3) {
        top4mxbhf8ps(accm, zmm_a, zmm_b, imm8);
    } else if (brg.dt_a == f8_e4m3 && brg.dt_b == f8_e4m3) {
        top4mxhf8ps(accm, zmm_a, zmm_b, imm8);
    } else if (brg.dt_a == f8_e4m3 && brg.dt_b == f8_e5m2) {
        top4mxhbf8ps(accm, zmm_a, zmm_b, imm8);
    } else {
        assert(!"Unsupported data type for outer product");
    }
}

void jit_brgemm_amx_uker_t::gemm_microkernel_ace(brgemm_iteration_t &bi) {
    prf0A.reset();
    prf1A.reset();
    prf2A.reset();
    prfntaA.reset();
    prf0B.reset();
    prf1B.reset();
    prf2B.reset();
    prfntaB.reset();

    // BSR selector of the current outer product. The A field of the BSR holds
    // 32 rows x 2 scale groups; `bdb` picks the 16-row half (bd_block is 16)
    // and the parity of the rd window picks the scale group. The B field
    // holds 64 N columns, of which `ldb` picks the 16-wide slice.
    // Both halves are addressed by the selector, so load_mxfp8_scales() only
    // has to refresh the BSR once per pair of rd windows.
    const auto bsr_selector = [&](int bdb, int ldb) {
        if (!is_mxfp8_compute()) return 0;
        assert(bdb < 2 && ldb < b_scales_perm_groups);
        const int selector_a = (bdb % 2) * 2 + (bi.rdi->pos(0) % 4) / 2;
        return selector_a | (ldb << 3);
    };

    if (brg.n_bcast_1_load) {
        for (int bdb = 0; bdb < bi.bdi->block2(); bdb++) {
            ace_load_A(bi, bdb, A_offset(bi, bdb));
        }

        // The ld loop is inside the rd loop so that consecutive outer
        // products target distinct accumulator tiles instead of chaining on
        // one of them.
        for (int rds = 0; rds < ace_rd_steps(bi.rdi->block(0)); rds++) {
            for (int ldb = 0; ldb < bi.ldi->block2(); ldb++) {
                // Load one line from B for the current ldb and rds into
                // ace_zmm_B(ldb, 0).
                ace_load_B(bi, ldb, B_offset(bi, ldb), rds);
                for (int bdb = 0; bdb < bi.bdi->block2(); bdb++) {
                    const auto &accm = Tmm(get_C_tensor(bi, bdb, ldb));
                    outer_product(ace_zmm_A(bdb, rds), ace_zmm_B(ldb, 0), accm,
                            bsr_selector(bdb, ldb));
                }
            }
        }
    } else {
        for (int ldb = 0; ldb < bi.ldi->block2(); ldb++) {
            // load several registers for each ldb
            ace_load_B(bi, ldb, B_offset(bi, ldb), -1);
        }

        for (int bdb = 0; bdb < bi.bdi->block2(); bdb++) {
            // Load the A registers for this bd block. A is deliberately
            // loaded here and not inside the rd loop below: ace_load_A()
            // fills all ace_zmms_per_bd_block registers at once, so calling
            // it per rd step would redo the whole masked load + transpose
            // (and, with ace_save_transform_A(), the transform stores too).
            ace_load_A(bi, bdb, A_offset(bi, bdb));
            // rd outside, ld inside: consecutive outer products write
            // different accumulator tiles, so the TMUL latency is hidden
            // instead of serializing on a single Tmm.
            for (int rds = 0; rds < ace_rd_steps(bi.rdi->block(0)); rds++) {
                for (int ldb = 0; ldb < bi.ldi->block2(); ldb++) {
                    const auto &accm = Tmm(get_C_tensor(bi, bdb, ldb));
                    outer_product(ace_zmm_A(bdb, rds), ace_zmm_B(ldb, rds),
                            accm, bsr_selector(bdb, ldb));
                }
            }
        }
    }
}

void jit_brgemm_amx_uker_t::gemm_microkernel_amx(brgemm_iteration_t &bi) {
    prf0A.reset();
    prf1A.reset();
    prf2A.reset();
    prfntaA.reset();
    prf0B.reset();
    prf1B.reset();
    prf2B.reset();
    prfntaB.reset();

    const auto store_by_vectors = get_store_by_vectors(bi.apply_postops);

    bool do_post_tilestore = (brg.interleave_tilestores_ && bi.last_bsi
            && imap_[bi.apply_postops].is_last_rdi(bi.rdi));

    bool do_pre_tilestore = (brg.interleave_tilestores_ && bi.first_bsi
            && bi.rdi->pos(0) == 0 && was_prev_bi_);

    if (store_by_vectors)
        mov(reg_stride_ld_block, ld_block_C_size_);
    else
        mov(reg_stride_ld_block, LDC_size_);

    for (int bdb = 0; bdb < bi.bdi->block2(); bdb++) {
        if (brg.fused_copy_a) {
            maybe_fused_copy_A_nt_load(bi, bdb);
        } else {
            maybe_tileloadd_nt(
                    bi, matrix_kind_t::matrix_A, bdb, A_offset(bi, bdb));
        }

        for (int ldb = 0; ldb < bi.ldi->block2(); ldb++) {
            if (bdb == 0)
                maybe_tileloadd_nt(
                        bi, matrix_kind_t::matrix_B, ldb, B_offset(bi, ldb));
            if (ldb == 0) {
                if (bdb > 0)
                    tdpbxxd(bi, bdb - 1, bi.ldi->block2() - 1, do_pre_tilestore,
                            do_post_tilestore);
            } else
                tdpbxxd(bi, bdb, ldb - 1, do_pre_tilestore, do_post_tilestore);
        }
    }
    // last tdpbxxd
    tdpbxxd(bi, bi.bdi->block2() - 1, bi.ldi->block2() - 1, do_pre_tilestore,
            do_post_tilestore);
}

void jit_brgemm_amx_uker_t::emit_ace_load_A_masks() {
    mov(reg_tmp_gpr, 0xf);
    kmovq(ace_load_A_mask_f, reg_tmp_gpr);
    mov(reg_tmp_gpr, 0xf0);
    kmovq(ace_load_A_mask_f0, reg_tmp_gpr);
    mov(reg_tmp_gpr, 0xf00);
    kmovq(ace_load_A_mask_f00, reg_tmp_gpr);
    mov(reg_tmp_gpr, 0xf000);
    kmovq(ace_load_A_mask_f000, reg_tmp_gpr);
}

void jit_brgemm_amx_uker_t::emit_mxfp8_tables() {
    // clang-format off
    static constexpr uint8_t
            permute_table[mxfp8_reduce_levels][zmm_width_in_bytes] = {
        {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111},
        {8, 9, 10, 11, 12, 13, 14, 15, 64, 65, 66, 67, 68, 69, 70, 71, 24, 25, 26, 27, 28, 29, 30, 31, 80, 81, 82, 83, 84, 85, 86, 87, 40, 41, 42, 43, 44, 45, 46, 47, 96, 97, 98, 99, 100, 101, 102, 103, 56, 57, 58, 59, 60, 61, 62, 63, 112, 113, 114, 115, 116, 117, 118, 119},
        {4, 5, 6, 7, 64, 65, 66, 67, 12, 13, 14, 15, 72, 73, 74, 75, 20, 21, 22, 23, 80, 81, 82, 83, 28, 29, 30, 31, 88, 89, 90, 91, 36, 37, 38, 39, 96, 97, 98, 99, 44, 45, 46, 47, 104, 105, 106, 107, 52, 53, 54, 55, 112, 113, 114, 115, 60, 61, 62, 63, 120, 121, 122, 123},
        {2, 3, 64, 65, 6, 7, 68, 69, 10, 11, 72, 73, 14, 15, 76, 77, 18, 19, 80, 81, 22, 23, 84, 85, 26, 27, 88, 89, 30, 31, 92, 93, 34, 35, 96, 97, 38, 39, 100, 101, 42, 43, 104, 105, 46, 47, 108, 109, 50, 51, 112, 113, 54, 55, 116, 117, 58, 59, 120, 121, 62, 63, 124, 125},
    };

    static constexpr uint64_t mask_table[mxfp8_reduce_levels] = {
        0xffff0000ffff0000,
        0xff00ff00ff00ff00,
        0xf0f0f0f0f0f0f0f0,
        0xcccccccccccccccc,
    };

    static constexpr uint8_t final_permute[zmm_width_in_bytes] = {
        0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 32, 48, 40, 56, 36, 52, 44, 60, 34, 50, 42, 58, 38, 54, 46, 62,
        1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31, 33, 49, 41, 57, 37, 53, 45, 61, 35, 51, 43, 59, 39, 55, 47, 63};

    static constexpr uint8_t final_permute_store[zmm_width_in_bytes] = {
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
        32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63};
    // clang-format on

    align(64);
    L(mxfp8_permute_table);
    for (int level = 0; level < mxfp8_reduce_levels; level++)
        for (int i = 0; i < zmm_width_in_bytes; i++)
            db(permute_table[level][i]);

    align(64);
    L(mxfp8_mask_table);
    for (int level = 0; level < mxfp8_reduce_levels; level++)
        dq(mask_table[level]);

    align(64);
    L(mxfp8_final_permute_table);
    for (int i = 0; i < zmm_width_in_bytes; i++)
        db(final_permute[i]);

    align(64);
    L(mxfp8_final_permute_store_table);
    for (int i = 0; i < zmm_width_in_bytes; i++)
        db(final_permute_store[i]);

    align(64);
    L(mxfp8_bf16_truncation_table);
    for (int i = 0; i < zmm_width_in_bytes / 2; i++)
        dw(2 * i + 1);
}

void jit_brgemm_amx_uker_t::quantize_to_mxfp8(
        brgemm_iteration_t &bi, bool tile_src, int node_start, int node_end) {

    // The code below is emitted as a sequence of numbered nodes and only the
    // nodes within the [node_start, node_end) range are actually emitted. This
    // allows the caller to interleave the quantization with the rest of the
    // kernel by requesting the nodes in several chunks.
    int current_node = 0;
    auto advance_node = [&]() { current_node++; };
    auto is_node_relevant = [&]() {
        return current_node >= node_start && current_node < node_end;
    };

    // Short aliases for the tile geometry, see the class scope declarations.
    // mxfp8_all_nodes is derived from them, so they must not be redefined here.
    constexpr int max_bd_blocks = mxfp8_max_bd_blocks;
    constexpr int max_ld_blocks = mxfp8_max_ld_blocks;
    constexpr int max_tiles = mxfp8_max_tiles;
    constexpr int tile_rows = mxfp8_tile_rows;
    constexpr int tiles_per_group = mxfp8_tiles_per_group;

    Zmm perm[mxfp8_reduce_levels];
    for (int i = 0; i < mxfp8_reduce_levels; i++)
        perm[i] = Zmm(29 - i);

    Opmask mask[mxfp8_reduce_levels];
    for (int i = 0; i < mxfp8_reduce_levels; i++)
        mask[i] = Opmask(i + 1);

    const Opmask int8_blend_mask = Opmask(7);
    // mask for the final f8 stores: guards partial ld blocks / missing tiles
    const Opmask d_store_mask = Opmask(4);
    // Set for the groups whose max is inf/NaN.
    const Opmask k_is_inf = k7;
    // Set for the groups whose scale is zero.
    const Opmask k_is_zero = k6;

    const Zmm zmm_inf = Zmm(29);
    const Zmm e8m0_zmm_inf = Zmm(29);
    const Zmm zmm_max_dt_exp = Zmm(28);
    const Zmm zmm_zero = Zmm(27);
    const Zmm zmm_e127 = Zmm(28);
    const Zmm zmm_final_permute_store = Zmm(26);
    const Zmm zmm_final_permute = Zmm(25);
    // Indices of the truncating f32 -> bf16 pack, see
    // mxfp8_bf16_truncation_table.
    const Zmm zmm_bf16_truncate_perm = Zmm(21);
    const Zmm zmm_inv_scale = Zmm(20);
    const Zmm zmm_max = Zmm(20);
    const Zmm zmm_scales = Zmm(30);
    Tmm tmm[max_tiles];

    vmovups(zmm_bf16_truncate_perm, ptr[rip + mxfp8_bf16_truncation_table]);

    if (is_node_relevant()) {
        for (int i = 0; i < mxfp8_reduce_levels; i++) {
            vmovdqu8(perm[i],
                    ptr[rip + mxfp8_permute_table + i * zmm_width_in_bytes]);
            kmovq(mask[i], ptr[rip + mxfp8_mask_table + i * sizeof(uint64_t)]);
        }

        mov(reg_tmp_gpr, 0xaaaaaaaaaaaaaaaa);
        kmovq(int8_blend_mask, reg_tmp_gpr);
    }
    advance_node();

    // Whether the tile (m, n) is a part of the current iteration.
    auto tile_exists = [&](int m, int n) {
        return static_cast<int>(bi.bdi->blocks.size()) > m
                && static_cast<int>(bi.ldi->blocks.size()) > n;
    };

    // Number of valid columns (ld dimension) in tile (m, n), 0 if the tile
    // does not exist at all.
    auto tile_ld_size = [&](int m, int n) {
        return tile_exists(m, n) ? bi.ldi->block(n) : 0;
    };

    // Number of valid rows (bd dimension) in the bdb-th block of tiles.
    auto tile_bd_size = [&](int m) {
        return static_cast<int>(bi.bdi->blocks.size()) > m ? bi.bdi->block(m)
                                                           : 0;
    };

    auto load_row_to_zmm = [&](const Zmm &into, const Tmm &maybe_tmm,
                                   int inp_bd, int ldb_local, int bdb_local) {
        if (tile_src) {
            tilemovrow(into, maybe_tmm, inp_bd);
        } else {
            vmovdqu16(into,
                    ptr[reg_buf
                            + C_offset_wsp(bi, bdb_local, ldb_local, inp_bd)]);
        }
    };

    // Loads the rows of the tile pair (m, n) and (m, n + 1) and keeps the
    // element-wise absolute max of them in `dst`. If only the first tile of
    // the pair is present, its row is used as is; if none of them is present,
    // `dst` is zeroed so that it does not contribute garbage to the reduction.
    auto load_pair_max = [&](const Zmm &dst, int inp_bd, int m, int n) {
        const int tile = m * max_ld_blocks + n;
        if (tile_exists(m, n) && tile_exists(m, n + 1)) {
            load_row_to_zmm(dst, tmm[tile], inp_bd, n, m);
            load_row_to_zmm(zmm30, tmm[tile + 1], inp_bd, n + 1, m);
            vminmaxps(dst, dst, zmm30, 0xb);
        } else if (tile_exists(m, n)) {
            load_row_to_zmm(dst, tmm[tile], inp_bd, n, m);
        } else {
            vpxord(dst, dst, dst);
        }
    };

    // Max-reduction tree over the rows [line_beg, line_end) of the tiles. The
    // leaves reduce a single row of all the tiles into a vector of exponents
    // packed as bytes, the internal levels combine two partial results with an
    // unsigned byte max.
    std::function<void(int, int, int, const Zmm &)> reduce_max_tree
            = [&](int level, int line_beg, int line_end, const Zmm &into) {
        if (line_end - line_beg == 1) {
            if (is_node_relevant()) {
                const Zmm zmm_level = Zmm(15);

                load_pair_max(into, line_beg, 0, 0);
                load_pair_max(zmm_level, line_beg, 0, 2);

                // Pack f32 -> bf16 keeping only what the byte-wise max below
                // needs: the 8 exponent bits. Truncation (keep the high word
                // of every f32) matches the e8m0 scale definition used by the
                // reference (float8_e8m0_t::operator=(float), round toward
                // zero).
                vpermt2w(into, zmm_bf16_truncate_perm, zmm_level);

                load_pair_max(zmm14, line_beg, 1, 0);
                load_pair_max(zmm_level, line_beg, 1, 2);
                vpermt2w(zmm14, zmm_bf16_truncate_perm, zmm_level);

                // move the exponents of both halves into byte lanes
                // so that the reduction can proceed as an unsigned
                // byte max
                vpsrlw(into, into, 7);
                vpsllw(zmm14, zmm14, 1);

                vmovdqu8(into | int8_blend_mask, zmm14);
            }
            advance_node();
            return;
        }

        const Zmm zmm_level = Zmm(level + 15);
        const Zmm zmm_tmp = Zmm(31);
        const Zmm zmm_tmp2 = Zmm(30);
        const int line_mid = line_beg + (line_end - line_beg) / 2;
        reduce_max_tree(level - 1, line_beg, line_mid, zmm_level);
        reduce_max_tree(level - 1, line_mid, line_end, zmm_tmp);
        if (is_node_relevant()) {
            vpblendmb(zmm_tmp2, zmm_level, zmm_tmp | mask[level - 1]);
            vpermt2b(zmm_level, perm[level - 1], zmm_tmp);
            vpmaxub(into, zmm_level, zmm_tmp2);
        }
        advance_node();
    };

    for (int m = 0; m < max_bd_blocks; m++)
        for (int n = 0; n < max_ld_blocks; n++)
            if (tile_exists(m, n))
                tmm[m * max_ld_blocks + n] = Tmm(get_C_tensor(bi, m, n));

    reduce_max_tree(mxfp8_reduce_levels, 0, tile_rows, zmm_max);

    if (is_node_relevant()) {
        vpxord(zmm_zero, zmm_zero, zmm_zero);

        // max exponent representable by the destination data type
        if (brg.dt_d == data_type::f8_e4m3)
            mov(reg_tmp_gpr, 8);
        else if (brg.dt_d == data_type::f8_e5m2)
            mov(reg_tmp_gpr, 15);
        else
            assert(!"unsupported dst dt");
        vpbroadcastb(zmm_max_dt_exp, reg_tmp_gpr.cvt8());

        // init e8m0 inf
        mov(reg_tmp_gpr, 0xff);
        vpbroadcastb(e8m0_zmm_inf, reg_tmp_gpr.cvt8());

        vmovups(zmm_final_permute_store,
                ptr[rip + mxfp8_final_permute_store_table]);
        vmovups(zmm_final_permute, ptr[rip + mxfp8_final_permute_table]);

        // gather the per-group exponents into consecutive bytes
        vpermb(zmm_scales, zmm_final_permute, zmm_max);
        // find inf/nan
        vpcmpub(k_is_inf, zmm_scales, e8m0_zmm_inf, 0x0);
        // scale = max_exponent - max_exponent_of_dst_dt
        vpsubusb(zmm_scales, zmm_scales, zmm_max_dt_exp);
        vpcmpub(k_is_zero, zmm_scales, zmm_zero, 0x0);
        // restore inf
        vmovdqu8(zmm_scales | k_is_inf, e8m0_zmm_inf);

        // reorder the scales into the layout expected by the copy kernel
        vpermb(zmm14, zmm_final_permute_store, zmm_scales);

        // save the scales
        reg_dst_scales.restore();
        vmovups(ptr[reg_dst_scales + D_scales_offset(bi, 0, 0, bi.ldi->pos(0))],
                zmm14);

        // f32 +inf, injected into the values of an inf/NaN group
        mov(reg_tmp_gpr, 0x7f800000);
        vpbroadcastd(zmm_inf, reg_tmp_gpr.cvt32());

        // f32 const 2^127, used as the inverse scale of a zero group
        mov(reg_tmp_gpr, 0x7f000000);
        vpbroadcastd(zmm_e127, reg_tmp_gpr.cvt32());
    }
    advance_node();

    // One vector of inverse scales covers a single tile pair, so it has to be
    // rebuilt for every (bdb, ldb). Note that `k_is_zero` is consumed 16 bits
    // at a time here, hence this must run for every tile pair, including the
    // ones that are skipped by the store mask below.
    auto prepare_inv_scales = [&](int bdb, int ldb) {
        const Xmm xmm_inv_scale(zmm_inv_scale.getIdx());
        vextracti64x2(xmm_inv_scale, zmm_scales,
                max_ld_blocks / tiles_per_group * bdb + ldb);
        vpmovzxbd(zmm_inv_scale, xmm_inv_scale);
        vpslld(zmm_inv_scale, zmm_inv_scale, 23);

        // scale^(-1). The scale is a power of two, so the 14-bit
        // approximation is exact here.
        vrcp14ps(zmm_inv_scale, zmm_inv_scale);
        // inject 2^127 for zero values
        vmovdqu32(zmm_inv_scale | k_is_zero, zmm_e127);
        kshiftrq(k_is_zero, k_is_zero, 16);
    };

    // Scales and down-converts row `i` of the tile pair (bdb, ldb) and stores
    // the resulting 32 f8 values.
    auto scale_and_store_row
            = [&](int bdb, int ldb, int i, bool use_store_mask) {
        const int tile = tiles_per_group * ldb + max_ld_blocks * bdb;

        // broadcast the scale of the row across the vector
        mov(reg_tmp_gpr, i);
        vpbroadcastd(zmm2, reg_tmp_gpr.cvt32());
        vpermd(zmm2, zmm2, zmm_inv_scale);

        // get the values from the tiles
        load_row_to_zmm(zmm0, tmm[tile], i, tiles_per_group * ldb, bdb);
        load_row_to_zmm(zmm1, tmm[tile + 1], i, tiles_per_group * ldb + 1, bdb);

        // the scale is zero for inf/nan, inject inf to get nan
        // after the multiplication
        vpcmpud(k_is_inf, zmm2, zmm_zero, 0x0);
        vmovdqu32(zmm0 | k_is_inf, zmm_inf);
        vmovdqu32(zmm1 | k_is_inf, zmm_inf);

        // scale^(-1) the values
        vmulps(zmm0, zmm0, zmm2);
        vmulps(zmm1, zmm1, zmm2);

        // quantize to f8. The saturating variants are required: the e8m0
        // scale truncates the exponent of the group max, so the scaled
        // values may exceed the dst dt range by up to one exponent.
        if (brg.dt_d == data_type::f8_e4m3) {
            vcvtps2hf8s(xmm0, zmm0);
            vcvtps2hf8s(xmm1, zmm1);
        } else if (brg.dt_d == data_type::f8_e5m2) {
            vcvtps2bf8s(xmm0, zmm0);
            vcvtps2bf8s(xmm1, zmm1);
        } else
            assert(!"unsupported dst dt");

        vinserti128(ymm0, ymm0, xmm1, 1);

        const auto addr_D = ptr[reg_D
                + D_offset(bi, bdb, i, bi.ldi->pos(tiles_per_group * ldb))];
        if (use_store_mask)
            vmovdqu8(addr_D, ymm0 | d_store_mask);
        else
            vmovdqu8(addr_D, ymm0);
    };

    for (int bdb = 0; bdb < bi.bdi->block2(); bdb++) {
        for (int ldb = 0; ldb < max_ld_blocks / tiles_per_group; ldb++) {
            if (is_node_relevant()) prepare_inv_scales(bdb, ldb);
            advance_node();

            // The stored ymm holds 32 f8 values: bytes [0..15] come from tile
            // (bdb, 2 * ldb) and bytes [16..31] from tile (bdb, 2 * ldb + 1).
            // Build the byte mask out of the number of valid ld elements of
            // each of the two tiles; a missing tile contributes 0 bits.
            const int ld_size_lo = tile_ld_size(bdb, tiles_per_group * ldb);
            const int ld_size_hi = tile_ld_size(bdb, tiles_per_group * ldb + 1);
            assert(ld_size_lo <= 16 && ld_size_hi <= 16);
            const uint32_t store_mask = (uint32_t)((1ULL << ld_size_lo) - 1)
                    | (uint32_t)(((1ULL << ld_size_hi) - 1) << 16);
            // nothing to store for this pair of tiles
            if (store_mask == 0) continue;
            const bool use_store_mask = store_mask != 0xffffffff;

            if (use_store_mask && is_node_relevant()) {
                mov(reg_tmp_gpr, store_mask);
                kmovd(d_store_mask, reg_tmp_gpr.cvt32());
            }
            advance_node();

            const int bd_size = tile_bd_size(bdb);
            for (int i = 0; i < tile_rows; i++) {
                // the line is outside of the valid bd range of both tiles
                if (i >= bd_size) continue;
                if (is_node_relevant())
                    scale_and_store_row(bdb, ldb, i, use_store_mask);
                advance_node();
            }
        }
    }

    // Callers pass [0, mxfp8_all_nodes) to emit everything, so the bound must
    // cover the whole sequence, otherwise the tail would be silently dropped.
    assert(current_node <= mxfp8_all_nodes);
}

void jit_brgemm_amx_uker_t::rdb_loop_body(brgemm_iteration_t &bi) {
    if (brg.is_ace())
        gemm_microkernel_ace(bi);
    else
        gemm_microkernel_amx(bi);
}
void jit_brgemm_amx_uker_t::rdb_loop_call_based(brgemm_iteration_t &bi) {
    // Note: the micro-kernel bodies registered below are entered with `call`,
    // so inside them rsp is 8 bytes lower than here (return address). The
    // register scratchpad is addressed relative to rsp, hence
    // reg64_savable_t::save()/restore() (and anything else touching the
    // scratchpad) must not be emitted inside a body. rdb_loop_body() only
    // reads A, B and the transform buffer through registers, so the
    // save/restore of these pointers stays here, in the caller.
    // reg_C, reg_D and reg_ldb_loop are not modified by rdb_loop_body() and
    // therefore are not preserved.
    const auto &tloop = imap_[bi.apply_postops];
    assert(tloop.rdis.size() > 1);

    const bool separate_last_iteration = brg.rdb_tail > 0;
    const auto *last_rdi = &tloop.rdis.back();
    auto rdb_loop_count = static_cast<int>(tloop.rdis.size());
    if (separate_last_iteration) rdb_loop_count--;

    // Calculate the A and B increments using the next rdi assuming that they
    // are the same for all iterations.
    bi.rdi = &tloop.rdis[0];
    brgemm_iteration_t next_rdi_bi = bi;
    next_rdi_bi.rdi = &tloop.rdis[1];
    const auto reg_A_increment = A_offset(next_rdi_bi, 0) - A_offset(bi, 0);
    const auto reg_B_increment = B_offset(next_rdi_bi, 0) - B_offset(bi, 0);

    // save registers advanced by the loop below
    reg_A.save();
    reg_B.save();
    if (brg.ace_save_transform_A()) reg_buf.save();

    // Defer the generation of the micro-kernel bodies: they are emitted out
    // of the hot path by top_loop(), which owns them.
    const int rd_unroll = call_based_rd_unroll;
    const auto n_variants = nstl::min(rdb_loop_count, rd_unroll);
    std::vector<std::shared_ptr<Label>> variant_labels;
    variant_labels.reserve(n_variants);
    for (int v = 0; v < n_variants; v++) {
        auto entry = std::make_shared<Label>();
        auto variant_bi = bi;
        variant_bi.rdi = &tloop.rdis[v];
        deferred_uk_bodies_.push_back({entry, [this, entry, variant_bi]() {
            auto body_bi = variant_bi;
            L(*entry);
            rdb_loop_body(body_bi);
            ret();
        }});
        variant_labels.push_back(std::move(entry));
    }

    for (int rdb = 0; rdb < rdb_loop_count; rdb += rd_unroll) {
        const auto n_calls = nstl::min(rd_unroll, rdb_loop_count - rdb);
        for (int v = 0; v < n_calls; v++) {
            // The BSR refresh has to stay in the caller: it restores the
            // scale pointers from the register scratchpad, which is addressed
            // relative to rsp and therefore off by the return address inside
            // a called body.
            //
            // Note that it is indexed by the *real* rd window (rdb + v), not
            // by the variant v: unlike reg_A / reg_B, the scale registers are
            // not advanced by the rd loop (reg_src_scales moves only between
            // bd iterations, in bs_loop()), so the offsets computed here are
            // absolute within the current bd iteration. The
            // shared body of variant v is still the right one, because the
            // only thing that varies within a group is the BSR selector, and
            // that depends on the window parity, which is v.
            bi.rdi = &tloop.rdis[rdb + v];
            maybe_load_mxfp8_scales(bi);
            call(*variant_labels[v]);
        }

        add(reg_A, reg_A_increment * n_calls);
        add(reg_B, reg_B_increment * n_calls);
        if (brg.ace_save_transform_A())
            add(reg_buf, brg.ace_transformed_A_bd_block2_size() * n_calls);
    }

    // restore registers
    if (brg.ace_save_transform_A()) reg_buf.restore();
    reg_B.restore();
    reg_A.restore();

    // separate last iteration if needed
    if (separate_last_iteration) {
        bi.rdi = last_rdi;
        maybe_load_mxfp8_scales(bi);
        rdb_loop_body(bi);
    } else {
        bi.rdi = last_rdi;
    }
}

void jit_brgemm_amx_uker_t::emit_deferred_uk_bodies() {
    if (deferred_uk_bodies_.empty()) return;

    Label end_uk_bodies;
    jmp(end_uk_bodies, T_NEAR);
    for (auto &body : deferred_uk_bodies_)
        body.emit();
    L(end_uk_bodies);

    // The labels are bound now and nothing may reference them anymore: they
    // are destroyed together with the emitters that own them.
    deferred_uk_bodies_.clear();
}

void jit_brgemm_amx_uker_t::rdb_loop(brgemm_iteration_t &bi) {
    const auto &tloop = imap_[bi.apply_postops];
    // The B-scale tail mask depends on bi.ldi only, so it is generated once
    // here instead of on every rd window.
    if (need_to_output_mxfp8()
            && !(brg.ace_save_transform_A() && bi.ldi->idx > 0))
        emit_ace_load_A_masks();
    if (call_based_rd_loop) {
        rdb_loop_call_based(bi);
        return;
    }
    for (auto &rdi : tloop.rdis) {
        bi.rdi = &rdi;
        maybe_load_mxfp8_scales(bi);
        rdb_loop_body(bi);
    }
}

void jit_brgemm_amx_uker_t::bs_loop_body(brgemm_iteration_t &bi) {
    if (brg.brgattr.var_bs) {
        set_A_B_matrices();
        reg_aux1_batch.restore();
        add(reg_aux1_batch, sizeof(brgemm_batch_element_t));
        prefetcht0(ptr[reg_aux1_batch]);
        reg_aux1_batch.save();
    } else {
        set_A_B_matrices(bi.bsi->pos);
    }

    rdb_loop(bi);
}

void jit_brgemm_amx_uker_t::bs_loop(brgemm_iteration_t &bi) {
    if (ununroll_bd_loop && bi.bdi->similar != nullptr) {
        // there is code for this iteration already, so we need to store
        // prev_bi_ only
        prev_bi_ = bi;
        was_prev_bi_ = true;
        return;
    }

    const auto &tloop = imap_[bi.apply_postops];
    if (ununroll_bd_loop && was_prev_bi_) {
        if (bi.bdi->idx != prev_bi_.bdi->idx) {
            add(reg_A, bi.bdi->A_shift);
            if (is_mxfp8_compute()) {
                reg_src_scales.restore();
                add(reg_src_scales, bi.bdi->A_scales_shift);
                reg_src_scales.save();
            }
            if (need_to_output_mxfp8()) {
                reg_dst_scales.restore();
                add(reg_dst_scales, bi.bdi->D_scales_shift);
                reg_dst_scales.save();
            }
        }

        const auto real_ils
                = actual_ils(bi.apply_postops, bi.skip_accumulation);

        brgemm_iteration_t *bi_shift = nullptr;
        if (!real_ils && bi.bdi->idx != prev_bi_.bdi->idx)
            bi_shift = &bi;
        else if (real_ils && prev_bi_.bdi->idx > 0 && prev_bi_.ldi->idx == 0)
            bi_shift = &prev_bi_;
        if (bi_shift != nullptr) {
            add(reg_C, bi_shift->bdi->C_shift);
            add(reg_D, bi_shift->bdi->D_shift);
            if (brg.req_comp_pads_with_bcast)
                add(reg_zp_comp_pad_a, bi_shift->bdi->zp_comp_pad_a_shift);
        }
    }

    if (bi.skip_accumulation) {
        store_accumulators(bi);
        return;
    }

    load_accumulators(bi);

    if (brg.brgattr.var_bs) {
        if (brg.alpha != 0.f) {
            Label BS_loop_label, end_BS_loop_label, first_BS_loop_label,
                    last_BS_loop_label;

            mov(reg_BS_loop, reg_BS);
            cmp(reg_BS_loop, 0);
            jz(end_BS_loop_label, T_NEAR);

            mov(reg_aux1_batch, reg_addr_batch);
            reg_aux1_batch.save();
            // first bs iteration
            cmp(reg_BS_loop, 1);
            jg(first_BS_loop_label, T_NEAR);

            bi.bsi = &(tloop.bsis[0]);
            // only one BS iteration: first and last
            bi.first_bsi = true;
            bi.last_bsi = true;
            bs_loop_body(bi);
            jmp(end_BS_loop_label, T_NEAR);

            // first BS iteration
            L_aligned(first_BS_loop_label, 64);
            bi.first_bsi = true;
            bi.last_bsi = false;
            bs_loop_body(bi);

            dec(reg_BS_loop);
            cmp(reg_BS_loop, 1);
            je(last_BS_loop_label, T_NEAR);

            // middle BS iterations
            L_aligned(BS_loop_label, 64);
            {
                bi.first_bsi = false;
                bi.last_bsi = false;
                bs_loop_body(bi);
                dec(reg_BS_loop);
                cmp(reg_BS_loop, 1);
                jg(BS_loop_label, T_NEAR);
            }
            // last BS iteration
            L_aligned(last_BS_loop_label, 64);
            bi.first_bsi = false;
            bi.last_bsi = true;
            bs_loop_body(bi);

            L_aligned(end_BS_loop_label, 64);
        }
        store_accumulators(bi);
    } else {
        if (brg.alpha != 0.f) {
            for (dim_t bs = 0; bs < brg.brgattr.max_bs; bs++) {
                bi.bsi = &(tloop.bsis[bs]);
                bi.first_bsi = bi.bsi->is_first;
                bi.last_bsi = bi.bsi->is_last;
                bs_loop_body(bi);
            }
        }
        store_accumulators(bi);
    }
}

void jit_brgemm_amx_uker_t::ldb_loop_body(brgemm_iteration_t &bi) {
    if (brg.innermost_loop == brgemm_bd_loop_innermost)
        bdb_loop(bi);
    else if (brg.innermost_loop == brgemm_ld_loop_innermost)
        bs_loop(bi);
    else
        assert(!"Unknown loop order!");
}

void jit_brgemm_amx_uker_t::ldb_loop(brgemm_iteration_t &bi) {
    // clear the transform cache for A, as the existing data is invalid as
    // we move to next bdb2 block.
    const auto &tloop = imap_[bi.apply_postops];
    transform_buf_map_A_.clear();
    for (auto &ldi : tloop.ldis) {
        bi.ldi = &ldi;
        ldb_loop_body(bi);
    }
}

jit_brgemm_amx_uker_t::bd_iteration_t *jit_brgemm_amx_uker_t::find_similar(
        const bd_iteration_t *bdi, bool apply_postops) {
    auto &tloop = imap_[apply_postops];
    const auto cidx = bdi->idx;
    // if wary_k_tail is true then last iteration is unique
    if (brg.amx_wary_k_tail() && cidx == tloop.bdis.size() - 1) return nullptr;

    for (size_t i = (actual_ils(apply_postops) ? 1 : 0); i < cidx; i++) {
        if (*bdi == tloop.bdis[i]
                && IMPLICATION(actual_ils(apply_postops),
                        tloop.bdis[cidx - 1] == tloop.bdis[i - 1])) {
            tloop.duplicated++;
            return &(tloop.bdis[i]);
        }
    }

    return nullptr;
}

void jit_brgemm_amx_uker_t::bdb_loop_body(brgemm_iteration_t &bi) {
    auto &tloop = imap_[bi.apply_postops];
    if (ununroll_bd_loop) {
        const auto cidx = bi.bdi->idx;
        if (bi.bdi->similar) {
            tloop.bdis[cidx].lstart = bi.bdi->similar->lstart;
        } else {
            align(64);
            L(tloop.bdis[cidx].lstart);
            reg_iter_labels_list.restore();
            mov(reg_iter_label, ptr[reg_iter_labels_list]);
            add(reg_iter_labels_list, 8);
            reg_iter_labels_list.save();
        }
    }

    if (brg.innermost_loop == brgemm_ld_loop_innermost)
        ldb_loop(bi);
    else if (brg.innermost_loop == brgemm_bd_loop_innermost)
        bs_loop(bi);
    else
        assert(!"Unknown loop order!");
    if (ununroll_bd_loop) { jmp(reg_iter_label); }
}

void jit_brgemm_amx_uker_t::bdb_loop(brgemm_iteration_t &bi) {
    const auto &tloop = imap_[bi.apply_postops];
    Label iteration_pointers;
    if (is_mxfp8_compute() && ununroll_bd_loop) {
        // bs_loop() advances reg_src_scales per bd iteration; keep the
        // start-of-loop value so the next ld iteration sees it again.
        reg_src_scales.restore();
        reg_src_scales_bd_loop.save();
    }
    if (need_to_output_mxfp8() && ununroll_bd_loop) {
        reg_dst_scales.restore();
        reg_dst_scales_bd_loop.save();
    }
    if (ununroll_bd_loop) {
        lea(reg_iter_labels_list, ptr[rip + iteration_pointers]);
        // shift to load address for jmp for next iteration
        add(reg_iter_labels_list, 8);
        reg_iter_labels_list.save();
    }

    for (auto &bdi : tloop.bdis) {
        bi.bdi = &bdi;
        bdb_loop_body(bi);
    }
    if (ununroll_bd_loop) {
        Label loop_end;
        jmp(loop_end, T_NEAR); //just skip list of iteration labels

        align(64);
        L(iteration_pointers);
        for (const auto &bdi : tloop.bdis) {
            putL(bdi.lstart);
        }
        putL(loop_end);
        L(loop_end);
    }
    if (is_mxfp8_compute() && ununroll_bd_loop) {
        reg_src_scales_bd_loop.restore();
        reg_src_scales.save();
    }
    if (need_to_output_mxfp8() && ununroll_bd_loop) {
        reg_dst_scales_bd_loop.restore();
        reg_dst_scales.save();
    }
}

void jit_brgemm_amx_uker_t::top_loop(brgemm_iteration_t &bi) {
    reg_C.restore();
    reg_D.restore();
    init(bi);
    if (brg.innermost_loop == brgemm_ld_loop_innermost)
        bdb_loop(bi);
    else if (brg.innermost_loop == brgemm_bd_loop_innermost)
        ldb_loop(bi);
    else
        assert(!"Unknown loop order!");

    // bi is last iteration now
    if (brg.interleave_tilestores_) {
        prev_bi_ = bi;
        was_prev_bi_ = true;
        for_(int bdb = 0; bdb < prev_bi_.bdi->block2(); bdb++)
        for (int ldb = 0; ldb < prev_bi_.ldi->block2(); ldb++) {
            maybe_tilestore(prev_bi_, bdb, ldb, true, false);
        }
    }

    const auto &tloop = imap_[bi.apply_postops];
    if (actual_ils(bi.apply_postops, bi.skip_accumulation) && ununroll_bd_loop
            && tloop.ldis.size() == 1) {
        // update reg_C and reg_D if they they were not updated yet
        add(reg_C, bi.bdi->C_shift);
        add(reg_D, bi.bdi->D_shift);
        if (brg.req_comp_pads_with_bcast)
            add(reg_zp_comp_pad_a, bi.bdi->zp_comp_pad_a_shift);
    }
    interleave_store(bi, true);

    // top_loop owns the deferred micro-kernel bodies registered by
    // rdb_loop_call_based(): emit them here, out of the hot path, and destroy
    // them together with their labels.
    emit_deferred_uk_bodies();
}

void jit_brgemm_amx_uker_t::fill_imap() {
    for (bool apply_postops : {false, true}) {
        auto &tloop = imap_[apply_postops];

        tloop.bdis.clear();
        tloop.ldis.clear();
        tloop.rdis.clear();
        tloop.bsis.clear();
        tloop.bdis.reserve(brg.bdb2);
        tloop.ldis.reserve(brg.ldb2);
        tloop.rdis.reserve(brg.rdb);
        tloop.bsis.reserve(brg.brgattr.max_bs);
        brgemm_iteration_t bi_prefetch;

        auto bdi_pos = skipped_bd_mask(0);
        bd_iteration_t bdi;
        bdi.blocks.reserve(brg.bd_block2);
        dim_t prefetch_distance_m = brg.bcast_dim;
        bd_iteration_t bdi_prefetch;
        bi_prefetch.bdi = &bdi_prefetch;

        for (dim_t bdb = 0; bdb < brg.bdb; bdb += brg.bd_block2) {
            bdi.blocks.clear();
            for (dim_t ibdb = 0; ibdb < brg.bd_block2; ibdb++) {
                auto abdb = bdb + ibdb;
                if (abdb >= brg.bdb) break;
                if (brg.bdb_tail && abdb == brg.bdb - 1) {
                    bdi.blocks.emplace_back(bdi_pos, brg.bdb_tail, true);
                    if (brg.prfA.sprinkled)
                        bdi_prefetch.blocks.emplace_back(
                                bdi_pos + prefetch_distance_m, brg.bdb_tail,
                                true);
                } else {
                    bdi.blocks.emplace_back(bdi_pos, brg.bd_block, false);
                    if (brg.prfA.sprinkled)
                        bdi_prefetch.blocks.emplace_back(
                                bdi_pos + prefetch_distance_m, brg.bd_block,
                                false);
                }
                bdi_pos += brg.bd_block;
                if (bdi_pos >= brg.bcast_dim) break;
                bdi_pos = skipped_bd_mask(bdi_pos);
            }
            bdi.idx = tloop.bdis.size();

            if (brg.brgattr.bd_mask_level > 0) {
                const auto lidx = bdi.blocks.size() - 1;
                const auto bdm_sz = bdi.rel_pos(lidx) + bdi.blocks[lidx].block;
                bdi.bd_mask.resize(bdm_sz);
                bdi.adj_bd_mask.resize(bdm_sz);
                for (dim_t i = 0; i < bdm_sz; i++) {
                    bdi.bd_mask[i] = bd_mask_buffer_ptr_[bdi.pos(0) + i];
                    bdi.adj_bd_mask[i] = adj_bd_mask_buffer_[bdi.pos(0) + i];
                }
            }

            if (ununroll_bd_loop && bdi.idx > 0) {
                const auto prev_bdi = &tloop.bdis[bdi.idx - 1];
                const auto inp_shift = (bdi.pos(0) - prev_bdi->pos(0));
                bdi.A_shift = inp_shift * LDA2_size_;
                if (is_mxfp8_compute()) {
                    // A_offset_scales(m, k) is the block-local repacked
                    // layout, which is all a single kernel invocation sees;
                    // the shift between two bd iterations is therefore the
                    // difference of their 32-row-aligned starts.
                    const dim_t m_prev = rnd_dn(prev_bdi->pos(0), 32);
                    const dim_t m_curr = rnd_dn(bdi.pos(0), 32);
                    bdi.A_scales_shift = A_offset_scales(m_curr, 0)
                            - A_offset_scales(m_prev, 0);
                }
                if (need_to_output_mxfp8()) {
                    bdi.D_scales_shift
                            = D_scales_offset(&bdi, 0, 0, 0, /*global=*/true)
                            - D_scales_offset(
                                    prev_bdi, 0, 0, 0, /*global=*/true);
                }

                const auto out_shift
                        = (get_out_bd(&bdi, 0, 0) - get_out_bd(prev_bdi, 0, 0));
                bdi.C_shift = out_shift * LDC2_size_M_;
                bdi.D_shift = out_shift * LDD_size_;
                bdi.zp_comp_pad_a_shift = out_shift * brg.LDB * sizeof(int32_t);
            }
            tloop.bdis.push_back(bdi);
        }

        dim_t ldi_pos = 0;
        dim_iteration_t ldi;
        ldi.blocks.reserve(brg.ld_block2);
        dim_t prefetch_distance_n = brg.ldb;
        bd_iteration_t ldi_prefetch;
        bi_prefetch.ldi = &ldi_prefetch;

        for (dim_t ldb = 0; ldb < brg.ldb; ldb += brg.ld_block2) {
            ldi.blocks.clear();
            for (dim_t ildb = 0; ildb < brg.ld_block2; ildb++) {
                auto aldb = ldb + ildb;
                if (aldb >= brg.ldb) break;
                if (brg.ldb_tail && aldb == brg.ldb - 1) {
                    ldi.blocks.emplace_back(ldi_pos, brg.ldb_tail, true);
                    if (brg.prfB.sprinkled)
                        ldi_prefetch.blocks.emplace_back(
                                ldi_pos + prefetch_distance_n, brg.ldb_tail,
                                true);

                } else {
                    ldi.blocks.emplace_back(ldi_pos, brg.ld_block, false);
                    if (brg.prfB.sprinkled)
                        ldi_prefetch.blocks.emplace_back(
                                ldi_pos + prefetch_distance_n, brg.ld_block,
                                false);
                }
                ldi_pos++;
            }
            ldi.idx = tloop.ldis.size();
            tloop.ldis.push_back(ldi);
        }

        dim_t rdi_pos = 0;
        dim_iteration_t rdi;
        rdi.blocks.reserve(1);
        dim_iteration_t rdi_prefetch;
        bi_prefetch.rdi = &rdi_prefetch;

        for (dim_t rdb = 0; rdb < brg.rdb; rdb++) {
            rdi.blocks.clear();
            rdi.blocks.emplace_back(rdi_pos, brg.rd_block);
            if (brg.prfA.sprinkled || brg.prfB.sprinkled) {
                rdi_prefetch.blocks.emplace_back(rdi_pos, brg.rd_block);
            }
            rdi.idx = tloop.rdis.size();
            tloop.rdis.push_back(rdi);
            rdi_pos++;
        }
        if (brg.rdb_tail > 0) {
            rdi.blocks.clear();
            rdi.blocks.emplace_back(rdi_pos, brg.rdb_tail, true);
            if (brg.prfA.sprinkled || brg.prfB.sprinkled) {
                rdi_prefetch.blocks.emplace_back(rdi_pos, brg.rdb_tail, true);
            }
            rdi.idx = tloop.rdis.size();
            tloop.rdis.push_back(std::move(rdi));
        }

        // The case where bs_max is > 1, and prefetches are enabled
        // is not supported. In order to support prefetches in this case,
        // current_num_amx_ops needs to be an array per bs in bs_max.
        bs_iteration_t bsi;
        for (dim_t bs = 0; bs < brg.brgattr.max_bs; bs++) {
            bsi.pos = bs;
            bsi.is_first = (bs == 0);
            bsi.is_last = (bs == brg.brgattr.max_bs - 1);
            bsi.idx = tloop.bsis.size();
            tloop.bsis.push_back(bsi);
        }

        if (ununroll_bd_loop) {
            for (size_t ibdi = 0; ibdi < tloop.bdis.size(); ibdi++) {
                tloop.bdis[ibdi].similar
                        = find_similar(&(tloop.bdis[ibdi]), apply_postops);
            }
        }
        const dim_t rdb_including_tail = brg.rdb + (brg.rdb_tail > 0 ? 1 : 0);
        num_amx_ops = brg.bdb * rdb_including_tail * brg.ldb;
        current_num_amx_ops = 0;
        // Calculate the offsets of A's cache lines to prefetch
        prf_sprinkled_a.reset();
        if (brg.prfA.sprinkled) {
            for (size_t bdb = 0; bdb < bdi_prefetch.blocks.size(); ++bdb) {
                for (size_t rdb = 0; rdb < rdi_prefetch.blocks.size(); ++rdb) {
                    const int bd_block_size = bdi_prefetch.blocks[bdb].block;
                    for (int bd = 0; bd < bd_block_size; bd++) {
                        prf_sprinkled_a.prefetch_offsets.push_back(
                                A_offset_line(bi_prefetch,
                                        static_cast<int>(bdb),
                                        static_cast<int>(rdb), bd));
                    }
                }
            }
            std::sort(prf_sprinkled_a.prefetch_offsets.begin(),
                    prf_sprinkled_a.prefetch_offsets.end());
        }

        // Calculate the offsets of B's cache lines to prefetch
        prf_sprinkled_b.reset();
        if (brg.prfB.sprinkled) {
            for (size_t ldb = 0; ldb < ldi_prefetch.blocks.size(); ++ldb) {
                for (size_t rdb = 0; rdb < rdi_prefetch.blocks.size(); ++rdb) {
                    const int rd_block_size = rdi_prefetch.blocks[rdb].block;
                    for (int rd = 0; rd < rd_block_size; rd += brg.rd_step) {
                        prf_sprinkled_b.prefetch_offsets.push_back(
                                B_offset_line(bi_prefetch,
                                        static_cast<int>(ldb),
                                        static_cast<int>(rdb), rd));
                    }
                }
            }
            std::sort(prf_sprinkled_b.prefetch_offsets.begin(),
                    prf_sprinkled_b.prefetch_offsets.end());
        }
    }
}

void jit_brgemm_amx_uker_t::init(brgemm_iteration_t &bi) {
    was_prev_bi_ = false;
    const auto bdb2_to_unroll = nstl::max<dim_t>(0,
            brg.bdb2
                    - (actual_ils(bi.apply_postops, bi.skip_accumulation) ? 1
                                                                          : 0));
    ununroll_bd_loop = brg.brgattr.hint_ununroll_bd_loop && bdb2_to_unroll > 1
            && (brg.innermost_loop == brgemm_ld_loop_innermost || brg.ldb2 == 1)
            && get_store_by_vectors(bi.apply_postops)
            && IMPLICATION(!bi.skip_accumulation,
                    (brg.brgattr.max_bs == 1 || brg.type == brgemm_static_offs)
                            && !brg.brgattr.var_bs);

    // TODO: extend the call based rd loop to non-ACE kernels and add a
    // heuristic based on the estimated kernel size.
    call_based_rd_loop = brg.is_ace() && brg.rdb > 1;

    if (brg.type == brgemm_static_offs && !bi.skip_accumulation) {
        reg_A.restore();
        reg_B.restore();
    } else if (brg.brgattr.max_bs == 1 && !bi.skip_accumulation) {
        assert(one_of(brg.type, brgemm_addr, brgemm_offs));
        if (brg.type == brgemm_addr) {
            if (brg.layout == brgemm_row_major) {
                mov(reg_A,
                        EVEX_compress_addr(
                                reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
                mov(reg_B,
                        EVEX_compress_addr(
                                reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
            } else {
                mov(reg_A,
                        EVEX_compress_addr(
                                reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.B)));
                mov(reg_B,
                        EVEX_compress_addr(
                                reg_addr_batch, GET_OFF_BATCH_ELEMENT(ptr.A)));
            }
        } else if (brg.type == brgemm_offs) {
            reg_A.restore();
            reg_B.restore();
        }
    }

    fill_imap();

    // for many primitives which use brgemm the brg.ldb2 is equal or less than 1
    // so we can read post ops data only once per brgemm call
    // For ACE the ZMM registers are used intensively to load data from A and
    // B, which prevents keeping the post-ops data until post-processing.
    if (brg.ldb2 > 1 || brg.is_ace()) {
        prepare_post_ops_registers_once_ = false;
    } else if (brg.ldb2 == 1) {
        if (brg.ldb2_tail == 0 && brg.ldb_tail == 0) {
            prepare_post_ops_registers_once_ = true;
            bi.ldi = &(imap_[true].ldis[0]);
            prepare_post_ops_registers(bi);
        }
    } else if (brg.ldb2_tail > 0) {
        if (brg.ldb_tail == 0) {
            prepare_post_ops_registers_once_ = true;
            bi.ldi = &(imap_[true].ldis[0]);
            prepare_post_ops_registers(bi);
        }
    } else {
        prepare_post_ops_registers_once_ = true;
        bi.ldi = &(imap_[true].ldis[0]);
        prepare_post_ops_registers(bi);
    }
    assert(IMPLICATION(brg.is_ace(), !prepare_post_ops_registers_once_));
    if (bi.apply_postops)
        dt_requires_saturation_ = one_of(
                brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    else {
        // if (brg.is_int8 && alpha_or_beta_applicable && !beta_uses_vadd) ->
        // accumulated values are converted to ps in apply_alpha_beta()
        const bool alpha_or_beta_applicable
                = brg.alpha != 1.0f || brg.beta != 0.f;
        const bool beta_uses_vadd = brg.beta == 1.f
                && IMPLICATION(brg.is_int8, brg.alpha == 1.0f);
        dt_requires_saturation_ = brg.is_int8 && !brg.has_per_k_scales()
                && !IMPLICATION(alpha_or_beta_applicable, beta_uses_vadd);
    }
    use_sat_cvt_ = dt_requires_saturation_
            && isa_has_sat_cvt(brg.isa_impl, brg.dt_d);
    if (dt_requires_saturation_) {
        init_saturate_f32(zmm_lbound, zmm_ubound, reg_tmp_gpr, data_type::f32,
                brg.dt_d, false, use_sat_cvt_);
    }

    if (bi.skip_accumulation) return;
    prf0A.set(brgemm_prf0, brg.prfA.dist0);
    prf1A.set(brgemm_prf1, brg.prfA.dist1);
    prf2A.set(brgemm_prf2, brg.prfA.dist2);
    prfntaA.set(brgemm_prfNTA, brg.prfA.distNTA);

    prf0B.set(brgemm_prf0, brg.prfB.dist0);
    prf1B.set(brgemm_prf1, brg.prfB.dist1);
    prf2B.set(brgemm_prf2, brg.prfB.dist2);
    prfntaB.set(brgemm_prfNTA, brg.prfB.distNTA);

    prf0C.set(brgemm_prf0, brg.prfC.dist0);
    prf1C.set(brgemm_prf1, brg.prfC.dist1);
}

void jit_brgemm_amx_uker_t::generate() {
    preamble();

    sub(rsp, regscratchpad_.Size());

    const auto tail_mask = size_t((1 << brg.ldb_tail) - 1);
    reg64_t reg_mask = rbx;

    // ld_full_mask is k0, which encodes as "no masking", so it is never read.
    mov(reg_mask, tail_mask);
    kmovq(ld_tail_mask, reg_mask);

    if (brg.is_ace()) {
        // Constant row masks of the ACE A load. ACE fp8 goes straight into
        // the outer product, so the fp8 up-convert paths that would clobber
        // these opmasks are never taken.
        assert(!brg.is_fp8_via_convert());
        // The fp8 post-op converters alias k2/k4 as well, but on ACE they
        // only ever take their native branches (max_cpu_isa is avx10_2_ace,
        // which is a superset of avx10_2_aux), and those touch no opmask.
        assert(IMPLICATION(
                brg.is_fp8, is_superset(max_cpu_isa(), avx10_2_aux)));
        // ld_scale_tail_mask aliases rd_tail_mask, which only the AMX k-tail
        // path writes. brgemm_desc_finalize() rejects that combination.
        assert(IMPLICATION(brg.is_mxfp8_ace, !brg.amx_wary_k_tail()));
        if (is_mxfp8_compute()) {
            // The BSR selector has 3 bits per operand and the A field spans
            // two bd blocks, so the blocking must stay within these bounds.
            // brgemm_blocking() caps ld_block2 accordingly.
            assert(brg.bd_block2 <= 2 && brg.ld_block2 <= b_scales_perm_groups);
            // The repacked A scales and the B scales are addressed per
            // (M_blk, K_blk) block with no batch term, so a batch of more
            // than one would read the same scales for every batch element.
            assert(brg.brgattr.max_bs == 1 && !brg.brgattr.var_bs);
        }
        // With MXFP8 dst quantization the masks are emitted by rdb_loop().
        if (!need_to_output_mxfp8()) emit_ace_load_A_masks();
    }

    LDA_size_ = static_cast<dim_t>(brg.typesize_A) * brg.LDA;
    LDB_size_ = static_cast<dim_t>(brg.typesize_B) * brg.LDB;
    LDC_size_ = static_cast<dim_t>(brg.typesize_C) * brg.LDC;
    LDD_size_ = static_cast<dim_t>(brg.typesize_D) * brg.LDD;

    LDA2_size_ = static_cast<dim_t>(brg.typesize_A) * brg.LDA2;
    LDB2_size_ = static_cast<dim_t>(brg.typesize_B) * brg.LDB2;
    LDC2_size_M_ = static_cast<dim_t>(brg.typesize_C) * brg.LDC2_M;
    LDC2_size_N_ = static_cast<dim_t>(brg.typesize_C) * brg.LDC2_N;

    ld_block_B_size_ = static_cast<dim_t>(brg.typesize_B)
            * ((brg.brgattr.LDB2 != 0) ? brg.brgattr.LDB2 : brg.ld_block);
    // tilestored stride for the f32 AMX accumulator tile (4-byte elements,
    // independent of logical dt_c).
    ld_block_C_size_ = static_cast<dim_t>(brgemm_desc_t::amx_c_tile_elem_size)
            * brg.ld_block;
    ld_block_D_size_ = static_cast<dim_t>(brg.typesize_D) * brg.ld_block;
    ld_block_bias_size_ = static_cast<dim_t>(brg.typesize_bias) * brg.ld_block;
    if (brg.with_wei_scales) {
        ld_block_scales_size_
                = static_cast<dim_t>(types::data_type_size(brg.dt_wei_scales))
                * brg.ld_block;
    }
    ld_block_zp_size_ = static_cast<dim_t>(sizeof(int32_t)) * brg.ld_block;
    ldb_tail_B_size_ = static_cast<dim_t>(brg.typesize_B) * brg.ldb_tail;
    ldb_tail_C_size_ = static_cast<dim_t>(brg.typesize_C) * brg.ldb_tail;
    ldb_tail_D_size_ = static_cast<dim_t>(brg.typesize_D) * brg.ldb_tail;
    ldb_tail_zp_size_ = static_cast<dim_t>(sizeof(int32_t)) * brg.ldb_tail;

    // if beta == 1 and C datatype is f32 it is better to perform addition by
    // reading tiles directly from C instead of by reading/writing by vectors
    // ACE palette does not support tileloadd, so we must use vector path
    // Per-K scales / per-(M,N) compensation must be applied to the current
    // K-block's partial result only, so previous results must not be
    // pre-loaded into the tile accumulators.
    may_load_accumulators_ = one_of(brg.alpha, 0.f, 1.f) && brg.beta == 1.f
            && !brg.is_ace() && brg.dt_c == brg.dt_d && !brg.has_per_k_scales()
            && !brg.with_per_mn_compensation
            && IMPLICATION(brg.is_input_convert(), brg.is_fp8_via_convert())
            && IMPLICATION(
                    brg.is_f32 || brg.is_bf16, brg.dt_c == data_type::f32)
            && IMPLICATION(brg.is_int8, brg.is_integer_acc())
            && brg.brgattr.bd_mask_level == 0;
    need_to_apply_alpha_beta_
            = (brg.beta != 0.f && !may_load_accumulators_) || brg.alpha != 1.f;
    are_post_ops_applicable_ = brg.are_post_ops_applicable();

    assert(IMPLICATION(brg.brgattr.LDB2 == 0, brg.load_dim <= brg.LDB));

    assert(IMPLICATION(brg.brgattr.var_bs,
            IMPLICATION(brg.is_input_convert(), brg.is_fp8_via_convert())));
    read_params();
    prepare_bd_mask();

    Label permute_index_table;
    if (brg.is_input_convert() || brg.amx_wary_k_tail() || brg.fused_copy_a) {
        // save tiles description for later use
        brgemm_init_tiles(brg, (char *)(&palette_));
        // load permute indices
        if (brg.is_bf32)
            vmovups(zmm_bf32_permute, ptr[rip + permute_index_table]);
    }

    mov(reg_stride_lda, lda() * (brg.is_ace() ? ace_zmms_per_bd_block : 1));
    mov(reg_stride_ldb, ldb());

    bool non_postops_generate
            = !are_post_ops_applicable_ || !brg.brgattr.postops_only;
    brgemm_iteration_t bi;

    // Per-compute-path ACE fp8 setup. It is emitted here rather than once up
    // front because the skip-accumulation path performs no outer product at
    // all, and its caller is not required to have brought the ACE state up.
    //  * unscaled fp8 expresses its product as the MX one over an all-ones
    //    Block Scale Register, so BSR0 is primed with bsrinit (every byte
    //    0x7F, i.e. an e8m0 exponent of 1.0). MXFP8 overwrites BSR0 on every
    //    K window, so priming it there would be dead work.
    //  * MXFP8 needs the B-scale permutation constant, which stays live for
    //    the whole compute path.
    auto init_ace_fp8_regs = [&]() {
        if (!brg.is_fp8 || !brg.is_ace()) return;
        if (is_mxfp8_compute())
            vmovdqu8(zmm_wei_scale_permute,
                    ptr[rip + b_scales_perm_index_table]);
        else
            bsrinit(bsr0);
    };

    Label label_to_ret;
    if (are_post_ops_applicable_) {
        Label label_store_without_post_ops;
        mov(reg_do_post_ops, ptr[param1 + GET_OFF(do_post_ops)]);
        cmp(reg_do_post_ops, 0);
        jz(label_store_without_post_ops, T_NEAR);
        bi.apply_postops = true;
        if (brg.with_binary) {
            // Scalar RHS values don't change within a call. Load them once
            // for both post-ops loops below instead of per stored group.
            injector_utils::vmm_index_set_t po_rhs_vmm_idxs;
            for (int i = 0; i < n_zmm_po_rhs(); i++)
                po_rhs_vmm_idxs.insert(zmm_po_rhs(i).getIdx());
            bi.preloaded_po_rhs
                    = postops_injector_->preload_scalar_vector_range(
                            po_rhs_vmm_idxs);
        }
        if (brg.brgattr.generate_skip_accumulation) {
            brgemm_iteration_t bi1;
            mov(reg_do_skip_accum, ptr[param1 + GET_OFF(skip_accm)]);
            cmp(reg_do_skip_accum, 0);
            Label label_do_not_skip_acc;
            jz(label_do_not_skip_acc, T_NEAR);

            bi1.skip_accumulation = true;
            bi1.apply_postops = true;
            bi1.preloaded_po_rhs = bi.preloaded_po_rhs;
            top_loop(bi1);
            jmp(label_to_ret, T_NEAR);

            L(label_do_not_skip_acc);
        }
        init_ace_fp8_regs();
        top_loop(bi);
        if (non_postops_generate) jmp(label_to_ret, T_NEAR);
        transform_buf_map_A_.clear();
        transform_buf_map_B_.clear();
        L(label_store_without_post_ops);
    }
    if (non_postops_generate) {
        bi.apply_postops = false;
        init_ace_fp8_regs();
        top_loop(bi);
    }
    L(label_to_ret);

    // Every deferred micro-kernel body must have been emitted by top_loop();
    // an undrained one would leave an unbound label referenced by a `call`.
    assert(deferred_uk_bodies_.empty());

    add(rsp, regscratchpad_.Size());

    postamble();

    if (brg.with_eltwise || brg.with_sum)
        postops_injector_->prepare_table(/* generate = */ true);

    if (brg.is_fp8_via_convert()) {
        if (f8_e5m2_cvt_) f8_e5m2_cvt_->prepare_table();
        if (f8_e4m3_cvt_) f8_e4m3_cvt_->prepare_table();
    }

    if (brg.is_bf32) {
        align(64);
        L(permute_index_table);
        const uint16_t _idx[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6,
                22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
                15, 31};
        for (size_t i = 0; i < 32; ++i)
            dw(_idx[i]);
    }

    if (is_mxfp8_compute()) {
        align(64);
        L(b_scales_perm_index_table);
        for (int i = 0; i < b_scales_perm_group_size; ++i)
            for (int j = 0; j < b_scales_perm_groups; ++j)
                db(i + b_scales_perm_group_size * j);
    }

    if (need_to_output_mxfp8()) emit_mxfp8_tables();
}

brgemm_kernel_t *create_brgemm_amx_uker_kernel(const brgemm_desc_t &brg) {
    assert(brg.can_dispatch_uker());
    return new jit_brgemm_amx_uker_t(brg);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
