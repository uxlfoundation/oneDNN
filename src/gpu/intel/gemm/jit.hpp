/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_JIT_HPP
#define GPU_INTEL_GEMM_JIT_HPP

#include <assert.h>
#include <limits>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/compute/zero_pool.hpp"
#include "gpu/intel/gemm/exec_types.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/gemm/jit/jit_gemm_pd.hpp"
#include "gpu/intel/gemm/utils.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

struct gen_t : public intel::primitive_t {
    struct pd_t : public jit::jit_gemm_pd_t {
        using jit::jit_gemm_pd_t::jit_gemm_pd_t;
        using kernel_desc_t = jit::gen_nocopy_desc_t;

        DECLARE_COMMON_PD_T("jit:gemm:any", gen_t);

        status_t init(impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;
            using namespace primitive_kind;
            using namespace alg_kind;
            using smask_t = primitive_attr_t::skip_mask_t;
            using arch_t = compute::gpu_arch_t;

            assert(engine->kind() == engine_kind::gpu);
            auto *intel_engine = utils::downcast<intel::engine_t *>(engine);

            // Collapse broadcast-batch dims to 2D / 3D BEFORE resolving
            // format_any. With concrete user-supplied tags the kernel-view
            // mds and attr in kernel_input_ land in the kernel-view shape
            // that catalog lookup expects; format_any falls through unchanged
            // (memory_desc_reshape rejects, bcast_ok=false). At end of init
            // commit_resolved_tags reflects resolved tags back to the user-
            // facing src_md_/weights_md_/dst_md_/bias_md_.
            CHECK(maybe_reshape_2d());

            // Resolve format_any before apply_swap_ab().
            CHECK(init_default_formats(false));

            dev_info_ = intel_engine->device_info();
            arch_ = dev_info_->gpu_arch();

            // Decide whether to keep matmul-natural orientation (un-swap) or
            // route through apply_swap_ab(). Un-swap is correct when the
            // kernel can compute the result directly in matmul-natural form:
            //   (a) skinny-N (gemm_n()==1) — column-vs-row distinction is trivial.
            //   (b) column-major user DST — kernel can write col-major C
            //       directly without the row-major-via-swap workaround.
            // Pre-swap, kernel-A = matmul SRC, kernel-B = matmul WEIGHTS,
            // kernel-C = matmul DST — read directly from kernel_input_ mds.
            const auto user_ldc
                    = jit::jit_gemm_pd_t::get_ld(*kernel_dst_md());
            const auto user_c_trans
                    = jit::jit_gemm_pd_t::get_trans(*kernel_dst_md());
            const auto natural_transb = gemm_transb();
            const auto natural_ldb
                    = jit::jit_gemm_pd_t::get_ld(*kernel_weights_md());
            const bool check_ldb = ((natural_transb == transpose::trans
                                            && natural_ldb == 1)
                    || (natural_transb == transpose::notrans));
            bool want_un_swap = (gemm_n() == 1 && user_ldc == 1 && check_ldb)
                    || (user_c_trans == transpose::trans);

            // Weights-only compression: catalog only has skinny-N for quant-in-A.
            want_un_swap &= !wei_decomp();

            // When matmul-WEIGHTS is INT4 (s4/u4) and we don't take the
            // wei_decomp path, the int-DT check at line 179-181 requires
            // gemm_b ∈ {u8, s8}. Un-swap puts gemm_b = matmul-WEIGHTS = INT4,
            // which the check rejects. Force the swap so gemm_a takes the
            // INT4 operand (matching base's gemm_a = matmul-WEIGHTS under
            // base.swap_ab=1 for cases like s8:u4:f16, s8:s4:f16, etc.).
            const bool wei_int4 = utils::one_of(
                    kernel_weights_md()->data_type, data_type::s4,
                    data_type::u4);
            want_un_swap &= !wei_int4;

            // Skinny-K (K==1) requires the column-major kernel: the matmul-
            // natural un-swap puts the large matmul-M-dim on kernel-A, after
            // which the (gemm_k()==1 && !gemm_trans_a()) pad below sets gemm_lda_ to 16 even
            // though actual stride is 1 → kernel over-reads matmul-SRC by 16x
            // and dst is garbage. Base always swaps for K==1 (its apply_swap_ab
            // mapped kernel-A to matmul-WEIGHTS, where pad-lda lands on a 1-row
            // matrix and is benign). Mirror that here, EXCEPT when DST is
            // col-major: in that case the VDISPATCH below requires un_swap, so
            // we keep un_swap on but suppress the lda pad in the K==1 + un_swap
            // path (the pad assumes lda is K-stride; for matmul-SRC col-major
            // K==1 it is M-stride and padding it desynchronizes the kernel's
            // row reads from the actual matmul-SRC layout).
            const bool keep_un_swap_for_colmajor_dst
                    = (user_c_trans == transpose::trans);
            want_un_swap &= (gemm_k() > 1) || keep_un_swap_for_colmajor_dst;

            // No kernels with transposed C if un-swap is disabled (e.g. due
            // to wei_decomp() or K==1) — this case cannot be handled.
            VDISPATCH_JIT_GEMM(
                    IMPLICATION(user_c_trans == transpose::trans, want_un_swap),
                    VERBOSE_UNSUPPORTED_TAG);

            if (!want_un_swap) CHECK(apply_swap_ab());

            // gen_t reads gemm_lda_/gemm_ldb_/gemm_transa_/gemm_transb_ for the skinny-N flip and
            // ld-pad logic below. Other impls (xe_hp_systolic, conv_t) don't
            // use these and can't safely run init_kernel_lds because get_ld
            // asserts on packed mds. See jit_gemm_pd.hpp init_kernel_lds doc.
            init_kernel_lds();

            CHECK(jit::jit_gemm_pd_t::init(engine, arch_));

            // Basic implementation attr support:
            auto attr_skip_mask = smask_t::post_ops | smask_t::fpmath_mode
                    | smask_t::accumulation_mode | smask_t::rounding_mode
                    | smask_t::scales | smask_t::scales_data_type
                    | smask_t::scales_groups | smask_t::precomputed_reductions
                    | smask_t::zero_points | smask_t::zero_points_data_type
                    | smask_t::zero_points_groups;
            VDISPATCH_JIT_GEMM(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_JIT_GEMM(!has_runtime_dims_or_strides(),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);

            auto m_ = gemm_m();
            auto n_ = gemm_n();

            // NOTE: previously this block had a `gemm_transb_ && n_ == 1` flip that
            // set gemm_transb_=false, gemm_ldb_=gemm_k(). It was inherited from a base
            // intermediate commit (8e31cc6245) that base itself later reverted
            // — base's HEAD has only the swap_ab_-gated A-side flip. The
            // worktree flip silently set ldb to a value assuming K-stride=1,
            // which is wrong for any matmul-SRC whose K-axis isn't contiguous
            // (e.g. `--stag=cab` with M=1, batch>1: K-stride=batch, not 1 →
            // GPU page fault from kernel reading OOB with stride-1 access).

            // Matmul-natural skinny-N (swap_ab_=false, N==1) mirrors base's
            // `if (!gemm_transa_ && m==1) { gemm_transa_=true; gemm_lda_=gemm_k(); }` under base's
            // swap_ab_=true. base's gemm-side m maps to matmul N; base's gemm_lda_
            // is the ld of base-A (= matmul-WEIGHTS after apply_swap_ab), so
            // the matmul-side mirror (no swap) targets matmul-WEIGHTS =
            // kernel-B: force trans-B and set ldb=gemm_k(). Without this, the
            // pad-ldb branch below (k==1 && !gemm_trans_b()) padded a 1-element
            // K-stride to 16 while base (with trans_a=true) skipped the pad —
            // the kernel sees a non-1 ld for a 1-element K dim and corrupts
            // output. Only fires when WEIGHTS is naturally notrans; otherwise
            // the natural ldb is already correct. Original buggy version
            // touched gemm_transa_/gemm_lda_, which collided with batched-non-skinny
            // N=1 cases where lda was the cache-padded matmul-SRC stride
            // (e.g., 2x10x30:2x30x1 with cache-aligned SRC stride=32 was
            // overwritten to lda=K=30 → wrong reads).
            if (!swap_ab_ && n_ == 1 && !gemm_trans_b()) {
                gemm_transb_ = true;
                gemm_ldb_ = gemm_k();
            }

            // Pad leading dimensions in case of a single row/column.
            // Under !swap_ab_ + K==1, kernel-A is matmul-SRC (full M dim) and
            // gemm_lda_ here is the M-stride (per get_ld's notrans degenerate-
            // K=1 branch), NOT the K-stride. Padding that to 16 desynchronizes
            // the kernel's row reads from the actual matmul-SRC layout (gotcha
            // #16 / #33). Skip the pad in that orientation.
            if ((gemm_k() == 1 && !gemm_trans_a() && swap_ab_)
                    || (m_ == 1 && gemm_trans_a())) {
                gemm_lda_ = utils::rnd_up(gemm_lda_, 16);
            }

            if ((n_ == 1 && !gemm_trans_b()) || (gemm_k() == 1 && gemm_trans_b())) {
                gemm_ldb_ = utils::rnd_up(gemm_ldb_, 16);
            }

            // Check parameters.
            if (utils::one_of(c_type(), s32, f16, bf16, f32, u8, s8)
                    && utils::one_of(gemm_a_type(), u8, s8, u4, s4)) {
                VDISPATCH_JIT_GEMM(
                        (utils::one_of(gemm_b_type(), u8, s8) || wei_decomp()),
                        VERBOSE_UNSUPPORTED_DT);

                VDISPATCH_JIT_GEMM(IMPLICATION(utils::one_of(c_type(), f32, s8,
                                                       u8, f16, bf16),
                                           arch_ >= arch_t::xe_hp),
                        VERBOSE_ISA_DT_MISMATCH);
            } else if (utils::one_of(gemm_a_type(), f16, bf16)) {
                VDISPATCH_JIT_GEMM(gemm_b_type() == gemm_a_type() || wei_decomp(),
                        VERBOSE_INCONSISTENT_DT, "a", "b");
                VDISPATCH_JIT_GEMM(utils::one_of(c_type(), gemm_a_type(), f32,
                                           f8_e5m2, f8_e4m3),
                        VERBOSE_INCONSISTENT_DT, "a", "c");
                VDISPATCH_JIT_GEMM(utils::one_of(acc_type(), gemm_a_type(), f32),
                        VERBOSE_INCONSISTENT_DT, "a", "acc");
            } else if (!wei_decomp()) {
                VDISPATCH_JIT_GEMM(utils::one_of(gemm_a_type(), f64, f32, f16, bf16,
                                           f8_e5m2, f8_e4m3, f4_e2m1, f4_e3m0),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_JIT_GEMM(
                        (gemm_b_type() == gemm_a_type()
                                || (utils::one_of(gemm_a_type(), f8_e5m2, f8_e4m3)
                                        && utils::one_of(
                                                gemm_b_type(), f8_e5m2, f8_e4m3))
                                || (utils::one_of(gemm_a_type(), f4_e2m1, f4_e3m0)
                                        && utils::one_of(gemm_b_type(), f4_e2m1,
                                                f4_e3m0))),
                        VERBOSE_INCONSISTENT_DT, "a", "b");
                VDISPATCH_JIT_GEMM(utils::one_of(acc_type(), gemm_a_type(), f32),
                        VERBOSE_UNSUPPORTED_DT);
                VDISPATCH_JIT_GEMM(IMPLICATION(utils::one_of(f64, gemm_a_type(),
                                                       gemm_b_type()),
                                           dev_info_->has_native(f64)),
                        VERBOSE_UNSUPPORTED_DT);
            }

            VDISPATCH_JIT_GEMM(!has_blocks(), VERBOSE_BLOCKING_FAIL, "");
            VDISPATCH_JIT_GEMM(
                    batch_dims() <= 4, VERBOSE_BAD_DIM, "batch", batch_dims());
            VDISPATCH_JIT_GEMM(intel_engine->mayiuse_ngen_kernels(),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "ngen_kernels");

            VDISPATCH_JIT_GEMM(utils::one_of(bias_type(), data_type::undef, f64,
                                       f32, bf16, f16, f8_e5m2, f8_e4m3)
                            && (kernel_bias_md()->ndims <= 6)
                            && gemm_bias_mask() < 8,
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_JIT_GEMM(
                    IMPLICATION(with_bias(),
                            (c_type() != f64 || bias_type() == f64)),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_JIT_GEMM(IMPLICATION(with_sum_ab(),
                                       !with_bias()
                                               && (attr()->zero_points_.has_default_values(
                                                       kC))),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_JIT_GEMM(attr()->post_ops_.check_sum_consistency(c_type(),
                                       utils::one_of(gemm_a_type(), s8, u8)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            auto c_kernel_type = jit::convert_dnnl_to_kernel_type(c_type());
            {
                const auto &kdst = *kernel_dst_md();
                for (int i = 0; i < kdst.ndims; i++) {
                    auto c_stride = kdst.format_desc.blocking.strides[i];
                    VDISPATCH_JIT_GEMM(
                            IMPLICATION(c_kernel_type.is4(),
                                    c_stride == 1 || c_stride % 2 == 0),
                            VERBOSE_SHAPE_RESTRICTION);
                }
            }

            const auto &k_post_ops = *kernel_post_ops();
            bool with_binary = (k_post_ops.find(binary) != -1)
                    || (k_post_ops.find(prelu) != -1);
            bool with_eltwise = (k_post_ops.find(eltwise) != -1);

            // Check GPU architecture.
            bool arch_ok = utils::one_of(arch_, arch_t::xe_lp, arch_t::xe_hp,
                    arch_t::xe_hpg, arch_t::xe_hpc, arch_t::xe2, arch_t::xe3);
            arch_ok |= (arch_ >= arch_t::xe3p);

            VDISPATCH_JIT_GEMM(arch_ok, VERBOSE_UNSUPPORTED_ARCH, "gpu");
            VDISPATCH_JIT_GEMM(IMPLICATION(with_binary, arch_ >= arch_t::xe_hp),
                    VERBOSE_UNSUPPORTED_ARCH, "gpu");

            // Grouped scales break pre-XeHPG kernels due to increased register
            // pressure. Read grouped-ness via swap-aware pd helper (matmul-side
            // attr); no pd-side intermediate state survives past init.
            VDISPATCH_JIT_GEMM(IMPLICATION(arch_ == compute::gpu_arch_t::xe_lp,
                                       !(a_grouped() || b_grouped())),
                    VERBOSE_UNSUPPORTED_FEATURE, "grouped scales");

            bool has_systolic
                    = intel_engine->mayiuse(compute::device_ext_t::
                                      intel_subgroup_matrix_multiply_accumulate)
                    || intel_engine->mayiuse(compute::device_ext_t::
                                    intel_subgroup_split_matrix_multiply_accumulate);

            bool is_integrated = dev_info_->is_integrated();

            // Size checks for fused reduction kernels.
            if (with_sum_ab()) {
                auto mnk = gemm_m() * gemm_n() * gemm_k();
                if (arch_ == arch_t::xe_hpc && gemm_a_type() == f32)
                    VDISPATCH_JIT_GEMM(
                            (mnk <= 256 * 1024 * 1024), VERBOSE_LARGE_SHAPES);
            }

            // Handle special compute modes.
            kernel_desc_t::compute_mode mode = kernel_desc_t::mode_default;

            if (attr()->mayiconvert(f32, tf32))
                set_mode(mode, kernel_desc_t::mode_tf32);
            if (attr()->mayiconvert(f32, bf16))
                set_mode(mode, kernel_desc_t::mode_bf16x1);
            if (attr()->mayiconvert(f32, f16))
                set_mode(mode, kernel_desc_t::mode_f16x1);
            if (attr()->mayiconvert(f32, f32))
                set_mode(mode, kernel_desc_t::mode_strict);
            if (attr()->deterministic_)
                set_mode(mode, kernel_desc_t::mode_deterministic);
            if (attr()->acc_mode_ == accumulation_mode::relaxed)
                set_mode(mode, kernel_desc_t::mode_relaxed_acc);

            if (wei_decomp()) { set_mode(mode, kernel_desc_t::mode_w_decomp); }

            // GEMM kernels down-convert the following parameters to
            // int/uint32_t.
            VDISPATCH_JIT_GEMM(std::max({m_, n_, gemm_k(), batch()})
                            <= std::numeric_limits<int32_t>::max(),
                    VERBOSE_SHAPE_RESTRICTION);
            VDISPATCH_JIT_GEMM(std::max({gemm_ld(kA), gemm_ld(kB), gemm_ld(kC)})
                            <= std::numeric_limits<uint32_t>::max(),
                    VERBOSE_SHAPE_RESTRICTION);

            gemmstone::GEMMProblem problem;
            CHECK(init_GEMMProblem(problem, intel_engine));

            VDISPATCH_JIT_GEMM(IMPLICATION(problem.Tc == gemmstone::Type::f64,
                                       !with_eltwise && !with_binary),
                    VERBOSE_UNSUPPORTED_POSTOP);

            if (arch_ >= arch_t::xe3p)
                kernel_desc_.set_efficient_64b(dev_info_->is_efficient_64bit());

            bool print_verbose = get_verbose(verbose_t::debuginfo) >= 5;
            bool kernel_success = false;
            auto lda = gemm_ld(kA);
            auto ldb = gemm_ld(kB);
            auto product = intel_engine->device_info()->gpu_product();
            int stepping = dev_info_->stepping_id();
            auto entries = kernel_desc_.select_kernel(product, stepping,
                    dev_info_->eu_count(), has_systolic, is_integrated, mode,
                    problem, alpha(), beta(), m_, n_, gemm_k(), lda, ldb,
                    gemm_ldc(), batch());

            for (auto &entry : entries) {
                kernel_desc_.set_entry(entry);
                kernel_desc_.set_problem(problem);
                auto status = kernel_desc_.finalize();
                bool valid = status == status::success;
                if (!valid && print_verbose)
                    dnnl::impl::verbose_printf(
                            "info,gpu,gemm,skipping:%s,Strategy finalization "
                            "failed.\n",
                            kernel_desc_.entry().str().c_str());
                if (kernel_desc_.driver_info()->kParallel()
                        && !kernel_desc_.driver_info()->fusedPostOps()) {
                    bool po_valid = !non_scale_po()
                            && !(with_sum() && with_c_scales())
                            && utils::one_of(c_type(), f32, s32);
                    if (!po_valid && print_verbose)
                        dnnl::impl::verbose_printf(
                                "info,gpu,gemm,skipping:%s,Invalid post op.\n",
                                kernel_desc_.entry().str().c_str());
                    valid &= po_valid;
                }
                if (kernel_desc_.problem()->Tc.size() < 4) {
                    bool need_x32_acc = with_binary
                            || !IMPLICATION(with_sum(), sum_at_begin());
                    valid &= !need_x32_acc;
                    if (need_x32_acc && print_verbose)
                        dnnl::impl::verbose_printf(
                                "info,gpu,gemm,skipping:%s,Invalid post op.\n",
                                kernel_desc_.entry().str().c_str());
                }
                if (attr()->deterministic_) {
                    bool deterministic
                            = !kernel_desc_.driver_info()->nondeterministic();
                    valid &= deterministic;
                    if (!deterministic && print_verbose)
                        dnnl::impl::verbose_printf(
                                "info,gpu,gemm,skipping:%s,Non deterministic "
                                "kernel.\n",
                                kernel_desc_.entry().str().c_str());
                }

                if (valid) {
                    auto try_create = [&]() {
                        std::vector<compute::kernel_t> kernel_(1);
                        auto *intel_engine
                                = utils::downcast<intel::engine_t *>(engine);
                        auto key = std::make_shared<
                                trivial_key_container_t<dnnl::impl::gpu::intel::
                                                gemm::jit::gen_nocopy_desc_t>>(
                                kernel_desc_, intel_engine->engine_id());
                        cache_state_t kernel_cache_status;
                        auto kernel_name = "gemm_kernel";
                        auto verbose
                                = get_verbose(verbose_t::create_profile) >= 1;
                        double start_ms = 0;
                        if (verbose) start_ms = get_msec();
                        status = get_cached_kernels<typename trivial_key_t<
                                dnnl::impl::gpu::intel::gemm::jit::
                                        gen_nocopy_desc_t>::value_type>(
                                std::move(key), intel_engine, kernel_,
                                {kernel_name}, kernel_cache_status);
                        if (verbose && status == status::success) {
                            double duration_ms = get_msec() - start_ms;
                            const char *str
                                    = cache_state2str(kernel_cache_status);
                            VPROF(start_ms, primitive, create, str,
                                    info(engine), duration_ms);
                        }
                        return status;
                    };
                    status = try_create();
                    if (status == status::success) {
                        kernel_success = true;
                        break;
                    }
                }
            }

            VDISPATCH_JIT_GEMM(
                    kernel_success, "matching kernel not found in catalog");

            init_scratchpad();

            // Project resolved format tags from the kernel-view mds back
            // onto the user-facing src_md_/weights_md_/dst_md_/bias_md_ so
            // the framework sees the resolved N-D shape.
            CHECK(commit_resolved_tags());

            return status::success;
        }

        status_t query(query_t what, int idx, void *result) const override {
            switch ((int)what) {
                case (int)query::preferred_gpu_threads_per_eu: {
                    int grfs = kernel_desc_.driver_info()->grfCount;
                    *(int *)result = (grfs > 128) ? 4 : 8;
                    break;
                }
                default:
                    return jit::jit_gemm_pd_t::query(what, idx, result);
            }
            return status::success;
        }

        status_t init_default_formats(bool no_transpose_c) {
            using namespace data_type;
            using namespace format_tag;
            using arch_t = compute::gpu_arch_t;

            auto m_ = M();
            auto n_ = N();
            auto k_ = K();
            // Pre-swap: kernel-A=SRC, kernel-B=WEIGHTS. Read from kernel-view
            // mds (kernel_input_.X_md).
            auto &k_src = kernel_input_.src_md;
            auto &k_wei = kernel_input_.weights_md;
            auto &k_dst = kernel_input_.dst_md;
            auto a_t = (utils::one_of(k_src.data_type, s4, u4))
                    ? s8
                    : k_src.data_type;
            auto b_t = (utils::one_of(k_wei.data_type, s4, u4))
                    ? s8
                    : k_wei.data_type;
            auto c_t = k_dst.data_type;

            bool is_f16 = utils::everyone_is(f16, a_t, b_t, c_t);
            bool is_bf16 = utils::everyone_is(bf16, a_t, b_t, c_t);
            bool is_xe_hp_plus = arch_ >= arch_t::xe_hp;

            memory_desc_wrapper a_mdw(&k_src);
            memory_desc_wrapper b_mdw(&k_wei);
            memory_desc_wrapper c_mdw(&k_dst);

            bool a_any = a_mdw.format_any();
            bool b_any = b_mdw.format_any();
            bool c_any = c_mdw.format_any();

            if (!a_any && !is_md_gemm_compatible_plain_format(&k_src))
                return status::unimplemented;
            if (!b_any && !is_md_gemm_compatible_plain_format(&k_wei))
                return status::unimplemented;
            if (!c_any
                    && !is_md_gemm_compatible_plain_format(
                            &k_dst, no_transpose_c))
                return status::unimplemented;

            // Pre-swap "is A trans" reads from natural a_md (= matmul SRC).
            bool is_a_trans = (get_trans(k_src) == transpose::trans);
            bool is_b_trans = (get_trans(k_wei) == transpose::trans);

            auto lda_choice = is_a_trans ? m_ : k_;
            auto ldb_choice = is_b_trans ? k_ : n_;

            auto is_aligned = [](dim_t ld, data_type_t dt, int byte) {
                return types::elements_to_bytes(dt, ld) % byte == 0;
            };

            bool a_4B_aligned = is_aligned(lda_choice, a_t, 4);
            bool b_4B_aligned = is_aligned(ldb_choice, b_t, 4);
            bool ab_4B_aligned = a_4B_aligned && b_4B_aligned;

            bool a_tn_4B_aligned = is_aligned(k_, a_t, 4);
            bool b_tn_4B_aligned = is_aligned(k_, b_t, 4);
            bool ab_tn_4B_aligned = a_tn_4B_aligned && b_tn_4B_aligned;

            bool use_tn = (m_ <= 32 || n_ <= 32) && !ab_4B_aligned
                    && ab_tn_4B_aligned;

            bool batch = batched();

            auto dotrans = batch ? acb : ba;
            auto notrans = batch ? abc : ab;

            auto cache_line_align_md = [&](memory_desc_t &md) {
                dnnl::impl::dims_t dims;
                dnnl::impl::utils::array_copy(dims, md.dims, md.ndims);

                auto kernel_type
                        = jit::convert_dnnl_to_kernel_type(md.data_type);
                size_t stride = [&](dim_t dim) {
                    auto stride = dim * kernel_type;

                    // Prefer cache line aligned sizes
                    if (stride > 32) {
                        stride = utils::rnd_up(stride, 64);
                        // Avoid conflicts in 8-way associative cache
                        if (stride % 256 == 0) stride += 64;
                        return stride / kernel_type;
                    }

                    int load_alignment = arch_ > arch_t::xe2 ? 16 : 4;
                    if (stride > load_alignment / 2)
                        return utils::rnd_up(stride, load_alignment)
                                / kernel_type;

                    return utils::rnd_up_pow2(stride) / kernel_type;
                }(md.dims[md.ndims - 1]);

                dnnl::impl::dims_t strides;
                strides[md.ndims - 1] = 1;
                strides[md.ndims - 2] = stride;
                for (int i = md.ndims - 3; i >= 0; i--)
                    strides[i] = strides[i + 1] * dims[i + 1];

                CHECK(memory_desc_init_by_strides(
                        md, md.ndims, dims, md.data_type, strides));
                return status::success;
            };
            if (a_any) CHECK(cache_line_align_md(k_src));
            if (b_any) CHECK(cache_line_align_md(k_wei));

            if ((is_f16 || is_bf16) && is_xe_hp_plus && use_tn) {
                if (a_any && b_any) {
                    CHECK(memory_desc_init_by_tag(k_src, dotrans));
                    CHECK(memory_desc_init_by_tag(k_wei, notrans));
                } else if (a_any && !is_b_trans) {
                    CHECK(memory_desc_init_by_tag(k_src, dotrans));
                } else if (b_any && is_a_trans) {
                    CHECK(memory_desc_init_by_tag(k_wei, notrans));
                }
            }

            return gemm_set_default_formats() ? status::success
                                              : status::unimplemented;
        }

        void init_scratchpad() {
            using namespace gemmstone;
            const auto *info = kernel_desc()->driver_info();
            if (info->needsTempC()) {
                auto scratchpad = scratchpad_registry().registrar();

                int temp_c_sz = nstl::max(
                        (int)types::data_type_size(c_type()), 4);
                int temp_c_elems = info->wgTile(LoopM) * info->wgTile(LoopN);
                if (with_sum_ab())
                    temp_c_elems += nstl::max(
                            info->wgTile(LoopM), info->wgTile(LoopN));
                temp_c_elems = utils::rnd_up(temp_c_elems, 64);
                temp_c_elems *= max_k_sliced_groups();

                scratchpad.book(memory_tracking::names::key_gemm_accumulator,
                        temp_c_elems, temp_c_sz, 64, 65536);
            }
        }

        const jit::gen_nocopy_desc_t *kernel_desc() const {
            return &kernel_desc_;
        }

        int max_k_sliced_groups() const {
            const auto *info = kernel_desc()->driver_info();
            bool large_grf_mode = (info->grfCount > 128);

            auto groups = dev_info_->hw_threads(large_grf_mode)
                    / (info->wg[gemmstone::LoopM] * info->wg[gemmstone::LoopN]);
            if (info->kParallelVariable()) groups *= 2;

            return groups;
        }

        size_t dyn_offset_a = 0;
        size_t dyn_offset_b = 0;
        size_t dyn_offset_c = 0;
        size_t dyn_offset_co = 0;

        const compute::device_info_t *dev_info_ = nullptr;

        kernel_desc_t kernel_desc_;
    };

    gen_t(const pd_t *apd) : intel::primitive_t(apd) {}

    ~gen_t() override {
        if (zero_pool_) release_zero_pool(zero_pool_);
    }

    status_t init(impl::engine_t *engine) override {
        return init_nocopy(engine);
    }

    status_t init_nocopy(impl::engine_t *engine) {
        using namespace data_type;
        auto kd = pd()->kernel_desc();

        CHECK(create_kernel(engine, nocopy_kernel_, "gemm_kernel", *kd));

        scalar_type_ = kd->scalar_type();
        const auto *info = nocopy_info();

        if (need_zero_pool()) {
            int zg_cl = 0;
            if (info->fusedBeta()) zg_cl++;
            if (info->fusedPostOps()) zg_cl++;

            zero_pool_bytes_ = pd()->max_k_sliced_groups() * 64 * zg_cl;

            auto zg_max = pd()->dev_info_->hw_threads(false);
            zero_pool_chunk_size_ = zg_max * 2 * 2 * 64;

            auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
            CHECK(lookup_zero_pool(
                    intel_engine, nullptr, zero_pool_chunk_size_, &zero_pool_));

            nocopy_kernel_.save_output_events();
        }

        return status::success;
    }

    status_t execute(const impl::exec_ctx_t &ctx) const override;

private:
    status_t launch_nocopy(const impl::exec_ctx_t &ctx, intel::stream_t *s,
            zero_pool_t *zero_pool, const memory_storage_t &a,
            const memory_storage_t &b, const memory_storage_t &c,
            const memory_storage_t *ao, const memory_storage_t *bo,
            int16_t ao_host_scalar, int16_t bo_host_scalar,
            const memory_storage_t *a_scales, const memory_storage_t *b_scales,
            const memory_storage_t *c_scales, const memory_storage_t *ag,
            const memory_storage_t *bg, const memory_storage_t &co,
            int16_t co_host_scalar, const memory_storage_t *c_temp,
            const memory_storage_t *sround_seed, int po_count,
            const memory_storage_t **po_src, int64_t offset_a, int64_t offset_b,
            int64_t offset_c, int64_t offset_aq, int64_t offset_bq,
            int64_t offset_co, int64_t *offset_po_src, int32_t lda, int32_t ldb,
            int32_t ldc, int32_t m, int32_t n, int32_t k, int32_t k0,
            float alpha, float beta, int32_t cmask, bool last_k_block,
            bool swap_ab, bool disable_hilbert) const;

    const pd_t *pd() const {
        return (const pd_t *)intel::primitive_t::pd().get();
    }
    const gemmstone::CommonDriverInfo *nocopy_info() const {
        return pd()->kernel_desc()->driver_info();
    }

    bool need_zero_pool() const {
        return nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps();
    }

    compute::kernel_t nocopy_kernel_;
    compute::scalar_type_t scalar_type_;
    zero_pool_t *zero_pool_ = nullptr;
    size_t zero_pool_bytes_ = 0;
    size_t zero_pool_chunk_size_ = 0;
};

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
