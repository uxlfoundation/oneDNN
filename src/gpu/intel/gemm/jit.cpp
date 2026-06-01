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

#include "gpu/intel/gemm/jit.hpp"
#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gemmstone/driver_info.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/host_scalars.hpp"
#include "gpu/intel/gemm/jit/pd.hpp"
#include "gpu/intel/gemm/jit/walk_orders.hpp"
#include "gpu/intel/jit/ir/block_2d_utils.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/logging.hpp"

#ifdef DNNL_WITH_SYCL
#include "gpu/intel/sycl/stream.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

bool check_memory_storage(const memory_storage_t *storage, const char *name) {
    if (storage && *storage) return true;
    VERROR(primitive, gpu, "%s,%s: %s", "jit::gemm", "argument is not set",
            name);
    return false;
}

status_t gen_t::launch_nocopy(const exec_ctx_t &ctx,
        intel::stream_t *compute_stream, zero_pool_t *zero_pool,
        const jit::exec_config_t &exec_cfg, const memory_storage_t *c_temp,
        int po_count, const memory_storage_t **po_srcs, int64_t offset_a,
        int64_t offset_b, int64_t offset_c, int64_t offset_aq,
        int64_t offset_bq, int64_t offset_co, int64_t *offset_po_src, int32_t m,
        int32_t n, int32_t k, int32_t k0, float alpha, float beta,
        bool last_k_block, bool disable_hilbert) const {
    if (pd()->desc()->batch() == 0) return status::success;

    uint32_t flags = 0;
    bool k_parallel_fixed
            = (nocopy_info()->kParallel() || nocopy_info()->kParallelLocal())
            && !nocopy_info()->kParallelVariable();

    auto problem = pd()->kernel_desc()->problem();

    if (!last_k_block) flags |= gemmstone::FlagNonfinalKBlock;
    if (exec_cfg.cmask & 1) flags |= gemmstone::FlagCOColumn;
    if (exec_cfg.cmask & 2) flags |= gemmstone::FlagCORow;

    const auto lda = into<int32_t>(exec_cfg.lda);
    const auto ldb = into<int32_t>(exec_cfg.ldb);
    const auto ldc = into<int32_t>(exec_cfg.ldc);

    compute::kernel_arg_list_t arg_list;
    int argn = 0;

    arg_list.set(argn++, *exec_cfg.a);
    arg_list.set(argn++, *exec_cfg.b);
    arg_list.set(argn++, *exec_cfg.c);
    arg_list.set(argn++, offset_a);
    arg_list.set(argn++, offset_b);
    arg_list.set(argn++, offset_c);
    arg_list.set(argn++, lda);
    arg_list.set(argn++, ldb);
    arg_list.set(argn++, ldc);
    arg_list.set(argn++, m);
    arg_list.set(argn++, n);
    arg_list.set(argn++, k);

    set_scalar_arg_cvt(arg_list, argn++, alpha, scalar_type_);
    set_scalar_arg_cvt(arg_list, argn++, beta, scalar_type_);

    bool a_zp_ptr
            = exec_cfg.with_a_zero_points && !problem->aOffsetHostScalar();
    bool b_zp_ptr
            = exec_cfg.with_b_zero_points && !problem->bOffsetHostScalar();

    if (a_zp_ptr) arg_list.set(argn++, *exec_cfg.ao);
    if (b_zp_ptr) arg_list.set(argn++, *exec_cfg.bo);
    if (problem->aOffsetHostScalar())
        arg_list.set(argn++, exec_cfg.ao_host_scalar);
    if (problem->bOffsetHostScalar())
        arg_list.set(argn++, exec_cfg.bo_host_scalar);
    if (problem->aScale2D()) arg_list.set(argn++, *exec_cfg.a_scales);
    if (problem->bScale2D()) arg_list.set(argn++, *exec_cfg.b_scales);
    if (pd()->with_mx_scale()) arg_list.set(argn++, *exec_cfg.c_scales);
    if (problem->needsAGroupSums()) {
        if (!check_memory_storage(exec_cfg.ag, "ag"))
            return status::runtime_error;
        arg_list.set(argn++, *exec_cfg.ag);
    }
    if (problem->needsBGroupSums()) {
        if (!check_memory_storage(exec_cfg.bg, "bg"))
            return status::runtime_error;
        arg_list.set(argn++, *exec_cfg.bg);
    }

    if (problem->aOffset2D() || problem->aScale2D()
            || problem->needsAGroupSums()) {
        auto layout = problem->needsAGroupSums() ? problem->Ag.layout
                : problem->aScale2D()            ? problem->A_scale.layout
                                                 : problem->AO.layout;
        auto ldaq = into<int32_t>(isColMajor(layout)
                        ? utils::div_up(m, problem->aqGroupM)
                        : utils::div_up(exec_cfg.k, problem->aqGroupK));
        arg_list.set(argn++, ldaq);
    }
    if (problem->bOffset2D() || problem->bScale2D()
            || problem->needsBGroupSums()) {
        auto layout = problem->needsBGroupSums() ? problem->Bg.layout
                : problem->bScale2D()            ? problem->B_scale.layout
                                                 : problem->BO.layout;
        auto ldbq = into<int32_t>(!isColMajor(layout)
                        ? utils::div_up(n, problem->bqGroupN)
                        : utils::div_up(exec_cfg.k, problem->bqGroupK));
        arg_list.set(argn++, ldbq);
    }
    if (exec_cfg.with_mx_scale) {
        auto ldcq = pd()->desc()->m() / problem->cqGroupM;
        arg_list.set(argn++, ldcq);
    }
    if (problem->usesCOPtr()) {
        if (exec_cfg.co->is_null()) return status::runtime_error;
        arg_list.set(argn++, *exec_cfg.co);
        arg_list.set(argn++, offset_co);
        if (exec_cfg.with_bias) {
            auto ldco = into<int32_t>(exec_cfg.ld_bias);
            arg_list.set(argn++, ldco);
        }
    } else if (problem->cOffsetHostScalar()) {
        arg_list.set(argn++, exec_cfg.co_host_scalar);
    }
    if (nocopy_info()->needsTempC()) arg_list.set(argn++, *c_temp);
    if (problem->postOps.cStochasticRound) {
        arg_list.set(argn++, *exec_cfg.sround_seed);
    }
    arg_list.set(argn++, flags);
    if (k_parallel_fixed) arg_list.set(argn++, k0);

    for (int i = 0; i < po_count; i++) {
        if (!po_srcs[i]) continue;
        arg_list.set(argn++, *po_srcs[i]);
        arg_list.set(argn++, offset_po_src[i]);

        if (problem->postOps.binaryRow[i] && problem->postOps.binaryCol[i])
            arg_list.set(argn++, int32_t(exec_cfg.ld_binary(i)));
    }

    std::unique_ptr<memory_storage_t> zeros;
    int zp_token = 0;
    if (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps()) {
        CHECK(zero_pool->claim(
                compute_stream, zero_pool_bytes_, zeros, &zp_token));
        arg_list.set(argn++, *zeros);
    }

    if (pd()->batch_dims() >= 1) {
        for (int i = pd()->batch_dims() - 1; i >= 0; i--) {
            auto stride_a = int64_t(exec_cfg.stride_a(i));
            auto stride_b = int64_t(exec_cfg.stride_b(i));
            auto stride_c = int64_t(exec_cfg.stride_c(i));
            if (jit::enable_generator_dsl()) {
                auto hw = ngen::getCore(
                        ((ngen::Product *)&utils::downcast<intel::engine_t *>(
                                 compute_stream->engine())
                                        ->device_info()
                                        ->gpu_product())
                                ->family);

                // 2d Surface pointer needs to be 64 byte aligned. When negative
                // bounds checking is unnecessary, this restriction can be
                // relaxed by rounding down the surface pointer and adjusting
                // the width accordingly.
                auto base_alignment = intel::jit::block_2d_base_alignment(hw);
                auto a_size = types::data_type_size(exec_cfg.a_type());
                if (stride_a * a_size % base_alignment) {
                    gpu_warning() << "Unimplemented load transform";
                    return status::runtime_error;
                }
                auto b_size = types::data_type_size(exec_cfg.b_type());
                if (stride_b * b_size % base_alignment) {
                    gpu_warning() << "Unimplemented load transform";
                    return status::runtime_error;
                }
            }
            arg_list.set(argn++, stride_a);
            arg_list.set(argn++, stride_b);
            arg_list.set(argn++, stride_c);
            if (problem->hasAScalePtr())
                arg_list.set(argn++, exec_cfg.scale_stride_a(i));
            if (problem->hasBScalePtr())
                arg_list.set(argn++, exec_cfg.scale_stride_b(i));
            if (problem->hasCMXScale())
                arg_list.set(argn++, stride_c / problem->cqGroupM);
            if (problem->hasAOffsetPtr())
                arg_list.set(argn++, exec_cfg.zp_stride_a(i));
            if (problem->hasBOffsetPtr())
                arg_list.set(argn++, exec_cfg.zp_stride_b(i));
            if (problem->needsAGroupSums())
                arg_list.set(argn++, exec_cfg.gs_stride_a(i));
            if (problem->needsBGroupSums())
                arg_list.set(argn++, exec_cfg.gs_stride_b(i));
        }
        for (int i = 0; i < po_count; i++) {
            if (problem->postOps.binaryBatch[i]) {
                for (int b = pd()->batch_dims() - 1; b >= 0; b--) {
                    arg_list.set(argn++, int64_t(exec_cfg.stride_binary(i, b)));
                }
            }
        }
        for (int i = 1; i < pd()->batch_dims(); i++) {
            auto batchSize = uint32_t(pd()->desc()->c_desc.dims[i]);
            arg_list.set(argn++, batchSize);
            if (jit::enable_generator_dsl()) {
                uint64_t magic = dnnl::impl::gpu::intel::jit::ir_utils::
                        idiv_magicgu_packed(batchSize);
                arg_list.set(argn++, magic);
            } else {
                uint32_t recipBatchSize = jit::uint32_reciprocal(batchSize);
                arg_list.set(argn++, recipBatchSize);
            }
        }
    }

    auto lws_k = pd()->kernel_desc()->aux_params()->wgK;

    compute::range_t gws = compute::range_t::empty();

    gws[0] = utils::div_up(m, nocopy_info()->unroll[gemmstone::LoopM]);
    gws[1] = utils::div_up(n, nocopy_info()->unroll[gemmstone::LoopN]);
    gws[2] = nocopy_info()->kParallel() ? nstl::max(1, utils::div_up(k, k0))
                                        : lws_k;

    compute::range_t lws = {size_t(nocopy_info()->wg[gemmstone::LoopM]),
            size_t(nocopy_info()->wg[gemmstone::LoopN]), size_t(lws_k)};

    // C Interleave: pad up gws[N] to a multiple of the chunk size and add to gws[M] if misaligned ldc
    auto info = nocopy_info();
    gws[1] = utils::rnd_up(
            gws[1], info->cInterleaveChunk(problem->Tc_ext) * lws[1]);
    if (info->cInterleaveEnabled()
            && (offset_c % 64 > 0 || ldc * problem->Tc % 64 > 0)) {
        auto wgTileM = info->wgTile(gemmstone::LoopM);
        auto maxShift = 64 / problem->Tc_ext.size() - 1;
        gws[0] += lws[0] * utils::div_up(wgTileM + maxShift, wgTileM);
    }

    if (nocopy_info()->isNMK()) {
        std::swap(lws[0], lws[1]);
        std::swap(gws[0], gws[1]);
    }

    if (nocopy_info()->fusedEUs() && (lws[0] > 1))
        gws[0] = utils::rnd_up(gws[0], 2);

    lws[2] = nstl::min(lws[2], gws[2]);

    if (nocopy_info()->kParallel() && nocopy_info()->kPadding())
        gws[2] += lws[2];

    int last_non_1 = 2;
    for (; last_non_1 >= 0 && (gws[last_non_1] == 1 || lws[last_non_1] == 1);
            last_non_1--)
        ;

    for (int d = 0; d < 3; d++) {
        if (nocopy_info()->fixedWG() || (gws[d] > lws[d]))
            gws[d] = utils::rnd_up(gws[d], lws[d]);
        else {
            // Workaround to avoid local ID reordering until reqd_walk_group_order implemented in UMD.
            if (pd()->arch_ >= compute::gpu_arch_t::xe_hp && d < last_non_1)
                gws[d] = utils::rnd_up_pow2(gws[d]);
            lws[d] = gws[d];
        }
    }

    lws[1] *= nocopy_info()->wgExpand;
    gws[1] *= nocopy_info()->wgExpand;

    gws[2] *= pd()->desc()->batch();

    jit::linear_order_args(arg_list, argn, lws, gws, m, n, k, disable_hilbert,
            *nocopy_info(), pd()->kernel_desc()->aux_params(), pd()->dev_info_);

    if (nocopy_info()->perKSLM > 0) {
        size_t slm = nocopy_info()->slm;
        if (lws[2] > 1) slm = nstl::max(slm, nocopy_info()->perKSLM * lws[2]);
        arg_list.set(argn++, slm, nullptr);
    }

    // Gate by the kernel-cfg problem (not matmul-frame pd()->a/b_quant), or
    // the arg list mismatches under swap_ab with differing A/B zp_ndims.
    if (problem->aoPtrDims > 0 || problem->aScale2D())
        arg_list.set(argn++, offset_aq);
    if (problem->boPtrDims > 0 || problem->bScale2D())
        arg_list.set(argn++, offset_bq);

    lws[0] *= nocopy_info()->subgroupSize;
    gws[0] *= nocopy_info()->subgroupSize;

    auto nd_range = compute::nd_range_t(gws, lws);
    auto status = parallel_for(ctx, nd_range, nocopy_kernel_, arg_list);

    if (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps())
        zero_pool->async_release(zp_token, compute_stream->ctx().get_deps());

    return status;
}

template <typename T>
static T pick_a(bool swap_ab, T a, T b) {
    return swap_ab ? b : a;
}
template <typename T>
static T pick_b(bool swap_ab, T a, T b) {
    return swap_ab ? a : b;
}

static status_t build_exec_config(jit::exec_config_t &exec_cfg,
        const gen_t::pd_t *pd, const exec_ctx_t &ctx) {
    static_cast<jit::kernel_config_t &>(exec_cfg) = pd->cfg();
    const bool s = exec_cfg.swap_ab;

    // swap_ab survives on the execute path only as the A/B routing decision,
    // applied once per runtime ctx-arg binding via pick_a/pick_b (no second
    // std::swap). All flags read the already-folded exec_cfg.problem directly.
    exec_cfg.pd = pd;
    exec_cfg.eff_a_arg = s ? DNNL_ARG_B : DNNL_ARG_A;
    exec_cfg.eff_b_arg = s ? DNNL_ARG_A : DNNL_ARG_B;
    exec_cfg.with_mx_scale = pd->with_mx_scale();

    auto &a_phys = GEMM_CTX_ARG_STORAGE(a);
    auto &b_phys = GEMM_CTX_ARG_STORAGE(b);
    exec_cfg.a = s ? &a_phys : &b_phys;
    exec_cfg.b = s ? &b_phys : &a_phys;
    exec_cfg.c = &GEMM_CTX_ARG_STORAGE(c);
    exec_cfg.sround_seed = &GEMM_CTX_ARG_STORAGE(sround_seed);

    // exec_cfg.cmask is already kernel-frame (seeded in init_post_ops, folded in
    // swap_fold); only the co storage / off_co0 selection is runtime here.
    const memory_storage_t *co = &GEMM_CTX_ARG_STORAGE(c_zero_point);
    if (pd->with_c_zero_points()) {
        exec_cfg.off_co0
                = types::bytes_to_elements(exec_cfg.c_type(), co->offset())
                + pd->dyn_offset_co;
        if (co->is_host_scalar()) {
            int co_host = 0;
            CHECK(maybe_get_host_scalar_value(*co, co_host));
            // DST zero point is added to result (not subtracted like SRC/WEI).
            exec_cfg.co_host_scalar = static_cast<int16_t>(co_host);
        }
    } else if (exec_cfg.with_bias) {
        co = &GEMM_CTX_ARG_STORAGE(bias);
        exec_cfg.off_co0
                = types::bytes_to_elements(exec_cfg.c_type(), co->offset());
    } else if (pd->with_sum_ab()) {
        co = &GEMM_CTX_ARG_STORAGE(sum_ab);
        exec_cfg.off_co0
                = types::bytes_to_elements(exec_cfg.c_type(), co->offset());
    }
    exec_cfg.co = co;

    // Read the folded problem directly: exec_cfg.problem is kernel-frame after
    // swap_fold, so aOffset==Calc is exactly kernel-A's zp state (Calc covers
    // tensor and host-scalar zps — a host scalar folds aoPtrDims to -1 but keeps
    // aOffset==Calc).
    using gemmstone::ABOffset;
    exec_cfg.with_a_zero_points = (exec_cfg.problem.aOffset == ABOffset::Calc);
    exec_cfg.with_b_zero_points = (exec_cfg.problem.bOffset == ABOffset::Calc);
    if (exec_cfg.with_a_zero_points || exec_cfg.with_b_zero_points) {
        const auto *ao_phys = &GEMM_CTX_ARG_STORAGE(a_zero_point);
        const auto *bo_phys = &GEMM_CTX_ARG_STORAGE(b_zero_point);
        int ao_host = 0, bo_host = 0;
        if (ao_phys->is_host_scalar())
            CHECK(maybe_get_host_scalar_value(*ao_phys, ao_host));
        if (bo_phys->is_host_scalar())
            CHECK(maybe_get_host_scalar_value(*bo_phys, bo_host));
        exec_cfg.ao = pick_a(s, ao_phys, bo_phys);
        exec_cfg.bo = pick_b(s, ao_phys, bo_phys);
        exec_cfg.ao_host_scalar
                = static_cast<int16_t>(pick_a(s, -ao_host, -bo_host));
        exec_cfg.bo_host_scalar
                = static_cast<int16_t>(pick_b(s, -ao_host, -bo_host));
    }

    // 2D-scale presence reads the folded problem; storage routes through swap.
    auto &a_scales_phys = GEMM_CTX_ARG_STORAGE(a_scales);
    auto &b_scales_phys = GEMM_CTX_ARG_STORAGE(b_scales);
    if (exec_cfg.problem.aScale2D())
        exec_cfg.a_scales = pick_a(s, &a_scales_phys, &b_scales_phys);
    if (exec_cfg.problem.bScale2D())
        exec_cfg.b_scales = pick_b(s, &a_scales_phys, &b_scales_phys);
    if (exec_cfg.with_mx_scale)
        exec_cfg.c_scales = &GEMM_CTX_ARG_STORAGE(c_scales);

    // Bind kernel-side directly: gating on problem.needs{A,B}GroupSums here
    // would mis-pair them under swap_ab and null out the slot the kernel reads.
    auto &ag_phys = GEMM_CTX_ARG_STORAGE(a_group_sums);
    auto &bg_phys = GEMM_CTX_ARG_STORAGE(b_group_sums);
    exec_cfg.ag = pick_a(s, &ag_phys, &bg_phys);
    exec_cfg.bg = pick_b(s, &ag_phys, &bg_phys);

    // dyn_offset_a/b are matmul-frame; route through swap to pair with the
    // kernel's A/B.
    const auto eff_dyn_offset_a = pick_a(s, pd->dyn_offset_a, pd->dyn_offset_b);
    const auto eff_dyn_offset_b = pick_b(s, pd->dyn_offset_a, pd->dyn_offset_b);
    exec_cfg.off_a0
            = types::bytes_to_elements(exec_cfg.a_type(), exec_cfg.a->offset())
            + eff_dyn_offset_a;
    exec_cfg.off_b0
            = types::bytes_to_elements(exec_cfg.b_type(), exec_cfg.b->offset())
            + eff_dyn_offset_b;
    exec_cfg.off_c0
            = types::bytes_to_elements(exec_cfg.c_type(), exec_cfg.c->offset())
            + pd->dyn_offset_c;

    exec_cfg.alpha = pd->alpha();
    if (pd->attr()->scales_.has_host_scalars()) {
        const auto &a_scales = pd->attr()->scales_.get(DNNL_ARG_A);
        const auto &b_scales = pd->attr()->scales_.get(DNNL_ARG_B);
        const auto &c_scales = pd->attr()->scales_.get(DNNL_ARG_C);
        const auto &a_scales_storage = GEMM_CTX_ARG_STORAGE(a_scales);
        const auto &b_scales_storage = GEMM_CTX_ARG_STORAGE(b_scales);
        const auto &c_scales_storage = GEMM_CTX_ARG_STORAGE(c_scales);
        exec_cfg.alpha = 1.0f;
        float scale_val = 0;
        if (a_scales.is_host_scalar()) {
            CHECK(maybe_get_host_scalar_value(a_scales_storage, scale_val));
            exec_cfg.alpha *= scale_val;
        }
        if (b_scales.is_host_scalar()) {
            CHECK(maybe_get_host_scalar_value(b_scales_storage, scale_val));
            exec_cfg.alpha *= scale_val;
        }
        // Limited support of host scalar dst scales.
        if (c_scales.is_host_scalar() && pd->attr()->post_ops_.len() == 0) {
            CHECK(maybe_get_host_scalar_value(c_scales_storage, scale_val));
            gpu_assert(scale_val != 0);
            exec_cfg.alpha /= scale_val;
        }
    }
    return status::success;
}

status_t gen_t::execute(const exec_ctx_t &ctx) const {
    auto *compute_stream = utils::downcast<intel::stream_t *>(ctx.stream());

    auto zero_pool = zero_pool_;

#ifdef DNNL_WITH_SYCL
    bool release_zp = false;
    const auto *sycl_stream
            = utils::downcast<const gpu::intel::sycl::stream_t *>(
                    compute_stream);

    if (need_zero_pool() && sycl_stream->recording()) {
        auto *intel_engine
                = utils::downcast<intel::engine_t *>(compute_stream->engine());
        CHECK(lookup_zero_pool(intel_engine, compute_stream,
                zero_pool_chunk_size_, &zero_pool));
        release_zp = true;
    }
#endif

    jit::exec_config_t exec_cfg;
    CHECK(build_exec_config(exec_cfg, pd(), ctx));

    const auto &problem = *pd()->kernel_desc()->problem();
    const auto m = into<int32_t>(exec_cfg.m);
    const auto n = into<int32_t>(exec_cfg.n);
    const auto k = into<int32_t>(exec_cfg.k);
    const auto lda = into<int32_t>(exec_cfg.lda);
    const auto ldb = into<int32_t>(exec_cfg.ldb);
    const auto ldc = into<int32_t>(exec_cfg.ldc);
    const auto ldco = into<int32_t>(exec_cfg.with_bias ? exec_cfg.ld_bias : 0);

    auto alpha = exec_cfg.alpha;
    auto beta = exec_cfg.beta;

    bool k_parallel_global = nocopy_info()->kParallel();
    bool k_parallel_fixed
            = (nocopy_info()->kParallel() || nocopy_info()->kParallelLocal())
            && !nocopy_info()->kParallelVariable();

    std::unique_ptr<memory_storage_t> c_temp;
    if (nocopy_info()->needsTempC()) {
        c_temp = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_accumulator);
    }

    const memory_storage_t *po_srcs[GEMM_MAX_PO];
    auto &bias = GEMM_CTX_ARG_STORAGE(bias);

    int po_count = int(exec_cfg.binary_srcs.size());
    assert(po_count <= GEMM_MAX_PO);

    for (int i = 0; i < po_count; i++) {
        auto &src = exec_cfg.binary_srcs[i];
        switch (src.type) {
            case jit::binary_src_t::binary:
                po_srcs[i]
                        = ctx.args()
                                  .exec_args
                                  .at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(src.index)
                                          | DNNL_ARG_SRC_1)
                                  .mem()
                                  ->memory_storage();
                break;
            case jit::binary_src_t::prelu:
                po_srcs[i]
                        = ctx.args()
                                  .exec_args
                                  .at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(src.index)
                                          | DNNL_ARG_WEIGHTS)
                                  .mem()
                                  ->memory_storage();
                break;
            case jit::binary_src_t::bias: po_srcs[i] = &bias; break;
            case jit::binary_src_t::scales:
                switch (src.index) {
                    case DNNL_ARG_WEIGHTS:
                        po_srcs[i] = &GEMM_CTX_ARG_STORAGE(a_scales);
                        break;
                    case DNNL_ARG_SRC:
                        po_srcs[i] = &GEMM_CTX_ARG_STORAGE(b_scales);
                        break;
                    case DNNL_ARG_DST:
                        po_srcs[i] = &GEMM_CTX_ARG_STORAGE(c_scales);
                        break;
                    default:
                        po_srcs[i] = nullptr;
                        assert(!"invalid scale type");
                        break;
                }
                break;
            default: po_srcs[i] = nullptr; break;
        }
    }

    int64_t po_offsets0[GEMM_MAX_PO] = {0}, po_offsets[GEMM_MAX_PO] = {0};
    for (int i = 0; i < po_count; i++)
        if (po_srcs[i])
            po_offsets0[i] = po_srcs[i]->offset() / problem.Tbinary[i];

    int64_t off_aq0 = 0, off_bq0 = 0;

    status_t status;

    auto block_m = nocopy_info()->blocking[0];
    auto block_n = nocopy_info()->blocking[1];
    auto block_k = nocopy_info()->blocking[2];

    bool disable_hilbert = (k <= 64) && nocopy_info()->isHilbert();
    if (disable_hilbert) {
        block_m = nocopy_info()->blockingAlt[0];
        block_n = nocopy_info()->blockingAlt[1];
    }

    if (!utils::one_of(exec_cfg.c_type(), data_type::f32, data_type::f16))
        block_k = k;
    if (problem.postOps.len() > 0 && !problem.postOps[0].is_sum()) block_k = k;

    if (k_parallel_fixed)
        block_k = into<int32_t>(pd()->kernel_desc()->aux_params()->k0);

    block_m = utils::rnd_up(block_m, nocopy_info()->wgTile(gemmstone::LoopM));
    block_n = utils::rnd_up(block_n, nocopy_info()->wgTile(gemmstone::LoopN));

    int32_t k0 = 1;
    if (k_parallel_fixed) {
        k0 = block_k;
        block_k = std::max(k, 1);

        if (k_parallel_global && !nocopy_info()->fusedBeta() && beta != 1.0f
                && (k > k0 * pd()->kernel_desc()->aux_params()->wgK)) {
            status = launch_nocopy(ctx, compute_stream, zero_pool, exec_cfg,
                    nullptr, po_count, po_srcs, exec_cfg.off_a0,
                    exec_cfg.off_b0, exec_cfg.off_c0, off_aq0, off_bq0,
                    exec_cfg.off_co0, po_offsets0, m, n, 0, 1, 1.0f, beta,
                    false, true);
            if (status) return status;
            beta = 1.0f;
        }
    }

    for (int64_t Bk = 0; Bk < nstl::max<dim_t>(k, 1); Bk += block_k) {
        int64_t size_k = k - Bk;
        bool last_k_block = (size_k <= block_k);
        if (!last_k_block) size_k = block_k;

        for (int64_t Bm = 0; Bm < m; Bm += block_m) {
            int64_t size_m = m - Bm;
            if (size_m > block_m) size_m = block_m;

            auto off_a_src = exec_cfg.off_a0
                    + (!exec_cfg.trans_a() ? (Bm + Bk * lda) : (Bk + Bm * lda));

            for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                int64_t size_n = n - Bn;
                if (size_n > block_n) size_n = block_n;

                auto off_b_src = exec_cfg.off_b0
                        + (!exec_cfg.trans_b() ? (Bk + Bn * ldb)
                                               : (Bn + Bk * ldb));

                auto off_c = exec_cfg.off_c0 + Bm + Bn * ldc;

                auto off_aq = off_aq0;
                auto off_bq = off_bq0;
                if (problem.aoPtrDims >= 1 || exec_cfg.a_scales) off_aq += Bm;
                if (problem.boPtrDims >= 1 || exec_cfg.b_scales) off_bq += Bn;

                auto off_co = exec_cfg.off_co0;
                switch (exec_cfg.cmask & 3) {
                    case 1: off_co += Bn; break;
                    case 2: off_co += Bm; break;
                    case 3:
                        off_co += isColMajor(problem.CO.layout)
                                ? (Bn * ldco + Bm)
                                : (Bm * ldco + Bn);
                        break;
                }

                for (int i = 0; i < po_count; i++) {
                    po_offsets[i] = po_offsets0[i];
                    bool row = problem.postOps.binaryRow[i],
                         col = problem.postOps.binaryCol[i];
                    if (row && col) {
                        auto ld = exec_cfg.ld_binary(i);
                        po_offsets[i] += isColMajor(problem.binary[i].layout)
                                ? (Bn * ld + Bm)
                                : (Bm * ld + Bn);
                    } else if (row)
                        po_offsets[i] += Bm;
                    else if (col)
                        po_offsets[i] += Bn;
                }

                float eff_beta = (Bk == 0) ? beta : 1.0f;
                status = launch_nocopy(ctx, compute_stream, zero_pool, exec_cfg,
                        c_temp.get(), po_count, po_srcs, off_a_src, off_b_src,
                        off_c, off_aq, off_bq, off_co, po_offsets,
                        into<int32_t>(size_m), into<int32_t>(size_n),
                        into<int32_t>(size_k), k0, alpha, eff_beta,
                        last_k_block, disable_hilbert);

                if (status) return status;
            }
        }
    }

#ifdef DNNL_WITH_SYCL
    if (release_zp) release_zero_pool(zero_pool);
#endif

    return status::success;
}

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
