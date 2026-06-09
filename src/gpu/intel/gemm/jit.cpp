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

namespace jit {
struct exec_args_t {
    const memory_storage_t *a = nullptr;
    const memory_storage_t *b = nullptr;
    const memory_storage_t *c = nullptr;

    const memory_storage_t *ao = nullptr;
    const memory_storage_t *bo = nullptr;
    const memory_storage_t *a_scales = nullptr;
    const memory_storage_t *b_scales = nullptr;
    const memory_storage_t *c_scales = nullptr;
    const memory_storage_t *ag = nullptr;
    const memory_storage_t *bg = nullptr;

    const memory_storage_t *co = nullptr;

    const memory_storage_t *sround_seed = nullptr;

    int16_t ao_host_scalar = 0;
    int16_t bo_host_scalar = 0;
    int16_t co_host_scalar = 0;

    float alpha = 1.0f;

    size_t off_a0 = 0;
    size_t off_b0 = 0;
    size_t off_c0 = 0;
    int64_t off_co0 = 0;
};
} // namespace jit

bool check_memory_storage(const memory_storage_t *storage, const char *name) {
    if (storage && *storage) return true;
    VERROR(primitive, gpu, "%s,%s: %s", "jit::gemm", "argument is not set",
            name);
    return false;
}

status_t gen_t::launch_nocopy(const exec_ctx_t &ctx,
        intel::stream_t *compute_stream, zero_pool_t *zero_pool,
        const jit::exec_args_t &exec_args, const memory_storage_t *c_temp,
        const memory_storage_t *zero_buf_scratchpad, int po_count,
        const memory_storage_t **po_srcs, int64_t offset_a, int64_t offset_b,
        int64_t offset_c, int64_t offset_aq, int64_t offset_bq,
        int64_t offset_co, int64_t *offset_po_src, int32_t m, int32_t n,
        int32_t k, int32_t k0, float alpha, float beta, bool last_k_block,
        bool disable_hilbert) const {
    const auto &cfg = pd()->cfg();
    if (cfg.batch() == 0) return status::success;

    std::unique_ptr<memory_storage_t> zeros;
    const memory_storage_t *zero_buf = nullptr;
    int zp_token = 0;
    if (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps()) {
        if (zero_pool) {
            CHECK(zero_pool->claim(
                    compute_stream, zero_buffer_bytes_, zeros, &zp_token));
            zero_buf = zeros.get();
        } else {
            zero_buf = zero_buf_scratchpad;
            auto nelems = uint32_t(zero_buffer_bytes_ / sizeof(uint32_t));
            compute::kernel_arg_list_t zero_arg_list;
            zero_arg_list.set(0, *zero_buf);
            zero_arg_list.set(1, nelems);
            compute::range_t zero_gws = {size_t(nelems)};
            compute::range_t zero_lws
                    = {size_t(std::min(256, (int)utils::max_pow2_div(nelems)))};
            auto zero_nd_range = compute::nd_range_t(zero_gws, zero_lws);
            CHECK(parallel_for(
                    ctx, zero_nd_range, zero_fill_kernel_, zero_arg_list));
        }
    }

    uint32_t flags = 0;
    bool k_parallel_fixed
            = (nocopy_info()->kParallel() || nocopy_info()->kParallelLocal())
            && !nocopy_info()->kParallelVariable();

    const auto &problem = cfg.problem;

    if (!last_k_block) flags |= gemmstone::FlagNonfinalKBlock;
    if (cfg.cmask & 1) flags |= gemmstone::FlagCOColumn;
    if (cfg.cmask & 2) flags |= gemmstone::FlagCORow;

    const auto lda = into<int32_t>(cfg.lda);
    const auto ldb = into<int32_t>(cfg.ldb);
    const auto ldc = into<int32_t>(cfg.ldc);

    compute::kernel_arg_list_t arg_list;
    int argn = 0;

    arg_list.set(argn++, *exec_args.a);
    arg_list.set(argn++, *exec_args.b);
    arg_list.set(argn++, *exec_args.c);
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

    bool a_zp_ptr = problem.hasAOffsetPtr();
    bool b_zp_ptr = problem.hasBOffsetPtr();

    if (a_zp_ptr) arg_list.set(argn++, *exec_args.ao);
    if (b_zp_ptr) arg_list.set(argn++, *exec_args.bo);
    if (problem.aOffsetHostScalar())
        arg_list.set(argn++, exec_args.ao_host_scalar);
    if (problem.bOffsetHostScalar())
        arg_list.set(argn++, exec_args.bo_host_scalar);
    if (problem.aScale2D()) arg_list.set(argn++, *exec_args.a_scales);
    if (problem.bScale2D()) arg_list.set(argn++, *exec_args.b_scales);
    if (problem.hasCMXScale()) arg_list.set(argn++, *exec_args.c_scales);
    if (problem.needsAGroupSums()) {
        if (!check_memory_storage(exec_args.ag, "ag"))
            return status::runtime_error;
        arg_list.set(argn++, *exec_args.ag);
    }
    if (problem.needsBGroupSums()) {
        if (!check_memory_storage(exec_args.bg, "bg"))
            return status::runtime_error;
        arg_list.set(argn++, *exec_args.bg);
    }

    if (problem.aOffset2D() || problem.aScale2D()
            || problem.needsAGroupSums()) {
        auto layout = problem.needsAGroupSums() ? problem.Ag.layout
                : problem.aScale2D()            ? problem.A_scale.layout
                                                : problem.AO.layout;
        auto ldaq = into<int32_t>(isColMajor(layout)
                        ? utils::div_up(cfg.m, problem.aqGroupM)
                        : utils::div_up(cfg.k, problem.aqGroupK));
        arg_list.set(argn++, ldaq);
    }
    if (problem.bOffset2D() || problem.bScale2D()
            || problem.needsBGroupSums()) {
        auto layout = problem.needsBGroupSums() ? problem.Bg.layout
                : problem.bScale2D()            ? problem.B_scale.layout
                                                : problem.BO.layout;
        auto ldbq = into<int32_t>(!isColMajor(layout)
                        ? utils::div_up(cfg.n, problem.bqGroupN)
                        : utils::div_up(cfg.k, problem.bqGroupK));
        arg_list.set(argn++, ldbq);
    }
    if (problem.hasCMXScale()) {
        auto ldcq = cfg.m / problem.cqGroupM;
        arg_list.set(argn++, ldcq);
    }
    if (problem.usesCOPtr()) {
        if (!check_memory_storage(exec_args.co, "co"))
            return status::runtime_error;
        arg_list.set(argn++, *exec_args.co);
        arg_list.set(argn++, offset_co);
        if (cfg.with_bias()) {
            auto ldco = into<int32_t>(cfg.ld_bias);
            arg_list.set(argn++, ldco);
        }
    } else if (problem.cOffsetHostScalar()) {
        arg_list.set(argn++, exec_args.co_host_scalar);
    }
    if (nocopy_info()->needsTempC()) arg_list.set(argn++, *c_temp);
    if (problem.postOps.cStochasticRound) {
        arg_list.set(argn++, *exec_args.sround_seed);
    }
    arg_list.set(argn++, flags);
    if (k_parallel_fixed) arg_list.set(argn++, k0);

    for (int i = 0; i < po_count; i++) {
        if (!po_srcs[i]) continue;
        arg_list.set(argn++, *po_srcs[i]);
        arg_list.set(argn++, offset_po_src[i]);

        if (problem.postOps.binaryRow[i] && problem.postOps.binaryCol[i])
            arg_list.set(argn++, int32_t(cfg.ld_binary(i)));
    }

    if (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps()) {
        arg_list.set(argn++, *zero_buf);
    }

    if (problem.batchDims >= 1) {
        for (int i = problem.batchDims - 1; i >= 0; i--) {
            auto stride_a = int64_t(cfg.a_batch_strides[i]);
            auto stride_b = int64_t(cfg.b_batch_strides[i]);
            auto stride_c = int64_t(cfg.c_batch_strides[i]);
            if (jit::enable_generator_dsl()) {
                auto *intel_engine = utils::downcast<intel::engine_t *>(
                        compute_stream->engine());
                auto hw = ngen::getCore(
                        intel_engine->device_info()->product().family);

                // 2d Surface pointer needs to be 64 byte aligned. When negative
                // bounds checking is unnecessary, this restriction can be
                // relaxed by rounding down the surface pointer and adjusting
                // the width accordingly.
                auto base_alignment = intel::jit::block_2d_base_alignment(hw);
                auto a_size = types::data_type_size(cfg.a_type());
                if (stride_a * a_size % base_alignment) {
                    gpu_warning() << "Unimplemented load transform";
                    return status::runtime_error;
                }
                auto b_size = types::data_type_size(cfg.b_type());
                if (stride_b * b_size % base_alignment) {
                    gpu_warning() << "Unimplemented load transform";
                    return status::runtime_error;
                }
            }
            arg_list.set(argn++, stride_a);
            arg_list.set(argn++, stride_b);
            arg_list.set(argn++, stride_c);
            if (problem.hasAScalePtr())
                arg_list.set(argn++, cfg.a_scale_stride(i));
            if (problem.hasBScalePtr())
                arg_list.set(argn++, cfg.b_scale_stride(i));
            if (problem.hasCMXScale())
                arg_list.set(argn++, stride_c / problem.cqGroupM);
            if (problem.hasAOffsetPtr())
                arg_list.set(argn++, cfg.a_zp_stride(i));
            if (problem.hasBOffsetPtr())
                arg_list.set(argn++, cfg.b_zp_stride(i));
            if (problem.needsAGroupSums())
                arg_list.set(argn++, cfg.a_gs_stride(i));
            if (problem.needsBGroupSums())
                arg_list.set(argn++, cfg.b_gs_stride(i));
        }
        for (int i = 0; i < po_count; i++) {
            if (problem.postOps.binaryBatch[i]) {
                for (int b = problem.batchDims - 1; b >= 0; b--) {
                    arg_list.set(argn++, int64_t(cfg.stride_binary(i, b)));
                }
            }
        }
        for (int i = 1; i < problem.batchDims; i++) {
            auto batchSize = uint32_t(cfg.c_batch_sizes[i]);
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

    auto lws_k = pd()->kernel_desc().aux_params()->wgK;

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
            gws[1], info->cInterleaveChunk(problem.Tc_ext) * lws[1]);
    if (info->cInterleaveEnabled()
            && (offset_c % 64 > 0 || ldc * problem.Tc % 64 > 0)) {
        auto wgTileM = info->wgTile(gemmstone::LoopM);
        auto maxShift = 64 / problem.Tc_ext.size() - 1;
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

    gws[2] *= cfg.batch();

    jit::linear_order_args(arg_list, argn, lws, gws, m, n, k, disable_hilbert,
            *nocopy_info(), pd()->kernel_desc().aux_params(), pd()->dev_info_);

    if (nocopy_info()->perKSLM > 0) {
        size_t slm = nocopy_info()->slm;
        if (lws[2] > 1) slm = nstl::max(slm, nocopy_info()->perKSLM * lws[2]);
        arg_list.set(argn++, slm, nullptr);
    }

    if (problem.aoPtrDims > 0 || problem.aScale2D())
        arg_list.set(argn++, offset_aq);
    if (problem.boPtrDims > 0 || problem.bScale2D())
        arg_list.set(argn++, offset_bq);

    lws[0] *= nocopy_info()->subgroupSize;
    gws[0] *= nocopy_info()->subgroupSize;

    auto nd_range = compute::nd_range_t(gws, lws);
    auto status = parallel_for(ctx, nd_range, nocopy_kernel_, arg_list);

    if (zero_pool
            && (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps()))
        zero_pool->async_release(zp_token, compute_stream->ctx().get_deps());

    return status;
}

static status_t init_exec_args(jit::exec_args_t &exec_args,
        const jit::kernel_config_t &cfg, const exec_ctx_t &ctx) {

    auto &a_phys = GEMM_CTX_ARG_STORAGE(a);
    auto &b_phys = GEMM_CTX_ARG_STORAGE(b);
    exec_args.a = cfg.swap_ab ? &a_phys : &b_phys;
    exec_args.b = cfg.swap_ab ? &b_phys : &a_phys;
    exec_args.c = &GEMM_CTX_ARG_STORAGE(c);
    exec_args.sround_seed = &GEMM_CTX_ARG_STORAGE(sround_seed);

    const memory_storage_t *co = &GEMM_CTX_ARG_STORAGE(c_zero_point);
    if (cfg.with_c_zero_points()) {
        exec_args.off_co0
                = types::bytes_to_elements(cfg.c_type(), co->offset());
        if (co->is_host_scalar()) {
            int co_host = 0;
            CHECK(maybe_get_host_scalar_value(*co, co_host));
            // DST zero point is added to result (not subtracted like SRC/WEI).
            exec_args.co_host_scalar = static_cast<int16_t>(co_host);
        }
    } else if (cfg.with_bias()) {
        co = &GEMM_CTX_ARG_STORAGE(bias);
        exec_args.off_co0
                = types::bytes_to_elements(cfg.c_type(), co->offset());
    } else if (cfg.with_sum_ab()) {
        co = &GEMM_CTX_ARG_STORAGE(sum_ab);
        exec_args.off_co0
                = types::bytes_to_elements(cfg.c_type(), co->offset());
    }
    exec_args.co = co;

    if (cfg.with_a_zero_points() || cfg.with_b_zero_points()) {
        const auto *ao_phys = &GEMM_CTX_ARG_STORAGE(a_zero_point);
        const auto *bo_phys = &GEMM_CTX_ARG_STORAGE(b_zero_point);
        int ao_host = 0, bo_host = 0;
        if (ao_phys->is_host_scalar())
            CHECK(maybe_get_host_scalar_value(*ao_phys, ao_host));
        if (bo_phys->is_host_scalar())
            CHECK(maybe_get_host_scalar_value(*bo_phys, bo_host));
        if (cfg.swap_ab) {
            std::swap(ao_phys, bo_phys);
            std::swap(ao_host, bo_host);
        }
        exec_args.ao = ao_phys;
        exec_args.bo = bo_phys;
        exec_args.ao_host_scalar = static_cast<int16_t>(-ao_host);
        exec_args.bo_host_scalar = static_cast<int16_t>(-bo_host);
    }

    auto &a_scales_phys = GEMM_CTX_ARG_STORAGE(a_scales);
    auto &b_scales_phys = GEMM_CTX_ARG_STORAGE(b_scales);
    if (cfg.problem.aScale2D())
        exec_args.a_scales = cfg.swap_ab ? &b_scales_phys : &a_scales_phys;
    if (cfg.problem.bScale2D())
        exec_args.b_scales = cfg.swap_ab ? &a_scales_phys : &b_scales_phys;
    if (cfg.problem.hasCMXScale())
        exec_args.c_scales = &GEMM_CTX_ARG_STORAGE(c_scales);

    auto &ag_phys = GEMM_CTX_ARG_STORAGE(a_group_sums);
    auto &bg_phys = GEMM_CTX_ARG_STORAGE(b_group_sums);
    exec_args.ag = cfg.swap_ab ? &bg_phys : &ag_phys;
    exec_args.bg = cfg.swap_ab ? &ag_phys : &bg_phys;

    exec_args.off_a0
            = types::bytes_to_elements(cfg.a_type(), exec_args.a->offset());
    exec_args.off_b0
            = types::bytes_to_elements(cfg.b_type(), exec_args.b->offset());
    exec_args.off_c0
            = types::bytes_to_elements(cfg.c_type(), exec_args.c->offset());

    exec_args.alpha = cfg.alpha();
    if (cfg.a_host_scale || cfg.b_host_scale || cfg.c_host_scale_to_alpha) {
        exec_args.alpha = 1.0f;
        float scale_val = 0;
        const auto &a_hs = cfg.swap_ab ? b_scales_phys : a_scales_phys;
        const auto &b_hs = cfg.swap_ab ? a_scales_phys : b_scales_phys;
        if (cfg.a_host_scale) {
            CHECK(maybe_get_host_scalar_value(a_hs, scale_val));
            exec_args.alpha *= scale_val;
        }
        if (cfg.b_host_scale) {
            CHECK(maybe_get_host_scalar_value(b_hs, scale_val));
            exec_args.alpha *= scale_val;
        }
        // Limited support of host scalar dst scales.
        if (cfg.c_host_scale_to_alpha) {
            CHECK(maybe_get_host_scalar_value(
                    GEMM_CTX_ARG_STORAGE(c_scales), scale_val));
            gpu_assert(scale_val != 0);
            exec_args.alpha /= scale_val;
        }
    }
    return status::success;
}

status_t gen_t::execute(const exec_ctx_t &ctx) const {
    auto *compute_stream = utils::downcast<intel::stream_t *>(ctx.stream());

    auto zero_pool = zero_pool_;

#ifdef DNNL_WITH_SYCL
    const auto *sycl_stream
            = utils::downcast<const gpu::intel::sycl::stream_t *>(
                    compute_stream);
    // The zero-pool global state management is not compatible with SYCL graph;
    // fall back to the graph-safe zero-buffer scratchpad path.
    if (sycl_stream->recording()) zero_pool = nullptr;
#endif

    const auto &cfg = pd()->cfg();
    jit::exec_args_t exec_args;
    CHECK(init_exec_args(exec_args, cfg, ctx));

    const auto &problem = cfg.problem;
    const auto m = into<int32_t>(cfg.m);
    const auto n = into<int32_t>(cfg.n);
    const auto k = into<int32_t>(cfg.k);
    const auto lda = into<int32_t>(cfg.lda);
    const auto ldb = into<int32_t>(cfg.ldb);
    const auto ldc = into<int32_t>(cfg.ldc);
    const auto ldco = into<int32_t>(cfg.with_bias() ? cfg.ld_bias : 0);

    auto alpha = exec_args.alpha;
    auto beta = cfg.beta;

    bool k_parallel_global = nocopy_info()->kParallel();
    bool k_parallel_fixed
            = (nocopy_info()->kParallel() || nocopy_info()->kParallelLocal())
            && !nocopy_info()->kParallelVariable();

    std::unique_ptr<memory_storage_t> c_temp;
    if (nocopy_info()->needsTempC()) {
        c_temp = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_accumulator);
    }

    std::unique_ptr<memory_storage_t> zero_buf_scratchpad;
    if (need_zero_buffer() && !zero_pool) {
        zero_buf_scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_zero_buffer);
    }

    const memory_storage_t *po_srcs[GEMM_MAX_PO];
    auto &bias = GEMM_CTX_ARG_STORAGE(bias);

    int po_count = int(cfg.binary_srcs.size());
    assert(po_count <= GEMM_MAX_PO);

    for (int i = 0; i < po_count; i++) {
        auto &src = cfg.binary_srcs[i];
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

    if (!utils::one_of(cfg.c_type(), data_type::f32, data_type::f16))
        block_k = k;
    if (problem.postOps.len() > 0 && !problem.postOps[0].is_sum()) block_k = k;

    if (k_parallel_fixed)
        block_k = into<int32_t>(pd()->kernel_desc().aux_params()->k0);

    block_m = utils::rnd_up(block_m, nocopy_info()->wgTile(gemmstone::LoopM));
    block_n = utils::rnd_up(block_n, nocopy_info()->wgTile(gemmstone::LoopN));

    int32_t k0 = 1;
    if (k_parallel_fixed) {
        k0 = block_k;
        block_k = std::max(k, 1);

        if (k_parallel_global && !nocopy_info()->fusedBeta() && beta != 1.0f
                && (k > k0 * pd()->kernel_desc().aux_params()->wgK)) {
            status = launch_nocopy(ctx, compute_stream, zero_pool, exec_args,
                    nullptr, zero_buf_scratchpad.get(), po_count, po_srcs,
                    exec_args.off_a0, exec_args.off_b0, exec_args.off_c0,
                    off_aq0, off_bq0, exec_args.off_co0, po_offsets0, m, n, 0,
                    1, 1.0f, beta, false, true);
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

            auto off_a_src = exec_args.off_a0
                    + (!cfg.trans_a() ? (Bm + Bk * lda) : (Bk + Bm * lda));

            for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                int64_t size_n = n - Bn;
                if (size_n > block_n) size_n = block_n;

                auto off_b_src = exec_args.off_b0
                        + (!cfg.trans_b() ? (Bk + Bn * ldb) : (Bn + Bk * ldb));

                auto off_c = exec_args.off_c0 + Bm + Bn * ldc;

                auto off_aq = off_aq0;
                auto off_bq = off_bq0;
                if (problem.aoPtrDims >= 1 || exec_args.a_scales) off_aq += Bm;
                if (problem.boPtrDims >= 1 || exec_args.b_scales) off_bq += Bn;

                auto off_co = exec_args.off_co0;
                switch (cfg.cmask & 3) {
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
                        auto ld = cfg.ld_binary(i);
                        po_offsets[i] += isColMajor(problem.binary[i].layout)
                                ? (Bn * ld + Bm)
                                : (Bm * ld + Bn);
                    } else if (row)
                        po_offsets[i] += Bm;
                    else if (col)
                        po_offsets[i] += Bn;
                }

                float eff_beta = (Bk == 0) ? beta : 1.0f;
                status = launch_nocopy(ctx, compute_stream, zero_pool,
                        exec_args, c_temp.get(), zero_buf_scratchpad.get(),
                        po_count, po_srcs, off_a_src, off_b_src, off_c, off_aq,
                        off_bq, off_co, po_offsets, into<int32_t>(size_m),
                        into<int32_t>(size_n), into<int32_t>(size_k), k0, alpha,
                        eff_beta, last_k_block, disable_hilbert);

                if (status) return status;
            }
        }
    }

    return status::success;
}

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
