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

#include <cmath>
#include <limits>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

// Internal alg_kind used by the GPU SDPA kernel. Must be removed once
// softmax_accurate_inf_as_zero is promoted to a public value.
#include "src/common/c_types_map.hpp"

#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "sdpa/sdpa.hpp"

namespace sdpa {

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> create_md(int ndims,
        const dims_t &dims, dnnl_data_type_t dt, const std::string &tag) {
    return dnn_mem_t::init_md(ndims, dims.data(), dt, tag);
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;
    const dir_t dir = init_pd_args.dir;

    auto q_dt = force_f32_dt ? dnnl_f32 : prb->q_dt();
    auto k_dt = force_f32_dt ? dnnl_f32 : prb->k_dt();
    auto v_dt = force_f32_dt ? dnnl_f32 : prb->v_dt();
    auto dst_dt = force_f32_dt ? dnnl_f32 : prb->dst_dt();

    auto q_d = create_md(prb->ndims, prb->q_dims(), q_dt, prb->qtag);
    auto k_d = create_md(prb->ndims, prb->k_dims(), k_dt, prb->ktag);
    auto v_d = create_md(prb->ndims, prb->v_dims(), v_dt, prb->vtag);
    auto dst_d = create_md(prb->ndims, prb->dst_dims, dst_dt, prb->dtag);

    // Attention mask (optional).
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> mask_d {};
    if (prb->with_mask()) {
        mask_d = create_md(prb->ndims, prb->msk_dims, prb->mdt, tag::abx);
    }

    // Always pass a valid md — the API unconditionally dereferences it.
    auto scale_d = dnn_mem_t::init_host_scalar_md(dnnl_f32);

    // Map benchdnn mask_type_t to API's dnnl_attn_mask_type_t values.
    // All buffer variants (full, 1D, 2D) use dnnl_attn_mask_buffer (1).
    int attn_mask_type_val = [](mask_type_t mt) {
        switch (mt) {
            case MASK_NONE: return 0; // dnnl_attn_mask_undef
            case MASK_BUFFER:
            case MASK_BUFFER_1D:
            case MASK_BUFFER_2D: return 1; // dnnl_attn_mask_buffer
            case MASK_CAUSAL_TOP_LEFT: return 2; // dnnl_attn_mask_top_left
            case MASK_CAUSAL_BOTTOM_RIGHT:
                return 3; // dnnl_attn_mask_bottom_right
            default: return 0;
        }
    }(prb->mask_type);
    dnnl_alg_kind_t softmax_alg = static_cast<dnnl_alg_kind_t>(
            dnnl::impl::alg_kind::softmax_accurate_inf_as_zero);

    // KV head count is always derived from the K tensor's head dimension.
    dnnl_dim_t kv_hn = prb->k_dims()[prb->ndims - 3];

    // Default to invert_scale=false; scale is filled in init_ref_memory_args.
    bool invert = prb->with_scale() ? prb->invert_scale() : false;

    // Build attr_args for dropout mask md if configured.
    attr_args_t attr_args;
    if (!prb->attr.dropout.is_def()) {
        attr_args.prepare_post_ops_mds(
                prb->attr, prb->ndims, prb->score_dims.data());
    }
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args, prb->ndims));

    const auto mask_ptr
            = prb->with_mask() ? (const_dnnl_memory_desc_t)mask_d : nullptr;

    if (dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;

        TIME_C_PD(DNN_SAFE_STATUS(sdpa_primitive_desc_create(&init_pd_args.pd,
                init_pd_args.engine, q_d, k_d, v_d, dst_d, mask_ptr, scale_d,
                invert, kv_hn, attn_mask_type_val, softmax_alg, prop, dnnl_attr,
                /* kq_attr = */ nullptr,
                /* vs_attr = */ nullptr)));
    } else {
        auto diff_q_d = create_md(prb->ndims, prb->q_dims(), q_dt, prb->qtag);
        auto diff_k_d = create_md(prb->ndims, prb->k_dims(), k_dt, prb->ktag);
        auto diff_v_d = create_md(prb->ndims, prb->v_dims(), v_dt, prb->vtag);
        auto diff_dst_d
                = create_md(prb->ndims, prb->dst_dims, dst_dt, prb->dtag);

        // Follow the implementation parameter order (mask/scale before
        // diff descs) which differs from the .hpp declaration.
        TIME_C_PD(DNN_SAFE_STATUS(sdpa_primitive_desc_create(&init_pd_args.pd,
                init_pd_args.engine, q_d, k_d, v_d, dst_d, mask_ptr, scale_d,
                diff_q_d, diff_k_d, diff_v_d, diff_dst_d,
                /* dS = */ nullptr, invert, kv_hn, attn_mask_type_val,
                softmax_alg, dnnl_attr, init_pd_args.hint)));
    }

    return dnnl_success;
}

int fill_data(int exec_arg, data_kind_t kind, const prb_t *prb,
        const cfg_t &cfg, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp)) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, res);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, res, get_perf_fill_cfg(mem_dt.dt()));
    }

    cfg_t::density_args_t density_args;
    density_args.data_kind = kind;
    density_args.n_acc = prb->head_size;
    const auto density = cfg.get_density(density_args);

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::bernoulli_distribution b_dist(density);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));

        // Make sure the first element is positive.
        if (idx_start == 0) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            mem_fp.set_f32_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = density == 1.f ? true : b_dist(b_seed);
            if (!is_one) {
                mem_fp.set_f32_elem(idx, 0.f);
                continue;
            }
            float val = gen(int_seed);
            mem_fp.set_f32_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });

    SAFE(mem_dt.reorder(mem_fp, cfg.get_swapped_dt(kind)), WARN);
    return OK;
}

// Fill attention mask with realistic 0 / -inf values (~25 % masked).
int fill_mask(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const float neg_inf = -std::numeric_limits<float>::infinity();
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand rng(nelems + idx_start + 1);
        rng.discard(1);

        for (int64_t idx = idx_start; idx < idx_end; ++idx)
            mem_fp.set_f32_elem(idx, (rng() % 4 == 0) ? neg_inf : 0.f);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->q_dt(), prb->k_dt(), prb->v_dt(), prb->dst_dt()}, prb->dir,
            res);

    // SDPA is currently only implemented for GPU.
    if (is_cpu()) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // SDPA API requires 4D tensors (batch x heads x seq x head_dim).
    // Guard must be here (not skip_unimplemented) because init_pd accesses
    // ndims-3 which would be OOB for 2D/3D inputs.
    if (prb->ndims < 4) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // Validate dimension consistency.
    const auto &qdims = prb->q_dims();
    const auto &kdims = prb->k_dims();
    const auto &vdims = prb->v_dims();
    int nd = prb->ndims;

    // Q head_size must match K row dim.
    if (qdims[nd - 1] != kdims[nd - 2]) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
    // K col dim must match V row dim.
    if (kdims[nd - 1] != vdims[nd - 2]) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    const bool is_bwd = (kind == SRC || kind == SRC_1 || kind == SRC_2);

    // Backward chains more matmuls and softmax_backward; needs looser
    // thresholds due to catastrophic cancellation in S*(dP - Di) and
    // accumulated atomic adds.
    const float trh_coeff = is_bwd ? 32.f : 8.f;
    const float trh = trh_coeff * (1 + prb->n_keys) * epsilon_dt(prb->dst_dt());

    // This standard point-to-point check  validates any point above 1e-5 against
    // the *relative* threshold. This works well for well-conditioned points,
    // but fails for small-magnitude points
    cmp.set_threshold(trh);

    // That alone is not sufficent SDPA's forward pass: it is a convex combination
    //     Out = sum(prob_k * V_k),   sum(prob_k) = 1,
    // so catastrophic cancellation can drive |Out| -> 0.f (observed ~5e-5) while
    // the absolute error stays pinned at the rounding floor, making
    // rel_diff = diff/|exp| blow up for a perfectly healthy kernel.
    //
    // The small-magnitude points are instead validated against an absolute floor
    if (!is_bwd) {
        const dnn_mem_t &absmag = ref_args.find(SDPA_REF_ARG_OUT_ABSMAG);
        const float eps_dst = epsilon_dt(prb->dst_dt());
        const dnn_mem_t *mag = &absmag;
        cmp.set_driver_check_function(
                [eps_dst, mag](
                        const compare::compare_t::driver_check_func_args_t &a)
                        -> bool {
            return a.diff <= eps_dst * mag->get_f32_elem(a.idx);
        });
    } else {
        // Backward chains more matmuls/softmax_bwd and produces element diffs up
        // to ~3e-2 for near-zero values; keep its looser empirical floor.
        const float abs_trh = 5e-2f;

        cmp.set_driver_check_function(
                [abs_trh](const compare::compare_t::driver_check_func_args_t &a)
                        -> bool { return a.diff <= abs_trh; });
    }

    cmp.set_zero_trust_percent(is_bwd ? 70.f : 90.f);
}

std::vector<int> supported_exec_args(dir_t dir) {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_QUERIES,
            DNNL_ARG_KEYS,
            DNNL_ARG_VALUES,
            DNNL_ARG_DST,
            DNNL_ARG_ATTN_MASK,
            DNNL_ARG_WORKSPACE,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_QUERIES,
            DNNL_ARG_KEYS,
            DNNL_ARG_VALUES,
            DNNL_ARG_DST,
            DNNL_ARG_ATTN_MASK,
            DNNL_ARG_DIFF_QUERIES,
            DNNL_ARG_DIFF_KEYS,
            DNNL_ARG_DIFF_VALUES,
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DS,
            DNNL_ARG_WORKSPACE,
    };
    return (dir & FLAG_FWD) ? exec_fwd_args : exec_bwd_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, DST});

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg == DNNL_ARG_SCRATCHPAD) continue;

        // Workspace connects forward and backward; not filled manually.
        if (exec_arg == DNNL_ARG_WORKSPACE) continue;

        if (exec_arg == DNNL_ARG_SCALE) {
            dnnl_dims_t s_dims = {1};
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(1, s_dims, dnnl_f32, tag::x, ref_engine,
                            /* prefill = */ false));
            auto &ref_mem = ref_mem_map[exec_arg];
            float scale_val = prb->invert_scale()
                    ? sqrtf(static_cast<float>(prb->head_size))
                    : 1.0f / sqrtf(static_cast<float>(prb->head_size));
            ref_mem.set_f32_elem(0, scale_val);
            mem.set_f32_elem(0, scale_val);
            continue;
        }

        ref_mem_map.emplace(exec_arg,
                dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                        /* prefill = */ false));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_QUERIES:
                SAFE(fill_data(exec_arg, SRC, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_KEYS:
            case DNNL_ARG_VALUES:
                SAFE(fill_data(exec_arg, WEI, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DST: break;
            case DNNL_ARG_ATTN_MASK: SAFE(fill_mask(mem, ref_mem), WARN); break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_data(exec_arg, DST, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_QUERIES:
            case DNNL_ARG_DIFF_KEYS:
            case DNNL_ARG_DIFF_VALUES:
            case DNNL_ARG_DS: break;
            default:
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
                break;
        }

        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    v_prim.resize(2); // just fwd or fwd + bwd.
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res, FLAG_FWD,
                 nullptr, /* is_service_prim = */ prb->dir & FLAG_BWD),
            WARN);
    if (prb->dir & FLAG_BWD) {
        SAFE(init_prim(prb->ctx_init, v_prim[1], init_pd, prb, res, FLAG_BWD,
                     query_pd(v_prim[0])),
                WARN);
    }
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        SAFE(check_total_size(res), WARN);
        if (v_prim[1]) SAFE(check_total_size(res), WARN);
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb, res), WARN);
        if (v_prim[1]) { SAFE(check_caches(v_prim[1], prb, res), WARN); }
    }
    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb, dir_t dir) {
    std::vector<data_kind_t> check_kinds;
    if (dir & FLAG_FWD) {
        // Skip forward validation when it runs as service for backward.
        if (!(prb->dir & FLAG_BWD)) check_kinds = {DST};
    } else {
        // Backward: check diff_Q (SRC), diff_K (SRC_1), diff_V (SRC_2).
        check_kinds = {SRC, SRC_1, SRC_2};
    }
    get_kinds_to_check_shared(check_kinds, prb->attr);
    return check_kinds;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

    const auto &prim = prb->dir & FLAG_FWD ? v_prim[0] : v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;

    init_memory_args<prb_t>(
            mem_map, prb, v_prim[0], supported_exec_args(FLAG_FWD));

    {
        auto scale_md = dnn_mem_t::init_host_scalar_md(dnnl_f32);
        mem_map.emplace(DNNL_ARG_SCALE, dnn_mem_t(scale_md));
    }

    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, v_prim[0], prb, res),
            WARN));

    // Reference-only buffer holding the per-element conditioning magnitude
    // sum_k prob_k*|V_k| (same layout as the reference DST). compute_ref fills
    // it; setup_cmp reads it to size a per-element DST threshold. Forward only.
    const auto &ref_dst = ref_mem_map.at(DNNL_ARG_DST);
    ref_mem_map.emplace(SDPA_REF_ARG_OUT_ABSMAG,
            dnn_mem_t(ref_dst.md_, get_cpu_engine(), /* prefill = */ false));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(v_prim[0], args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb, FLAG_FWD), args, ref_args,
            setup_cmp, res, FLAG_FWD);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb, FLAG_FWD), args, prb->attr,
                 prb->inplace, res),
            WARN);

    if (prb->dir & FLAG_BWD) {
        // Extend memory map with backward args.
        init_memory_args<prb_t>(
                mem_map, prb, v_prim[1], supported_exec_args(FLAG_BWD));

        // Re-add scale (pruned by init_memory_args since SCALE is not in
        // exec_bwd_args; init_ref_memory_args will fill the value).
        {
            auto scale_md = dnn_mem_t::init_host_scalar_md(dnnl_f32);
            mem_map.emplace(DNNL_ARG_SCALE, dnn_mem_t(scale_md));
        }

        TIME_FILL(SAFE(
                init_ref_memory_args(ref_mem_map, mem_map, v_prim[1], prb, res),
                WARN));

        args = args_t(mem_map);
        ref_args = args_t(ref_mem_map);

        SAFE(run_execution(v_prim[1], args, res), WARN);

        check_correctness(prb, get_kinds_to_check(prb, FLAG_BWD), args,
                ref_args, setup_cmp, res, FLAG_BWD);
        SAFE(check_bitwise(prim, get_kinds_to_check(prb, FLAG_BWD), args,
                     prb->attr, prb->inplace, res),
                WARN);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace sdpa
