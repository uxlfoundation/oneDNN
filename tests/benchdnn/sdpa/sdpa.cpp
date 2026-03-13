/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "sdpa/sdpa.hpp"

namespace sdpa {

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> create_md(
        int ndims, const dims_t &dims, dnnl_data_type_t dt,
        const std::string &tag) {
    return dnn_mem_t::init_md(ndims, dims.data(), dt, tag);
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    auto q_dt = force_f32_dt ? dnnl_f32 : prb->q_dt();
    auto k_dt = force_f32_dt ? dnnl_f32 : prb->k_dt();
    auto v_dt = force_f32_dt ? dnnl_f32 : prb->v_dt();
    auto dst_dt_val = force_f32_dt ? dnnl_f32 : prb->dst_dt();

    auto q_d = create_md(prb->ndims, prb->q_dims(), q_dt, prb->qtag);
    auto k_d = create_md(prb->ndims, prb->k_dims(), k_dt, prb->ktag);
    auto v_d = create_md(prb->ndims, prb->v_dims(), v_dt, prb->vtag);
    auto dst_d = create_md(prb->ndims, prb->dst_dims, dst_dt_val, prb->dtag);

    // Attention mask (optional).
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> mask_d {};
    if (prb->with_mask()) {
        mask_d = create_md(prb->ndims, prb->msk_dims, dnnl_f32, tag::abx);
    }

    // Scale memory descriptor - always pass a valid md because the internal
    // create_sdpa_desc() unconditionally dereferences it (no null check).
    auto scale_d = dnn_mem_t::init_host_scalar_md(dnnl_f32);

    int attn_mask_type_val = static_cast<int>(prb->mask_type);
    dnnl_alg_kind_t softmax_alg = dnnl_softmax_accurate;

    // When kv_head_number is 0 (default / standard MHA), the library expects
    // the actual number of KV heads (= K's dim[1]).  A value of 0 would cause
    // a divide-by-zero inside the GPU implementation.
    dnnl_dim_t kv_hn = prb->kv_head_number;
    if (kv_hn == 0) kv_hn = prb->k_dims()[1];

    // When no explicit scale is requested, use invert_scale=false — the scale
    // memory will be filled with 1/sqrt(head_size) in init_ref_memory_args.
    bool invert = prb->with_scale() ? prb->invert_scale() : false;

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t(), prb->ndims));

    TIME_C_PD(DNN_SAFE_STATUS(sdpa_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.engine, q_d, k_d, v_d, dst_d,
            prb->with_mask() ? (const_dnnl_memory_desc_t)mask_d : nullptr,
            scale_d, invert, kv_hn, attn_mask_type_val,
            softmax_alg, dnnl_attr,
            /* kq_attr = */ nullptr, /* vs_attr = */ nullptr)));

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

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type(
            {prb->q_dt(), prb->k_dt(), prb->v_dt(), prb->dst_dt()}, prb->dir,
            res);

    // SDPA is currently only implemented for GPU.
    if (is_cpu()) {
        res->state = SKIPPED;
        res->reason = skip_reason::case_not_supported;
        return;
    }
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // Validate dimension consistency.
    const auto &qdims = prb->q_dims();
    const auto &kdims = prb->k_dims();
    const auto &vdims = prb->v_dims();
    int nd = prb->ndims;

    // Q head_size must match K row dim.
    if (qdims[nd - 1] != kdims[nd - 2]) {
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }
    // K col dim must match V row dim.
    if (kdims[nd - 1] != vdims[nd - 2]) {
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    // SDPA chains two matmuls and softmax; use a generous relative threshold.
    const float trh = 8.f * (1 + prb->n_keys) * epsilon_dt(prb->dst_dt());
    cmp.set_threshold(trh);

    // Near-zero softmax tails can have high rdiff despite tiny absolute diff.
    // Allow elements whose absolute diff < abs_trh even when rdiff > trh.
    // 2e-3f is a small absolute tolerance tuned for fp outputs: it is large
    // enough to ignore numerically insignificant differences in softmax tails
    // while staying in the same order of magnitude as typical gtest float
    // comparison thresholds.
    const float abs_trh = 2e-3f;
    cmp.set_driver_check_function(
            [abs_trh](const compare::compare_t::driver_check_func_args_t &a)
                    -> bool { return a.diff <= abs_trh; });

    cmp.set_zero_trust_percent(90.f);
}

std::vector<int> supported_exec_args(dir_t dir) {
    // DNNL_ARG_SHIFT is DNNL_ARG_ATTN_MASK for SDPA — init_memory_args
    // will skip it when no mask is present (ndims=0 md).
    static const std::vector<int> exec_args = {
            DNNL_ARG_SRC_0, // Q (DNNL_ARG_QUERIES)
            DNNL_ARG_SRC_1, // K (DNNL_ARG_KEYS)
            DNNL_ARG_SRC_2, // V (DNNL_ARG_VALUES)
            DNNL_ARG_DST,
            DNNL_ARG_SHIFT, // Attention mask (DNNL_ARG_ATTN_MASK)
    };
    return exec_args;
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

        // Scale uses host_scalar memory - handle specially.
        if (exec_arg == DNNL_ARG_SCALE) {
            // Create a regular f32 1-element memory for reference.
            dnnl_dims_t s_dims = {1};
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(1, s_dims, dnnl_f32, tag::x, ref_engine,
                            /* prefill = */ false));
            auto &ref_mem = ref_mem_map[exec_arg];
            // SCALE_DIV stores sqrt(H) (the kernel divides, giving 1/sqrt(H)).
            // SCALE_MUL and SCALE_NONE store 1/sqrt(H) directly.
            float scale_val = prb->invert_scale()
                    ? sqrtf(static_cast<float>(prb->head_size))
                    : 1.0f / sqrtf(static_cast<float>(prb->head_size));
            ref_mem.set_f32_elem(0, scale_val);
            // Set same value on device (host_scalar is host-accessible).
            mem.set_f32_elem(0, scale_val);
            continue;
        }

        ref_mem_map.emplace(exec_arg,
                dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                        /* prefill = */ false));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC_0: // Queries
                SAFE(fill_data(exec_arg, SRC, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_SRC_1: // Keys
                SAFE(fill_data(exec_arg, WEI, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_SRC_2: // Values
                SAFE(fill_data(exec_arg, WEI, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DST: break;
            case DNNL_ARG_SHIFT: // Attention mask
                SAFE(fill_data(exec_arg, SRC, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
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
    v_prim.resize(1);
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        SAFE(check_total_size(res), WARN);
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb, res), WARN);
    }
    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds = {DST};
    get_kinds_to_check_shared(check_kinds, prb->attr);
    return check_kinds;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const prb_t *prb, res_t *res) {
    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

    const auto &prim = v_prim[0];

    dnn_mem_map_t mem_map, ref_mem_map;

    init_memory_args<prb_t>(mem_map, prb, prim, supported_exec_args(prb->dir));

    // Scale is a host_scalar memory - not queryable through query_md,
    // so always create it manually.  The GPU kernel requires DNNL_ARG_SCALE
    // at execution time even when no explicit scale knob was set.
    // init_ref_memory_args will fill it with 1/sqrt(head_size).
    {
        auto scale_md = dnn_mem_t::init_host_scalar_md(dnnl_f32);
        mem_map.emplace(DNNL_ARG_SCALE, dnn_mem_t(scale_md));
    }

    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, prim, prb, res), WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(prim, args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb), args, ref_args, setup_cmp,
            res, prb->dir);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->attr,
                 prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace sdpa
