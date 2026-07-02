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

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/numeric.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "gated_mlp/gated_mlp.hpp"

namespace gated_mlp {

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> create_md(
        const dims_t &dims, dnnl_data_type_t dt, const std::string &tag) {
    return dnn_mem_t::init_md(
            static_cast<int>(dims.size()), dims.data(), dt, tag);
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    auto src_dt = force_f32_dt ? dnnl_f32 : prb->src_dt();
    auto w_gate_dt = force_f32_dt ? dnnl_f32 : prb->w_gate_dt();
    auto w_up_dt = force_f32_dt ? dnnl_f32 : prb->w_up_dt();
    auto w_down_dt = force_f32_dt ? dnnl_f32 : prb->w_down_dt();
    auto dst_dt = force_f32_dt ? dnnl_f32 : prb->dst_dt();

    auto src_d = create_md(prb->src_dims, src_dt, prb->stag);
    auto w_gate_d = create_md(prb->w_gate_dims, w_gate_dt, prb->wtag);
    auto w_up_d = create_md(prb->w_up_dims, w_up_dt, prb->wtag);
    auto w_down_d = create_md(prb->w_down_dims, w_down_dt, prb->wtag);
    auto dst_d = create_md(prb->dst_dims, dst_dt, prb->dtag);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr,
            static_cast<int>(prb->dst_dims.size()), prb->dst_dims.data(),
            dnnl_matmul);
    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args, prb->ndims));

    TIME_C_PD(DNN_SAFE_STATUS(dnnl_gated_mlp_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.engine, src_d, w_gate_d, w_up_d,
            w_down_d, dst_d, prb->activation, dnnl_attr)));

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
    // Chained matmul pipeline amplifies magnitudes quadratically (up*gate).
    // ZP adds per-group bias on top. Inflate n_acc to reduce density.
    const bool has_zp = !prb->attr.zero_points.is_def();
    density_args.n_acc = std::max(prb->ic, prb->oc) * (has_zp ? 64 : 4);
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

    // Scale SRC to tame quadratic amplification in the chained pipeline.
    // For int types this is done through scales instead (see below).
    if (kind == SRC && !is_integral_dt(cfg.get_dt(kind))) {
        const float coeff = 0.125f;
        for (int64_t idx = 0; idx < nelems; ++idx) {
            mem_fp.set_f32_elem(idx, mem_fp.get_f32_elem(idx) * coeff);
        }
    }

    SAFE(mem_dt.reorder(mem_fp, cfg.get_swapped_dt(kind)), WARN);
    return OK;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this;
    skip_unimplemented_data_type(
            {prb->src_dt(), prb->w_gate_dt(), prb->w_up_dt(), prb->w_down_dt(),
                    prb->dst_dt()},
            prb->dir, res);

    // GatedMLP is currently only implemented for GPU.
    if (is_cpu()) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb = this;
    // Validate dimensions are positive.
    if (prb->mb <= 0 || prb->ic <= 0 || prb->oc <= 0) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    const int64_t max_acc = std::max(prb->ic, prb->oc);
    const auto dst_dt = prb->dst_dt();

    // Relaxed fpmath may accumulate in f16 precision; error ~ sqrt(K)*eps.
    const bool relaxed_acc = prb->attr.acc_mode != dnnl_accumulation_mode_strict
            && prb->attr.acc_mode != dnnl_accumulation_mode_f32;
    const float eps_acc
            = relaxed_acc ? epsilon_dt(dnnl_f16) : epsilon_dt(dst_dt);
    const float rel_trh = std::max(5e-2f, 3.f * sqrtf(max_acc) * eps_acc);
    cmp.set_threshold(rel_trh);

    // Absolute threshold for near-zero outputs where relative check is
    // too strict. Floor at rel_trh covers GPU-vs-CPU reduction order diffs.
    const float abs_trh = std::max(rel_trh, 20.f * sqrtf(max_acc) * eps_acc);
    const float dst_max = max_dt(dst_dt);
    cmp.set_driver_check_function(
            [abs_trh, dst_max](
                    const compare::compare_t::driver_check_func_args_t &a)
                    -> bool {
        if (a.diff <= abs_trh) return true;
        // Accept overflow saturation in either direction:
        // ref→±inf with GPU→±max, or GPU→±inf with ref→±max.
        if (std::isinf(a.exp) && fabsf(a.got) == dst_max) return true;
        if (std::isinf(a.got) && fabsf(a.exp) == dst_max) return true;
        return false;
    });

    cmp.set_zero_trust_percent(90.f);
}

std::vector<int> prb_t::supported_exec_args(bool override_dir_with_fwd) const {
    static const std::vector<int> exec_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS_GATE,
            DNNL_ARG_WEIGHTS_UP,
            DNNL_ARG_WEIGHTS_DOWN,
            DNNL_ARG_DST,
    };
    return exec_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const prb_t *prb, res_t *res,
        dnnl_primitive_t prim_ref = nullptr) {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, DST});

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive.
        if (exec_arg == DNNL_ARG_SCRATCHPAD) continue;

        ref_mem_map.emplace(exec_arg,
                dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                        /* prefill = */ false));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data(exec_arg, SRC, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_WEIGHTS_GATE:
            case DNNL_ARG_WEIGHTS_UP:
            case DNNL_ARG_WEIGHTS_DOWN:
                SAFE(fill_data(exec_arg, WEI, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(attr_t::post_ops_t::SUM) >= 0) {
                    SAFE(fill_data(exec_arg, DST, prb, cfg, mem, ref_mem, res),
                            WARN);
                }
                break;
            default:
                // Apply coeff to scales to prevent overflow. For int SRC
                // without own scales, go through WEI gate/up scales instead.
                if (exec_arg & DNNL_ARG_ATTR_SCALES) {
                    const int true_arg = exec_arg ^ DNNL_ARG_ATTR_SCALES;
                    const bool is_src_scale = (true_arg == DNNL_ARG_SRC);
                    const bool int_src_no_scale = is_integral_dt(prb->src_dt())
                            && prb->attr.scales.get(DNNL_ARG_SRC).is_def();
                    const bool is_gate_up_scale
                            = (true_arg == DNNL_ARG_WEIGHTS_GATE
                                    || true_arg == DNNL_ARG_WEIGHTS_UP);
                    if (is_src_scale
                            || (int_src_no_scale && is_gate_up_scale)) {
                        const float coeff = 0.125f;
                        dnn_mem_t null;
                        SAFE(fill_scales(
                                     prb->attr, true_arg, null, ref_mem, res),
                                WARN);
                        for (int64_t i = 0; i < ref_mem.nelems(); i++)
                            ref_mem.set_f32_elem(
                                    i, ref_mem.get_f32_elem(i) * coeff);
                        SAFE(mem.reorder(ref_mem), WARN);
                        break;
                    }
                }
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
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    v_prim.resize(1);
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        SAFE(check_total_size(res), WARN);
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb->ctx_init, res), WARN);
    }
    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds = {DST};
    get_kinds_to_check_shared(check_kinds, prb->attr);
    return check_kinds;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

    const auto &prim = v_prim[0];

    dnn_mem_map_t mem_map, ref_mem_map;

    init_memory_args(mem_map, prb, prim);

    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, prim, prb, res), WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(prim, args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb), args, ref_args, compute_ref,
            setup_cmp, res, prb->dir);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->attr,
                 prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace gated_mlp
