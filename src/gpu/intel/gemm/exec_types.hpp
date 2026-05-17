/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_EXEC_TYPES_HPP
#define GPU_INTEL_GEMM_EXEC_TYPES_HPP

#include <utility>

#include "common/memory_storage.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_attr_quant.hpp"
#include "common/primitive_exec_types.hpp"
#include "gpu/intel/gemm/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

#define GEMM_ARG_STORAGE(argument) \
    (args.argument ? *(args.argument) \
                   : dnnl::impl::memory_storage_t::empty_storage())

struct exec_args_t {
    const memory_storage_t *a = nullptr;
    const memory_storage_t *b = nullptr;
    const memory_storage_t *c = nullptr;
    const memory_storage_t *a_zero_point = nullptr;
    const memory_storage_t *b_zero_point = nullptr;
    const memory_storage_t *c_zero_point = nullptr;
    const memory_storage_t *bias = nullptr;
    const memory_storage_t *a_scales = nullptr;
    const memory_storage_t *b_scales = nullptr;
    const memory_storage_t *c_scales = nullptr;
    const memory_storage_t *a_group_sums = nullptr;
    const memory_storage_t *b_group_sums = nullptr;
    const memory_storage_t *sum_ab = nullptr;
    const memory_storage_t *sround_seed = nullptr;
    const memory_storage_t *dropout_mask = nullptr;
    const memory_storage_t *dropout_seed = nullptr;
    const memory_storage_t *dropout_offset = nullptr;
    const memory_storage_t *dropout_prob = nullptr;
    impl::exec_args_t exec_args;

    // Self-inverse: swaps all A<->B pointer pairs.
    void route_by_swap_ab(bool swap_ab) {
        if (!swap_ab) return;
        std::swap(a, b);
        std::swap(a_zero_point, b_zero_point);
        std::swap(a_scales, b_scales);
        std::swap(a_group_sums, b_group_sums);
    }
};

// Build a JIT-kernel-internal exec_args_t from a matmul exec context.
//
// Matmul row-major: SRC = kernel-A (M×K), WEIGHTS = kernel-B (K×N),
// DST = kernel-C. JIT GEMM kernels are column-major; swap_ab flips A↔B (and
// matching attr buffers) so a row-major matmul becomes column-major BLAS.
//
// This replaces the BLAS-style remap previously in gemm::primitive_t::execute
// (`args.a = WEIGHTS, args.b = SRC` — the dnnl_gemm convention) for impls that
// are registered against matmul directly.
//
// Dynamic scales (mx, dynamic_fp) are registered by the framework as OUTPUT
// args even on the src/wei side — the kernel WRITES the per-block scale
// exponents. ctx.input() asserts on non-const args, so we cannot blindly
// route every scale through in_storage(). auto_storage() picks input vs
// output based on the is_const bit the framework recorded.
inline exec_args_t exec_args_from_matmul(
        const impl::exec_ctx_t &ctx, bool swap_ab) {
    const auto in_storage = [&](int arg) -> const memory_storage_t * {
        auto *m = ctx.input(arg);
        return m ? m->memory_storage() : nullptr;
    };
    const auto out_storage = [&](int arg) -> const memory_storage_t * {
        auto *m = ctx.output(arg);
        return m ? m->memory_storage() : nullptr;
    };
    const auto auto_storage = [&](int arg) -> const memory_storage_t * {
        const auto &args_map = ctx.args();
        const auto it = args_map.find(arg);
        if (it == args_map.end()) return nullptr;
        auto *m = it->second.is_const() ? ctx.input(arg) : ctx.output(arg);
        return m ? m->memory_storage() : nullptr;
    };

    exec_args_t args;
    args.a = in_storage(DNNL_ARG_SRC);
    args.b = in_storage(DNNL_ARG_WEIGHTS);
    args.c = out_storage(DNNL_ARG_DST);
    args.bias = in_storage(DNNL_ARG_BIAS);

    // Scales can be input (static) or output (dynamic_mx / dynamic_fp). The
    // framework's arg_usage classifier on primitive_desc_t routes dynamic
    // scales to OUTPUT for any arg (SRC/WEIGHTS/DST). Use auto_storage.
    args.a_scales = auto_storage(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    args.b_scales = auto_storage(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    args.c_scales = auto_storage(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    args.a_zero_point
            = in_storage(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    args.b_zero_point
            = in_storage(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    args.c_zero_point
            = in_storage(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);
    args.a_group_sums = in_storage(
            DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC);
    args.b_group_sums = in_storage(
            DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_WEIGHTS);

    // Reduce slot subsumes the old DNNL_ARG_DIFF_BIAS hack used by ip BWD_W.
    args.sum_ab = out_storage(DNNL_ARG_REDUCE);

    args.sround_seed = in_storage(DNNL_ARG_ATTR_ROUNDING_SEED);
    args.dropout_mask = out_storage(DNNL_ARG_ATTR_DROPOUT_MASK);
    args.dropout_seed = in_storage(DNNL_ARG_ATTR_DROPOUT_SEED);
    args.dropout_prob = in_storage(DNNL_ARG_ATTR_DROPOUT_PROBABILITY);

    args.route_by_swap_ab(swap_ab);
    return args;
}

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
