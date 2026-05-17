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

#include "common/gemm_arg.hpp"
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

// Rekey user matmul-named attr keys (SRC/WEIGHTS/DST) to gemm A/B/C.
// Called on a pd-local attr copy at the wrapper -> gemm boundary.
inline void rekey_attr_for_gemm(primitive_attr_t &a) {
    auto rekey = [](quant_entries_t &q) {
        q.swap_entries(DNNL_ARG_SRC, gemm_arg::A);
        q.swap_entries(DNNL_ARG_WEIGHTS, gemm_arg::B);
        q.swap_entries(DNNL_ARG_DST, gemm_arg::C);
    };
    rekey(a.scales_);
    rekey(a.zero_points_);
    rekey(a.precomputed_reductions_);
}

// Resolve a slot pointer into a reference (empty_storage if unbound).
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

    // Swaps all A<->B pointer pairs when swap_ab is set. Self-inverse.
    void route_by_swap_ab(bool swap_ab) {
        if (!swap_ab) return;
        std::swap(a, b);
        std::swap(a_zero_point, b_zero_point);
        std::swap(a_scales, b_scales);
        std::swap(a_group_sums, b_group_sums);
    }
};

struct exec_ctx_t : impl::exec_ctx_t {
    exec_ctx_t(impl::stream_t *stream, const exec_args_t &args,
            const desc_t *desc = nullptr)
        : impl::exec_ctx_t(stream), args_(args), desc_(desc) {}
    exec_ctx_t(const impl::exec_ctx_t &other, const exec_args_t &args,
            const desc_t *desc = nullptr)
        : impl::exec_ctx_t(other, {}), args_(args), desc_(desc) {}

    const exec_args_t &args() const { return args_; }
    const desc_t *desc() const { return desc_; }

private:
    exec_args_t args_;
    const desc_t *desc_ = nullptr;
};

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
