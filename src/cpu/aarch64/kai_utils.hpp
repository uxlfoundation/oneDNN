/*******************************************************************************
* Copyright 2023-2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_KAI_UTILS_HPP
#define CPU_AARCH64_KAI_UTILS_HPP

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "kai/ops/gemm/gemm_common.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include "kai/ops/gemm/ndrange.hpp"
#include "kai/ops/newgemm_lib.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace kai_utils {

struct post_ops_fusion_t {
    int fallback_start_index = 0;
    bool accumulate = false;
    kai::ops::Activation activation;

    bool has_fallback(const post_ops_t &post_ops) const {
        return post_ops.len() > fallback_start_index;
    }
};

struct thread_partition_t {
    // KleidiAI exposes work as a 6D window. We currently split only dimensions 0 and
    // 1; the remaining dimensions stay wholly assigned to each participating
    // thread.
    //
    // oneDNN's balance2D takes the first balanced dimension followed by the
    // second balanced dimension and calls the second dimension's thread-group
    // count nx_divider. Here that is nthr_dim1. Each dim1 group contains
    // nthr_dim0 threads, which balance2D uses to split window dimension 0.
    // KleidiAI also needs the per-dimension thread coordinates and counts in
    // thread_locator so kernels with internal barriers or per-thread scratch can
    // identify their position in this 2D team.
    // nthr is the active KAI worker count; team_nthr is the OpenMP team size
    // requested from oneDNN. Keep team_nthr at max-or-1 to avoid libgomp
    // overhead from spawning different team sizes across primitives.
    int nthr = 1;
    int team_nthr = 1;
    unsigned int nthr_dim0 = 1;
    unsigned int nthr_dim1 = 1;
};

CPUInfo *get_cpu_info();

int interleave_by(const kai::ops::WeightFormat wf);
int block_by(const kai::ops::WeightFormat wf);

bool is_fixed_format(kai::ops::WeightFormat wf);
bool is_bf16_weight_format(kai::ops::WeightFormat wf);
bool is_bf16(const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const primitive_attr_t &attr);
bool use_fast_mode(const memory_desc_t &src_md, const primitive_attr_t &attr);

int threads_for_work(double work, double min_work_per_thread, int max_threads);
int threads_for_kernel_execute(double work, int max_threads);

void parallel_pretranspose_B_array(kai::ops::IGemmCommon &kernel, void *out,
        const void *in, int row_stride, int multi_stride, bool transposed,
        int num_threads);

// Create a post ops fusion struct which describes what a kai op will fuse
// into the kernel itself and what will go to fallback. Note that this cannot fail
// because the worst case is that everything goess to fallback.
post_ops_fusion_t create_post_ops_fusion(
        const post_ops_t &post_ops, bool allow_sum_fusion);

int num_sum_post_ops(const post_ops_t &po);

std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm(
        const kai::ops::GemmArgs &base_args, data_type_t src_dt,
        data_type_t weights_dt, data_type_t dst_dt, int max_threads = 0);

std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm_dequant(
        const kai::ops::GemmArgs &base_args, data_type_t src_dt,
        data_type_t weights_dt, data_type_t dst_dt,
        const kai::ops::DequantizeFloat &dequant, int max_threads = 0);

thread_partition_t make_thread_partition(
        int requested_nthr, const kai::ops::ndrange_t &window_size);

void parallel_execute(kai::ops::IGemmCommon &kernel,
        const kai::ops::ndrange_t &window_size,
        const thread_partition_t &partition);

void weight_format_to_memory_desc(memory_desc_t &md, kai::ops::WeightFormat wf,
        dim_t i_dim, dim_t o_dim, const std::vector<dim_t> &spatial_dims,
        const std::vector<dim_t> &batch_dims = {});

bool memory_desc_matches_weight_format(const memory_desc_t &md,
        kai::ops::WeightFormat wf, dim_t i_dim, dim_t o_dim,
        const std::vector<dim_t> &spatial_dims,
        const std::vector<dim_t> &batch_dims = {});

} // namespace kai_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
