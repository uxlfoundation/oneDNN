/*******************************************************************************
* Copyright 2022-2024, 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_POST_OPS_FALLBACK_HPP
#define CPU_AARCH64_POST_OPS_FALLBACK_HPP

#include "common/eltwise_pd.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct post_ops_fallback_t {

    post_ops_fallback_t() = default;

    // init the post_ops_fallback_t. Note that this function modifies the passed in
    // post ops by setting the preferred memory formats
    status_t init(const engine_t *engine, post_ops_t &post_ops,
            const memory_desc_t &dst_md, int post_op_start_index = 0);

    bool has_sum() const { return sum_index >= 0; }

    void init_scratchpad(memory_tracking::registrar_t &scratchpad) const;

    status_t execute(
            const exec_ctx_t &ctx, void *src, void *dst = nullptr) const;

private:
    status_t create_binary_primitive(const engine_t *engine,
            const binary_desc_t &binary_desc,
            std::shared_ptr<primitive_t> &primitive) const;

    status_t create_eltwise_primitive(const engine_t *engine,
            const eltwise_desc_t &eltwise_desc,
            std::shared_ptr<primitive_t> &primitive) const;

    status_t execute_binary(const exec_ctx_t &ctx, const primitive_t *post_op,
            const void *src0, const void *src1, const void *src2, void *dst,
            int primitive_index) const;

    status_t execute_eltwise(const exec_ctx_t &ctx, const primitive_t *post_op,
            void *src, int primitive_index) const;

    // Index of the sum post op if there is one, < 0 means no sum
    int sum_index = -1;
    // Index of the first post op this primitive executes. This is typically the
    // number of post ops which were fused.
    int post_op_start_index_ = 0;
    data_type_t dst_data_type;
    // Vector of primitives used to execute the post ops.
    std::vector<std::shared_ptr<primitive_t>> post_op_primitives;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
