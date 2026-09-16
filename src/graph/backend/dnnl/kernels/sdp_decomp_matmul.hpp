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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_MATMUL_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_MATMUL_HPP

#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#include "graph/backend/dnnl/common.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct sdp_matmul_operand_plan_t {
    memory::desc selected_md;
};

struct sdp_decomp_matmul_base_t {
public:
    const memory::desc &scratchpad_desc() const { return scratchpad_md_; }

    status_t execute(const dnnl::stream &astream,
            const std::unordered_map<int, dnnl::memory> &args) const;

protected:
    status_t init_matmul(const dnnl::engine &engine,
            const memory::desc &direct_src_md,
            const memory::desc &direct_weights_md,
            const memory::desc &direct_dst_md, const memory::desc &dense_src_md,
            const memory::desc &dense_weights_md,
            const memory::desc &dense_dst_md, const primitive_attr &attr);

    const memory::desc &src_desc() const { return src_md_; }
    const memory::desc &weights_desc() const { return weights_md_; }
    const memory::desc &dst_desc() const { return dst_md_; }

private:
    dnnl::primitive matmul_prim_;
    memory::desc src_md_;
    memory::desc weights_md_;
    memory::desc dst_md_;
    memory::desc scratchpad_md_;
};

struct sdp_decomp_bmm1_t : public sdp_decomp_matmul_base_t {
public:
    status_t init(const dnnl::engine &engine, const memory::desc &query_user_md,
            const memory::desc &query_dense_md,
            const primitive_attr &query_reorder_attr,
            const memory::desc &key_user_md, const memory::desc &key_dense_md,
            const primitive_attr &key_reorder_attr,
            const memory::desc &scores_md, const primitive_attr &matmul_attr);

    const sdp_matmul_operand_plan_t &query_plan() const { return query_plan_; }
    const sdp_matmul_operand_plan_t &key_plan() const { return key_plan_; }

private:
    sdp_matmul_operand_plan_t query_plan_;
    sdp_matmul_operand_plan_t key_plan_;
};

struct sdp_decomp_bmm2_t : public sdp_decomp_matmul_base_t {
public:
    status_t init(const dnnl::engine &engine, const memory::desc &src_md,
            const memory::desc &value_user_md,
            const memory::desc &value_dense_md,
            const primitive_attr &value_reorder_attr,
            const memory::desc &output_user_md,
            const memory::desc &output_dense_md,
            const primitive_attr &output_reorder_attr,
            const primitive_attr &matmul_attr);

    const sdp_matmul_operand_plan_t &value_plan() const { return value_plan_; }
    const sdp_matmul_operand_plan_t &output_plan() const {
        return output_plan_;
    }

private:
    sdp_matmul_operand_plan_t value_plan_;
    sdp_matmul_operand_plan_t output_plan_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
