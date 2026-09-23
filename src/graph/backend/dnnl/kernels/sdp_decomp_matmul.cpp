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

#include "graph/backend/dnnl/kernels/sdp_decomp_matmul.hpp"

#include <cstring>

#include "common/primitive_attr.hpp"
#include "common/verbose.hpp"
#include "graph/utils/utils.hpp"

#define VDISPATCH_SDP_DECOMP(msg, ...) \
    VINFO(graph, create, dispatch, sdp_decomp_matmul, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

namespace {

bool is_brg_matmul(const matmul::primitive_desc &pd) {
    // TODO(xxx): this check is conservative and specific to x64 platforms.
    return pd && std::strstr(pd.impl_info_str(), "brg_matmul") != nullptr;
}

bool requires_dense_md(const memory::desc &user_md,
        const memory::desc &dense_md, const primitive_attr &attr) {
    const bool has_value_transform
            = user_md.get_data_type() != dense_md.get_data_type()
            || !attr.get()->has_default_values();
    const bool force_densify = graph::utils::getenv_int_internal(
                                       "GRAPH_SDPA_DECOMP_FORCE_DENSIFY", 0)
            == 1;
    return has_value_transform || (force_densify && user_md != dense_md);
}

} // namespace

status_t sdp_decomp_matmul_base_t::init_matmul(const dnnl::engine &engine,
        const memory::desc &direct_src_md,
        const memory::desc &direct_weights_md,
        const memory::desc &direct_dst_md, const memory::desc &dense_src_md,
        const memory::desc &dense_weights_md, const memory::desc &dense_dst_md,
        const primitive_attr &attr) {
    auto pd = matmul::primitive_desc(engine, direct_src_md, direct_weights_md,
            direct_dst_md, attr, /*allow_empty=*/true);
    const bool use_direct = is_brg_matmul(pd);
    if (use_direct) {
        src_md_ = direct_src_md;
        weights_md_ = direct_weights_md;
        dst_md_ = direct_dst_md;
    } else {
        src_md_ = dense_src_md;
        weights_md_ = dense_weights_md;
        dst_md_ = dense_dst_md;
        pd = matmul::primitive_desc(
                engine, src_md_, weights_md_, dst_md_, attr);
    }

    matmul_prim_ = matmul(pd);
    scratchpad_md_ = pd.scratchpad_desc();
    VDISPATCH_SDP_DECOMP("matmul %s layout: impl:%s",
            use_direct ? "direct" : "dense", pd.impl_info_str());
    return status::success;
}

status_t sdp_decomp_matmul_base_t::execute(const dnnl::stream &astream,
        const std::unordered_map<int, dnnl::memory> &args) const {
    return dnnl_primitive_execute_without_tp_hook(matmul_prim_, astream, args);
}

status_t sdp_decomp_bmm1_t::init(const dnnl::engine &engine,
        const memory::desc &query_user_md, const memory::desc &query_dense_md,
        const primitive_attr &query_reorder_attr,
        const memory::desc &key_user_md, const memory::desc &key_dense_md,
        const primitive_attr &key_reorder_attr, const memory::desc &scores_md,
        const primitive_attr &matmul_attr) {
    const bool requires_dense_q = requires_dense_md(
            query_user_md, query_dense_md, query_reorder_attr);
    const bool requires_dense_k
            = requires_dense_md(key_user_md, key_dense_md, key_reorder_attr);
    const auto &direct_query_md
            = requires_dense_q ? query_dense_md : query_user_md;
    const auto &direct_key_md = requires_dense_k ? key_dense_md : key_user_md;
    VDISPATCH_SDP_DECOMP("bmm1: requires_dense query:%d key:%d",
            requires_dense_q, requires_dense_k);

    CHECK(init_matmul(engine, direct_query_md, direct_key_md, scores_md,
            query_dense_md, key_dense_md, scores_md, matmul_attr));
    query_plan_.selected_md = src_desc();
    key_plan_.selected_md = weights_desc();
    return status::success;
}

status_t sdp_decomp_bmm2_t::init(const dnnl::engine &engine,
        const memory::desc &src_md, const memory::desc &value_user_md,
        const memory::desc &value_dense_md,
        const primitive_attr &value_reorder_attr,
        const memory::desc &output_user_md, const memory::desc &output_dense_md,
        const primitive_attr &output_reorder_attr,
        const primitive_attr &matmul_attr) {
    const bool requires_dense_v = requires_dense_md(
            value_user_md, value_dense_md, value_reorder_attr);
    const bool requires_dense_o = requires_dense_md(
            output_user_md, output_dense_md, output_reorder_attr);
    const auto &direct_value_md
            = requires_dense_v ? value_dense_md : value_user_md;
    const auto &direct_output_md
            = requires_dense_o ? output_dense_md : output_user_md;
    VDISPATCH_SDP_DECOMP("bmm2: requires_dense value:%d output:%d",
            requires_dense_v, requires_dense_o);

    CHECK(init_matmul(engine, src_md, direct_value_md, direct_output_md, src_md,
            value_dense_md, output_dense_md, matmul_attr));
    value_plan_.selected_md = weights_desc();
    output_plan_.selected_md = dst_desc();
    return status::success;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
