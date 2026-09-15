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

#include "graph/backend/dnnl/kernels/sdp_decomp_reorder.hpp"

#include "common/primitive_attr.hpp"
#include "graph/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t sdp_decomp_reorder_t::init(const dnnl::engine &engine,
        const memory::desc &src_md, const memory::desc &dst_md,
        const primitive_attr &attr, sdp_reorder_hint_t hint) {
    const bool has_value_transform
            = src_md.get_data_type() != dst_md.get_data_type()
            || !attr.get()->has_default_values();

    bool direct_layout_supported = src_md == dst_md;
    if (hint != sdp_reorder_hint_t::none) {
        const bool is_dst = hint == sdp_reorder_hint_t::matmul_dst;
        const auto &user_md = is_dst ? dst_md : src_md;
        const auto strides = user_md.get_strides();
        direct_layout_supported = is_dst ? strides.back() == 1
                                         : strides[strides.size() - 1] == 1
                        || strides[strides.size() - 2] == 1;
    }

    // Setting the internal testing control to 1 restores the legacy policy,
    // which aliases only when the source and destination descriptors match.
    const bool use_legacy_policy = graph::utils::getenv_int_internal(
                                           "GRAPH_SDPA_DECOMP_FORCE_DENSIFY", 0)
            == 1;
    const bool densify = has_value_transform || !direct_layout_supported
            || (use_legacy_policy && src_md != dst_md);

    is_alias_ = !densify;
    if (is_alias_) return status::success;

    primitive_attr reorder_attr = attr;
    reorder_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pd = reorder::primitive_desc(
            engine, src_md, engine, dst_md, reorder_attr);
    reorder_prim_ = reorder(pd);
    scratchpad_md_ = pd.scratchpad_desc();
    return status::success;
}

status_t sdp_decomp_reorder_t::execute(const dnnl::stream &astream,
        const std::unordered_map<int, dnnl::memory> &args) const {
    if (is_alias_) {
        void *handle = args.at(DNNL_ARG_SRC).get_data_handle();
        args.at(DNNL_ARG_DST).set_data_handle(handle);
        return status::success;
    }

    return dnnl_primitive_execute_without_tp_hook(reorder_prim_, astream, args);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
