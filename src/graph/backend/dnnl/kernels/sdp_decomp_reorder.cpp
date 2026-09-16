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
namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t sdp_decomp_reorder_t::init(const dnnl::engine &engine,
        const memory::desc &src_md, const memory::desc &dst_md,
        const primitive_attr &attr) {
    const bool has_value_transform
            = src_md.get_data_type() != dst_md.get_data_type()
            || !attr.get()->has_default_values();
    is_alias_ = src_md == dst_md && !has_value_transform;
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
