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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_REORDER_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_REORDER_HPP

#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#include "graph/backend/dnnl/common.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

enum class sdp_reorder_hint_t { none, matmul_src, matmul_weights, matmul_dst };

// Selects between aliasing and a real SDPA reorder based on descriptor and
// attribute semantics, consumer layout support, and the SDPA densification
// policy. A real reorder primitive is created only when required.
struct sdp_decomp_reorder_t {
public:
    status_t init(const dnnl::engine &engine, const memory::desc &src_md,
            const memory::desc &dst_md, const primitive_attr &attr,
            sdp_reorder_hint_t hint = sdp_reorder_hint_t::none);

    bool is_alias() const { return is_alias_; }
    const memory::desc &scratchpad_desc() const { return scratchpad_md_; }

    status_t execute(const dnnl::stream &astream,
            const std::unordered_map<int, dnnl::memory> &args) const;

private:
    dnnl::primitive reorder_prim_;
    memory::desc scratchpad_md_;
    bool is_alias_ = false;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
