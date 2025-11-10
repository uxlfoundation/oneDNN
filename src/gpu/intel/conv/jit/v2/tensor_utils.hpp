/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_CONV_JIT_V2_TENSOR_UTILS_HPP
#define GPU_INTEL_CONV_JIT_V2_TENSOR_UTILS_HPP

#include "gpu/intel/jit/ir/problem.hpp"
#include "gpu/intel/jit/ir/v2/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {

using namespace intel::jit;

namespace v2 {

using namespace intel::jit::v2;

layout_desc_t make_layout_desc(
        tensor_kind_t tensor_kind, bool src_dst_with_group = false);
layout_desc_t make_algo_layout_desc(
        prop_kind_t prop, tensor_kind_t tensor_kind);
layout_tag_t make_layout_tag(tensor_kind_t tensor_kind, const std::string &s);
layout_tag_t make_layout_tag(
        tensor_kind_t tensor_kind, dim_idx_t ndims, const memory_desc_t &md);
v2::layout_t make_layout(tensor_kind_t tensor_kind, const layout_tag_t &_tag,
        bool is_dw, const prb_reqs_t &reqs, uint32_t mask = 0xFFFFFFFF);
layout_tag_t append_groups(
        tensor_kind_t tensor_kind, const layout_tag_t &layout_tag, bool is_dw);

class dim_mapper_manager_t {
public:
    dim_mapper_manager_t() = default;
    dim_mapper_manager_t(prop_kind_t prop, const prb_reqs_t &reqs);
    const dim_mapper_t &mapper(tensor_kind_t tensor) const;

private:
    expr_t kw_idx = index_var(pvars::kw);
    expr_t kh_idx = index_var(pvars::kh);
    expr_t kd_idx = index_var(pvars::kd);
    expr_t id_idx = index_var(pvars::id);
    expr_t ih_idx = index_var(pvars::ih);
    expr_t iw_idx = index_var(pvars::iw);
    expr_t od_idx = index_var(pvars::od);
    expr_t oh_idx = index_var(pvars::oh);
    expr_t ow_idx = index_var(pvars::ow);

    dim_mapper_t init_src_mapper() const;
    dim_mapper_t init_wei_mapper() const;
    dim_mapper_t init_dst_mapper() const;
    dim_mapper_t init_bias_mapper() const;

    prop_kind_t prop_ = prop_kind::undef;
    prb_reqs_t reqs_;
    dim_mapper_t src_mapper_;
    dim_mapper_t wei_mapper_;
    dim_mapper_t dst_mapper_;
    dim_mapper_t bias_mapper_;
};

dim_mapper_t extend_mapper(
        const dim_mapper_t &mapper, const pvar_t &extra_dim, char letter);

std::vector<pvar_t> skip_mask(
        const v2::view_t &view, const tile_t &tile, const prb_reqs_t &reqs);

} // namespace v2
} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
