/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_CONV_JIT_V2_BRIDGE_HPP
#define GPU_INTEL_CONV_JIT_V2_BRIDGE_HPP

#include "gpu/intel/conv/config.hpp"
#include "gpu/intel/conv/jit/v2/plan.hpp"
#include "gpu/intel/conv/jit/v2/problem.hpp"
#include "gpu/intel/conv/jit/v2/tensor_utils.hpp"
#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/ir/v2/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {
namespace v2 {

using namespace intel::jit::v2;

inline tile_t to_shape(const pd_t *pd) {
    tile_t shape;
    shape[pvars::mb] = pd->MB();
    shape[pvars::ic] = ir_utils::safe_div(pd->IC(), pd->G());
    shape[pvars::oc] = ir_utils::safe_div(pd->OC(), pd->G());
    shape[pvars::g] = pd->G();
    shape[pvars::id] = pd->ID();
    shape[pvars::ih] = pd->IH();
    shape[pvars::iw] = pd->IW();
    shape[pvars::od] = pd->OD();
    shape[pvars::oh] = pd->OH();
    shape[pvars::ow] = pd->OW();
    shape[pvars::kd] = pd->KD();
    shape[pvars::kh] = pd->KH();
    shape[pvars::kw] = pd->KW();
    shape[pvars::sd] = pd->KSD();
    shape[pvars::sh] = pd->KSH();
    shape[pvars::sw] = pd->KSW();
    shape[pvars::dd] = pd->KDD();
    shape[pvars::dh] = pd->KDH();
    shape[pvars::dw] = pd->KDW();
    shape[pvars::pd] = pd->padFront();
    shape[pvars::ph] = pd->padT();
    shape[pvars::pw] = pd->padL();
    memory_desc_wrapper mdw_src(pd->invariant_src_md());
    memory_desc_wrapper mdw_wei(pd->invariant_wei_md());
    memory_desc_wrapper mdw_dst(pd->invariant_dst_md());

    bool src_strided = (mdw_src.is_plain() && !mdw_src.is_dense());
    bool wei_strided = (mdw_wei.is_plain() && !mdw_wei.is_dense());
    bool dst_strided = (mdw_dst.is_plain() && !mdw_dst.is_dense());
    auto get_stride = [&](const memory_desc_t *md, int index,
                              int non_spatial_ndims = 2) -> dim_t {
        if (pd->ndims() > index) {
            index = index >= non_spatial_ndims
                    ? pd->ndims() - (index + 1 - non_spatial_ndims)
                    : index;
            if (md->dims[index] == 1) return 0;
            return md->format_desc.blocking.strides[index];
        }
        return 0;
    };
    if (src_strided) {
        shape[prb_stride(pvars::mb, tensor_kind_t::src)]
                = get_stride(pd->invariant_src_md(), 0);
        shape[prb_stride(pvars::ic, tensor_kind_t::src)]
                = get_stride(pd->invariant_src_md(), 1);
        shape[prb_stride(pvars::g, tensor_kind_t::src)]
                = shape[prb_stride(pvars::ic, tensor_kind_t::src)]
                * shape[pvars::ic];
        shape[prb_stride(pvars::id, tensor_kind_t::src)]
                = get_stride(pd->invariant_src_md(), 4);
        shape[prb_stride(pvars::ih, tensor_kind_t::src)]
                = get_stride(pd->invariant_src_md(), 3);
        shape[prb_stride(pvars::iw, tensor_kind_t::src)]
                = get_stride(pd->invariant_src_md(), 2);
    }
    if (wei_strided) {
        shape[prb_stride(pvars::g, tensor_kind_t::wei)] = pd->with_groups()
                ? get_stride(pd->invariant_wei_md(), 0, 3)
                : 0;
        shape[prb_stride(pvars::oc, tensor_kind_t::wei)]
                = get_stride(pd->invariant_wei_md(), 1, 3);
        shape[prb_stride(pvars::ic, tensor_kind_t::wei)]
                = get_stride(pd->invariant_wei_md(), 2, 3);
        shape[prb_stride(pvars::kd, tensor_kind_t::wei)]
                = get_stride(pd->invariant_wei_md(), 5, 3);
        shape[prb_stride(pvars::kh, tensor_kind_t::wei)]
                = get_stride(pd->invariant_wei_md(), 4, 3);
        shape[prb_stride(pvars::kw, tensor_kind_t::wei)]
                = get_stride(pd->invariant_wei_md(), 3, 3);
    }
    if (dst_strided) {
        shape[prb_stride(pvars::mb, tensor_kind_t::dst)]
                = get_stride(pd->invariant_dst_md(), 0);
        shape[prb_stride(pvars::oc, tensor_kind_t::dst)]
                = get_stride(pd->invariant_dst_md(), 1);
        shape[prb_stride(pvars::g, tensor_kind_t::dst)]
                = shape[prb_stride(pvars::oc, tensor_kind_t::dst)]
                * shape[pvars::oc];
        shape[prb_stride(pvars::od, tensor_kind_t::dst)]
                = get_stride(pd->invariant_dst_md(), 4);
        shape[prb_stride(pvars::oh, tensor_kind_t::dst)]
                = get_stride(pd->invariant_dst_md(), 3);
        shape[prb_stride(pvars::ow, tensor_kind_t::dst)]
                = get_stride(pd->invariant_dst_md(), 2);
    }
    return shape;
}

inline problem_t to_problem(const pd_t *pd, const impl::engine_t *engine) {
    auto prop = pd->desc()->prop_kind;
    auto src = make_layout_tag(
            tensor_kind_t::src, pd->ndims(), *pd->invariant_src_md());
    auto wei = make_layout_tag(
            tensor_kind_t::wei, pd->ndims(), *pd->invariant_wei_md());
    auto dst = make_layout_tag(
            tensor_kind_t::dst, pd->ndims(), *pd->invariant_dst_md());
    auto shape = to_shape(pd);

    problem_t prb;
    prb.set_hw(hw_t(engine));
    prb.set_prop(prop);
    prb.set_with_groups(pd->with_groups());
    prb.set_bias_type(to_ir(pd->invariant_bia_md()->data_type));
    prb.set_src_tag(src);
    prb.set_wei_tag(wei);
    prb.set_dst_tag(dst);
    prb.set_shape(shape);
    if (pd->attr()->post_ops_.len() > 0) prb.set_with_post_ops(true);
    prb.set_deterministic(pd->attr()->deterministic_);
    prb.normalize();

    return prb;
}

inline jit::layout_t to_layout(const layout_tag_t &_tag,
        const memory_desc_t &md, bool remove_a_dim = false) {
    auto tag = _tag.raw_tag();
    dim_idx_t non_spatial_ndims = tag.ndims() - 3;
    if (remove_a_dim) {
        tag.remove_dim('a');
        non_spatial_ndims--;
    }
    while (tag.ndims() > into<dim_idx_t>(md.ndims)) {
        tag.remove_dim(dim_idx::as_tag(non_spatial_ndims));
    }
    return jit::layout_t(md, tag.str(), /*do_normalize=*/false);
}

inline jit::layout_t to_layout(const layout_tag_t &_tag, const tile_t &shape) {
    int ndims = _tag.desc().ndims();
    const auto &tag = _tag.raw_tag();
    std::vector<dim_t> dims(ndims);
    for (int i = 0; i < ndims; i++) {
        auto d = _tag.desc().prb_dim(i);
        dims[i] = shape.at(d);
    }
    return jit::layout_t(_tag.type(), expr_t(0), tag.str(), dims);
}

inline jit::grid_info_t to_grid_info(const grid_t &grid, const tile_t &tile) {
    std::vector<dim_t> dims;
    std::vector<expr_t> idxs;
    for (int i = 0; i < grid_t::N; i++) {
        dim_t size = grid.size(i, tile);
        dims.push_back(size);
        idxs.push_back(size == 1 ? 0 : grid.index_var(i));
    }
    return jit::grid_info_t(dims, idxs);
}

} // namespace v2
} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
