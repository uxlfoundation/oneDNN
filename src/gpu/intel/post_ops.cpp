/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/post_ops.hpp"
#include "gemmstone/dsl/ir/object.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace post_op {

std::string relative_md_t::ocl_defines(const std::string &prefix,
        const std::array<std::string, MAX_NDIMS> &strides, int ndims) const {
    std::array<int, MAX_NDIMS> current_alignment = {1, 1, 1, 1, 1, 1};
    std::array<const char *, MAX_NDIMS> args
            = {"(x0)", "(x1)", "(x2)", "(x3)", "(x4)", "(x5)"};
    std::string offset;
    int inner_stride = 1;
    for (int i = 0; i < blocking_t::max_dims; i++) {
        const idx_t idx = inner_layout.idxs[i];
        const uint8_t block = inner_layout.blocks[i];
        if (!idx.is_unset()) {
            auto md_idx = to_md_idx(idx, ndims);
            auto &align = current_alignment[md_idx];
            auto total_block = align * block;

            if (!offset.empty()) offset += "+";

            offset += args[md_idx];

            if (total_block != 1) offset += "%" + std::to_string(total_block);
            if (align != 1) offset += "/" + std::to_string(align);
            if (align * inner_stride != 1)
                offset += "*" + std::to_string(align * inner_stride);

            align = total_block;
            inner_stride *= block;
        }
    }

    for (int i = 0; i < MAX_NDIMS; i++) {
        if (!is_broadcast(i, ndims)) {
            auto &align = current_alignment[i];
            if (!offset.empty()) offset += "+";
            offset += args[i];
            if (align != 1) offset += "/" + std::to_string(align);
            offset += "*"
                    + (is_inner_dim(i, ndims) ? std::to_string(inner_stride)
                                              : strides[i]);
        }
    }

    std::string offset_macro = " -D" + prefix + "_RMD_OFF(x0,x1,x2,x3,x4,x5)="
            + (offset.empty() ? "0" : offset) + "";

    return offset_macro;
}

gemmstone::dsl::ir::expr_t relative_md_t::get_offset(
        const std::vector<gemmstone::dsl::ir::expr_t> &dim_idxs,
        const std::vector<gemmstone::dsl::ir::expr_t> &strides) const {
    using namespace gemmstone::dsl::ir;
    std::array<int, MAX_NDIMS> current_alignment = {1, 1, 1, 1, 1, 1};
    expr_t offset = 0;
    int inner_stride = 1;
    int ndims = into<int>(dim_idxs.size());
    for (int i = 0; i < relative_md_t::blocking_t::max_dims; i++) {
        const relative_md_t::idx_t idx = inner_layout.idxs[i];
        const uint8_t block = inner_layout.blocks[i];
        if (!idx.is_unset()) {
            auto md_idx = relative_md_t::to_md_idx(idx, ndims);
            auto &align = current_alignment[md_idx];
            auto total_block = align * block;

            expr_t block_offset = dim_idxs[md_idx];
            if (total_block != 1) block_offset = block_offset % total_block;
            if (align != 1) block_offset = block_offset / align;
            if (align * inner_stride != 1)
                block_offset = block_offset * (align * inner_stride);

            offset = offset + block_offset;

            align = total_block;
            inner_stride *= block;
        }
    }

    for (int i = 0; i < ndims; i++) {
        if (!is_broadcast(i, ndims)) {
            auto &align = current_alignment[i];
            expr_t dim_offset = dim_idxs[i];
            if (align != 1) dim_offset = dim_offset / align;
            dim_offset = dim_offset
                    * (is_inner_dim(i, ndims) ? inner_stride : strides[i]);
            offset = offset + dim_offset;
        }
    }

    return offset;
}

} // namespace post_op
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
