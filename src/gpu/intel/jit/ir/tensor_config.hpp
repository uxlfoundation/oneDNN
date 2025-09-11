/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_TENSOR_CONFIG_HPP
#define GPU_INTEL_JIT_IR_TENSOR_CONFIG_HPP

#include <vector>

#include "gpu/intel/jit/ir/post_ops.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct tensor_info_t {
    std::string name;
    int arg_key;
    bool is_input;
    bool is_output;
    layout_t compute_layout;
    layout_t user_layout;

    bool needs_reorder;
    bool needs_zero_out;
};

class tensor_config_t {
public:
    const std::vector<tensor_info_t> &tensors() const { return tensors_; }

    void add_tensor(const std::string &name, int arg_key, bool is_input,
            bool is_output, const layout_t &user_layout) {
        tensors_.emplace_back();
        auto &t = tensors_.back();
        t.name = name;
        t.arg_key = arg_key;
        t.is_input = is_input;
        t.is_output = is_output;
        t.compute_layout = user_layout;
        t.user_layout = user_layout;
        t.needs_reorder = false;
        t.needs_zero_out = false;
    }

    void add_tensor(const std::string &name, int arg_key, bool is_input,
            bool is_output, const layout_t &compute_layout,
            const layout_t &user_layout) {
        tensors_.emplace_back();
        auto &t = tensors_.back();
        t.name = name;
        t.arg_key = arg_key;
        t.is_input = is_input;
        t.is_output = is_output;
        t.compute_layout = compute_layout;
        t.user_layout = user_layout;
        t.needs_reorder = (t.compute_layout != t.user_layout);
        t.needs_zero_out = false;
    }

    void set_compute_layout(
            const std::string &name, const layout_t &compute_layout) {
        auto &t = find_tensor(name);
        t.compute_layout = compute_layout;
        t.needs_reorder = (t.compute_layout != t.user_layout);
    }

    const layout_t &compute_layout(const std::string &name) const {
        return find_tensor(name).compute_layout;
    }

    const layout_t &user_layout(const std::string &name) const {
        return find_tensor(name).user_layout;
    }

    void require_zero_out(const std::string &name) {
        auto &t = find_tensor(name);
        t.needs_zero_out = true;
    }

private:
    const tensor_info_t &find_tensor(const std::string &name) const {
        for (auto &t : tensors_) {
            if (t.name == name) return t;
        }
        gpu_error_not_expected() << "Can't find tensor " << name;
        return tensors_.front();
    }

    tensor_info_t &find_tensor(const std::string &name) {
        auto *const_this = const_cast<const tensor_config_t *>(this);
        return const_cast<tensor_info_t &>(const_this->find_tensor(name));
    }

    std::vector<tensor_info_t> tensors_;
};

// Returns vector of <dimension index, block size> pairs.
std::vector<layout_block_t> parse_format(
        const std::string &format, int ndims_hint);

// Returns vector of <dimension letter, block size> pairs.
std::vector<std::pair<char, dim_t>> parse_letter_blocks(
        const std::string &format);

inline layout_t make_layout(const type_t &type, const expr_t &offset,
        const std::string &format, const std::vector<dim_t> &dims = {}) {
    auto parts = parse_format(format, into<dim_idx_t>(dims.size()));
    std::vector<layout_block_t> blocks;

    // Iterate from right to left (innermost to outermost).
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        auto &dim = it->dim;
        dim_t block = it->block;
        if (block == 0) {
            dim_t full_block = 1;
            for (auto &b : blocks)
                if (b.dim == dim) full_block *= b.block;

            block = utils::div_up(dims[dim], full_block);
        }

        blocks.emplace_back(dim, block);
    }

    return layout_t(type, blocks, offset, into<dim_idx_t>(dims.size()),
            /*do_normalize=*/false);
}

inline layout_t make_layout(const type_t &type, const std::vector<dim_t> &dims,
        const std::string &tag) {
    return make_layout(type, 0, tag, dims);
}

// Note, the default value of do_normalize is the opposite of the default value
// for layout_t. The reason behind this is that most practical uses of this
// interface do not perform normalization.
inline layout_t make_layout(
        const memory_desc_t &md, bool do_normalize = false) {
    if (md.format_kind == format_kind::any) return layout_t();

    auto mdw = memory_desc_wrapper(md);
    block_layout_t layout(
            mdw, /* inner_only */ false, /* do_normalize */ false);
    std::vector<layout_block_t> blocks;
    for (const auto &block : layout) {
        blocks.emplace_back(block.dim_idx, block.block, block.stride);
    }

    return layout_t(to_ir(mdw.data_type()), blocks, mdw.offset0(), mdw.ndims(),
            do_normalize);
}

inline layout_t make_layout(const memory_desc_t &md, const std::string &tag) {
    if (tag == "user") return make_layout(md);
    auto mdw = memory_desc_wrapper(md);
    return make_layout(to_ir(mdw.data_type()), mdw.offset0(), tag,
            std::vector<dim_t>(mdw.dims(), mdw.dims() + mdw.ndims()));
}

bool matches_tag(const layout_t &layout, const std::string &tag,
        const std::vector<dim_t> &dims);

inline bool matches_tag(const layout_t &layout, const std::string &tag) {
    return matches_tag(layout, tag, layout.dims());
}

inline bool matches_tag(const memory_desc_t &md, const std::string &tag) {
    if (md.format_kind == format_kind::any) return false;
    std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
    return matches_tag(make_layout(md), tag, dims);
}

inline void set_default_format(memory_desc_t &md, const std::string &tag) {
    if (md.format_kind != format_kind::any) return;
    md = to_md(make_layout(md, tag), md);
}

inline std::vector<std::pair<const char *, int>> get_scale_args() {
    std::vector<std::pair<const char *, int>> ret = {
            {"src_scales", DNNL_ARG_SRC},
            {"wei_scales", DNNL_ARG_WEIGHTS},
            {"dst_scales", DNNL_ARG_DST},
    };
    return ret;
}

inline std::vector<dim_t> get_prelu_weights_dims(
        uint32_t mask, const memory_desc_t &md) {
    std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
    for (int i = 0; i < md.ndims; ++i)
        dims[i] = (mask & (1 << i)) ? dims[i] : 1;
    return dims;
}

void init_extra_tensors(const zero_points_config_t &zp_cfg,
        const primitive_attr_t &attr, const memory_desc_t *zp_src,
        const memory_desc_t &dst_md, dim_t ic, dim_t oc,
        tensor_config_t &tensor_cfg);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
