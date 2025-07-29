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

#ifndef GPU_INTEL_JIT_DSL_DSL_HPP
#define GPU_INTEL_JIT_DSL_DSL_HPP

#include <stack>

#include "gpu/intel/jit/ir/blocking.hpp"
#include "gpu/intel/jit/ir/fma.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/ir_builder.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/message_patterns.hpp"
#include "gpu/intel/jit/pass/pass.hpp"
#include "gpu/intel/jit/v2/ir/bridge.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"
#include "ngen_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace dsl {

int grf_size();
int min_align_2d();
int min_pitch_2d();

struct transform_t {
    // Sample transforms on bf16 data with pack_size 16:
    // none:           64a64b -> 64a64b
    // block:          64a64b -> 4b64a16b
    // vnni:           64a64b -> 4b32a16b2a
    // transpose_vnni: 64a64b -> 4a32b16a2b
    enum class kind_t { none, block, vnni, transpose_vnni };

    transform_t() = default;

    transform_t(kind_t t_kind, int pack_size, ngen::CacheSettingsLSC cache_hint,
            std::array<pvar_t, 2> dims)
        : kind(t_kind)
        , pack_size(pack_size)
        , cache_hint(to_ir(cache_hint))
        , dims(std::move(dims)) {}

    v2::layout_t get_layout(const tile_t &sizes, type_t type,
            const v2::layout_desc_t &desc) const {

        auto col_var = dims[0];
        auto col = sizes[dims[0]];
        auto row_var = dims[1];
        auto row = sizes[dims[1]];
        auto t = type.size();

        auto normalized = kind;
        if (normalized == kind_t::transpose_vnni) {
            std::swap(col_var, row_var);
            std::swap(col, row);
            normalized = kind_t::vnni;
        }

        if (normalized == kind_t::vnni && t >= 4) normalized = kind_t::block;

        int col_inner = pack_size ? pack_size : grf_size();
        if (normalized == kind_t::block && col <= col_inner)
            normalized = kind_t::none;

        switch (normalized) {
            case kind_t::none:
                return v2::layout_t(desc, type, 0,
                        {{col_var, col, 1}, {row_var, row, col}});

            case kind_t::block: {
                int col_outer = (int)(col / col_inner);
                return v2::layout_t(desc, type, 0,
                        {{col_var, col_inner, 1}, {row_var, row, col_inner},
                                {col_var, col_outer, row * col_inner}});
            }

            case kind_t::vnni: {
                int row_inner = 4 / t;
                int row_outer = (int)(row / row_inner);
                int col_outer = (int)(col / col_inner);
                return v2::layout_t(desc, type, 0,
                        {{row_var, row_inner, 1},
                                {col_var, col_inner, row_inner},
                                {row_var, row_outer, col_inner * row_inner},
                                {col_var, col_outer,
                                        row_outer * col_inner * row_inner}});
            }

            // Impossible to hit due to normalization
            case kind_t::transpose_vnni:
            default: gpu_assert(false); return {};
        }
    }

    // Tile used for 2d messages
    tile_t get_2d_tile(type_t type) const {
        if (kind == kind_t::transpose_vnni) {
            auto width = pack_size ? pack_size
                                   : grf_size() / std::max(type.size(), 4);
            auto height = 32;
            return {{dims[1], width}, {dims[0], height}};
        }

        auto width = pack_size ? pack_size : grf_size() / type.size();
        auto height = 32;
        return {{dims[0], width}, {dims[1], height}};
    }

    // Tile used for block loads
    tile_t get_block_tile(type_t type) const {
        if (kind == kind_t::none) {
            return {{dims[0], 8 * grf_size() / type.size()}, {dims[1], 1}};
        } else if (kind == kind_t::block) {
            return {{dims[0], pack_size ? pack_size : grf_size() / type.size()},
                    {dims[1], 1}};
        } else {
            gpu_assert(false);
            return {};
        }
    }

    static send_cache_hint_t to_ir(ngen::CacheSettingsLSC hint) {
        switch (hint) {
            case ngen::CacheSettingsLSC::L1C_L3C:
                return send_cache_hint_t::load_once;
            case ngen::CacheSettingsLSC::Default:
                return send_cache_hint_t::hw_default;
            default: gpu_assert(false); return send_cache_hint_t::undef;
        }
    }

    kind_t kind = kind_t::none;
    int pack_size = 0;
    send_cache_hint_t cache_hint = send_cache_hint_t::undef;
    std::array<pvar_t, 2> dims = {};
};

struct tensor_t {
    std::string str() const {
        std::ostringstream oss;
        oss << "buffer:    " << buf.str();
        oss << "layout: " << layout.str();
        return oss.str();
    }
    IR_DEFINE_DUMP()
    expr_t buf;
    v2::layout_t layout;
};

struct global_tensor_t {
    expr_t buf;
    type_t type;
    expr_t base_offset;
    pvar_map_t<expr_t> idxs;
    pvar_map_t<expr_t> strides;
    pvar_map_t<expr_t> sizes;
    tile_t tile;

    expr_t offset(const icoord_t &coord) const {
        expr_t ret = base_offset;
        for (auto &c : coord) {
            ret += (idxs[c] + coord[c]) * strides[c];
        }
        return simplify(ret * type.size());
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "(" << buf << "+" << base_offset << ")." << type << " : ";
        for (auto &k : idxs) {
            oss << " " << k << " - (idx: " << idxs[k]
                << ", stride: " << strides[k] << ", size: " << sizes[k];
            if (!tile.is_empty()) oss << ", tile: " << tile[k];
            oss << ")";
        }
        return oss.str();
    }
};

void declare_kernel(const kernel_iface_t &interface, ir_context_t &ctx);
stmt_t end_kernel();

void begin_scope();
void end_scope();
stmt_t pop_scope(); // Ends current scope and removes it from the kernel

void assume(expr_t e);

const std::array<expr_t, 3> &group_ids();
const expr_t &group_id(int idx);
const std::array<expr_t, 3> &local_ids();
const expr_t &local_id(int idx);
const std::array<expr_t, 3> &local_sizes();
const expr_t &local_size(int idx);

class lval_t {

public:
    lval_t(const expr_t &v) : var(v) {}

    lval_t &operator=(const expr_t &obj);

    lval_t sub(int off, int elems) const {
        assert(var.is<var_t>());
        return lval_t(ref_t::make(var, off, elems));
    }
    lval_t operator[](int off) const { return sub(off, 1); }
    operator expr_t() const { return var; }

#define DEFINE_BINARY_ASSIGN_OPERATOR(op) \
    lval_t &operator op##=(const expr_t &rhs) { \
        (*this) = (*this)op rhs; \
        return *this; \
    }

    DEFINE_BINARY_ASSIGN_OPERATOR(+)
    DEFINE_BINARY_ASSIGN_OPERATOR(-)
    DEFINE_BINARY_ASSIGN_OPERATOR(*)
    DEFINE_BINARY_ASSIGN_OPERATOR(/)
    DEFINE_BINARY_ASSIGN_OPERATOR(%)
    DEFINE_BINARY_ASSIGN_OPERATOR(&)

#undef DEFINE_BINARY_ASSIGN_OPERATOR

    std::string str() const {
        std::ostringstream oss;
        oss << "lval->var: " << var.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()
    expr_t var;
};

expr_t arg(const std::string &name);
lval_t def(type_t type, const std::string &name, const expr_t &value = {},
        bool force_alloc = false);
lval_t def(const std::string &name, const expr_t &value);

tensor_t def(const v2::layout_t &layout, const std::string &name,
        const expr_t &value = {});
expr_t let(type_t type, const std::string &name, const expr_t &value);
expr_t let(const std::string &name, const expr_t &value);

void prefetch(const global_tensor_t &g, const transform_t &transform,
        const icoord_t &base);
void load(const tensor_t &t, const global_tensor_t &g,
        const transform_t &transform, const icoord_t &base);
void store(const global_tensor_t &g, const tensor_t &t,
        const transform_t &transform, const icoord_t &base);

void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B,
        const tile_t &tile, const icoord_t &base, bool is_systolic);

void assign(const expr_t &var, const expr_t &value);

template <typename F>
void if_(const expr_t &cond, F if_body) {
    begin_scope();
    if_body();
    if_(cond, pop_scope());
}
template <>
void if_(const expr_t &cond, const stmt_t &if_body);

template <typename F, typename G>
void if_(const expr_t &cond, const F &if_body, const G &else_body) {
    begin_scope();
    if_body();
    auto if_body_stmt = pop_scope();

    begin_scope();
    else_body();
    auto else_body_stmt = pop_scope();

    if_(cond, if_body_stmt, else_body_stmt);
}
template <>
void if_(const expr_t &cond, const stmt_t &if_body, const stmt_t &else_body);

template <typename F>
void while_(const expr_t &cond, F body) {
    begin_scope();
    body();
    while_(cond, pop_scope());
}
template <>
void while_(const expr_t &cond, const stmt_t &body);

} // namespace dsl
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
