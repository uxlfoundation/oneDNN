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

#include "gpu/intel/jit/dsl/decl.hpp"
#include "gpu/intel/jit/dsl/tensor.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/message_patterns.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace dsl {

int grf_size();
int min_align_2d();
int min_pitch_2d();

using layout_t = dnnl::impl::gpu::intel::jit::layout_t;
using expr_t = dnnl::impl::gpu::intel::jit::expr_t;

struct send_hint_t {
    send_cache_hint_t cache;
};

struct tensor_t {
    tensor_t() = default;
    tensor_t(const expr_t &buf, const layout_t &layout)
        : buf(buf), layout(layout) {}
    const type_t &type() const { return layout.type(); }
    tensor_t sub(const icoord_t &coord, const tile_t &tile) const {
        // coord is not measured relative to tile size
        for (auto &var : coord)
            gpu_assert(coord[var] % tile[var] == 0);
        return {subbuf(coord), layout.sub(tile)};
    }
    expr_t subbuf(const icoord_t &coord) const {
        return buf[offset_in_bytes(layout, coord)];
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "buffer:    " << buf.str() << "\n";
        oss << "layout: " << layout.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()

    expr_t buf;
    layout_t layout;
};

struct global_tensor_t {
    expr_t buf;
    type_t type;
    expr_t base_offset;
    coord_t coord;
    idx_map_t<expr_t> sizes;
    idx_map_t<expr_t> strides;
    tile_t tile;

    global_tensor_t() = default;
    global_tensor_t(const expr_t &buf, const idx_map_t<expr_t> &sizes,
            const idx_map_t<expr_t> &strides)
        : buf(buf), type(buf.type().scalar()), sizes(sizes), strides(strides) {}
    global_tensor_t(const expr_t &buf, const type_t &type,
            const expr_t &base_offset, const coord_t &coord,
            const idx_map_t<expr_t> &sizes, const idx_map_t<expr_t> &strides,
            const tile_t &tile)
        : buf(buf)
        , type(type)
        , base_offset(base_offset)
        , coord(coord)
        , sizes(sizes)
        , strides(strides)
        , tile(tile) {}

    expr_t offset(const icoord_t &sub_coord) const {
        expr_t ret = base_offset;
        for (auto &c : sub_coord) {
            ret += (coord[c] + sub_coord[c]) * strides[c];
        }
        return simplify(ret * type.size());
    }

    global_tensor_t map(const tile_t &tile, const coord_t &coord) const {
        global_tensor_t ret = *this;
        ret.coord = coord;
        ret.tile = tile;
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "(" << buf << "+" << base_offset << ")." << type << " : ";
        for (auto &k : coord) {
            oss << " " << k << " - (coord: " << coord[k]
                << ", stride: " << strides[k] << ", size: " << sizes[k];
            if (!tile.is_empty()) oss << ", tile: " << tile[k];
            oss << ")";
        }
        return oss.str();
    }
};

struct kernel_t {
    kernel_t() : iface("invalid_dsl_kernel") {}
    kernel_t(kernel_iface_t iface, stmt_t body, const exec_config_t &exec_cfg)
        : iface(std::move(iface)), body(std::move(body)), exec_cfg(exec_cfg) {}

    kernel_iface_t iface;
    stmt_t body;
    exec_config_t exec_cfg;
    ngen::DebugConfig debug_cfg;
};

void declare_kernel(const kernel_iface_t &interface, ir_context_t &ctx,
        bool new_ir_api = false);
kernel_t end_kernel();

void begin_scope();
void end_scope();
stmt_t pop_scope(); // Ends current scope and removes it from the kernel
void append(stmt_t stmt); // Adds statement to the current scope

void assume(const expr_t &e);

const std::array<expr_t, 3> &group_ids();
const expr_t &group_id(int idx);
const std::array<expr_t, 3> &local_ids();
const expr_t &local_id(int idx);
const std::array<expr_t, 3> &local_sizes();
const expr_t &local_size(int idx);

class lval_t {
public:
    lval_t() = default;
    lval_t(const type_t &type, const std::string &name)
        : var(var_t::make(type, name)) {}
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
    DEFINE_BINARY_ASSIGN_OPERATOR(|)
    DEFINE_BINARY_ASSIGN_OPERATOR(&)
    DEFINE_BINARY_ASSIGN_OPERATOR(^)

#undef DEFINE_BINARY_ASSIGN_OPERATOR

    std::string str() const {
        std::ostringstream oss;
        oss << "lval->var: " << var.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()
    expr_t var;
};

expr_t subgroup_id(int idx = 0);
expr_t arg(const std::string &name, bool allow_empty = false);
// TODO: Unify def() API, keep three versions:
// 1. def(name, type, value)
// 2. def(name, type)
// 3. def(name, value)
// name goes first in all three for consistency.
lval_t def(type_t type, const std::string &name, const expr_t &value = {},
        bool force_alloc = false);
lval_t def(
        const std::string &name, const type_t &type, const expr_t &value = {});
lval_t def(const std::string &name, const expr_t &value);
tensor_t def(const layout_t &layout, const std::string &name,
        const expr_t &value = {});
expr_t let(type_t type, const std::string &name, const expr_t &value);
expr_t let(const std::string &name, const expr_t &value);
tensor_t def_slm(layout_t layout, const std::string &name);

expr_t iif(
        const expr_t &cond, const expr_t &true_expr, const expr_t &false_expr);
expr_t extract(const expr_t &expr, int lane);

void assign(const expr_t &var, const expr_t &value);

void prefetch(const global_tensor_t &g, const icoord_t &base = {},
        const send_hint_t &hint = {});
void load(const tensor_t &t, const global_tensor_t &g,
        const icoord_t &base = {}, const send_hint_t &hint = {});
void store(const global_tensor_t &g, const tensor_t &t,
        const icoord_t &base = {}, const send_hint_t &hint = {});

void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B,
        const tile_t &tile, const icoord_t &base, bool is_systolic);

template <typename F>
void _if(const expr_t &cond, F if_body) {
    if (is_const(cond)) {
        if (to_bool(cond)) {
            begin_scope();
            if_body();
            end_scope();
        }
    } else {
        begin_scope();
        if_body();
        append(if_t::make(cond, pop_scope()));
    }
}

template <typename F, typename G>
void _if(const expr_t &cond, const F &if_body, const G &else_body) {
    if (is_const(cond)) {
        begin_scope();
        if (to_bool(cond)) {
            if_body();
        } else {
            else_body();
        }
        end_scope();
    } else {
        begin_scope();
        if_body();
        auto if_body_stmt = pop_scope();

        begin_scope();
        else_body();
        append(if_t::make(cond, if_body_stmt, pop_scope()));
    }
}

template <typename F>
void _for(const expr_t &var, const expr_t &bound, const expr_t &step,
        const F &body) {
    begin_scope();
    body();
    append(for_t::make(var, 0, bound, pop_scope(), step));
}

template <typename F>
void _for(const expr_t &var, const expr_t &bound, const F &body) {
    _for(var, bound, 1, body);
}

template <typename F>
void _while(const expr_t &cond, F body) {
    if (is_const(cond) && !to_bool(cond)) return;
    begin_scope();
    body();
    append(while_t::make(cond, pop_scope()));
}

void binary(op_kind_t op, const tensor_t &dst, const tensor_t &src0,
        const tensor_t &src1);

void barrier();

} // namespace dsl
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
