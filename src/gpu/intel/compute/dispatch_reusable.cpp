/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "common/c_types_map.hpp"
#include "gemmstone/../../dsl/ir/core.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/data_type_converter.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/logging.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

status_t reusable_dispatch_config_t::define_dim_index(
        name_id_t dim_name, dim_idx_t dim_id, dim_t size) {
    memory_desc_t md = types::zero_md();
    md.ndims = 1;
    md.dims[0] = size;
    md.padded_dims[0] = size;
    md.data_type = data_type::f32; // doesn't matter
    md.format_kind = format_kind::blocked;
    md.format_desc.blocking.strides[0] = 1;
    md.format_desc.blocking.inner_nblks = 0;

    named_buffer_t buf(dim_name, md, {dim_id});
    CHECK(register_buffer(buf));
    return status::success;
}

// Validate whether the given buffer is consistent with existing buffer layouts,
// and then add to the internal buffer registry.
status_t reusable_dispatch_config_t::register_buffer(
        const named_buffer_t &buffer) {
    if (buffers.size() >= MAX_REGISTERED_BUFFERS) return status::unimplemented;

    for (const auto &dim : dispatched_dims) {
        size_t canonical_idx = buffer.get_dim_idx(dim);
        if (canonical_idx == dim_not_found) continue;
        auto size = buffer.dims[canonical_idx];
        auto padded_size = buffer.padded_dims[canonical_idx];
        if (padded_size == 1) continue; // Skip potentially broadcast dimensions

        auto it = dim_sizes.emplace(dim, dim_size_t {size, padded_size});
        if (!it.second) {
            // Support the buffer with the most padding,every dimensions must
            // have a consistent size or be broadcasted
            auto &value = it.first->second;
            gpu_assert(size == value.size || (size == 1 && padded_size == 1));
            value.padded_size = std::max(value.padded_size, padded_size);
        }
    }
    buffers.emplace_back(buffer);
    return status::success;
}

namespace dsl = jit::dsl;
namespace ir = gemmstone::dsl::ir;

const dsl::expr_t &global_ids(size_t idx) {
    static const thread_local std::array<dsl::expr_t, 3> ids([] {
        std::array<dsl::expr_t, 3> ret;
        for (int i = 0; i < 3; i++) {
            ret[i] = ir::var_t::make(
                    dsl::u64, "get_global_id(" + std::to_string(i) + ")");
        }
        return ret;
    }());
    return ids[idx];
}

struct term_registry_t {

    dsl::expr_t add(int64_t value, const std::string &name) {
        auto name_it = name_map.emplace(name, dsl::expr_t());
        auto &expr = name_it.first->second;
        if (name_it.second) expr = ir::var_t::make(dsl::s64, name);

        auto value_it = value_map.emplace(expr, value);
        gpu_assert(value == value_it.first->second);
        return expr;
    }
    std::unordered_map<std::string, dsl::expr_t> name_map;
    ir::object_map_t<dsl::expr_t, int64_t> value_map;
};

// Encodes an dsl::expr into a compressed format optimized for calculating
// buffer offsets. The general format of an expression is a prefix (kind_t)
// followed by a fixed size sequence of bytes corresponding any following
// expressions.
struct expr_encoder_t : public ir::ir_visitor_t {
    uint8_t operator()(
            const dsl::expr_t &offset, const term_registry_t &registry) {
        registry_ = &registry;
        ir::ir_visitor_t::visit(offset);
        return expr_locations[offset];
    }
    const std::vector<uint8_t> &expr_data() const { return expr_data_; }
    const std::vector<int64_t> &term_list() const { return term_list_; }

    static std::string decode(const uint8_t *expr_data, uint8_t offset,
            std::vector<int64_t> *term_list = nullptr) {
        auto dec = [&](uint8_t off) {
            return decode(expr_data, expr_data[offset + off], term_list);
        };
        auto sym = [](uint8_t kind) {
            switch (kind_t(kind)) {
                case kind_t::add: return "+";
                case kind_t::sub: return "-";
                case kind_t::mul: return "*";
                case kind_t::div: return "/";
                case kind_t::mod: return "%";
                case kind_t::_or: return "||";
                case kind_t::_and: return "&&";
                case kind_t::lt: return "<";
                case kind_t::le: return "<=";
                case kind_t::gt: return ">";
                case kind_t::ge: return ">=";
                case kind_t::ne: return "!=";
                case kind_t::eq: return "==";
                default: gpu_error_not_expected(); return "";
            }
        };
        ostringstream_t oss;
        switch (kind_t(expr_data[offset])) {
            case kind_t::add:
            case kind_t::sub:
            case kind_t::mul:
            case kind_t::div:
            case kind_t::mod:
            case kind_t::_or:
            case kind_t::_and:
            case kind_t::lt:
            case kind_t::le:
            case kind_t::gt:
            case kind_t::ge:
            case kind_t::ne:
            case kind_t::eq:
                oss << "(" << dec(1) << sym(expr_data[offset]) << dec(2) << ")";
                break;
            case kind_t::idiv: {
                auto rt_off = expr_data[offset + 2];
                oss << "idiv(" << dec(1) << ","
                    << "rt.params[" + std::to_string(rt_off) << "],"
                    << "rt.params[" + std::to_string(rt_off + 1) << "])";
                break;
            }
            case kind_t::runtime_term:
                oss << "rt.params["
                    << std::to_string(int(expr_data[offset + 1])) << "]";
                if (term_list)
                    oss << "(="
                        << std::to_string((*term_list)[expr_data[offset + 1]])
                        << ")";
                break;
            case kind_t::constant_true: oss << "true"; break;
            case kind_t::constant_false: oss << "false"; break;
            case kind_t::constant_s8:
                oss << std::to_string(int8_t(expr_data[offset + 1]));
                break;
            case kind_t::constant_u8:
                oss << std::to_string(expr_data[offset + 1]) << "u";
                break;
            case kind_t::constant_s16:
                oss << std::to_string(int16_t(
                        expr_data[offset + 1] + (expr_data[offset + 2] << 8)));
                break;
            case kind_t::constant_u16:
                oss << std::to_string(uint16_t(
                        expr_data[offset + 1] + (expr_data[offset + 2] << 8)))
                    << "u";
                break;
            case kind_t::constant_s32:
                oss << std::to_string(int32_t(expr_data[offset + 1]
                        + (expr_data[offset + 2] << 8)
                        + (expr_data[offset + 3] << 16)
                        + (expr_data[offset + 4] << 24)));
                break;
            case kind_t::constant_u32:
                oss << std::to_string(uint32_t(expr_data[offset + 1]
                        + (expr_data[offset + 2] << 8)
                        + (expr_data[offset + 3] << 16)
                        + (expr_data[offset + 4] << 24)))
                    << "u";
                break;

            case kind_t::global_id0: oss << global_ids(0).str(); break;
            case kind_t::global_id1: oss << global_ids(1).str(); break;
            case kind_t::global_id2: oss << global_ids(2).str(); break;
            default: gpu_error_not_expected();
        }
        return oss.str();
    }
    std::string decode(uint8_t offset) {
        return decode(expr_data_.data(), offset, &term_list_);
    }

    void _visit(const ir::unary_op_t &obj) override {
        gpu_except_not_implemented();
    }
    void _visit(const ir::binary_op_t &obj) override {
        auto it = expr_locations.emplace(obj, into<uint8_t>(expr_data_.size()));
        if (!it.second) return;

        auto offset = it.first->second;
        expr_data_.resize(expr_data_.size() + 3);

        ir::ir_visitor_t::visit(obj.a);
        ir::ir_visitor_t::visit(obj.b);

        expr_data_[offset] = uint8_t(to_kind(obj.op_kind));
        expr_data_[offset + 1] = expr_locations.at(obj.a);
        expr_data_[offset + 2] = expr_locations.at(obj.b);
    }

    void _visit(const ir::ternary_op_t &obj) override {
        auto it = expr_locations.emplace(obj, into<uint8_t>(expr_data_.size()));
        if (!it.second) return;

        gpu_assert(obj.op_kind == ir::op_kind_t::_idiv);
        auto offset = it.first->second;
        expr_data_.resize(expr_data_.size() + 3);

        // Directly encode constants for idiv to save encoding space, since
        // the magic numbers are uniquely associated with this object.
        ir::ir_visitor_t::visit(obj.a);
        auto b = into<int8_t>(term_list_.size());
        term_list_.emplace_back(registry_->value_map.at(obj.b));
        term_list_.emplace_back(registry_->value_map.at(obj.c));

        expr_data_[offset] = uint8_t(to_kind(obj.op_kind));
        expr_data_[offset + 1] = expr_locations.at(obj.a);
        expr_data_[offset + 2] = b;
    }

    void _visit(const ir::var_t &obj) override {
        auto it = expr_locations.emplace(obj, into<uint8_t>(expr_data_.size()));
        if (!it.second) return;

        int idx = gid_idx(obj);
        if (idx >= 0) {
            expr_data_.emplace_back(uint8_t(kind_t::global_id) + idx);
        } else {
            expr_data_.emplace_back(uint8_t(kind_t::runtime_term));
            expr_data_.emplace_back(into<uint8_t>(term_list_.size()));
            term_list_.emplace_back(registry_->value_map.at(obj));
        }
    }

    void _visit(const ir::bool_imm_t &obj) override {
        if (obj.value)
            expr_data_.emplace_back(uint8_t(kind_t::constant_true));
        else
            expr_data_.emplace_back(uint8_t(kind_t::constant_false));
    }

    void _visit(const ir::int_imm_t &obj) override {
        auto it = expr_locations.emplace(obj, into<uint8_t>(expr_data_.size()));
        if (!it.second) return;

        if (obj.value == uint8_t(obj.value) || obj.value == int8_t(obj.value)) {
            expr_data_.emplace_back(obj.value == uint8_t(obj.value)
                            ? uint8_t(kind_t::constant_u8)
                            : uint8_t(kind_t::constant_s8));
            expr_data_.emplace_back(into<uint8_t>(obj.value));
            return;
        }
        if (obj.value == uint16_t(obj.value)
                || obj.value == int16_t(obj.value)) {
            expr_data_.emplace_back(obj.value == uint16_t(obj.value)
                            ? uint16_t(kind_t::constant_u16)
                            : uint16_t(kind_t::constant_s16));
            expr_data_.emplace_back(uint8_t(kind_t::constant_s16));
            expr_data_.emplace_back(uint8_t(obj.value));
            expr_data_.emplace_back(uint8_t(obj.value >> 8));
            return;
        }
        if (obj.value == uint32_t(obj.value)
                || obj.value == int32_t(obj.value)) {
            expr_data_.emplace_back(obj.value == uint32_t(obj.value)
                            ? uint32_t(kind_t::constant_u32)
                            : uint32_t(kind_t::constant_s32));
            expr_data_.emplace_back(uint8_t(obj.value));
            expr_data_.emplace_back(uint8_t(obj.value >> 8));
            expr_data_.emplace_back(uint8_t(obj.value >> 16));
            expr_data_.emplace_back(uint8_t(obj.value >> 24));
            return;
        }
        gpu_except_not_implemented();
    }

    int gid_idx(const ir::var_t &var) {
        if (var == global_ids(0).as<ir::var_t>()) return 0;
        if (var == global_ids(1).as<ir::var_t>()) return 1;
        if (var == global_ids(2).as<ir::var_t>()) return 2;
        return -1;
    }

    enum class kind_t : uint8_t {
        undef = 0,
        add, // encoded operands (expr, expr)
        sub, // encoded operands (expr, expr)
        mul, // encoded operands (expr, expr)
        div, // encoded operands (expr, expr)
        idiv, // encoded operands (expr, term)
        mod, // encoded operands (expr, expr)
        _or, // encoded operands (expr, expr)
        _and, // encoded operands (expr, expr)
        lt, // encoded operands (expr, expr)
        le, // encoded operands (expr, expr)
        gt, // encoded operands (expr, expr)
        ge, // encoded operands (expr, expr)
        ne, // encoded operands (expr, expr)
        eq, // encoded operands (expr, expr)
        runtime_term, // encoded operands (term)
        constant_true, // encodes true, no operands
        constant_false, // encodes false, no operands
        constant_s8, // encoded operands (int8_t)
        constant_u8, // encoded operands (uint8_t)
        constant_s16, // encoded operands (int16_t)
        constant_u16, // encoded operands (uint16_t)
        constant_s32, // encoded operands (int32_t)
        constant_u32, // encoded operands (int32_t)
        global_id, // encodes expression for global_id, no operands
        global_id0 = global_id,
        global_id1 = global_id + 1,
        global_id2 = global_id + 2,
    };

    kind_t to_kind(ir::op_kind_t op_kind) {
        switch (op_kind) {
            case ir::op_kind_t::_add: return kind_t::add;
            case ir::op_kind_t::_sub: return kind_t::sub;
            case ir::op_kind_t::_mul: return kind_t::mul;
            case ir::op_kind_t::_div: return kind_t::div;
            case ir::op_kind_t::_idiv: return kind_t::idiv;
            case ir::op_kind_t::_mod: return kind_t::mod;
            case ir::op_kind_t::_or: return kind_t::_or;
            case ir::op_kind_t::_and: return kind_t::_and;
            case ir::op_kind_t::_lt: return kind_t::lt;
            case ir::op_kind_t::_le: return kind_t::le;
            case ir::op_kind_t::_gt: return kind_t::gt;
            case ir::op_kind_t::_ge: return kind_t::ge;
            case ir::op_kind_t::_ne: return kind_t::ne;
            case ir::op_kind_t::_eq: return kind_t::eq;
            default: gpu_except_not_implemented();
        }
        return kind_t::undef;
    }

    std::vector<uint8_t> expr_data_;
    std::vector<int64_t> term_list_;
    ir::object_eq_map_t<dsl::expr_t, uint8_t> expr_locations;
    const term_registry_t *registry_;
};

struct fused_dim_t {
    fused_dim_t(dim_idx_t base, int64_t size, int64_t padded_size)
        : base(base), size(size), padded_size(padded_size) {}
    dim_idx_t base = dim_idx::invalid;
    int64_t size = 0;
    int64_t padded_size = 0;
    dim_idx_t gws_idx = dim_idx::invalid;
    ir::expr_t idx;
    std::string str() const {
        ostringstream_t oss;
        oss << "root dimension " << base << ": size - " << size
            << " padded size - " << padded_size << " idx - " << idx;
        return oss.str();
    }
};

// Represents dimensions within a list of buffers which may be fused for
// offset calculations. The fused dimensions are represented as linked lists
// stored in a hash map.
struct fused_dim_set_t {
    using dim_size_t = reusable_dispatch_config_t::dim_size_t;
    struct node_t {
        bool is_root() const { return prev == dim_idx::invalid; }
        bool is_end() const { return next == dim_idx::invalid; }
        dim_idx_t prev = dim_idx::invalid;
        dim_idx_t next = dim_idx::invalid;
    };

    fused_dim_set_t(const named_buffer_t &buf, const block_layout_t &layout,
            const std::map<dim_idx_t, dim_size_t> &dim_sizes,
            const lws_strategy_t &lws_strategy)
        : dim_sizes(dim_sizes) {
        for (auto &b : layout) {
            nodes[b.dim_idx] = {};
        }
        for (int64_t i = 0; i < into<int64_t>(layout.size()); i++) {
            auto dim_size = dim_sizes.find(layout[i].dim_idx);

            // Dimension not part of the workgroup
            if (dim_size == dim_sizes.end()) {
                unlink(layout[i].dim_idx);
                nodes.erase(layout[i].dim_idx);
                continue;
            }

            // Dimensions can only be fused when they are contiguous.
            bool is_dense = (i < into<int64_t>(layout.size() - 1))
                    && layout[i].stride * layout[i].block
                            == layout[i + 1].stride;

            // Blocked Dimensions cannot be fused since we need to know
            // their value to calculate offsets into the blocking.
            bool is_blocked = false;
            for (size_t j = 0; j < layout.size(); j++) {
                if (layout[j].dim_idx == layout[i].dim_idx && (j != size_t(i)))
                    is_blocked = true;
            }

            // Local dimensions must start a fused dimension so they can be
            // mapped into the gws.
            bool is_local = false;
            for (auto &ld : lws_strategy.local()) {
                if (layout[i].dim_idx == ld.idx) is_local = true;
            }

            // Padded dimensions cannot be fused with any further dimensions
            // so that padding checks can be performed.
            bool is_padded = buf.dims[buf.get_dim_idx(layout[i].dim_idx)]
                    < dim_size->second.padded_size;

            if (is_dense && !is_blocked && !is_padded) {
                link(layout[i].dim_idx, layout[i + 1].dim_idx);
            } else if (is_blocked || is_local) {
                unlink(layout[i].dim_idx);
            }
        }
    }
    fused_dim_set_t(const std::vector<named_buffer_t> &buffers,
            const std::vector<block_layout_t> &layouts,
            const std::map<dim_idx_t, dim_size_t> &dim_sizes,
            const lws_strategy_t &lws_strategy)
        : dim_sizes(dim_sizes) {
        for (size_t i = 0; i < buffers.size(); i++) {
            fused_dim_set_t buffer_set(
                    buffers[i], layouts[i], dim_sizes, lws_strategy);
            if (nodes.empty()) {
                nodes = std::move(buffer_set.nodes);
                continue;
            }
            for (auto &e : buffer_set.nodes) {
                if (nodes[e.first].prev != e.second.prev) unlink(e.first);
                if (!nodes[e.first].is_end()
                        && nodes[e.first].next != e.second.next)
                    unlink(nodes[e.first].next);
            }
        }
    }

    std::vector<fused_dim_t> as_list() const {
        std::vector<fused_dim_t> ret;
        for (auto &e : nodes) {
            if (e.second.is_root()) {
                size_t size = 1;
                size_t padded_size = 1;
                auto next = e.first;
                while (next != dim_idx::invalid) {
                    size *= dim_sizes.at(next).size;
                    padded_size *= dim_sizes.at(next).padded_size;
                    next = nodes.at(next).next;
                }
                ret.emplace_back(e.first, size, padded_size);
            }
        }
        return ret;
    }

    void link(dim_idx_t idx, dim_idx_t next) {
        gpu_assert(nodes[idx].next == dim_idx::invalid);
        gpu_assert(nodes[next].prev == dim_idx::invalid);
        nodes[idx].next = next;
        nodes[next].prev = idx;
    }

    void unlink(dim_idx_t idx) {
        auto &prev = nodes[idx].prev;
        if (prev != dim_idx::invalid) {
            nodes[prev].next = dim_idx::invalid;
            prev = dim_idx::invalid;
        }
    }

    std::string str() const {
        ostringstream_t oss;
        const char *prefix = "";
        oss << "{";
        for (auto &e : nodes) {
            if (e.second.is_root()) {
                dim_idx_t next = e.first;
                while (next != dim_idx::invalid) {
                    oss << prefix << next;
                    prefix = " -> ";
                    next = nodes.at(next).next;
                }
                prefix = ", ";
            }
        }
        oss << "}";
        return oss.str();
    }

    std::map<dim_idx_t, node_t> nodes;
    const std::map<dim_idx_t, dim_size_t> &dim_sizes;
};

std::unordered_map<dim_idx_t, int> get_dim_pack_order(
        const lws_strategy_t &lws_strategy,
        const std::vector<dim_idx_t> &dispatched_dims,
        const std::vector<named_buffer_t> &buffers,
        const std::vector<block_layout_t> &layouts) {
    std::unordered_map<dim_idx_t, int> ret;
    int64_t idx = -1;
    for (size_t i = 0; i < buffers.size(); i++) {
        if (utils::one_of(buffers[i].get_name_id(), name_id_t::dst,
                    name_id_t::diff_src)) {
            idx = i;
            break;
        }
    }

    auto local_info = lws_strategy.local();

    int priority = 0;
    for (auto &local_dim : local_info)
        ret.emplace(local_dim.idx, priority++);
    if (idx >= 0) {
        for (auto &block : layouts[idx])
            ret.emplace(block.dim_idx, priority++);
    }
    for (auto &dim_idx : dispatched_dims)
        ret.emplace(dim_idx, priority++);
    return ret;
}

// XXX: Mapping blocks into the gws cannot happen until all necessary dim indices
// have been requested and all buffers have been registered. Only then can the terms
// be computed, thus it's all done in the generate function
status_t reusable_dispatch_config_t::generate(reusable_dispatch_t &dispatch) {
    // The reusable dispatcher must have at least one buffer to dispatch against
    gpu_assert(!buffers.empty());

    term_registry_t registry;
    expr_encoder_t encoder;

    // Sort to enable deterministic output and simplify serialization
    std::sort(buffers.begin(), buffers.end(),
            [&](const named_buffer_t &a, const named_buffer_t &b) {
        return a.get_name_id() < b.get_name_id();
    });

    std::vector<block_layout_t> layouts;
    layouts.reserve(buffers.size());
    for (auto &b : buffers) {
        layouts.emplace_back(b.layout());
    }

    // Ensure subgroups and local ids are uniform
    auto local_info = lws_strategy.local();
    auto subgroup_info = lws_strategy.subgroup();
    if (subgroup_info.idx != dim_idx::invalid) {
        gpu_assert(local_info[0].idx == subgroup_info.idx
                && local_info[0].size % subgroup_info.size == 0);
    }

    for (auto &local_dim : local_info) {
        if (local_dim.size == 0) continue;
        auto &padded_size = dim_sizes[local_dim.idx].padded_size;
        padded_size = utils::rnd_up(padded_size, local_dim.size);
    }

    // Every dispatched dim must have a defined size. For unregistered
    // dimensions, we assume they have size 1.
    for (dim_idx_t id : dispatched_dims) {
        dim_sizes.emplace(id, dim_size_t {1, 1});
    }

    // Generate fused dimensions used for offset calculations
    auto fused_dims_set
            = fused_dim_set_t(buffers, layouts, dim_sizes, lws_strategy);
    std::vector<fused_dim_t> fused_dims = fused_dims_set.as_list();
    {
        // TODO: replace `get_dim_pack_order` with a function which direcltly
        // computes fused_dims;
        auto pack_order = get_dim_pack_order(
                lws_strategy, dispatched_dims, buffers, layouts);

        std::sort(fused_dims.begin(), fused_dims.end(),
                [&](const fused_dim_t &a, const fused_dim_t &b) {
            return pack_order[a.base] < pack_order[b.base];
        });

        std::array<size_t, 3> gws_idx = {0, 0, 0};
        size_t pack_size = fused_dims.size() / 3;
        dim_idx_t g_i = 0;
        for (auto &fd : fused_dims) {
            for (int i = 0; i < 3; i++) {
                if (local_info[i].idx == fd.base) {
                    fd.gws_idx = i;
                    gws_idx[i]++;
                    break;
                }
            }
            if (fd.gws_idx != dim_idx::invalid) continue;

            // Distribute dimensions as evenly as possible to reduce operations
            // require to compute the fused dimensions index.
            while (gws_idx[g_i] > pack_size
                    || (gws_idx[g_i] == pack_size
                            && g_i >= fused_dims.size() % 3))
                g_i++;
            fd.gws_idx = g_i;
            gws_idx[g_i]++;
        }
    }

    // Map fused dimensions into work groups
    range_t gws_base = {1, 1, 1};
    std::array<int64_t, 3> gws_max_idx = {0, 0, 0};
    for (auto &fd : fused_dims) {
        gws_base[fd.gws_idx] *= fd.padded_size;
        gws_max_idx[fd.gws_idx]++;
    }

    range_t gws = gws_base;
    range_t lws = lws_strategy.create_lws(gws);

    // Determine calculation for fused dimensions based on the mapping into work
    // groups.
    std::array<int64_t, 3> strides = {1, 1, 1};
    std::array<int64_t, 3> gws_idx = {0, 0, 0};

    for (auto &fd : fused_dims) {
        size_t g_i = fd.gws_idx;

        auto scale = [&]() {
            if (strides[fd.gws_idx] == 1) return global_ids(g_i);

            auto stride_name = "gws" + std::to_string(g_i) + "_stride["
                    + std::to_string(gws_idx[g_i]) + "]";

            if (gws[g_i] > INT_MAX) {
                auto stride = registry.add(strides[g_i], stride_name);
                return global_ids(g_i) / stride;
            }

            uint32_t m, p;
            jit::ir_utils::idiv_magicgu(into<uint32_t>(strides[g_i]), m, p);
            dsl::expr_t magic_m(registry.add(m, stride_name + "_magic_m"));
            dsl::expr_t magic_p(registry.add(p, stride_name + "_magic_p"));
            return ir::ternary_idiv(global_ids(g_i), magic_m, magic_p);
        }();

        auto clamp = [&]() {
            if (gws_idx[g_i] == gws_max_idx[g_i] - 1) return scale;
            if (fd.padded_size == 1) return ir::expr_t(0);

            auto size_name = "gws" + std::to_string(g_i) + "_size["
                    + std::to_string(gws_idx[g_i]) + "]";
            auto size = registry.add(fd.padded_size, size_name);

            if (gws[g_i] > INT_MAX) { return scale % size; }

            uint32_t m, p;
            jit::ir_utils::idiv_magicgu(into<uint32_t>(fd.padded_size), m, p);
            dsl::expr_t magic_m(registry.add(m, size_name + "_magic_m"));
            dsl::expr_t magic_p(registry.add(p, size_name + "_magic_p"));
            return scale - (ir::ternary_idiv(scale, magic_m, magic_p) * size);
        }();

        fd.idx = clamp;
        strides[g_i] *= fd.padded_size;
        gws_idx[g_i]++;
    }

    gpu_info() << "dimension fusions: " << fused_dims_set;
    for (auto &fd : fused_dims) {
        gpu_info() << "   " << fd;
    }

    gpu_info() << "gws: " << gws.str() << " lws: " << lws.str()
               << " (gws_before_lws_rounding: " << gws_base << ")";

    // Determine calculation required for workgroup overflows due to lws
    // divisibility requirements.
    auto gws_overflow = [&]() -> uint8_t {
        auto ret = ir::expr_t(false);
        for (int i = 0; i < 3; i++) {
            if (gws_base[i] == gws[i]) continue;
            auto idx_check = global_ids(i) >= registry.add(gws_base[i],
                                     "gws" + std::to_string(i) + "_max");
            ret = ret.is(false) ? idx_check : (ret | idx_check);
        }
        gpu_info() << "    gws overflow: " << ret;
        return encoder(ret, registry);
    }();

    // Determine calculations required for buffer padding.
    auto in_padding = [&]() -> uint8_t {
        auto ret = ir::expr_t(false);
        for (size_t i = 0; i < fused_dims.size(); i++) {
            auto &fd = fused_dims[i];
            if (fd.size == fd.padded_size) continue;
            auto idx_check = (fd.idx >= registry.add(fd.size,
                                      "fused_size[" + std::to_string(i) + "]"));
            ret = ret.is(false) ? idx_check : ret | idx_check;
        }
        gpu_info() << "    gws in padding: " << ret;
        return encoder(ret, registry);
    }();

    // Determine calculations for the offset into each buffer.
    std::vector<uint8_t> buffer_exprs;
    buffer_exprs.reserve(buffers.size());
    for (size_t buf_idx = 0; buf_idx < buffers.size(); buf_idx++) {
        // Deduplicate buffers with the same layout
        for (size_t i = 0; i < buf_idx; i++) {
            if (layouts[i] == layouts[buf_idx]) {
                buffer_exprs.emplace_back(buffer_exprs[i]);
                break;
            }
        }
        if (buffer_exprs.size() > buf_idx) continue;
        ir::expr_t expr(0);
        for (auto &fd : fused_dims) {
            auto dim_stride = 1;

            const block_t *outer = nullptr;
            for (auto &e : layouts[buf_idx]) {
                if (e.dim_idx == fd.base) outer = &e;
            }
            if (outer == nullptr) continue;

            for (auto &e : layouts[buf_idx]) {
                if (e.dim_idx == fd.base) {
                    if (e.stride == 0) continue;
                    auto term = dim_stride == 1 ? fd.idx : fd.idx / dim_stride;
                    if (outer == &e && e.stride != 1) {
                        term = term
                                * registry.add(int64_t(e.stride),
                                        buffers[buf_idx].name() + "_stride["
                                                + std::to_string(e.dim_idx)
                                                + "]");
                    } else if (outer != &e) {
                        if (e.block > 1) term = term % e.block;
                        if (e.stride != 1) term = term * int64_t(e.stride);
                    }
                    expr = expr.is(0) ? term : expr + term;

                    dim_stride *= e.block;
                }
            }
        }
        gpu_info() << buffers[buf_idx].name() << ": " << expr;

        buffer_exprs.emplace_back(encoder(expr, registry));
    }

    gpu_info() << "total expr data: " << encoder.expr_data().size()
               << " bytes, total runtime terms: " << encoder.term_list().size();

    if (encoder.expr_data().size() >= MAX_EXPR_TERMS)
        return status::unimplemented;
    if (encoder.term_list().size() >= MAX_RUNTIME_TERMS)
        return status::unimplemented;
    if (buffer_exprs.size() >= MAX_REGISTERED_BUFFERS)
        return status::unimplemented;

    dispatch = reusable_dispatch_t(nd_range_t(gws, lws), subgroup_info.size,
            buffers, gws_overflow, in_padding, buffer_exprs,
            encoder.expr_data(), encoder.term_list());

    return status::success;
}

void dispatch_compile_params_t::def_kernel_macros(
        kernel_ctx_t &kernel_ctx, const char *suffix) const {
    kernel_ctx.define_int("GWS_WITH_RUNTIME_PARAMS", 1);
    if (use_int32_offset) kernel_ctx.define_int("GWS_USE_PARAMS32", 1);
    kernel_ctx.use_int32_offset(use_int32_offset);

    // Define data types for conversion (Ignore the default suffix)
    std::string conv_suff = (suffix == std::string("DEFAULT"))
            ? ""
            : utils::format("_%s", suffix);

    kernel_ctx.add_option(utils::format("-DGWS_%s_OVERFLOW(rt)=(%s)", suffix,
            expr_encoder_t::decode(exprs, gws_overflow_expr)));
    kernel_ctx.add_option(utils::format("-DGWS_%s_IN_PADDING(rt)=(%s)", suffix,
            expr_encoder_t::decode(exprs, in_padding_expr)));

    // For each buffer, define the sum that leads to the offset calculation
    data_type_converter_t converter;
    int buf_idx = 0;
    for (size_t bit_idx = 0; bit_idx < sizeof(name_id_t) * 8; bit_idx++) {
        name_id_t name_id = name_id_t(uint64_t(1) << bit_idx);
        if (!static_cast<uint64_t>(buffer_set & name_id)) continue;

        if (buffer_types[buf_idx] != data_type::undef) {
            converter.register_type(
                    to_string(name_id) + conv_suff, buffer_types[buf_idx]);
        }

        std::string equation
                = expr_encoder_t::decode(exprs, buffer_off_expr[buf_idx]);
        kernel_ctx.add_option(utils::format("-DGWS_%s_%s_OFF(rt)=(%s)",
                to_string(name_id), suffix, equation));
        buf_idx++;
    }
    converter.def_kernel_macros(kernel_ctx);

    kernel_ctx.define_int(
            utils::format("GWS_WITH_SG_%s", suffix), bool(subgroup_size));
    if (subgroup_size) {
        kernel_ctx.define_int(
                utils::format("GWS_SGS_%s", suffix), uint32_t(subgroup_size));
    }
}

bool dispatch_compile_params_t::has_padding() const {
    return expr_encoder_t::kind_t(exprs[in_padding_expr])
            != expr_encoder_t::kind_t::constant_false;
}

std::string dispatch_compile_params_t::str() const {
    ostringstream_t ss;
    auto buf_idx = 0;
    ss << "{";
    for (size_t bit_idx = 0; bit_idx < 8 * sizeof(name_id_t); bit_idx++) {
        name_id_t name_id = name_id_t(uint64_t(1) << bit_idx);
        if (!static_cast<uint64_t>(buffer_set & name_id)) continue;
        if (buf_idx > 0) ss << ", ";

        ss << to_string(name_id)
           << " - [ type: " << dnnl_dt2str(buffer_types[buf_idx])
           << " offset : "
           << expr_encoder_t::decode(exprs, buffer_off_expr[buf_idx]) << "]";
        buf_idx++;
    }
    ss << "}";
    return ss.str();
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
