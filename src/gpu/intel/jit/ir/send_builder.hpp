/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_SEND_BUILDER_HPP
#define GPU_INTEL_JIT_IR_SEND_BUILDER_HPP

#include "gpu/intel/jit/ir/send.hpp"

#include "gpu/intel/jit/ir/fma.hpp"
#include "gpu/intel/jit/ir/gemm_schedule.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class memory_walker_t;
class layout_walker_t;

struct send_2d_hint_t {
    dsl::type_t type;
    bool enable = false;
    bool vnni = false;
    bool transpose = false;
    int vnni_permute_factor = 0;
    int width = 0;
    int height = 0;
};

struct send_params_t {
    send_params_t() = default;
    send_params_t(
            const hw_t &hw, const dsl::type_t &mem_type, send_op_t send_op)
        : hw(hw), mem_type(mem_type), send_op(send_op), use_send_plan(true) {}

    send_op_t convert(const send_op_t &op) const {
        if (hint_2d.enable) {
            if (op == send_op_t::load) return send_op_t::load_2d;
            if (op == send_op_t::store) return send_op_t::store_2d;
            if (op == send_op_t::prefetch) return send_op_t::prefetch_2d;
        }
        return op;
    }

    bool is_slm() const { return send_address == send_address_t::slm; }
    bool is_prefetch() const {
        return utils::one_of(
                send_op, send_op_t::prefetch, send_op_t::prefetch_2d);
    }

    hw_t hw;
    dsl::type_t mem_type;
    send_op_t send_op;
    send_address_t send_address;
    send_cache_hint_t cache_hint;
    send_2d_hint_t hint_2d;
    bool prefer_dense = false;
    bool use_send_plan = false;
    bool try_legacy = true;
};

// Generates loads or stores to move data between memory (global or SLM) and
// GRF. Memory view is a parameter. GRF payload layout is deduced
// automatically, according to the decomposition into messages.
class access_builder_t {
public:
    access_builder_t(ir_context_t &ir_ctx, const view_t &mem_view,
            const expr_t &mem_buf, const expr_t &reg_buf, send_op_t send_op,
            send_address_t send_address, send_cache_hint_t send_cache_hint,
            send_params_t &send_params, bool zero_out);
    access_builder_t(access_builder_t &&);
    ~access_builder_t();

    const layout_t &reg_layout() const { return reg_layout_; }
    int reg_buf_size() const {
        if (reg_buf_size_ == 0)
            return into<int>(size_bytes(reg_layout_, grf_size()));
        return reg_buf_size_;
    }
    const stmt_t &stmt() const { return stmt_; }

    std::string str() const {
        ostringstream_t oss;
        oss << "Memory view:          " << mem_view_ << std::endl;
        oss << "Register layout:      " << reg_layout_ << std::endl;
        oss << "Register buffer:      " << reg_buf_ << std::endl;
        oss << "Register buffer size: " << reg_buf_size() << " ("
            << reg_buf_size() / grf_size() << " regs)" << std::endl;
        oss << "Statement:            " << std::endl << stmt_;
        return oss.str();
    }

private:
    void build();
    bool try_build(const layout_t &try_layout, memory_walker_t &mem_walker);
    bool try_build_2d(send_params_t &send_params);
    bool fixup_send_2d_params(const dsl::type_t &send_type, bool vnni,
            bool transpose, bool use_xy, int &W, int &H, int &P, int &w, int &h,
            int &c, int &vnni_permute_factor);

    bool check_2d_mask(const tile_t &tile, const coord_t &coord,
            bool use_virtual_surface, size_t w_idx, size_t h_idx,
            expr_t &mask) const;

    std::vector<layout_t> candidate_payload_layouts() const;
    stmt_t create_send_stmt(
            const send_t &send, const memory_walker_t &memory_walker);
    int grf_size() const { return ir_ctx_->hw().grf_size(); }

    ir_context_t *ir_ctx_ = nullptr;
    view_t mem_view_;
    expr_t mem_buf_;
    expr_t reg_buf_;
    send_op_t send_op_;
    send_address_t send_address_;
    send_cache_hint_t send_cache_hint_;

    dsl::type_t mem_type_;

    std::unique_ptr<layout_walker_t> reg_layout_walker_;

    layout_t reg_layout_;
    int reg_buf_size_ = 0;
    bool zero_out_ = true;
    stmt_t stmt_;
};

send_params_t get_send_params(const kernel::options_t &options,
        send_op_t send_op, send_address_t send_address, const view_t &view,
        send_cache_hint_t cache_hint = send_cache_hint_t::undef,
        fma_kind_t fma_kind = fma_kind_t::undef,
        abc_kind_t abc_kind = abc_kind_t::undef, bool allow_2d = false);

send_params_t get_send_params(const kernel::options_t &options,
        send_op_t send_op, send_address_t send_address, fma_kind_t fma_kind,
        abc_kind_t abc_kind, const view_t &view,
        const gemm_schedule_t &gemm_schedule, bool allow_2d = true);

inline send_params_t get_send_params(const kernel::options_t &options,
        send_op_t send_op, send_address_t send_address, const view_t &view,
        bool allow_2d) {
    return get_send_params(options, send_op, send_address, view,
            send_cache_hint_t::undef, fma_kind_t::undef, abc_kind_t::undef,
            allow_2d);
}

inline access_builder_t make_access_builder(ir_context_t &ir_ctx,
        const view_t &mem_view, const expr_t &mem_buf, const expr_t &reg_buf,
        send_params_t &send_params, bool zero_out = true) {
    return access_builder_t(ir_ctx, mem_view, mem_buf, reg_buf,
            send_params.send_op, send_params.send_address,
            send_params.cache_hint, send_params, zero_out);
}

inline access_builder_t make_access_builder(ir_context_t &ir_ctx,
        const view_t &mem_view, const expr_t &mem_buf, const expr_t &reg_buf,
        send_op_t send_op, send_address_t send_address,
        send_cache_hint_t cache_hint = send_cache_hint_t::undef,
        bool zero_out = true) {
    auto send_params = get_send_params(
            ir_ctx.options(), send_op, send_address, mem_view, cache_hint);
    return make_access_builder(
            ir_ctx, mem_view, mem_buf, reg_buf, send_params, zero_out);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
