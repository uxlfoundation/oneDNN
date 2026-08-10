/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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
#include <cassert>
#include "common/verbose.hpp"
#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace injector {

#define VCHECK_PO_INJ_BOOL(cond, msg) \
    VCONDCHECK(primitive, create, check, postops_injector, cond, false, msg);

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(
        jit_generator_t *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params,
        const lambda_jit_injectors_t &lambda_jit_injectors)
    : post_ops_(post_ops)
    , host_(host)
    , binary_injector_(nullptr)
    , lambda_jit_injectors_(lambda_jit_injectors) {

    const auto &esp = eltwise_static_params;
    bool is_like_binary = false;

    for (int i = 0; i < post_ops.len(); i++) {
        const auto &post_op = post_ops.entry_[i];
        if (post_op.is_eltwise()) {
            alg_to_eltwise_injector_.emplace(i,
                    jit_uni_eltwise_injector_t<isa>(
                            host_, post_op.eltwise, esp));
        } else if (post_op.is_like_binary()) {
            is_like_binary = true;
        }
    }

    if (is_like_binary)
        binary_injector_ = utils::make_unique<
                binary_injector::jit_uni_binary_injector_t<isa>>(
                host, binary_static_params);
}

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(
        jit_generator_t *host, const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
              eltwise_static_params, lambda_jit_injectors_t()) {}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
        size_t group_stride) {

    // [start_idx, end_idx) is the half-open physical register span (x64
    // semantics); group_stride is the registers-per-accumulator (LMUL/EMUL).
    // make_vmm_group_set steps by it so an m>1 run maps to its true group
    // bases; both sub-injectors then consume the explicit set.
    compute_vector_range(injector_utils::make_vmm_group_set<isa>(
                                 start_idx, end_idx, group_stride),
            rhs_arg_params, group_stride);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx, size_t group_stride) {
    compute_vector_range(start_idx, end_idx,
            binary_injector::rhs_arg_dynamic_params_t(), group_stride);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
        size_t group_stride) {

    // group_stride is each accumulator's e32 LMUL (1/2/4); the binary injector
    // needs it to pick the narrow-dtype rhs load LMUL. The eltwise injector
    // computes at the host vtype (it issues no vsetvli) and so is LMUL-agnostic.
    std::size_t rhs_arg_idx = 0;
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &post_op = post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            alg_to_eltwise_injector_.at(i).compute_vector_range(
                    vmm_idxs, group_stride);
        } else if (post_op.is_like_binary()) {
            binary_injector_->compute_vector_range(vmm_idxs, rhs_arg_idx,
                    post_op, rhs_arg_params, group_stride);
            ++rhs_arg_idx;
            // Ternary op handles two arguments at the same time, thus,
            // skipping one more.
            if (post_op.is_binary_with_ternary_op()) ++rhs_arg_idx;
        } else {
            const auto lam = lambda_jit_injectors_.find(post_op.kind);
            if (lam != lambda_jit_injectors_.end()) lam->second();
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs, size_t group_stride) {
    compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t(),
            group_stride);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector(size_t idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
        size_t group_stride) {
    compute_vector_range({idx}, rhs_arg_params, group_stride);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector(
        size_t idx, size_t group_stride) {
    compute_vector_range({idx}, group_stride);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::set_lambda_injector(
        dnnl_primitive_kind_t kind, const std::function<void()> &jit_injector) {
    lambda_jit_injectors_[kind] = jit_injector;
}

post_ops_ok_args_t::post_ops_ok_args_t(const cpu_isa_t isa,
        const std::vector<post_op_type> &accepted_post_op_types,
        const post_ops_t &post_ops, const memory_desc_wrapper *dst_d,
        const bool sum_at_pos_0_only, const bool sum_requires_scale_one,
        const bool sum_requires_zp_zero, const bool sum_requires_same_params,
        const bcast_set_t &enabled_bcast_strategy, const int n_vaux,
        const bool allow_binary_select)
    : isa(isa)
    , accepted_post_op_types(accepted_post_op_types)
    , post_ops(post_ops)
    , dst_d(dst_d)
    , sum_at_pos_0_only(sum_at_pos_0_only)
    , sum_requires_scale_one(sum_requires_scale_one)
    , sum_requires_zp_zero(sum_requires_zp_zero)
    , sum_requires_same_params(sum_requires_same_params)
    , enabled_bcast_strategy(enabled_bcast_strategy)
    , n_vaux(n_vaux)
    , allow_binary_select(allow_binary_select) {};

bool post_ops_ok(const post_ops_ok_args_t &post_ops_ok_args) {
    const cpu_isa_t isa = post_ops_ok_args.isa;
    const std::vector<post_op_type> &accepted_post_op_types
            = post_ops_ok_args.accepted_post_op_types;
    const post_ops_t &post_ops = post_ops_ok_args.post_ops;
    const memory_desc_wrapper *dst_d = post_ops_ok_args.dst_d;
    const bool sum_at_pos_0_only = post_ops_ok_args.sum_at_pos_0_only;
    const bool sum_requires_scale_one = post_ops_ok_args.sum_requires_scale_one;
    const bool sum_requires_zp_zero = post_ops_ok_args.sum_requires_zp_zero;
    const bool sum_requires_same_params
            = post_ops_ok_args.sum_requires_same_params;
    const auto &enabled_bcast_strategy
            = post_ops_ok_args.enabled_bcast_strategy;

    VCHECK_PO_INJ_BOOL(dst_d && dst_d->md_->format_kind != dnnl_format_kind_any,
            VERBOSE_UNSUPPORTED_FORMAT_KIND);

    // Save scale and zero point of first sum postop in order to check that any
    // subsequent sum postops have the same values. This check is necessary
    // because there is only one lambda injector.
    const auto sum_idx = post_ops.find(primitive_kind::sum);
    const bool with_sum = sum_idx != -1;
    const auto &entry
            = with_sum ? post_ops.entry_[sum_idx] : dnnl_post_ops::entry_t();
    const auto sum_scale = with_sum ? entry.sum.scale : 0;
    const auto sum_zero_point = with_sum ? entry.sum.zero_point : 0;

    const auto is_accepted_postop = [&](const int idx) {
        for (const auto &post_op : accepted_post_op_types) {
            const auto &entry = post_ops.entry_[idx];
            // Note: check for post-op kinds is needed as `post_op` value
            // represents all supported but not only passed kinds.
            switch (post_op) {
                case sum:
                    if (!entry.is_sum(false, false)) continue;
                    if (sum_requires_same_params) {
                        VCHECK_PO_INJ_BOOL(entry.sum.scale == sum_scale,
                                "Unsupported sum scale value");
                        VCHECK_PO_INJ_BOOL(
                                entry.sum.zero_point == sum_zero_point,
                                "Unsupported sum zero-point value");
                    }
                    if (sum_requires_scale_one) {
                        VCHECK_PO_INJ_BOOL(entry.sum.scale == 1.f,
                                "Unsupported sum scale value");
                    }
                    if (sum_requires_zp_zero) {
                        VCHECK_PO_INJ_BOOL(entry.sum.zero_point == 0,
                                "Unsupported sum zero-point value");
                    }
                    VCHECK_PO_INJ_BOOL(IMPLICATION(sum_at_pos_0_only, idx == 0),
                            "Unsupported sum position in post-ops");
                    return true;
                case eltwise: {
                    if (!entry.is_eltwise()) continue;
                    // The heavy algs (log/soft_relu/gelu_erf) need a fourth
                    // aux group from the host; x64 instead checks
                    // eltwise_injector::is_supported(isa, alg, f32) -- the
                    // aux budget is an rv64 host contract (no stack spill).
                    const auto alg = entry.eltwise.alg;
                    return post_ops_ok_args.n_vaux >= 4
                            ? eltwise_injector::is_supported(alg)
                            : eltwise_injector::is_alg_supported(alg);
                }
                case binary:
                    if (entry.is_binary()) {
                        assert(dst_d != nullptr && "dst_d is null");
                        bool ok = binary_injector::is_supported(isa,
                                binary_injector::get_src1_desc(entry, *dst_d),
                                *dst_d, enabled_bcast_strategy);
                        if (entry.is_binary_with_ternary_op()) {
                            // rv64: the select condition goes through the
                            // same load machinery as src1, so it gets the
                            // full is_supported check (x64 checks only the
                            // src2 data type).
                            VCHECK_PO_INJ_BOOL(
                                    post_ops_ok_args.allow_binary_select,
                                    "Unsupported binary select post-op");
                            const auto src2_d = binary_injector::get_src2_desc(
                                    entry, *dst_d);
                            ok = ok
                                    && binary_injector::is_supported(isa,
                                            src2_d, *dst_d,
                                            enabled_bcast_strategy);
                            // The injector addresses the condition through
                            // the no_broadcast path only (x64 likewise
                            // supports no src2 broadcasting).
                            VCHECK_PO_INJ_BOOL(
                                    get_rhs_arg_broadcasting_strategy(src2_d,
                                            *dst_d, enabled_bcast_strategy)
                                            == broadcasting_strategy_t::
                                                    no_broadcast,
                                    "Unsupported broadcast for binary select "
                                    "condition");
                        }
                        return ok;
                    }
                    break;
                default: assert(!"Unhandled post_op type");
            }
        }
        return false;
    };

    for (int i = 0; i < post_ops.len(); i++) {
        if (!is_accepted_postop(i)) return false;
    }

    return true;
}

template class jit_uni_postops_injector_t<v>;
template class jit_uni_postops_injector_t<zvfh>;
template class jit_uni_postops_injector_t<zvfbfwma>;

#undef VCHECK_PO_INJ_BOOL

} // namespace injector
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
