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
#ifndef CPU_RV64_INJECTORS_JIT_UNI_POSTOPS_INJECTOR_HPP
#define CPU_RV64_INJECTORS_JIT_UNI_POSTOPS_INJECTOR_HPP

#include <functional>
#include <map>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "cpu/rv64/injectors/injector_utils.hpp"
#include "cpu/rv64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace injector {

/*
 * Allows specifying custom injector function for given post-op type - one
 * function per primitive. There are post-ops type (example: sum) that don't
 * have specialized injector. They heavily rely on kernel specific intrnals,
 * which makes the generalization unreasonable. As so user can prepare internal
 * kernel lambda and pass it explicitly to injector.
 */
using lambda_jit_injectors_t
        = std::map<dnnl_primitive_kind_t, std::function<void()>>;

struct post_ops_ok_args_t;
/*
 * Checks if postops injection for given args is supported.
 */
bool post_ops_ok(const post_ops_ok_args_t &args);

/*
 * Main mechanism of handling various post-ops types. It utilizes internally
 * specialized injectors to generate post-ops code to host primitive. Random
 * order of post-ops is supported.
 */
template <cpu_isa_t isa>
class jit_uni_postops_injector_t {
public:
    /*
     * @param host <required> - user primitive where post-ops generated code is
     * injected
     * @param post_ops <required> - struct representing requested post-ops chain
     * @binary_static_params <reguired> - static params needed for binary_injector.
     * see: jit_uni_binary_injector.hpp for more info.
     * @param eltwise_static_params <required> - static params needed for
     * eltwise_injector. Unlike x64 (where a default-constructed value exists)
     * the rv64 eltwise params carry the host-reserved aux register groups and
     * are therefore mandatory.
     * @param lambda_jit_injectors <optional> - allows user specify custom injector
     * function for given post-op type
     */
    jit_uni_postops_injector_t(jit_generator_t *host,
            const post_ops_t &post_ops,
            const binary_injector::static_params_t &binary_static_params,
            const eltwise_injector::static_params_t &eltwise_static_params);
    jit_uni_postops_injector_t(jit_generator_t *host,
            const post_ops_t &post_ops,
            const binary_injector::static_params_t &binary_static_params,
            const eltwise_injector::static_params_t &eltwise_static_params,
            const lambda_jit_injectors_t &lambda_jit_injectors);

    /*
     * Generates code of post_ops chain injected to host primitive. Applied to
     * ordered set of vector registers' indexes. group_stride is each
     * accumulator's e32 LMUL (registers per accumulator, one of 1/2/4); the
     * binary sub-injector uses it to pick the narrow-dtype rhs load LMUL, the
     * eltwise sub-injector is LMUL-agnostic (it issues no vsetvli).
     *
     * @rhs_arg_params: see jit_uni_binary_injector description
     */
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
            size_t group_stride);

    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            size_t group_stride);

    /*
     * Generates code of post_ops chain injected to host primitive. Applied to
     * the accumulators occupying the half-open physical register span
     * <start_idx, end_idx). group_stride is the vector registers per
     * accumulator (LMUL/EMUL, one of 1/2/4; m8 is unsupported -- the helper
     * scratch is m4); the run's group bases are enumerated by stepping it (see
     * make_vmm_group_set), so unlike x64 an m>1 run maps to its true bases. It
     * is mandatory so an x64 compute_vector_range(start, end) cannot be copied
     * as a silent stride-1 enumeration.
     *
     * @rhs_arg_params: see jit_uni_binary_injector description
     */
    void compute_vector_range(size_t start_idx, size_t end_idx,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
            size_t group_stride);

    void compute_vector_range(
            size_t start_idx, size_t end_idx, size_t group_stride);

    /*
     * Generates code of post_ops chain injected to host primitive. Applied to
     * a single vector register index. group_stride is the accumulator's e32
     * LMUL (1/2/4); see the set overload.
     *
     * @rhs_arg_params: see jit_uni_binary_injector description
     */
    void compute_vector(size_t idx,
            const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params,
            size_t group_stride);
    void compute_vector(size_t idx, size_t group_stride);

    // x64's prepare_table thin wrapper has no rv64 analog: the rv64 eltwise
    // injector materializes constants inline (li + fmv.w.x) instead of a
    // rodata table.
    void set_lambda_injector(lambda_jit_injectors_t::key_type,
            const lambda_jit_injectors_t::mapped_type &jit_injector);

private:
    post_ops_t post_ops_;
    jit_generator_t *host_;
    // Key is a numerical order of a post-op in attributes.
    std::map<int, jit_uni_eltwise_injector_t<isa>> alg_to_eltwise_injector_;
    std::unique_ptr<binary_injector::jit_uni_binary_injector_t<isa>>
            binary_injector_;
    lambda_jit_injectors_t lambda_jit_injectors_;
};

// prelu is not supported by the rv64 binary injector (an intentional gap next
// to bf16), so unlike x64 it is not an accepted post-op type.
enum post_op_type { sum = 0, eltwise, binary };

struct post_ops_ok_args_t {
    // The two trailing arguments are rv64-only:
    // @param n_vaux - how many vector aux groups the host supplies to the
    // eltwise injector static params; the heavy eltwise algs (log/soft_relu/
    // gelu_erf) require n_vaux >= 4 (x64 spills to stack and has no such
    // budget).
    // @param allow_binary_select - opts a host into the select (ternary)
    // post-op. The injected select consumes the v0 mask and reuses the shared
    // rhs helper for the condition load (rv64 cannot stack-spill a vector
    // like x64's push_vmm), so only a host that keeps v0 dead across the
    // inject point and audits the extra rhs pointer-array slot enables it
    // (x64 has no such gate).
    post_ops_ok_args_t(const cpu_isa_t isa,
            const std::vector<post_op_type> &accepted_post_op_types,
            const post_ops_t &post_ops, const memory_desc_wrapper *dst_d,
            const bool sum_at_pos_0_only, const bool sum_requires_scale_one,
            const bool sum_requires_zp_zero = true,
            const bool sum_requires_same_params = true,
            const bcast_set_t &enabled_bcast_strategy = default_strategies(),
            const int n_vaux = 3, const bool allow_binary_select = false);

    const cpu_isa_t isa;
    const std::vector<post_op_type> &accepted_post_op_types;
    const post_ops_t &post_ops;
    const memory_desc_wrapper *dst_d;
    const bool sum_at_pos_0_only;
    const bool sum_requires_scale_one;
    const bool sum_requires_zp_zero;
    const bool sum_requires_same_params;
    const bcast_set_t enabled_bcast_strategy;
    const int n_vaux;
    const bool allow_binary_select;
};

} // namespace injector
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
