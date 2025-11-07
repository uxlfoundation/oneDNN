/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_MATMUL_KAI_MATMUL_HPP
#define CPU_AARCH64_MATMUL_KAI_MATMUL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/aarch64/post_ops_fallback.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"

// Forward declare so that we can have pointers to these in kai_matmul_t::pd_t.
namespace kai {
namespace ops {
// NOLINTBEGIN(readability-identifier-naming)
struct GemmArgs;
struct DequantizeFloat;
class IGemmCommon;
// NOLINTEND(readability-identifier-naming)
} // namespace ops
} // namespace kai

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct kai_matmul_t : public primitive_t {
    struct pd_t : public cpu::matmul::cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T(impl_name(), kai_matmul_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(const engine_t *engine);
        std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm() const;
        std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm_dequant(
                const kai::ops::DequantizeFloat &) const;

        bool is_dequant() const {
            return kai_gemm_type_ == kai_gemm_type::dequant;
        }
        bool is_batches() const { return batch_mode_ == batch_mode::batches; }
        bool is_multis() const { return batch_mode_ == batch_mode::multis; }
        bool fixed_format() const;
        bool kai_pack_weights() const { return kai_pack_weights_; }
        bool broadcast_src_batch() const { return broadcast_src_batch_; }
        const post_ops_fallback_t &post_ops_fallback() const {
            return post_ops_fallback_;
        }

        int kernel_maxthreads() const;

    private:
        // dequant is i8*i8->f32, requant is i8*i8->i8 (not supported in this wrapper yet)
        enum class kai_gemm_type { noquant, dequant, requant };
        kai_gemm_type kai_gemm_type_ = kai_gemm_type::noquant;

        enum class batch_mode { none, batches, multis };
        batch_mode batch_mode_ = batch_mode::none;

        std::shared_ptr<kai::ops::GemmArgs> args_ = nullptr;
        data_type_t kai_src_dt_ = data_type::undef;
        data_type_t kai_weights_dt_ = data_type::undef;
        data_type_t kai_dst_dt_ = data_type::undef;

        // KleidiAI calls its weight packing transform "pretranspose". Call it
        // kai_pack here to keep it separate from oneDNN layout selection.
        bool kai_pack_weights_ = false;

        bool broadcast_src_batch_ = false;

        post_ops_fallback_t post_ops_fallback_;

        const char *impl_name() const {
            return post_ops_fallback_.len() > 0 ? "kai+post_ops_fallback"
                                                : "kai";
        }
    };

    kai_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    post_ops_fallback_t post_ops_fallback_;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

}; // kai_matmul_t

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
