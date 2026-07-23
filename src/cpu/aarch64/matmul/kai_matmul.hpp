/*******************************************************************************
* Copyright 2021-2026 Arm Ltd. and affiliates
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

// Forwward declare so that we can have pointers to these in kai_matmul_t::pd_t
namespace kai::ops {
struct GemmConfig;
struct GemmArgs;
struct DequantizeFloat;
struct IGemmCommon;
} // namespace kai::ops

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct kai_matmul_t : public primitive_t {
    struct pd_t : public cpu::matmul::cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T(impl_name(), kai_matmul_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine);
        std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm() const;
        std::unique_ptr<kai::ops::IGemmCommon> create_kai_gemm_dequant(
                const kai::ops::DequantizeFloat &) const;

        enum class kai_gemm_type { noquant, dequant, requant };
        kai_gemm_type kai_gemm_type_ = kai_gemm_type::noquant;
        bool is_dequant() const {
            return kai_gemm_type_ == kai_gemm_type::dequant;
        }

        enum class batch_mode { none, batches, multis };
        batch_mode batch_mode_ = batch_mode::none;
        bool is_batches() const { return batch_mode_ == batch_mode::batches; }
        bool is_multis() const { return batch_mode_ == batch_mode::multis; }

        std::shared_ptr<kai::ops::GemmConfig> _cfg = nullptr;
        std::shared_ptr<kai::ops::GemmArgs> _args = nullptr;
        data_type_t _kai_src_dt = data_type::undef;
        data_type_t _kai_weights_dt = data_type::undef;
        data_type_t _kai_dst_dt = data_type::undef;
        bool _fixed_format = false;
        bool _run_weight_reorder = false;
        unsigned int _ag_nbatches = 1;
        unsigned int _ag_nmulti = 1;
        bool _src_broadcast_batch_dims = false;
        bool _has_post_ops_fallback = false;
        // Rename to post_ops_fallback_ to avoid confusion with the post_ops object in the base class
        post_ops_fallback_t post_ops;

        bool swd_dt(data_type_t s, data_type_t w, data_type_t d) const;

    private:
        const char *impl_name() const {
            return _has_post_ops_fallback ? "kai+post_ops_fallback" : "kai";
        }
    };

    kai_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

}; // kai_matmul_t

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
