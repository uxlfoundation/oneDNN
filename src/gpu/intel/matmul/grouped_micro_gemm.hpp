/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP
#define GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP

#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_types.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "common/utils.hpp"
#include "gemmstone/microkernel/package.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/matmul/grouped_post_ops_gen.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

#include <array>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

// Which axis of the problem is partitioned by the grouped offsets array.
//   m_axis: src [total_M, K] and dst [total_M, N] grouped, wei dense 3D
//           [G, K, N]. The token count varies per expert (MoE forward).
//           Called 2Dx3D by the common layer.
//   k_axis: src [M, total_K] and wei [total_K, N] grouped with the same
//           partition, dst dense 3D [G, M, N]. The contraction dim varies
//           per group (MoE backward).
enum class grouped_axis_t { undefined = -1, m_axis, k_axis, n_axis };

struct grouped_micro_params_t
    : trivially_serializable_t<grouped_micro_params_t> {

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names
                = {"grouped_micro_gemm_m_axis", "grouped_micro_gemm_k_axis"};
        return kernel_names;
    }

    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        compute::kernel_ctx_t kernel_ctx;
        CHECK(get_kernel_ctx(kernel_ctx));
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
        return status;
    }

    status_t get_kernel_ctx(compute::kernel_ctx_t &) const;
};

struct grouped_micro_gemm_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;
        const char *kernel_name_ = "grouped_gemm:micro";
        DECLARE_COMMON_PD_T(kernel_name_, grouped_micro_gemm_t);

        status_t init(const impl::engine_t *engine);
        status_t init_m_axis(const impl::engine_t *engine);
        status_t init_k_axis(const impl::engine_t *engine);
        status_t init_microkernels(const impl::engine_t *engine);
        status_t init_kernel_ctx_m_axis();
        status_t init_kernel_ctx_k_axis();

        bool transc() const {
            if (dst_md()->format_kind != dnnl_format_kind_sparse)
                return gemm_desc_t::get_trans(*dst_md()) == transpose::trans;
            else
                return false;
        }

        grouped_axis_t grouped_axis_ = grouped_axis_t::undefined;

        bool is_gemv_ = false;
        bool with_post_op_ = false;
        po_kind_t po_chain_[3]
                = {po_kind_t::none, po_kind_t::none, po_kind_t::none};
        data_type_t binary_scale_dts_[2] = {data_type::undef, data_type::undef};
        int sg_size_ = 0;
        int strategyGRFs_ = 0;
        dim_t ngroups_ = 0;
        std::array<int, 2> src_group_sizes_ = {0, 0};
        std::array<int, 3> wei_group_sizes_ = {0, 0, 0};
        quantization_t src_quant_;
        quantization_t wei_quant_;
        gemmstone::microkernel::Package gemm_;
        compute::kernel_ctx_t kernel_ctx_;
    };
    status_t init(impl::engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

private:
    // grouped_micro_gemm_k_axis.cl, dispatched for the k_axis workload
    status_t execute_k_axis(const exec_ctx_t &ctx) const;
    // grouped_micro_gemm_m_axis.cl, dispatched for the m_axis workload
    status_t execute_m_axis(const exec_ctx_t &ctx) const;
    compute::kernel_t kernel_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
#endif // GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP
