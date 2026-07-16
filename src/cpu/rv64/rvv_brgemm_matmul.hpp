/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#ifndef CPU_RV64_RVV_BRGEMM_MATMUL_HPP
#define CPU_RV64_RVV_BRGEMM_MATMUL_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

#include "cpu/rv64/brgemm/brgemm.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_uni_postops_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

struct jit_pack_a_tile_t;

struct rvv_brgemm_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("brgemm:", isa_, ""), rvv_brgemm_matmul_t);

        status_t init(const engine_t *engine);

        std::shared_ptr<brgemm_kernel_t> brg_kernel_;
        std::shared_ptr<jit_pack_a_tile_t> pack_kernel_;
        std::shared_ptr<jit_uni_postops_kernel_t> postops_kernel_;

        dim_t M_ = 0;
        dim_t N_ = 0;
        dim_t K_ = 0;
        dim_t batch_ = 0;
        bool weights_are_broadcast_ = false;
        // Input element size in bytes (4=f32, 2=bf16/f16, 1=int8). dst is f32 or s32.
        int input_typesize_ = 4;
        // Kernel isa from the input dtype (f32->v / f16->zvfh / bf16->zvfbfwma);
        // drives the impl name (brgemm:rvv / brgemm:rvv_zvfh / ..._zvfbfwma).
        cpu_isa_t isa_ = v;

    private:
        void init_scratchpad();

        static bool is_row_major(const memory_desc_wrapper &mdw) {
            const int ndims = mdw.ndims();
            if (ndims < 2) return false;
            const auto &strides = mdw.blocking_desc().strides;
            if (strides[ndims - 1] != 1) return false;
            dim_t expected_stride = mdw.dims()[ndims - 1];
            for (int d = ndims - 2; d >= 0; --d) {
                if (strides[d] != expected_stride) return false;
                expected_stride *= mdw.dims()[d];
            }
            return true;
        }
    };

    rvv_brgemm_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    ~rvv_brgemm_matmul_t() override = default;

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    status_t init(engine_t *engine) override { return status::success; }

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
};

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_BRGEMM_MATMUL_HPP
