/*******************************************************************************
* Copyright 2022 IBM Corporation
* Copyright 2025 Intel Corporation
* Copyright 2025 Institute of Software, Chinese Academy of Sciences
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

#ifndef CPU_RV64_RVV_GEMM_REORDER_HPP
#define CPU_RV64_RVV_GEMM_REORDER_HPP

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace format_tag;

using bd = block_dim_t;
using ib = inner_blk_t;

struct rvv_matrixA_reorder_t : public primitive_t {

    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rvv_matrixA_reorder_t", rvv_matrixA_reorder_t);

        status_t init(
                engine_t *engine, engine_t *src_engine, engine_t *dst_engine);

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        void init_scratchpad() {}
        friend dnnl::impl::impl_list_item_t;
    };
    rvv_matrixA_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override { return status::success; }

private:
    status_t execute_body(const exec_ctx_t &ctx) const;
    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_body(ctx);
    }
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
